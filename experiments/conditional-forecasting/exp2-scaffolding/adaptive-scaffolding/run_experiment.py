#!/usr/bin/env python3
"""Adaptive scaffolding experiment.

Tests whether routing pairs to different scaffolds based on context improves
overall performance compared to using a single scaffold uniformly.

Usage:
    uv run python experiments/fb-conditional/scaffolding/adaptive-scaffolding/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/adaptive-scaffolding/run_experiment.py --limit 5
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

import litellm


# ============================================================================
# PROMPTS
# ============================================================================

BASELINE_PROMPT = """Question: "{q_a}"

What is the probability this resolves YES? Give only your estimate, no explanation.

Return only JSON: {{"p_a": 0.XX}}"""


BRACKET_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Step 1: Relationship
Are these questions positively correlated, negatively correlated, or independent?
- Positive: A=YES makes B=YES more likely (or vice versa)
- Negative: A=YES makes B=YES less likely (or vice versa)
- Independent: Knowing one tells you nothing about the other

Step 2: Probabilities
Give your estimates for:
- P(A): probability A resolves YES
- P(A|B=YES): probability A resolves YES, given B resolved YES
- P(A|B=NO): probability A resolves YES, given B resolved NO

CONSTRAINT: If you said "positive" or "negative", your P(A) MUST fall between P(A|B=YES) and P(A|B=NO). If "independent", all three should be approximately equal.

Think through the relationship carefully before giving numbers.

Return only valid JSON:
{{
  "direction": "positive|negative|independent",
  "mechanism": "one sentence explaining why",
  "p_a": 0.XX,
  "p_a_given_b1": 0.XX,
  "p_a_given_b0": 0.XX
}}"""


TWOSTAGE_CLASSIFY_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Most question pairs are UNRELATED. Before assuming a connection, consider:

1. Are these about the same entity, event, or domain?
2. Is there a DIRECT causal mechanism linking outcomes?
3. Would a subject matter expert see an obvious connection?

If you cannot identify a clear, specific mechanism—not just thematic similarity—classify as independent.

Classification:
- "correlated": Clear causal or logical link exists
- "independent": No meaningful connection, or only superficial similarity

Return only JSON: {{"classification": "correlated|independent", "reasoning": "one sentence"}}"""


JOINT_TABLE_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Estimate the joint probability distribution over outcomes. All four cells must sum to 1.0.

        B=YES    B=NO
A=YES   [  ]     [  ]
A=NO    [  ]     [  ]

Think carefully about how A and B relate before filling in values.

Return only valid JSON: {{"a1_b1": 0.XX, "a1_b0": 0.XX, "a0_b1": 0.XX, "a0_b0": 0.XX}}"""


# ============================================================================
# LLM HELPERS
# ============================================================================

async def call_llm(prompt: str, model: str, thinking: bool) -> str | None:
    """Make an LLM call and return the content."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["temperature"] = 1
    else:
        kwargs["temperature"] = 0.3

    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"    LLM Error: {e}")
        return None


def extract_json(content: str) -> dict | None:
    """Extract JSON from response content."""
    if not content:
        return None
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
    return None


# ============================================================================
# SCAFFOLD IMPLEMENTATIONS
# ============================================================================

async def run_baseline(q_a: str, q_b: str, model: str, thinking: bool) -> dict:
    """Run baseline direct elicitation."""
    prompt = BASELINE_PROMPT.format(q_a=q_a)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "p_a" in result:
        return {
            "scaffold": "baseline",
            "p_a": result["p_a"],
            "p_a_given_b1": None,
            "p_a_given_b0": None,
            "direction": None,
            "success": True,
        }
    return {"scaffold": "baseline", "success": False}


async def run_bracket(q_a: str, q_b: str, model: str, thinking: bool) -> dict:
    """Run bracket elicitation."""
    prompt = BRACKET_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["direction", "p_a", "p_a_given_b1", "p_a_given_b0"]):
        return {
            "scaffold": "bracket",
            "p_a": result["p_a"],
            "p_a_given_b1": result["p_a_given_b1"],
            "p_a_given_b0": result["p_a_given_b0"],
            "direction": result["direction"].lower().strip(),
            "mechanism": result.get("mechanism"),
            "success": True,
        }
    return {"scaffold": "bracket", "success": False}


async def run_two_stage(q_a: str, q_b: str, model: str, thinking: bool) -> dict:
    """Run two-stage elicitation (classify then route)."""
    # Stage 1: Classification
    classify_prompt = TWOSTAGE_CLASSIFY_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(classify_prompt, model, thinking)
    classify_result = extract_json(content)

    if not classify_result or "classification" not in classify_result:
        return {"scaffold": "two_stage", "success": False}

    classification = classify_result["classification"].lower().strip()

    # Stage 2: Route based on classification
    if classification == "independent":
        baseline_result = await run_baseline(q_a, q_b, model, thinking)
        if baseline_result["success"]:
            return {
                "scaffold": "two_stage",
                "routed_to": "baseline",
                "classification": classification,
                "classification_reasoning": classify_result.get("reasoning"),
                "p_a": baseline_result["p_a"],
                "p_a_given_b1": None,
                "p_a_given_b0": None,
                "direction": None,
                "success": True,
            }
    else:
        bracket_result = await run_bracket(q_a, q_b, model, thinking)
        if bracket_result["success"]:
            return {
                "scaffold": "two_stage",
                "routed_to": "bracket",
                "classification": classification,
                "classification_reasoning": classify_result.get("reasoning"),
                "p_a": bracket_result["p_a"],
                "p_a_given_b1": bracket_result["p_a_given_b1"],
                "p_a_given_b0": bracket_result["p_a_given_b0"],
                "direction": bracket_result["direction"],
                "success": True,
            }

    return {"scaffold": "two_stage", "success": False}


async def run_joint_table(q_a: str, q_b: str, model: str, thinking: bool) -> dict:
    """Run joint table elicitation."""
    prompt = JOINT_TABLE_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["a1_b1", "a1_b0", "a0_b1", "a0_b0"]):
        a1_b1 = result["a1_b1"]
        a1_b0 = result["a1_b0"]
        a0_b1 = result["a0_b1"]
        a0_b0 = result["a0_b0"]

        # Derive probabilities
        p_a = a1_b1 + a1_b0
        p_b = a1_b1 + a0_b1
        p_a_given_b1 = a1_b1 / (a1_b1 + a0_b1) if (a1_b1 + a0_b1) > 0.001 else 0.5
        p_a_given_b0 = a1_b0 / (a1_b0 + a0_b0) if (a1_b0 + a0_b0) > 0.001 else 0.5

        # Infer direction
        diff = p_a_given_b1 - p_a_given_b0
        if abs(diff) < 0.02:
            direction = "independent"
        elif diff > 0:
            direction = "positive"
        else:
            direction = "negative"

        return {
            "scaffold": "joint_table",
            "table": result,
            "table_valid": abs(a1_b1 + a1_b0 + a0_b1 + a0_b0 - 1.0) <= 0.01,
            "p_a": p_a,
            "p_b": p_b,
            "p_a_given_b1": p_a_given_b1,
            "p_a_given_b0": p_a_given_b0,
            "direction": direction,
            "success": True,
        }
    return {"scaffold": "joint_table", "success": False}


# ============================================================================
# ADAPTIVE ROUTING
# ============================================================================

Context = Literal["general", "trading", "research"]


async def run_adaptive(
    q_a: str, q_b: str, model: str, thinking: bool, context: Context
) -> dict:
    """Run adaptive routing based on context."""

    # Decision tree from Coherence Scaffolding Experiments:
    # - General: Route to Bracket (best overall tradeoff)
    # - Trading (FP costly): Route to Two-Stage (0% FP)
    # - Research (coherence required): Route to Joint Table (100% coherent)

    if context == "general":
        result = await run_bracket(q_a, q_b, model, thinking)
        result["context"] = context
        result["routing_reason"] = "default_bracket"
        return result

    elif context == "trading":
        result = await run_two_stage(q_a, q_b, model, thinking)
        result["context"] = context
        result["routing_reason"] = "fp_costly"
        return result

    elif context == "research":
        result = await run_joint_table(q_a, q_b, model, thinking)
        result["context"] = context
        result["routing_reason"] = "coherence_required"
        return result

    return {"scaffold": "adaptive", "context": context, "success": False}


# ============================================================================
# METRICS
# ============================================================================

def compute_brier(result: dict, resolution_a: float, resolution_b: float) -> float | None:
    """Compute Brier score based on scaffold type."""
    if not result.get("success"):
        return None

    p_a = result.get("p_a")
    if p_a is None:
        return None

    # For scaffolds with conditionals, use P(A|B=actual)
    if result.get("p_a_given_b1") is not None:
        p_a_given_b_actual = result["p_a_given_b1"] if resolution_b == 1.0 else result["p_a_given_b0"]
        return (p_a_given_b_actual - resolution_a) ** 2
    else:
        # Baseline uses P(A) directly
        return (p_a - resolution_a) ** 2


def compute_brier_improvement(result: dict, resolution_a: float, resolution_b: float) -> float | None:
    """Compute Brier improvement over independence assumption."""
    if not result.get("success"):
        return None

    p_a = result.get("p_a")
    if p_a is None:
        return None

    brier_independence = (p_a - resolution_a) ** 2
    brier_actual = compute_brier(result, resolution_a, resolution_b)

    if brier_actual is None:
        return None

    return brier_independence - brier_actual


def check_direction_correct(result: dict) -> bool | None:
    """Check if stated direction matches probability pattern."""
    if not result.get("success"):
        return None

    direction = result.get("direction")
    p_a_given_b1 = result.get("p_a_given_b1")
    p_a_given_b0 = result.get("p_a_given_b0")

    if direction is None or p_a_given_b1 is None or p_a_given_b0 is None:
        return None

    diff = p_a_given_b1 - p_a_given_b0

    if abs(diff) < 0.02:
        return direction == "independent"
    elif diff > 0:
        return direction == "positive"
    else:
        return direction == "negative"


def check_false_positive(result: dict, category: str) -> bool:
    """Check if this is a false positive (none pair with spurious correlation)."""
    if category != "none":
        return False

    if not result.get("success"):
        return False

    p_a_given_b1 = result.get("p_a_given_b1")
    p_a_given_b0 = result.get("p_a_given_b0")

    if p_a_given_b1 is None or p_a_given_b0 is None:
        return False

    sensitivity = abs(p_a_given_b1 - p_a_given_b0)
    return sensitivity > 0.05


def check_bayes_consistent(result: dict) -> bool | None:
    """Check if result is Bayes-consistent (for joint table)."""
    if not result.get("success"):
        return None

    # Joint table is always Bayes-consistent by construction
    if result.get("scaffold") == "joint_table":
        return True

    # For bracket, check if P(A) is between conditionals
    p_a = result.get("p_a")
    p_a_given_b1 = result.get("p_a_given_b1")
    p_a_given_b0 = result.get("p_a_given_b0")

    if p_a is None or p_a_given_b1 is None or p_a_given_b0 is None:
        return None

    min_cond = min(p_a_given_b1, p_a_given_b0)
    max_cond = max(p_a_given_b1, p_a_given_b0)

    return min_cond <= p_a <= max_cond


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

async def run_all_conditions(pair: dict, model: str, thinking: bool) -> dict:
    """Run all conditions on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]
    category = pair.get("category")
    resolution_a = pair["resolution_a"]
    resolution_b = pair["resolution_b"]

    # Run all conditions
    results = {}

    # Uniform conditions
    results["baseline"] = await run_baseline(q_a, q_b, model, thinking)
    results["bracket"] = await run_bracket(q_a, q_b, model, thinking)
    results["two_stage"] = await run_two_stage(q_a, q_b, model, thinking)

    # Adaptive conditions (three contexts)
    results["adaptive_general"] = await run_adaptive(q_a, q_b, model, thinking, "general")
    results["adaptive_trading"] = await run_adaptive(q_a, q_b, model, thinking, "trading")
    results["adaptive_research"] = await run_adaptive(q_a, q_b, model, thinking, "research")

    # Compute metrics for each condition
    for cond_name, cond_result in results.items():
        if cond_result.get("success"):
            cond_result["brier"] = compute_brier(cond_result, resolution_a, resolution_b)
            cond_result["brier_improvement"] = compute_brier_improvement(cond_result, resolution_a, resolution_b)
            cond_result["direction_correct"] = check_direction_correct(cond_result)
            cond_result["false_positive"] = check_false_positive(cond_result, category)
            cond_result["bayes_consistent"] = check_bayes_consistent(cond_result)

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": category,
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": resolution_a,
        "resolution_b": resolution_b,
        "conditions": results,
    }


# ============================================================================
# ANALYSIS AND OUTPUT
# ============================================================================

def aggregate_metrics(results: list[dict], condition: str) -> dict:
    """Aggregate metrics for a single condition across all pairs."""
    valid = [r for r in results if r["conditions"].get(condition, {}).get("success")]

    if not valid:
        return {"n": 0}

    by_category = {"strong": [], "weak": [], "none": []}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat in by_category:
            by_category[cat].append(r)

    metrics = {"n": len(valid)}

    # Brier by category
    for cat in ["strong", "weak", "none"]:
        cat_results = by_category[cat]
        if cat_results:
            briers = [r["conditions"][condition].get("brier_improvement", 0) or 0 for r in cat_results]
            metrics[f"brier_{cat}"] = sum(briers) / len(briers)
        else:
            metrics[f"brier_{cat}"] = None

    # Overall Brier
    all_briers = [r["conditions"][condition].get("brier_improvement", 0) or 0 for r in valid]
    metrics["brier_overall"] = sum(all_briers) / len(all_briers)

    # False positive rate (on none pairs)
    none_pairs = by_category["none"]
    if none_pairs:
        fps = sum(1 for r in none_pairs if r["conditions"][condition].get("false_positive"))
        metrics["fp_rate"] = 100 * fps / len(none_pairs)
    else:
        metrics["fp_rate"] = None

    # Direction accuracy (on strong pairs)
    strong_pairs = by_category["strong"]
    if strong_pairs:
        direction_checks = [r["conditions"][condition].get("direction_correct") for r in strong_pairs]
        correct = sum(1 for d in direction_checks if d is True)
        total = sum(1 for d in direction_checks if d is not None)
        metrics["direction_accuracy"] = 100 * correct / total if total > 0 else None
    else:
        metrics["direction_accuracy"] = None

    # Coherence rate
    coherence_checks = [r["conditions"][condition].get("bayes_consistent") for r in valid]
    coherent = sum(1 for c in coherence_checks if c is True)
    total = sum(1 for c in coherence_checks if c is not None)
    metrics["coherence_rate"] = 100 * coherent / total if total > 0 else None

    return metrics


def print_summary(results: list[dict]):
    """Print summary tables."""
    conditions = ["baseline", "bracket", "two_stage", "adaptive_general", "adaptive_trading", "adaptive_research"]

    print("\n" + "=" * 90)
    print("ADAPTIVE SCAFFOLDING EXPERIMENT RESULTS")
    print("=" * 90)

    # Aggregate metrics for each condition
    all_metrics = {cond: aggregate_metrics(results, cond) for cond in conditions}

    # Overall Performance Table
    print("\n" + "-" * 90)
    print("OVERALL PERFORMANCE")
    print("-" * 90)

    print("\n┌─────────────────────┬─────────────┬────────────────┬──────────────┬─────────┬───────────┬───────────┐")
    print("│      Condition      │ Brier (All) │ Brier (Strong) │ Brier (None) │ FP Rate │ Direction │ Coherence │")
    print("├─────────────────────┼─────────────┼────────────────┼──────────────┼─────────┼───────────┼───────────┤")

    for cond in conditions:
        m = all_metrics[cond]
        if m["n"] == 0:
            print(f"│ {cond:19} │ {'N/A':>11} │ {'N/A':>14} │ {'N/A':>12} │ {'N/A':>7} │ {'N/A':>9} │ {'N/A':>9} │")
            continue

        brier_all = f"{m['brier_overall']:+.3f}" if m.get("brier_overall") is not None else "N/A"
        brier_strong = f"{m['brier_strong']:+.3f}" if m.get("brier_strong") is not None else "N/A"
        brier_none = f"{m['brier_none']:+.3f}" if m.get("brier_none") is not None else "N/A"
        fp_rate = f"{m['fp_rate']:.0f}%" if m.get("fp_rate") is not None else "N/A"
        direction = f"{m['direction_accuracy']:.0f}%" if m.get("direction_accuracy") is not None else "N/A"
        coherence = f"{m['coherence_rate']:.0f}%" if m.get("coherence_rate") is not None else "N/A"

        print(f"│ {cond:19} │ {brier_all:>11} │ {brier_strong:>14} │ {brier_none:>12} │ {fp_rate:>7} │ {direction:>9} │ {coherence:>9} │")

    print("└─────────────────────┴─────────────┴────────────────┴──────────────┴─────────┴───────────┴───────────┘")

    # Routing Distribution (for adaptive conditions)
    print("\n" + "-" * 90)
    print("ROUTING DISTRIBUTION (Adaptive General = Bracket, Trading = Two-Stage, Research = Joint Table)")
    print("-" * 90)

    # Count routing by category
    by_category = {"strong": [], "weak": [], "none": []}
    for r in results:
        cat = r.get("category", "unknown")
        if cat in by_category:
            by_category[cat].append(r)

    print("\n┌───────────┬─────────────────────────────────────────────────────────────────┐")
    print("│  Category │ General→Bracket | Trading→TwoStage | Research→JointTable        │")
    print("├───────────┼─────────────────────────────────────────────────────────────────┤")

    for cat in ["strong", "weak", "none"]:
        cat_results = by_category[cat]
        n = len(cat_results)
        if n == 0:
            continue

        # Count successes for each adaptive context
        general_ok = sum(1 for r in cat_results if r["conditions"].get("adaptive_general", {}).get("success"))
        trading_ok = sum(1 for r in cat_results if r["conditions"].get("adaptive_trading", {}).get("success"))
        research_ok = sum(1 for r in cat_results if r["conditions"].get("adaptive_research", {}).get("success"))

        print(f"│ {cat:9} │ {general_ok:>2}/{n} Bracket     | {trading_ok:>2}/{n} Two-Stage   | {research_ok:>2}/{n} Joint Table       │")

    print("└───────────┴─────────────────────────────────────────────────────────────────┘")

    # Context Comparison
    print("\n" + "-" * 90)
    print("CONTEXT COMPARISON")
    print("-" * 90)

    m_general = all_metrics["adaptive_general"]
    m_trading = all_metrics["adaptive_trading"]
    m_research = all_metrics["adaptive_research"]

    print("\n┌───────────┬─────────┬──────────────────────┬───────────────────────┐")
    print("│  Metric   │ General │       Trading        │       Research        │")
    print("├───────────┼─────────┼──────────────────────┼───────────────────────┤")

    fp_general = f"{m_general['fp_rate']:.0f}%" if m_general.get("fp_rate") is not None else "N/A"
    fp_trading = f"{m_trading['fp_rate']:.0f}% (target: lowest)" if m_trading.get("fp_rate") is not None else "N/A"
    fp_research = f"{m_research['fp_rate']:.0f}%" if m_research.get("fp_rate") is not None else "N/A"
    print(f"│ FP Rate   │ {fp_general:>7} │ {fp_trading:>20} │ {fp_research:>21} │")

    coh_general = f"{m_general['coherence_rate']:.0f}%" if m_general.get("coherence_rate") is not None else "N/A"
    coh_trading = f"{m_trading['coherence_rate']:.0f}%" if m_trading.get("coherence_rate") is not None else "N/A"
    coh_research = f"{m_research['coherence_rate']:.0f}% (target: highest)" if m_research.get("coherence_rate") is not None else "N/A"
    print(f"│ Coherence │ {coh_general:>7} │ {coh_trading:>20} │ {coh_research:>21} │")

    brier_general = f"{m_general['brier_overall']:+.3f}" if m_general.get("brier_overall") is not None else "N/A"
    brier_trading = f"{m_trading['brier_overall']:+.3f}" if m_trading.get("brier_overall") is not None else "N/A"
    brier_research = f"{m_research['brier_overall']:+.3f}" if m_research.get("brier_overall") is not None else "N/A"
    print(f"│ Brier     │ {brier_general:>7} │ {brier_trading:>20} │ {brier_research:>21} │")

    print("└───────────┴─────────┴──────────────────────┴───────────────────────┘")


def print_analysis(results: list[dict]):
    """Print analysis answering key questions."""
    conditions = ["baseline", "bracket", "two_stage", "adaptive_general", "adaptive_trading", "adaptive_research"]
    all_metrics = {cond: aggregate_metrics(results, cond) for cond in conditions}

    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    # 1. Is adaptive better than uniform?
    m_baseline = all_metrics["baseline"]
    m_bracket = all_metrics["bracket"]
    m_adaptive = all_metrics["adaptive_general"]

    print("\n1. IS ADAPTIVE BETTER THAN UNIFORM?")
    print(f"   Baseline Brier: {m_baseline.get('brier_overall', 'N/A'):+.3f}" if m_baseline.get("brier_overall") else "   Baseline Brier: N/A")
    print(f"   Bracket Brier:  {m_bracket.get('brier_overall', 'N/A'):+.3f}" if m_bracket.get("brier_overall") else "   Bracket Brier: N/A")
    print(f"   Adaptive (General) Brier: {m_adaptive.get('brier_overall', 'N/A'):+.3f}" if m_adaptive.get("brier_overall") else "   Adaptive Brier: N/A")

    # Find best
    briers = {
        "baseline": m_baseline.get("brier_overall"),
        "bracket": m_bracket.get("brier_overall"),
        "adaptive_general": m_adaptive.get("brier_overall"),
    }
    valid_briers = {k: v for k, v in briers.items() if v is not None}
    if valid_briers:
        best = max(valid_briers, key=valid_briers.get)
        print(f"   → Best: {best} ({valid_briers[best]:+.3f})")

    # 2. Do contexts matter?
    m_trading = all_metrics["adaptive_trading"]
    m_research = all_metrics["adaptive_research"]

    print("\n2. DO CONTEXTS MATTER?")
    print(f"   Trading FP rate: {m_trading.get('fp_rate', 'N/A'):.0f}%" if m_trading.get("fp_rate") else "   Trading FP rate: N/A")
    print(f"   General FP rate: {m_adaptive.get('fp_rate', 'N/A'):.0f}%" if m_adaptive.get("fp_rate") else "   General FP rate: N/A")

    if m_trading.get("fp_rate") is not None and m_adaptive.get("fp_rate") is not None:
        if m_trading["fp_rate"] < m_adaptive["fp_rate"]:
            print("   → YES, Trading context achieves lower FP rate")
        else:
            print("   → NO, Trading context doesn't reduce FP")

    print(f"   Research coherence: {m_research.get('coherence_rate', 'N/A'):.0f}%" if m_research.get("coherence_rate") else "   Research coherence: N/A")
    print(f"   General coherence: {m_adaptive.get('coherence_rate', 'N/A'):.0f}%" if m_adaptive.get("coherence_rate") else "   General coherence: N/A")

    if m_research.get("coherence_rate") is not None and m_adaptive.get("coherence_rate") is not None:
        if m_research["coherence_rate"] > m_adaptive["coherence_rate"]:
            print("   → YES, Research context achieves higher coherence")
        else:
            print("   → NO, Research context doesn't improve coherence")

    # 3. Routing accuracy
    print("\n3. ROUTING ACCURACY")
    print("   Optimal routing (post-hoc):")
    print("   - Strong → Bracket (want 100% direction)")
    print("   - Weak → Bracket (want direction benefit)")
    print("   - None → Baseline (want lowest FP)")
    print("   Adaptive General routes ALL to Bracket, so:")

    by_category = {"strong": [], "weak": [], "none": []}
    for r in results:
        cat = r.get("category", "unknown")
        if cat in by_category:
            by_category[cat].append(r)

    strong_n = len(by_category["strong"])
    weak_n = len(by_category["weak"])
    none_n = len(by_category["none"])
    total = strong_n + weak_n + none_n

    # General routes all to Bracket
    optimal_for_bracket = strong_n + weak_n  # Strong and Weak should go to Bracket
    optimal_for_baseline = none_n  # None should go to Baseline

    print(f"   - Strong/Weak → Bracket: {optimal_for_bracket}/{optimal_for_bracket} correct (100%)")
    print(f"   - None → Baseline: 0/{none_n} correct (0%)")
    print(f"   Overall routing accuracy: {optimal_for_bracket}/{total} ({100*optimal_for_bracket/total:.0f}%)")

    # 4. Cost-benefit
    print("\n4. COST-BENEFIT")
    print("   Two-Stage requires 2 LLM calls (classify + elicit)")
    print("   Bracket requires 1 LLM call")
    print("   Joint Table requires 1 LLM call")
    print(f"   Trading context FP rate: {m_trading.get('fp_rate', 'N/A'):.0f}%" if m_trading.get("fp_rate") else "   Trading FP rate: N/A")
    print(f"   Bracket FP rate: {m_bracket.get('fp_rate', 'N/A'):.0f}%" if m_bracket.get("fp_rate") else "   Bracket FP rate: N/A")

    if m_trading.get("fp_rate") is not None and m_bracket.get("fp_rate") is not None:
        fp_reduction = m_bracket["fp_rate"] - m_trading["fp_rate"]
        print(f"   → FP reduction from 2x calls: {fp_reduction:+.0f}%")

    # 5. Recommendation
    print("\n5. RECOMMENDATION")

    # Compare adaptive_general to bracket
    if m_adaptive.get("brier_overall") is not None and m_bracket.get("brier_overall") is not None:
        diff = m_adaptive["brier_overall"] - m_bracket["brier_overall"]
        if abs(diff) < 0.005:
            print("   Adaptive (General) ≈ Bracket (difference < 0.005)")
            print("   → Use Bracket for simplicity (no routing overhead)")
        elif diff > 0:
            print(f"   Adaptive (General) outperforms Bracket by {diff:+.3f}")
            print("   → Adaptive routing may be worth it for slight improvement")
        else:
            print(f"   Bracket outperforms Adaptive (General) by {-diff:+.3f}")
            print("   → Use Bracket uniformly")

    # Context-specific recommendations
    if m_trading.get("fp_rate") is not None and m_trading["fp_rate"] == 0:
        print("   → For FP-sensitive applications: Use Trading context (Two-Stage)")

    if m_research.get("coherence_rate") is not None and m_research["coherence_rate"] == 100:
        print("   → For coherence-critical applications: Use Research context (Joint Table)")


async def main():
    parser = argparse.ArgumentParser(description="Adaptive scaffolding experiment")
    parser.add_argument("--pairs", type=str, default="experiments/fb-conditional/pairs_filtered.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_false", dest="thinking")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load pairs
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        return

    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])
    if args.limit:
        pairs = pairs[:args.limit]

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"experiments/fb-conditional/scaffolding/adaptive-scaffolding/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running adaptive scaffolding experiment on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()
    print("Conditions: baseline, bracket, two_stage, adaptive_general, adaptive_trading, adaptive_research")
    print()

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(3)  # Lower concurrency since each pair runs 6 conditions

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:50]}...")
            result = await run_all_conditions(pair, args.model, args.thinking)

            # Quick summary
            successes = sum(1 for c in result["conditions"].values() if c.get("success"))
            print(f"    {successes}/6 conditions succeeded")

            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print outputs
    print_summary(results)
    print_analysis(results)

    # Save results
    output = {
        "results": results,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "model": args.model,
            "thinking": args.thinking,
            "num_pairs": len(pairs),
            "conditions": ["baseline", "bracket", "two_stage", "adaptive_general", "adaptive_trading", "adaptive_research"],
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
