#!/usr/bin/env python3
"""Anchor and Adjust experiment.

Tests whether eliciting adjustments from an anchor (e.g., "+5 percentage points")
produces better-calibrated conditional probabilities than direct absolute elicitation.

Hypothesis: Models may be better at relative adjustments than absolute probability
estimates. Anchoring on P(A) first, then asking for the delta, could improve calibration.

Usage:
    uv run python experiments/magnitude/anchor-adjust/run_experiment.py
    uv run python experiments/magnitude/anchor-adjust/run_experiment.py --limit 5
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm


# Stage 1: Independence classification (same as Two-Stage)
STAGE1_PROMPT = """Questions:
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


# Two-Stage Baseline prompt (for comparison)
TWOSTAGE_BRACKET_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

You determined these are correlated. Now estimate:

Step 1: Is the correlation positive or negative?
- Positive: A=YES makes B=YES more likely (or vice versa)
- Negative: A=YES makes B=NO more likely (or vice versa)

Step 2: Give P(A), P(A|B=YES), P(A|B=NO).

Constraint: P(A) MUST fall between P(A|B=YES) and P(A|B=NO).

Return JSON: {{
  "direction": "positive|negative",
  "p_a": 0.XX,
  "p_a_given_b1": 0.XX,
  "p_a_given_b0": 0.XX
}}"""


# Anchor and Adjust prompt
ANCHOR_ADJUST_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

You've determined these questions are correlated.

Step 1: ANCHOR
What is P(A) unconditionally? This is your anchor.

Step 2: DIRECTION
Is the correlation positive or negative?
- Positive: A=YES makes B=YES more likely
- Negative: A=YES makes B=NO more likely

Step 3: ADJUSTMENT
Starting from your anchor P(A), by how many percentage points should the probability change?

Think about this as: "If I learned B=YES, I would adjust P(A) by ___ percentage points."

Guidelines for adjustment size:
- 1-5 pp: Weak correlation (knowing B shifts belief slightly)
- 5-15 pp: Moderate correlation (meaningful but not determinative)
- 15-30 pp: Strong correlation (substantial causal/evidential link)
- 30+ pp: Very strong (one nearly determines the other)

Return JSON:
{{
  "p_a_anchor": 0.XX,
  "direction": "positive|negative",
  "adjustment_pp": X,
  "reasoning": "one sentence on adjustment size"
}}"""


# Baseline prompt (for independent pairs)
BASELINE_PROMPT = """Question: "{q_a}"

What is the probability this resolves YES? Give only your estimate, no explanation.

Return only JSON: {{"p_a": 0.XX}}"""


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


def compute_from_adjustment(p_a: float, direction: str, adjustment_pp: float) -> tuple[float, float]:
    """
    Apply symmetric adjustment to anchor to get conditionals.

    adjustment_pp: percentage points (e.g., 10 means 0.10)
    """
    shift = adjustment_pp / 100

    if direction == "positive":
        p_a_given_b1 = min(0.99, p_a + shift)
        p_a_given_b0 = max(0.01, p_a - shift)
    else:  # negative
        p_a_given_b1 = max(0.01, p_a - shift)
        p_a_given_b0 = min(0.99, p_a + shift)

    return p_a_given_b1, p_a_given_b0


async def stage1_classify(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Stage 1: Classify pair as correlated or independent."""
    prompt = STAGE1_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "classification" in result:
        result["classification"] = result["classification"].lower().strip()
        return result
    return None


async def get_baseline(q_a: str, model: str, thinking: bool) -> dict | None:
    """Get baseline P(A) for independent pairs."""
    prompt = BASELINE_PROMPT.format(q_a=q_a)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "p_a" in result:
        return {"p_a": result["p_a"]}
    return None


async def get_twostage_bracket(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Get Two-Stage baseline (bracket elicitation) for comparison."""
    prompt = TWOSTAGE_BRACKET_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["direction", "p_a", "p_a_given_b1", "p_a_given_b0"]):
        return {
            "direction": result["direction"].lower().strip(),
            "p_a": result["p_a"],
            "p_a_given_b1": result["p_a_given_b1"],
            "p_a_given_b0": result["p_a_given_b0"],
        }
    return None


async def get_anchor_adjust(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Get anchor-adjust elicitation."""
    prompt = ANCHOR_ADJUST_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["p_a_anchor", "direction", "adjustment_pp"]):
        return {
            "p_a_anchor": result["p_a_anchor"],
            "direction": result["direction"].lower().strip(),
            "adjustment_pp": float(result["adjustment_pp"]),
            "reasoning": result.get("reasoning", ""),
        }
    return None


def compute_brier(p_a_given_b_actual: float, resolution_a: float) -> float:
    """Compute Brier score."""
    return (p_a_given_b_actual - resolution_a) ** 2


async def run_pair(pair: dict, model: str, thinking: bool) -> dict:
    """Run experiment on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]
    resolution_a = pair["resolution_a"]
    resolution_b = pair["resolution_b"]

    # Stage 1: Classification
    stage1 = await stage1_classify(q_a, q_b, model, thinking)

    if stage1 is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Stage 1 classification failed",
        }

    result = {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": resolution_a,
        "resolution_b": resolution_b,
        "stage1": stage1,
    }

    # If independent, just get baseline
    if stage1["classification"] == "independent":
        baseline = await get_baseline(q_a, model, thinking)
        if baseline is None:
            result["error"] = "Baseline elicitation failed"
            return result

        result["baseline"] = {
            "p_a": baseline["p_a"],
            "p_a_given_b1": baseline["p_a"],
            "p_a_given_b0": baseline["p_a"],
        }
        result["anchor_adjust"] = None
        result["computed"] = None

        # Brier for baseline
        brier_baseline = compute_brier(baseline["p_a"], resolution_a)
        result["brier"] = {
            "baseline": brier_baseline,
            "anchor_adjust": brier_baseline,
            "improvement": 0,
        }
        return result

    # For correlated pairs, run both methods
    # 1. Two-Stage baseline (bracket)
    twostage = await get_twostage_bracket(q_a, q_b, model, thinking)
    if twostage is None:
        result["error"] = "Two-Stage bracket elicitation failed"
        return result

    result["baseline"] = twostage

    # 2. Anchor and Adjust elicitation
    anchor_adjust = await get_anchor_adjust(q_a, q_b, model, thinking)
    if anchor_adjust is None:
        result["error"] = "Anchor-adjust elicitation failed"
        return result

    result["anchor_adjust"] = anchor_adjust

    # 3. Compute conditionals from adjustment
    p_a = anchor_adjust["p_a_anchor"]
    direction = anchor_adjust["direction"]
    adjustment_pp = anchor_adjust["adjustment_pp"]

    p_a_given_b1, p_a_given_b0 = compute_from_adjustment(p_a, direction, adjustment_pp)

    result["computed"] = {
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
    }

    # 4. Compute Brier scores
    # Baseline: use Two-Stage bracket result
    baseline_p_given_actual = twostage["p_a_given_b1"] if resolution_b == 1.0 else twostage["p_a_given_b0"]
    brier_baseline = compute_brier(baseline_p_given_actual, resolution_a)

    # Anchor-adjust method
    anchor_p_given_actual = p_a_given_b1 if resolution_b == 1.0 else p_a_given_b0
    brier_anchor = compute_brier(anchor_p_given_actual, resolution_a)

    result["brier"] = {
        "baseline": brier_baseline,
        "anchor_adjust": brier_anchor,
        "improvement": brier_baseline - brier_anchor,
    }

    return result


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        print("\nNo valid results to summarize.")
        return

    print("\n" + "=" * 70)
    print("ANCHOR AND ADJUST EXPERIMENT RESULTS")
    print("=" * 70)

    # Group by category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Overall summary table
    print("\n" + "-" * 70)
    print("BRIER SCORES BY CATEGORY")
    print("-" * 70)

    print("\n| Category | N | Baseline | Anchor-Adjust | Improvement |")
    print("|----------|---|----------|---------------|-------------|")

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue
        cat_results = by_category[cat]
        n = len(cat_results)

        mean_baseline = sum(r["brier"]["baseline"] for r in cat_results) / n
        mean_anchor = sum(r["brier"]["anchor_adjust"] for r in cat_results) / n
        mean_improvement = sum(r["brier"]["improvement"] for r in cat_results) / n

        print(f"| {cat:8} | {n:2} | {mean_baseline:.4f}   | {mean_anchor:.4f}        | {mean_improvement:+.4f}      |")

    # Adjustment distribution
    print("\n" + "-" * 70)
    print("ADJUSTMENT SIZES BY CATEGORY (percentage points)")
    print("-" * 70)

    for cat in ["strong", "weak"]:
        if cat not in by_category:
            continue
        cat_results = [r for r in by_category[cat] if r["anchor_adjust"] is not None]
        if not cat_results:
            continue

        adjustments = [r["anchor_adjust"]["adjustment_pp"] for r in cat_results]
        mean_adj = sum(adjustments) / len(adjustments)
        min_adj = min(adjustments)
        max_adj = max(adjustments)

        print(f"\n{cat.upper()} (n={len(cat_results)}):")
        print(f"  Mean adjustment: {mean_adj:.1f} pp")
        print(f"  Range: {min_adj:.0f} - {max_adj:.0f} pp")
        print(f"  Distribution: {sorted([int(a) for a in adjustments])}")


def print_analysis(results: list[dict]):
    """Print analysis answering key questions."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    strong = by_category.get("strong", [])
    weak = by_category.get("weak", [])
    none_cat = by_category.get("none", [])

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # 1. Does anchoring improve Brier?
    strong_correlated = [r for r in strong if r["anchor_adjust"] is not None]
    if strong_correlated:
        mean_imp = sum(r["brier"]["improvement"] for r in strong_correlated) / len(strong_correlated)
        wins = sum(1 for r in strong_correlated if r["brier"]["improvement"] > 0)
        n = len(strong_correlated)

        print(f"\n1. DOES ANCHORING IMPROVE BRIER? (strong pairs)")
        print(f"   Mean improvement: {mean_imp:+.4f}")
        print(f"   Win rate: {wins}/{n} ({100*wins/n:.0f}%)")
        if mean_imp > 0:
            print(f"   -> YES, anchor-adjust improves by {mean_imp:+.4f} on average")
        else:
            print(f"   -> NO, anchor-adjust is {mean_imp:+.4f} worse on average")

    # 2. Do adjustment sizes discriminate?
    strong_adjustments = [r["anchor_adjust"]["adjustment_pp"] for r in strong if r["anchor_adjust"]]
    weak_adjustments = [r["anchor_adjust"]["adjustment_pp"] for r in weak if r["anchor_adjust"]]

    if strong_adjustments and weak_adjustments:
        mean_strong = sum(strong_adjustments) / len(strong_adjustments)
        mean_weak = sum(weak_adjustments) / len(weak_adjustments)

        print(f"\n2. DO ADJUSTMENT SIZES DISCRIMINATE?")
        print(f"   Mean adjustment (strong pairs): {mean_strong:.1f} pp")
        print(f"   Mean adjustment (weak pairs):   {mean_weak:.1f} pp")
        print(f"   Difference: {mean_strong - mean_weak:.1f} pp")

        if mean_strong > mean_weak:
            print(f"   -> YES, strong pairs get larger adjustments by {mean_strong - mean_weak:.1f} pp")
        else:
            print(f"   -> NO, weak pairs actually get equal or larger adjustments")

    # 3. Anchor quality
    if strong_correlated:
        pa_diffs = [abs(r["baseline"]["p_a"] - r["anchor_adjust"]["p_a_anchor"]) for r in strong_correlated]
        mean_pa_diff = sum(pa_diffs) / len(pa_diffs) if pa_diffs else 0

        print(f"\n3. ANCHOR QUALITY")
        print(f"   Mean |P(A) baseline - P(A) anchor|: {mean_pa_diff:.3f}")
        if mean_pa_diff < 0.05:
            print(f"   -> Anchors are consistent with baseline estimates")
        elif mean_pa_diff < 0.1:
            print(f"   -> Moderate divergence in anchor vs baseline P(A)")
        else:
            print(f"   -> Large divergence - anchor P(A) differs significantly")

    # 4. Asymmetric patterns (implicit in symmetric formula)
    print(f"\n4. ASYMMETRIC PATTERNS")
    print(f"   Using symmetric adjustment formula (same shift up/down)")
    print(f"   Consider testing asymmetric variant if results mixed")

    # 5. Direction accuracy
    if strong_correlated:
        direction_matches = sum(
            1 for r in strong_correlated
            if r["baseline"]["direction"] == r["anchor_adjust"]["direction"]
        )
        n = len(strong_correlated)
        print(f"\n5. DIRECTION AGREEMENT (baseline vs anchor-adjust)")
        print(f"   Agreement: {direction_matches}/{n} ({100*direction_matches/n:.0f}%)")

    # 6. Key finding
    print(f"\n6. KEY FINDING")
    if strong_correlated:
        mean_imp = sum(r["brier"]["improvement"] for r in strong_correlated) / len(strong_correlated)
        if mean_imp > 0.01:
            print(f"   -> ANCHOR-ADJUST IS BETTER: +{mean_imp:.4f} Brier improvement")
        elif mean_imp < -0.01:
            print(f"   -> BASELINE IS BETTER: {mean_imp:.4f} Brier")
        else:
            print(f"   -> NO CLEAR WINNER: {mean_imp:+.4f} Brier (essentially equivalent)")

        if strong_adjustments and weak_adjustments:
            if mean_strong > mean_weak + 3:
                print(f"   -> ADJUSTMENTS DO DISCRIMINATE: {mean_strong - mean_weak:.1f} pp difference")
            else:
                print(f"   -> ADJUSTMENTS DON'T DISCRIMINATE WELL: only {mean_strong - mean_weak:.1f} pp difference")


async def main():
    parser = argparse.ArgumentParser(description="Anchor and Adjust experiment")
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
        args.output = f"experiments/magnitude/anchor-adjust/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running anchor-adjust experiment on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(5)

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:50]}...")
            result = await run_pair(pair, args.model, args.thinking)
            if "error" not in result:
                if result["anchor_adjust"]:
                    adj = result["anchor_adjust"]["adjustment_pp"]
                    imp = result["brier"]["improvement"]
                    print(f"    Adjustment: {adj:.0f} pp, Improvement: {imp:+.4f}")
                else:
                    print(f"    Independent (no adjustment)")
            else:
                print(f"    Error: {result['error']}")
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
            "num_successful": len([r for r in results if "error" not in r]),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
