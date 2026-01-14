#!/usr/bin/env python3
"""
Enhanced E2E evaluation: Combining ρ estimation with coherence scaffolding.

Tests whether adding Bracket constraints to E2E improves performance.

Conditions:
1. E2E 1-call (baseline): ρ + conditional in one prompt
2. E2E + Bracket: ρ + direction commitment + P(A) between conditionals
3. E2E + Skeptical: Skeptical classification first, then ρ + Bracket for correlated only

Usage: python evaluate_e2e_enhanced.py
"""

import json
from pathlib import Path
from datetime import datetime
import anthropic
from dotenv import load_dotenv
import numpy as np

# Load environment from repo root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"


def load_curated_pairs():
    with open(Path(__file__).parent.parent / "data" / "curated_pairs.json") as f:
        return json.load(f)["curated_pairs"]


def call_model(prompt: str) -> dict:
    """Call model and parse JSON response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"  [PARSE ERROR] {text[:100]}...")
        return {"error": "parse_error", "raw": text}


# =============================================================================
# PROMPTS
# =============================================================================

def get_e2e_1call_prompt(question_a: str, question_b: str) -> str:
    """E2E 1-call (baseline from Exp 4): Estimate ρ and use it in one prompt."""
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

Step 1: First, estimate the correlation coefficient (ρ) between these questions.
- ρ = +1: Perfect positive correlation
- ρ = 0: Independent
- ρ = -1: Perfect negative correlation

Step 2: Then, estimate P(A) and P(A|B=YES), using your ρ estimate to calibrate the magnitude of update.

Respond with JSON only:
{{"rho_estimate": <float from -1 to +1>, "p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_e2e_bracket_prompt(question_a: str, question_b: str) -> str:
    """E2E + Bracket: ρ estimation with Bracket constraints."""
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

Step 1: Estimate the correlation coefficient (ρ) between these questions.
- ρ > 0: Positive correlation (B=YES makes A more likely)
- ρ = 0: Independent (B tells you nothing about A)
- ρ < 0: Negative correlation (B=YES makes A less likely)

Step 2: Based on your ρ, commit to a direction:
- If ρ > 0.1: direction = "positive" (P(A|B=YES) > P(A) > P(A|B=NO))
- If ρ < -0.1: direction = "negative" (P(A|B=YES) < P(A) < P(A|B=NO))
- If |ρ| <= 0.1: direction = "none" (P(A|B=YES) ≈ P(A) ≈ P(A|B=NO))

Step 3: Give P(A), P(A|B=YES), and P(A|B=NO).

CRITICAL CONSTRAINT: Your probabilities MUST be consistent with your stated direction:
- If direction = "positive": P(A|B=YES) > P(A) > P(A|B=NO)
- If direction = "negative": P(A|B=YES) < P(A) < P(A|B=NO)
- If direction = "none": All three should be approximately equal

Respond with JSON only:
{{"rho_estimate": <float -1 to +1>, "direction": "positive|negative|none", "p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "p_a_given_b_no": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_skeptical_classification_prompt(question_a: str, question_b: str) -> str:
    """Stage 1 of E2E + Skeptical: Skeptical classification."""
    return f"""You are analyzing whether two prediction market questions are related.

Question A: "{question_a}"
Question B: "{question_b}"

Most question pairs are UNRELATED. Before assuming a connection, consider:

1. Are these about the same entity, event, or domain?
2. Is there a DIRECT causal mechanism linking outcomes?
3. Would a subject matter expert see an obvious connection?

If you cannot identify a clear, specific mechanism—not just thematic similarity—classify as independent.

Classification:
- "correlated": Clear causal or logical link exists
- "independent": No meaningful connection, or only superficial similarity

Respond with JSON only:
{{"classification": "correlated|independent", "reasoning": "<one sentence>"}}"""


def get_e2e_bracket_with_rho_prompt(question_a: str, question_b: str) -> str:
    """Stage 2 of E2E + Skeptical: ρ + Bracket for correlated pairs."""
    return f"""You are a forecaster estimating conditional probabilities for questions that ARE correlated.

Question A: "{question_a}"
Question B: "{question_b}"

These questions have been identified as correlated. Now:

Step 1: Estimate the correlation coefficient (ρ).
- ρ > 0: Positive correlation (B=YES makes A more likely)
- ρ < 0: Negative correlation (B=YES makes A less likely)

Step 2: Commit to direction based on ρ sign.

Step 3: Give P(A), P(A|B=YES), and P(A|B=NO).

CRITICAL CONSTRAINT: P(A) MUST lie between P(A|B=YES) and P(A|B=NO).
- If ρ > 0: P(A|B=YES) > P(A) > P(A|B=NO)
- If ρ < 0: P(A|B=YES) < P(A) < P(A|B=NO)

Respond with JSON only:
{{"rho_estimate": <float -1 to +1>, "direction": "positive|negative", "p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "p_a_given_b_no": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_independent_baseline_prompt(question_a: str, question_b: str) -> str:
    """For independent pairs in E2E + Skeptical: just get P(A)."""
    return f"""You are a forecaster.

Question: "{question_a}"

What is the probability this resolves YES?

Respond with JSON only:
{{"p_a": <float 0-1>, "reasoning": "<brief explanation>"}}"""


# =============================================================================
# EVALUATION LOGIC
# =============================================================================

def get_direction(p_a: float, p_a_given_b: float, threshold: float = 0.03) -> str:
    """Classify direction of update."""
    diff = p_a_given_b - p_a
    if abs(diff) < threshold:
        return "none"
    return "positive" if diff > 0 else "negative"


def get_expected_direction(rho: float, threshold: float = 0.1) -> str:
    """Expected direction based on true ρ."""
    if abs(rho) < threshold:
        return "none"
    return "positive" if rho > 0 else "negative"


def check_bracket_constraint(p_a: float, p_a_b1: float, p_a_b0: float, direction: str) -> bool:
    """Check if probabilities satisfy Bracket constraint."""
    if direction == "positive":
        return p_a_b1 > p_a > p_a_b0
    elif direction == "negative":
        return p_a_b1 < p_a < p_a_b0
    else:  # none
        return abs(p_a_b1 - p_a) < 0.05 and abs(p_a_b0 - p_a) < 0.05


def check_impossible_update(p_a: float, p_a_b1: float, p_a_b0: float) -> bool:
    """Check if both conditionals move in same direction (impossible)."""
    delta_b1 = p_a_b1 - p_a
    delta_b0 = p_a_b0 - p_a
    # Both positive or both negative = impossible
    if abs(delta_b1) < 0.02 or abs(delta_b0) < 0.02:
        return False  # One is ~no update, not impossible
    return (delta_b1 > 0 and delta_b0 > 0) or (delta_b1 < 0 and delta_b0 < 0)


def run_evaluation():
    pairs = load_curated_pairs()
    results = []

    for pair in pairs:
        pair_id = pair["id"]
        question_a = pair["question_a"]
        question_b = pair["question_b"]
        true_rho = pair["rho"]
        category = pair["category"]
        expected_dir = get_expected_direction(true_rho)

        print(f"\n{'='*60}")
        print(f"Pair {pair_id}: {category}")
        print(f"  True ρ: {true_rho:.2f} (expected: {expected_dir})")
        print("-" * 60)

        pair_result = {
            "pair_id": pair_id,
            "category": category,
            "true_rho": true_rho,
            "expected_direction": expected_dir,
            "conditions": {}
        }

        # -----------------------------------------------------------------
        # Condition 1: E2E 1-call (baseline)
        # -----------------------------------------------------------------
        print("  [E2E 1-call]    ", end="")
        response = call_model(get_e2e_1call_prompt(question_a, question_b))

        rho_est = response.get("rho_estimate", 0.0)
        p_a = response.get("p_a", 0.5)
        p_a_b1 = response.get("p_a_given_b_yes", 0.5)
        pred_dir = get_direction(p_a, p_a_b1)

        pair_result["conditions"]["e2e_1call"] = {
            "rho_estimate": rho_est,
            "rho_error": rho_est - true_rho,
            "p_a": p_a,
            "p_a_given_b_yes": p_a_b1,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
        }
        status = "✓" if pred_dir == expected_dir else "✗"
        print(f"ρ̂={rho_est:.2f} P(A)={p_a:.2f}→{p_a_b1:.2f} [{pred_dir}] {status}")

        # -----------------------------------------------------------------
        # Condition 2: E2E + Bracket
        # -----------------------------------------------------------------
        print("  [E2E+Bracket]   ", end="")
        response = call_model(get_e2e_bracket_prompt(question_a, question_b))

        rho_est = response.get("rho_estimate", 0.0)
        stated_dir = response.get("direction", "none")
        p_a = response.get("p_a", 0.5)
        p_a_b1 = response.get("p_a_given_b_yes", 0.5)
        p_a_b0 = response.get("p_a_given_b_no", 0.5)
        pred_dir = get_direction(p_a, p_a_b1)

        bracket_satisfied = check_bracket_constraint(p_a, p_a_b1, p_a_b0, stated_dir)
        impossible = check_impossible_update(p_a, p_a_b1, p_a_b0)

        pair_result["conditions"]["e2e_bracket"] = {
            "rho_estimate": rho_est,
            "rho_error": rho_est - true_rho,
            "stated_direction": stated_dir,
            "p_a": p_a,
            "p_a_given_b_yes": p_a_b1,
            "p_a_given_b_no": p_a_b0,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
            "bracket_satisfied": bracket_satisfied,
            "impossible_update": impossible,
        }
        status = "✓" if pred_dir == expected_dir else "✗"
        constraint = "☐" if bracket_satisfied else "⚠"
        print(f"ρ̂={rho_est:.2f} P(A)={p_a:.2f}→{p_a_b1:.2f} [{pred_dir}] {status} {constraint}")

        # -----------------------------------------------------------------
        # Condition 3: E2E + Skeptical (two-stage)
        # -----------------------------------------------------------------
        print("  [E2E+Skeptical] ", end="")

        # Stage 1: Skeptical classification
        class_response = call_model(get_skeptical_classification_prompt(question_a, question_b))
        classification = class_response.get("classification", "independent")

        if classification == "correlated":
            # Stage 2: ρ + Bracket
            response = call_model(get_e2e_bracket_with_rho_prompt(question_a, question_b))
            rho_est = response.get("rho_estimate", 0.0)
            stated_dir = response.get("direction", "none")
            p_a = response.get("p_a", 0.5)
            p_a_b1 = response.get("p_a_given_b_yes", 0.5)
            p_a_b0 = response.get("p_a_given_b_no", 0.5)
            pred_dir = get_direction(p_a, p_a_b1)
            bracket_satisfied = check_bracket_constraint(p_a, p_a_b1, p_a_b0, stated_dir)
            impossible = check_impossible_update(p_a, p_a_b1, p_a_b0)
        else:
            # Independent: just get P(A), no update
            response = call_model(get_independent_baseline_prompt(question_a, question_b))
            rho_est = 0.0
            stated_dir = "none"
            p_a = response.get("p_a", 0.5)
            p_a_b1 = p_a  # No update
            p_a_b0 = p_a
            pred_dir = "none"
            bracket_satisfied = True
            impossible = False

        pair_result["conditions"]["e2e_skeptical"] = {
            "classification": classification,
            "rho_estimate": rho_est,
            "rho_error": rho_est - true_rho if classification == "correlated" else None,
            "stated_direction": stated_dir,
            "p_a": p_a,
            "p_a_given_b_yes": p_a_b1,
            "p_a_given_b_no": p_a_b0,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
            "bracket_satisfied": bracket_satisfied,
            "impossible_update": impossible,
        }
        status = "✓" if pred_dir == expected_dir else "✗"
        print(f"[{classification[:3]}] ρ̂={rho_est:.2f} P(A)={p_a:.2f}→{p_a_b1:.2f} [{pred_dir}] {status}")

        results.append(pair_result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    conditions = ["e2e_1call", "e2e_bracket", "e2e_skeptical"]
    summary = {}

    for cond in conditions:
        cond_results = [r["conditions"][cond] for r in results]

        direction_correct = sum(1 for c in cond_results if c["direction_correct"])
        direction_accuracy = direction_correct / len(results)

        # Bracket-specific metrics
        if cond in ["e2e_bracket", "e2e_skeptical"]:
            bracket_satisfied = sum(1 for c in cond_results if c.get("bracket_satisfied", True))
            impossible_updates = sum(1 for c in cond_results if c.get("impossible_update", False))
        else:
            bracket_satisfied = None
            impossible_updates = None

        # For skeptical: classification accuracy
        if cond == "e2e_skeptical":
            # True positives: correlated pairs classified as correlated
            correlated_pairs = [r for r in results if abs(r["true_rho"]) > 0.1]
            independent_pairs = [r for r in results if abs(r["true_rho"]) <= 0.1]

            tp = sum(1 for r in correlated_pairs if r["conditions"][cond]["classification"] == "correlated")
            fn = len(correlated_pairs) - tp
            tn = sum(1 for r in independent_pairs if r["conditions"][cond]["classification"] == "independent")
            fp = len(independent_pairs) - tn
        else:
            tp = fn = tn = fp = None

        summary[cond] = {
            "direction_accuracy": direction_accuracy,
            "direction_correct": direction_correct,
            "total": len(results),
            "bracket_satisfied": bracket_satisfied,
            "impossible_updates": impossible_updates,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp
        }

        print(f"\n{cond}:")
        print(f"  Direction accuracy: {direction_accuracy:.1%} ({direction_correct}/{len(results)})")
        if bracket_satisfied is not None:
            print(f"  Bracket satisfied: {bracket_satisfied}/{len(results)}")
            print(f"  Impossible updates: {impossible_updates}/{len(results)}")
        if tp is not None:
            print(f"  Classification: TP={tp} FN={fn} TN={tn} FP={fp}")

    # Comparison table
    print("\n" + "-" * 70)
    print("COMPARISON TABLE")
    print("-" * 70)
    print(f"{'Condition':<18} {'Dir Acc':>10} {'Bracket':>10} {'Impossible':>12}")
    print("-" * 50)
    for cond in conditions:
        s = summary[cond]
        bracket_str = f"{s['bracket_satisfied']}/{s['total']}" if s['bracket_satisfied'] is not None else "—"
        imposs_str = f"{s['impossible_updates']}/{s['total']}" if s['impossible_updates'] is not None else "—"
        print(f"{cond:<18} {s['direction_accuracy']:>9.1%} {bracket_str:>10} {imposs_str:>12}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "model": MODEL,
            "n_pairs": len(results)
        },
        "summary": summary,
        "results": results
    }

    output_path = Path(__file__).parent.parent / "results" / f"evaluation_e2e_enhanced_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    run_evaluation()
