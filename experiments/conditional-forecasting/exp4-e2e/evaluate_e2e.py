#!/usr/bin/env python3
"""
End-to-end evaluation: Does ρ estimation compose with conditional forecasting?

Tests four conditions:
1. Baseline: No ρ provided, estimate P(A|B=YES) directly
2. Oracle: Correct market-derived ρ provided
3. E2E (2-call): Estimate ρ first, then use that estimate for P(A|B=YES)
4. E2E (1-call): Estimate ρ and P(A|B=YES) in single prompt

Key question: Do errors compound, or does the pipeline still work?

Usage: python evaluate_e2e.py
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
        max_tokens=800,
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
        print(f"Failed to parse: {text}")
        return {"error": "parse_error", "raw": text}


# =============================================================================
# PROMPTS FOR EACH CONDITION
# =============================================================================

def get_baseline_prompt(question_a: str, question_b: str) -> str:
    """Baseline: No ρ information at all."""
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

First, estimate P(A) - the unconditional probability that A occurs.
Then, estimate P(A|B=YES) - the probability that A occurs given that B resolves YES.

Think about whether these questions are related and how learning B=YES would update your belief about A.

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_oracle_prompt(question_a: str, question_b: str, rho: float) -> str:
    """Oracle: Provide correct ρ from market data."""
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

These questions have a correlation coefficient of ρ = {rho:.2f} based on historical prediction market price movements.
- ρ > 0 means they tend to move together (if one becomes more likely, so does the other)
- ρ < 0 means they move oppositely (if one becomes more likely, the other becomes less likely)
- ρ ≈ 0 means they are roughly independent

First, estimate P(A) - the unconditional probability that A occurs.
Then, estimate P(A|B=YES) - the probability that A occurs given that B resolves YES.

Use the correlation coefficient to calibrate the magnitude of your update.

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_rho_estimation_prompt(question_a: str, question_b: str) -> str:
    """Step 1 of E2E (2-call): Estimate ρ."""
    return f"""You are estimating the correlation between two prediction market questions.

Question A: "{question_a}"
Question B: "{question_b}"

Estimate the correlation coefficient (ρ) between these two questions. This measures how much knowing the outcome of one question tells you about the other:
- ρ = +1: Perfect positive correlation (if A is YES, B is definitely YES)
- ρ = 0: Independent (knowing A tells you nothing about B)
- ρ = -1: Perfect negative correlation (if A is YES, B is definitely NO)

Think about:
- Are these questions about related events?
- Would one outcome make the other more or less likely?
- Are they measuring the same underlying phenomenon?

Respond with JSON only:
{{"rho_estimate": <float from -1 to +1>, "reasoning": "<brief explanation>"}}"""


def get_conditional_with_estimated_rho_prompt(question_a: str, question_b: str, estimated_rho: float) -> str:
    """Step 2 of E2E (2-call): Use estimated ρ for conditional."""
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

Based on analysis, these questions have an estimated correlation coefficient of ρ = {estimated_rho:.2f}.
- ρ > 0 means they tend to move together (if one becomes more likely, so does the other)
- ρ < 0 means they move oppositely (if one becomes more likely, the other becomes less likely)
- ρ ≈ 0 means they are roughly independent

First, estimate P(A) - the unconditional probability that A occurs.
Then, estimate P(A|B=YES) - the probability that A occurs given that B resolves YES.

Use the correlation coefficient to calibrate the magnitude of your update.

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_e2e_single_call_prompt(question_a: str, question_b: str) -> str:
    """E2E (1-call): Estimate ρ and use it in one prompt."""
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
        print(f"  A: {question_a[:60]}...")
        print(f"  B: {question_b[:60]}...")
        print(f"  True ρ: {true_rho:.2f} (expected direction: {expected_dir})")
        print("-" * 60)

        pair_result = {
            "pair_id": pair_id,
            "category": category,
            "question_a": question_a,
            "question_b": question_b,
            "true_rho": true_rho,
            "expected_direction": expected_dir,
            "conditions": {}
        }

        # -----------------------------------------------------------------
        # Condition 1: Baseline (no ρ)
        # -----------------------------------------------------------------
        print("  [Baseline]", end=" ")
        response = call_model(get_baseline_prompt(question_a, question_b))
        p_a = response.get("p_a", 0.5)
        p_a_given_b = response.get("p_a_given_b_yes", 0.5)
        pred_dir = get_direction(p_a, p_a_given_b)

        pair_result["conditions"]["baseline"] = {
            "p_a": p_a,
            "p_a_given_b_yes": p_a_given_b,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
            "rho_used": None,
            "reasoning": response.get("reasoning", "")
        }
        print(f"P(A)={p_a:.2f} → P(A|B=YES)={p_a_given_b:.2f} [{pred_dir}] {'✓' if pred_dir == expected_dir else '✗'}")

        # -----------------------------------------------------------------
        # Condition 2: Oracle (correct ρ)
        # -----------------------------------------------------------------
        print("  [Oracle]  ", end=" ")
        response = call_model(get_oracle_prompt(question_a, question_b, true_rho))
        p_a = response.get("p_a", 0.5)
        p_a_given_b = response.get("p_a_given_b_yes", 0.5)
        pred_dir = get_direction(p_a, p_a_given_b)

        pair_result["conditions"]["oracle"] = {
            "p_a": p_a,
            "p_a_given_b_yes": p_a_given_b,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
            "rho_used": true_rho,
            "reasoning": response.get("reasoning", "")
        }
        print(f"P(A)={p_a:.2f} → P(A|B=YES)={p_a_given_b:.2f} [{pred_dir}] {'✓' if pred_dir == expected_dir else '✗'}")

        # -----------------------------------------------------------------
        # Condition 3: E2E 2-call (estimate ρ, then use it)
        # -----------------------------------------------------------------
        print("  [E2E 2-call]", end=" ")

        # Step 1: Estimate ρ
        rho_response = call_model(get_rho_estimation_prompt(question_a, question_b))
        estimated_rho = rho_response.get("rho_estimate", 0.0)
        rho_error = estimated_rho - true_rho

        # Step 2: Use estimated ρ for conditional
        response = call_model(get_conditional_with_estimated_rho_prompt(question_a, question_b, estimated_rho))
        p_a = response.get("p_a", 0.5)
        p_a_given_b = response.get("p_a_given_b_yes", 0.5)
        pred_dir = get_direction(p_a, p_a_given_b)

        pair_result["conditions"]["e2e_2call"] = {
            "p_a": p_a,
            "p_a_given_b_yes": p_a_given_b,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
            "rho_used": estimated_rho,
            "rho_error": rho_error,
            "rho_reasoning": rho_response.get("reasoning", ""),
            "reasoning": response.get("reasoning", "")
        }
        print(f"ρ̂={estimated_rho:.2f} (err={rho_error:+.2f}) → P(A)={p_a:.2f} → P(A|B=YES)={p_a_given_b:.2f} [{pred_dir}] {'✓' if pred_dir == expected_dir else '✗'}")

        # -----------------------------------------------------------------
        # Condition 4: E2E 1-call (estimate ρ and use it in one prompt)
        # -----------------------------------------------------------------
        print("  [E2E 1-call]", end=" ")
        response = call_model(get_e2e_single_call_prompt(question_a, question_b))
        estimated_rho = response.get("rho_estimate", 0.0)
        rho_error = estimated_rho - true_rho
        p_a = response.get("p_a", 0.5)
        p_a_given_b = response.get("p_a_given_b_yes", 0.5)
        pred_dir = get_direction(p_a, p_a_given_b)

        pair_result["conditions"]["e2e_1call"] = {
            "p_a": p_a,
            "p_a_given_b_yes": p_a_given_b,
            "predicted_direction": pred_dir,
            "direction_correct": pred_dir == expected_dir,
            "rho_used": estimated_rho,
            "rho_error": rho_error,
            "reasoning": response.get("reasoning", "")
        }
        print(f"ρ̂={estimated_rho:.2f} (err={rho_error:+.2f}) → P(A)={p_a:.2f} → P(A|B=YES)={p_a_given_b:.2f} [{pred_dir}] {'✓' if pred_dir == expected_dir else '✗'}")

        results.append(pair_result)

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    conditions = ["baseline", "oracle", "e2e_2call", "e2e_1call"]

    summary = {}
    for cond in conditions:
        direction_correct = sum(1 for r in results if r["conditions"][cond]["direction_correct"])
        direction_accuracy = direction_correct / len(results)

        # Compute MAE of probability shift vs expected shift
        # Expected shift ≈ ρ * 0.3 (simplified heuristic from v2)
        mae_values = []
        for r in results:
            c = r["conditions"][cond]
            actual_shift = c["p_a_given_b_yes"] - c["p_a"]
            expected_shift = r["true_rho"] * 0.3
            mae_values.append(abs(actual_shift - expected_shift))

        mae = np.mean(mae_values)

        summary[cond] = {
            "direction_accuracy": direction_accuracy,
            "direction_correct": direction_correct,
            "direction_total": len(results),
            "mae": mae
        }

        print(f"\n{cond}:")
        print(f"  Direction accuracy: {direction_accuracy:.1%} ({direction_correct}/{len(results)})")
        print(f"  MAE (shift vs expected): {mae:.3f}")

    # E2E-specific analysis: breakdown by ρ estimation accuracy
    print("\n" + "-" * 70)
    print("E2E ANALYSIS: Breakdown by ρ estimation accuracy")
    print("-" * 70)

    for cond in ["e2e_2call", "e2e_1call"]:
        good_rho = [r for r in results if abs(r["conditions"][cond].get("rho_error", 999)) < 0.3]
        bad_rho = [r for r in results if abs(r["conditions"][cond].get("rho_error", 999)) >= 0.3]

        if good_rho:
            good_acc = sum(1 for r in good_rho if r["conditions"][cond]["direction_correct"]) / len(good_rho)
        else:
            good_acc = 0

        if bad_rho:
            bad_acc = sum(1 for r in bad_rho if r["conditions"][cond]["direction_correct"]) / len(bad_rho)
        else:
            bad_acc = 0

        print(f"\n{cond}:")
        print(f"  Good ρ estimate (|error| < 0.3): {good_acc:.1%} direction accuracy (n={len(good_rho)})")
        print(f"  Bad ρ estimate (|error| >= 0.3): {bad_acc:.1%} direction accuracy (n={len(bad_rho)})")

        summary[cond]["good_rho_accuracy"] = good_acc
        summary[cond]["good_rho_n"] = len(good_rho)
        summary[cond]["bad_rho_accuracy"] = bad_acc
        summary[cond]["bad_rho_n"] = len(bad_rho)

    # Comparison table
    print("\n" + "-" * 70)
    print("COMPARISON TABLE")
    print("-" * 70)
    print(f"{'Condition':<15} {'Direction Acc':>15} {'MAE':>10}")
    print("-" * 40)
    for cond in conditions:
        s = summary[cond]
        print(f"{cond:<15} {s['direction_accuracy']:>14.1%} {s['mae']:>10.3f}")

    # Key question: does E2E beat baseline?
    print("\n" + "-" * 70)
    print("KEY COMPARISONS")
    print("-" * 70)
    baseline_acc = summary["baseline"]["direction_accuracy"]
    oracle_acc = summary["oracle"]["direction_accuracy"]
    e2e_2call_acc = summary["e2e_2call"]["direction_accuracy"]
    e2e_1call_acc = summary["e2e_1call"]["direction_accuracy"]

    print(f"E2E 2-call vs Baseline: {e2e_2call_acc:.1%} vs {baseline_acc:.1%} ({'+' if e2e_2call_acc > baseline_acc else ''}{(e2e_2call_acc - baseline_acc)*100:.1f}pp)")
    print(f"E2E 2-call vs Oracle:   {e2e_2call_acc:.1%} vs {oracle_acc:.1%} ({'+' if e2e_2call_acc > oracle_acc else ''}{(e2e_2call_acc - oracle_acc)*100:.1f}pp)")
    print(f"E2E 1-call vs Baseline: {e2e_1call_acc:.1%} vs {baseline_acc:.1%} ({'+' if e2e_1call_acc > baseline_acc else ''}{(e2e_1call_acc - baseline_acc)*100:.1f}pp)")
    print(f"E2E 1-call vs Oracle:   {e2e_1call_acc:.1%} vs {oracle_acc:.1%} ({'+' if e2e_1call_acc > oracle_acc else ''}{(e2e_1call_acc - oracle_acc)*100:.1f}pp)")

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

    output_path = Path(__file__).parent.parent / "results" / f"evaluation_e2e_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_evaluation()
