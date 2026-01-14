#!/usr/bin/env python3
"""
Evaluate whether few-shot examples improve magnitude calibration.
Compares baseline (no examples) vs few-shot (with correlation examples).
"""

import json
import os
import random
from pathlib import Path

import litellm
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = os.environ.get("MODEL", "claude-sonnet-4-20250514")

BASELINE_PROMPT = """You are a forecaster estimating conditional probabilities.

Question A: "{q_a}"
Question B: "{q_b}"

Current market probability for A: P(A) = {p_a:.0%}
Current market probability for B: P(B) = {p_b:.0%}

If we learn that B resolves YES, what should P(A | B=YES) be?

Respond with ONLY a number between 0 and 1 (e.g., 0.65). No explanation."""


FEW_SHOT_PROMPT = """You are a forecaster estimating conditional probabilities.

Here are examples of how market correlation affects conditional updates:

{examples}

---

Now estimate for this pair:

Question A: "{q_a}"
Question B: "{q_b}"

Current market probability for A: P(A) = {p_a:.0%}
Current market probability for B: P(B) = {p_b:.0%}

If we learn that B resolves YES, what should P(A | B=YES) be?

Respond with ONLY a number between 0 and 1 (e.g., 0.65). No explanation."""


def parse_probability(text: str) -> float | None:
    """Parse a probability from LLM response."""
    text = text.strip()
    try:
        # Try direct float parse
        val = float(text.replace("%", ""))
        if val > 1:
            val /= 100
        return max(0, min(1, val))
    except ValueError:
        # Try to find a number in the text
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            val = float(match.group(1))
            if val > 1:
                val /= 100
            return max(0, min(1, val))
        return None


def expected_conditional(p_a: float, p_b: float, rho: float) -> float:
    """
    Estimate P(A|B=YES) from correlation.

    Using the approximation for binary variables:
    P(A|B) ≈ P(A) + rho * sqrt(P(A)(1-P(A)) * P(B)(1-P(B))) / P(B)

    This is a rough heuristic, not exact.
    """
    if p_b <= 0 or p_b >= 1:
        return p_a

    std_a = np.sqrt(p_a * (1 - p_a))
    std_b = np.sqrt(p_b * (1 - p_b))

    # Shift proportional to correlation
    shift = rho * std_a * (1 - p_b) / (p_b * std_b + 1e-6)

    result = p_a + shift
    return max(0.01, min(0.99, result))


def evaluate_pair(
    q_a: str, q_b: str, p_a: float, p_b: float, rho: float,
    examples_text: str = "",
) -> tuple[float | None, float]:
    """
    Ask LLM for P(A|B=YES) and compare to expected.

    Returns (predicted, expected) or (None, expected) if parse failed.
    """
    if examples_text:
        prompt = FEW_SHOT_PROMPT.format(
            examples=examples_text,
            q_a=q_a, q_b=q_b, p_a=p_a, p_b=p_b,
        )
    else:
        prompt = BASELINE_PROMPT.format(
            q_a=q_a, q_b=q_b, p_a=p_a, p_b=p_b,
        )

    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
    )

    predicted = parse_probability(response.choices[0].message.content)
    expected = expected_conditional(p_a, p_b, rho)

    return predicted, expected


def main():
    # Load training examples
    examples_path = DATA_DIR / "training_examples.json"
    if not examples_path.exists():
        print("Run generate_examples.py first")
        return

    with open(examples_path) as f:
        training_examples = json.load(f)

    # Load all pairs for test set
    pairs_path = DATA_DIR / "pairs.json"
    with open(pairs_path) as f:
        all_pairs = json.load(f)

    # Exclude training pairs from test set
    training_ids = {
        (ex["pair"]["market_a"]["condition_id"], ex["pair"]["market_b"]["condition_id"])
        for ex in training_examples
    }

    test_pairs = [
        p for p in all_pairs
        if (p["market_a"]["condition_id"], p["market_b"]["condition_id"]) not in training_ids
        and p["market_a"]["question"] and p["market_b"]["question"]
    ]

    # Sample test pairs (stratified by bucket)
    n_test_per_bucket = 10
    test_sample = []
    for bucket in ["independent", "weak", "moderate", "strong"]:
        bucket_pairs = [p for p in test_pairs if p["bucket"] == bucket]
        random.shuffle(bucket_pairs)
        test_sample.extend(bucket_pairs[:n_test_per_bucket])

    print(f"Testing on {len(test_sample)} pairs...")

    # Format training examples for few-shot prompt
    examples_text = "\n\n".join([
        ex["example_text"] for ex in training_examples[:10]  # Use top 10
    ])

    # Run evaluation
    results = {"baseline": [], "few_shot": []}

    for i, pair in enumerate(test_sample):
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        rho = pair["rho"]
        bucket = pair["bucket"]

        # Use current market price as P (simulated since we don't have real-time)
        # For now, use 0.5 as default
        p_a = 0.5
        p_b = 0.5

        print(f"[{i+1}/{len(test_sample)}] {bucket}: {q_a[:30]}...")

        try:
            # Baseline
            pred_base, expected = evaluate_pair(q_a, q_b, p_a, p_b, rho, "")
            if pred_base is not None:
                error_base = abs(pred_base - expected)
                results["baseline"].append({
                    "pair": pair,
                    "predicted": pred_base,
                    "expected": expected,
                    "error": error_base,
                })

            # Few-shot
            pred_fs, _ = evaluate_pair(q_a, q_b, p_a, p_b, rho, examples_text)
            if pred_fs is not None:
                error_fs = abs(pred_fs - expected)
                results["few_shot"].append({
                    "pair": pair,
                    "predicted": pred_fs,
                    "expected": expected,
                    "error": error_fs,
                })

        except Exception as e:
            print(f"  Error: {e}")

    # Compute summary statistics
    def summarize(result_list):
        if not result_list:
            return {"n": 0, "mae": None, "median_error": None}
        errors = [r["error"] for r in result_list]
        return {
            "n": len(errors),
            "mae": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
        }

    summary = {
        "baseline": summarize(results["baseline"]),
        "few_shot": summarize(results["few_shot"]),
    }

    # Improvement
    if summary["baseline"]["mae"] and summary["few_shot"]["mae"]:
        improvement = 1 - summary["few_shot"]["mae"] / summary["baseline"]["mae"]
        summary["improvement"] = f"{improvement:.1%}"
    else:
        summary["improvement"] = None

    # Save results
    output_path = RESULTS_DIR / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "baseline_results": results["baseline"],
            "few_shot_results": results["few_shot"],
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    print(f"  Baseline MAE:  {summary['baseline']['mae']:.3f}")
    print(f"  Few-shot MAE:  {summary['few_shot']['mae']:.3f}")
    print(f"  Improvement:   {summary['improvement']}")


if __name__ == "__main__":
    main()
