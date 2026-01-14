#!/usr/bin/env python3
"""
Test: Can LLMs estimate ρ accurately without market data?

Takes curated pairs, asks model to estimate correlation coefficient,
compares to actual market-derived ρ values.
"""

import json
from pathlib import Path
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


def get_rho_estimation_prompt(question_a: str, question_b: str) -> str:
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


def call_model(prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
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
        return {"rho_estimate": 0.0, "reasoning": "parse error"}


def run_evaluation():
    pairs = load_curated_pairs()

    results = []

    for pair in pairs:
        print(f"\nPair {pair['id']}: {pair['category']}")
        print(f"  A: {pair['question_a'][:60]}...")
        print(f"  B: {pair['question_b'][:60]}...")

        prompt = get_rho_estimation_prompt(pair["question_a"], pair["question_b"])
        response = call_model(prompt)

        estimated_rho = response.get("rho_estimate", 0.0)
        actual_rho = pair["rho"]
        error = estimated_rho - actual_rho

        results.append({
            "pair_id": pair["id"],
            "category": pair["category"],
            "actual_rho": actual_rho,
            "estimated_rho": estimated_rho,
            "error": error,
            "abs_error": abs(error),
            "reasoning": response.get("reasoning", "")
        })

        print(f"  Actual ρ: {actual_rho:.2f}")
        print(f"  Estimated ρ: {estimated_rho:.2f}")
        print(f"  Error: {error:+.2f}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    actual = [r["actual_rho"] for r in results]
    estimated = [r["estimated_rho"] for r in results]
    errors = [r["error"] for r in results]
    abs_errors = [r["abs_error"] for r in results]

    correlation = np.corrcoef(actual, estimated)[0, 1]
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    bias = np.mean(errors)

    print(f"\nCorrelation (actual vs estimated): {correlation:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Bias (mean error): {bias:+.3f}")

    # Direction accuracy (sign match)
    direction_correct = sum(
        1 for r in results
        if (r["actual_rho"] > 0.1 and r["estimated_rho"] > 0.1) or
           (r["actual_rho"] < -0.1 and r["estimated_rho"] < -0.1) or
           (abs(r["actual_rho"]) <= 0.1 and abs(r["estimated_rho"]) <= 0.1)
    )
    print(f"Direction accuracy: {direction_correct}/{len(results)} ({100*direction_correct/len(results):.1f}%)")

    # Breakdown by category
    print("\nBy category:")
    categories = set(r["category"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_mae = np.mean([r["abs_error"] for r in cat_results])
        print(f"  {cat}: MAE={cat_mae:.3f} (n={len(cat_results)})")

    # Save results
    output = {
        "summary": {
            "correlation": correlation,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "direction_accuracy": direction_correct / len(results)
        },
        "results": results
    }

    with open(Path(__file__).parent.parent / "results" / "rho_estimation.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to ../results/rho_estimation.json")


if __name__ == "__main__":
    run_evaluation()
