#!/usr/bin/env python3
"""
Evaluate v2: Curated pairs with explicit formula.

Tests three conditions:
1. Baseline: No examples, no formula
2. Formula only: Explicit ρ→magnitude formula
3. Curated examples + formula: Both

Usage: python evaluate_v2.py
"""

import json
import os
from pathlib import Path
import anthropic
from dataclasses import dataclass
import random

# Load environment from repo root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

@dataclass
class EvalResult:
    pair_id: int
    condition: str
    question_a: str
    question_b: str
    true_rho: float
    expected_update: str
    baseline_prob: float  # P(A)
    predicted_prob: float  # P(A|B=YES)
    actual_direction: str  # 'positive', 'negative', 'none'
    predicted_direction: str
    magnitude_error: float


def load_curated_pairs():
    with open(Path(__file__).parent / "curated_pairs.json") as f:
        return json.load(f)["curated_pairs"]


def load_explicit_formula():
    with open(Path(__file__).parent / "explicit_formula.txt") as f:
        return f.read()


def get_baseline_prompt(question_a: str, question_b: str) -> str:
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

First, estimate P(A) - the unconditional probability that A occurs.
Then, estimate P(A|B=YES) - the probability that A occurs given that B resolves YES.

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_rho_only_prompt(question_a: str, question_b: str, rho: float) -> str:
    """Just provide ρ - no formula, no examples."""
    return f"""You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"
Correlation coefficient ρ = {rho:.2f}

First, estimate P(A) - the unconditional probability that A occurs.
Then, estimate P(A|B=YES) - the probability that A occurs given that B resolves YES.

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_formula_prompt(question_a: str, question_b: str, rho: float, formula: str) -> str:
    return f"""You are a forecaster estimating conditional probabilities.

{formula}

---

Now apply this to the following pair:

Question A: "{question_a}"
Question B: "{question_b}"
Correlation coefficient ρ = {rho:.2f}

First, estimate P(A) - the unconditional probability that A occurs.
Then, using the formula above and the given ρ, estimate P(A|B=YES).

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation including how you applied the formula>"}}"""


def get_examples_only_prompt(question_a: str, question_b: str, rho: float, examples: list) -> str:
    """Curated examples WITHOUT the formula - to isolate effect of curation vs formula."""
    examples_text = "\n\n".join([
        f"**{ex['category'].replace('_', ' ').title()}**\n"
        f"A: {ex['question_a']}\n"
        f"B: {ex['question_b']}\n"
        f"ρ = {ex['rho']:.2f}\n"
        f"Rationale: {ex['rationale']}\n"
        f"Expected update: {ex['expected_update']}"
        for ex in examples
    ])

    return f"""You are a forecaster estimating conditional probabilities.

Here are curated examples of correlated and uncorrelated prediction market pairs:

{examples_text}

---

Now apply this to the following pair:

Question A: "{question_a}"
Question B: "{question_b}"
Correlation coefficient ρ = {rho:.2f}

First, estimate P(A) - the unconditional probability that A occurs.
Then, using the examples above as guidance, estimate P(A|B=YES).

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def get_examples_prompt(question_a: str, question_b: str, rho: float, formula: str, examples: list) -> str:
    examples_text = "\n\n".join([
        f"**{ex['category'].replace('_', ' ').title()}**\n"
        f"A: {ex['question_a']}\n"
        f"B: {ex['question_b']}\n"
        f"ρ = {ex['rho']:.2f}\n"
        f"Rationale: {ex['rationale']}\n"
        f"Expected update: {ex['expected_update']}"
        for ex in examples
    ])

    return f"""You are a forecaster estimating conditional probabilities.

{formula}

---

Here are curated examples of correlated and uncorrelated prediction market pairs:

{examples_text}

---

Now apply this to the following pair:

Question A: "{question_a}"
Question B: "{question_b}"
Correlation coefficient ρ = {rho:.2f}

First, estimate P(A) - the unconditional probability that A occurs.
Then, using the formula and examples above, estimate P(A|B=YES).

Respond with JSON only:
{{"p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def call_model(prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    # Extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"Failed to parse: {text}")
        return {"p_a": 0.5, "p_a_given_b_yes": 0.5, "reasoning": "parse error"}


def get_direction(p_a: float, p_a_given_b: float) -> str:
    diff = p_a_given_b - p_a
    if abs(diff) < 0.05:
        return "none"
    return "positive" if diff > 0 else "negative"


def get_expected_direction(expected_update: str) -> str:
    if "positive" in expected_update.lower():
        return "positive"
    elif "negative" in expected_update.lower():
        return "negative"
    else:
        return "none"


def run_evaluation():
    pairs = load_curated_pairs()
    formula = load_explicit_formula()

    # Use 5 diverse examples for few-shot
    example_ids = [1, 5, 7, 11, 16]  # timeline, crypto, mutually exclusive, independent, same race
    examples = [p for p in pairs if p["id"] in example_ids]

    # Test on remaining pairs
    test_pairs = [p for p in pairs if p["id"] not in example_ids]

    results = []

    for pair in test_pairs:
        print(f"\nEvaluating pair {pair['id']}: {pair['category']}")

        for condition in ["baseline", "rho_only", "examples_only", "formula_only", "formula_plus_examples"]:
            if condition == "baseline":
                prompt = get_baseline_prompt(pair["question_a"], pair["question_b"])
            elif condition == "rho_only":
                prompt = get_rho_only_prompt(pair["question_a"], pair["question_b"], pair["rho"])
            elif condition == "examples_only":
                prompt = get_examples_only_prompt(pair["question_a"], pair["question_b"], pair["rho"], examples)
            elif condition == "formula_only":
                prompt = get_formula_prompt(pair["question_a"], pair["question_b"], pair["rho"], formula)
            else:
                prompt = get_examples_prompt(pair["question_a"], pair["question_b"], pair["rho"], formula, examples)

            response = call_model(prompt)

            p_a = response.get("p_a", 0.5)
            p_a_given_b = response.get("p_a_given_b_yes", 0.5)

            predicted_dir = get_direction(p_a, p_a_given_b)
            expected_dir = get_expected_direction(pair["expected_update"])

            # For magnitude error, we use the correlation-implied expected shift
            expected_shift = pair["rho"] * 0.3  # Simplified expected magnitude
            actual_shift = p_a_given_b - p_a
            magnitude_error = abs(actual_shift - expected_shift)

            result = EvalResult(
                pair_id=pair["id"],
                condition=condition,
                question_a=pair["question_a"],
                question_b=pair["question_b"],
                true_rho=pair["rho"],
                expected_update=pair["expected_update"],
                baseline_prob=p_a,
                predicted_prob=p_a_given_b,
                actual_direction=expected_dir,
                predicted_direction=predicted_dir,
                magnitude_error=magnitude_error
            )
            results.append(result)

            print(f"  {condition}: P(A)={p_a:.2f} → P(A|B=YES)={p_a_given_b:.2f} "
                  f"(predicted: {predicted_dir}, expected: {expected_dir})")

    # Compute summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for condition in ["baseline", "rho_only", "examples_only", "formula_only", "formula_plus_examples"]:
        cond_results = [r for r in results if r.condition == condition]

        direction_correct = sum(1 for r in cond_results if r.predicted_direction == r.actual_direction)
        direction_accuracy = direction_correct / len(cond_results) if cond_results else 0

        mean_magnitude_error = sum(r.magnitude_error for r in cond_results) / len(cond_results) if cond_results else 0

        print(f"\n{condition}:")
        print(f"  Direction accuracy: {direction_accuracy:.1%} ({direction_correct}/{len(cond_results)})")
        print(f"  Mean magnitude error: {mean_magnitude_error:.3f}")

    # Save results
    results_data = {
        "pairs_tested": len(test_pairs),
        "examples_used": len(examples),
        "results": [
            {
                "pair_id": r.pair_id,
                "condition": r.condition,
                "true_rho": r.true_rho,
                "expected_update": r.expected_update,
                "baseline_prob": r.baseline_prob,
                "predicted_prob": r.predicted_prob,
                "predicted_direction": r.predicted_direction,
                "actual_direction": r.actual_direction,
                "magnitude_error": r.magnitude_error
            }
            for r in results
        ]
    }

    with open(Path(__file__).parent / "results" / "evaluation_v2.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to results/evaluation_v2.json")


if __name__ == "__main__":
    run_evaluation()
