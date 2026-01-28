#!/usr/bin/env python3
"""
Test market-aware ρ estimation prompts.

Experiment: Compare different prompt variants for estimating correlation
between prediction market questions. The key insight is that markets show
POSITIVE correlation between mutually exclusive outcomes when they share
an underlying driver (e.g., "Will Trump nominate someone?").

Tests 4 prompt variants:
1. baseline: Current two-step entity/competition prompt
2. warning_only: Baseline + market logic warning
3. shared_drivers: New framing focused on what moves prices together
4. shared_drivers_with_warning: Combined approach

Success metric: Direction accuracy on mutually_exclusive pairs (n=13)
Currently at 54% — goal is >70%

Usage:
    uv run python experiments/question-generation/voi-validation/test_market_prompts.py
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import numpy as np

import litellm

from llm_forecasting.voi import (
    RHO_DIRECTION_PROMPT,
    RHO_DIRECTION_WITH_WARNING_PROMPT,
    RHO_SHARED_DRIVERS_PROMPT,
    RHO_SHARED_DRIVERS_WITH_WARNING_PROMPT,
    DEFAULT_RHO_MODEL,
)

# Paths
VOI_DIR = Path(__file__).parent
RESULTS_DIR = VOI_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model for testing (use Sonnet for consistency with previous experiments)
TEST_MODEL = "anthropic/claude-sonnet-4-20250514"


def load_pairs():
    """Load curated pairs with ground-truth ρ and classifications."""
    with open(VOI_DIR / "curated_pairs_nontrivial.json") as f:
        data = json.load(f)
    return data["curated_pairs"]


async def estimate_direction_with_prompt(
    question_a: str,
    question_b: str,
    prompt_template: str,
    model: str = TEST_MODEL,
) -> tuple[str, str, dict]:
    """Estimate direction using a specific prompt template.

    Returns:
        Tuple of (direction, reasoning, full_parsed_response)
    """
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt_template.format(
                    question_a=question_a,
                    question_b=question_b,
                )
            }],
            max_tokens=500,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        direction = result.get("direction", "no_effect")
        reasoning = result.get("reasoning", "")

        return direction, reasoning, result

    except Exception as e:
        return "error", str(e), {}


def direction_to_sign(direction: str) -> int:
    """Convert direction string to sign for comparison with ground truth ρ."""
    if direction == "more_likely":
        return 1
    elif direction == "less_likely":
        return -1
    else:
        return 0


def sign(x: float) -> int:
    """Get sign of a number."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


async def test_prompt_variant(
    pairs: list[dict],
    prompt_template: str,
    variant_name: str,
) -> dict:
    """Test a single prompt variant on all pairs.

    Returns:
        Dict with results and accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing: {variant_name}")
    print(f"{'='*60}")

    results = []

    for i, pair in enumerate(pairs):
        question_a = pair["question_a"]
        question_b = pair["question_b"]
        ground_truth_rho = pair["rho"]
        category = pair["classification"]["category"]

        direction, reasoning, full_response = await estimate_direction_with_prompt(
            question_a, question_b, prompt_template
        )

        predicted_sign = direction_to_sign(direction)
        actual_sign = sign(ground_truth_rho)
        correct = predicted_sign == actual_sign

        results.append({
            "question_a": question_a,
            "question_b": question_b,
            "category": category,
            "ground_truth_rho": ground_truth_rho,
            "actual_sign": actual_sign,
            "predicted_direction": direction,
            "predicted_sign": predicted_sign,
            "correct": correct,
            "reasoning": reasoning,
            "full_response": full_response,
        })

        status = "✓" if correct else "✗"
        print(f"  [{i+1}/{len(pairs)}] {status} {category[:15]:15s} | GT={actual_sign:+d} Pred={predicted_sign:+d} | {question_a[:40]}...")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.1)

    # Compute metrics
    all_correct = sum(1 for r in results if r["correct"])
    all_total = len(results)

    # By category
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0, "results": []}
        by_category[cat]["total"] += 1
        if r["correct"]:
            by_category[cat]["correct"] += 1
        by_category[cat]["results"].append(r)

    for cat in by_category:
        by_category[cat]["accuracy"] = by_category[cat]["correct"] / by_category[cat]["total"]

    # Focus on mutually_exclusive
    me_stats = by_category.get("mutually_exclusive", {"correct": 0, "total": 0, "accuracy": 0})

    print(f"\n  Overall accuracy: {all_correct}/{all_total} = {all_correct/all_total:.1%}")
    print(f"  mutually_exclusive accuracy: {me_stats['correct']}/{me_stats['total']} = {me_stats['accuracy']:.1%}")

    return {
        "variant": variant_name,
        "overall_accuracy": all_correct / all_total,
        "overall_correct": all_correct,
        "overall_total": all_total,
        "mutually_exclusive_accuracy": me_stats["accuracy"],
        "mutually_exclusive_correct": me_stats["correct"],
        "mutually_exclusive_total": me_stats["total"],
        "by_category": {
            cat: {
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for cat, stats in by_category.items()
        },
        "results": results,
    }


async def main():
    print("=" * 70)
    print("MARKET-AWARE ρ ESTIMATION PROMPT EXPERIMENT")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {TEST_MODEL}")

    pairs = load_pairs()
    print(f"\nLoaded {len(pairs)} pairs")

    # Count by category
    categories = {}
    for p in pairs:
        cat = p["classification"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\nBy category:")
    for cat, n in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")

    # Define prompt variants to test
    prompt_variants = {
        "baseline": RHO_DIRECTION_PROMPT,
        "warning_only": RHO_DIRECTION_WITH_WARNING_PROMPT,
        "shared_drivers": RHO_SHARED_DRIVERS_PROMPT,
        "shared_drivers_with_warning": RHO_SHARED_DRIVERS_WITH_WARNING_PROMPT,
    }

    # Run tests
    all_results = {}
    for variant_name, prompt_template in prompt_variants.items():
        result = await test_prompt_variant(pairs, prompt_template, variant_name)
        all_results[variant_name] = result

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:30s} {:>12s} {:>20s}".format("Variant", "Overall", "mutually_exclusive"))
    print("-" * 64)

    for variant_name, result in all_results.items():
        overall = f"{result['overall_correct']}/{result['overall_total']} ({result['overall_accuracy']:.1%})"
        me = f"{result['mutually_exclusive_correct']}/{result['mutually_exclusive_total']} ({result['mutually_exclusive_accuracy']:.1%})"
        print(f"{variant_name:30s} {overall:>12s} {me:>20s}")

    # Identify best variant
    best_overall = max(all_results.items(), key=lambda x: x[1]["overall_accuracy"])
    best_me = max(all_results.items(), key=lambda x: x[1]["mutually_exclusive_accuracy"])

    print(f"\nBest overall: {best_overall[0]} ({best_overall[1]['overall_accuracy']:.1%})")
    print(f"Best on mutually_exclusive: {best_me[0]} ({best_me[1]['mutually_exclusive_accuracy']:.1%})")

    # Breakdown by category for each variant
    print("\n" + "-" * 70)
    print("ACCURACY BY CATEGORY")
    print("-" * 70)

    # Get all categories
    all_cats = set()
    for result in all_results.values():
        all_cats.update(result["by_category"].keys())
    all_cats = sorted(all_cats)

    # Header
    header = "{:25s}".format("Category")
    for variant_name in prompt_variants.keys():
        header += f" {variant_name[:12]:>12s}"
    print(header)
    print("-" * (25 + 13 * len(prompt_variants)))

    # Rows
    for cat in all_cats:
        row = f"{cat[:24]:25s}"
        for variant_name in prompt_variants.keys():
            stats = all_results[variant_name]["by_category"].get(cat, {})
            if stats:
                acc = stats["accuracy"]
                row += f" {acc:>11.0%} "
            else:
                row += "          -- "
        print(row)

    # Analysis: Where does each variant fail?
    print("\n" + "-" * 70)
    print("FAILURE ANALYSIS: mutually_exclusive pairs")
    print("-" * 70)

    me_pairs = [p for p in pairs if p["classification"]["category"] == "mutually_exclusive"]

    for pair in me_pairs:
        qa = pair["question_a"][:50]
        gt_rho = pair["rho"]
        gt_sign = sign(gt_rho)

        print(f"\n{qa}...")
        print(f"  Ground truth ρ = {gt_rho:+.3f} (sign={gt_sign:+d})")

        for variant_name, result in all_results.items():
            # Find this pair in results
            for r in result["results"]:
                if r["question_a"] == pair["question_a"]:
                    pred = r["predicted_sign"]
                    status = "✓" if r["correct"] else "✗"
                    print(f"  {variant_name:30s}: {status} pred={pred:+d}")
                    break

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": TEST_MODEL,
            "n_pairs": len(pairs),
            "n_mutually_exclusive": len(me_pairs),
        },
        "summary": {
            variant_name: {
                "overall_accuracy": result["overall_accuracy"],
                "mutually_exclusive_accuracy": result["mutually_exclusive_accuracy"],
                "by_category": result["by_category"],
            }
            for variant_name, result in all_results.items()
        },
        "best": {
            "overall": best_overall[0],
            "mutually_exclusive": best_me[0],
        },
        "full_results": all_results,
    }

    output_path = RESULTS_DIR / "market_prompt_experiment.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    baseline_me = all_results["baseline"]["mutually_exclusive_accuracy"]
    best_me_acc = best_me[1]["mutually_exclusive_accuracy"]

    if best_me_acc > 0.7:
        print(f"\n✅ SUCCESS: {best_me[0]} achieves {best_me_acc:.0%} on mutually_exclusive")
        print("   Proceed with full VOI calibration using this variant")
    elif best_me_acc > baseline_me:
        print(f"\n⚠️ PARTIAL: {best_me[0]} improves to {best_me_acc:.0%} (from {baseline_me:.0%})")
        print("   Some improvement but may need additional prompt engineering")
    else:
        print(f"\n❌ NO IMPROVEMENT: Best is {best_me_acc:.0%}, baseline was {baseline_me:.0%}")
        print("   Problem may be harder than prompting — consider other approaches")


if __name__ == "__main__":
    asyncio.run(main())
