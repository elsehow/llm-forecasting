#!/usr/bin/env python3
"""
Test magnitude calibration prompts for ρ estimation.

Experiment: Phase 3 found that improved direction accuracy (55.9% vs 44.1%) did NOT
translate to VOI improvement (r=-0.049 vs r=0.138). Root cause: the model anchors
on high magnitudes (predicting |ρ|=0.8 when ground-truth is |ρ|=0.1).

Tests 4 magnitude prompt variants while keeping direction fixed:
1. baseline: Current magnitude prompt
2. calibrated: Add explicit calibration warning about weak correlations
3. discount: Ask model to apply 50% discount to logical strength
4. anchored: Provide actual market correlation examples

Direction is fixed using `shared_drivers_with_warning` which achieves 77% accuracy
on mutually_exclusive pairs.

Success metric: VOI r with actual_shift (baseline r=0.138 from Phase 3)
Target: r > 0.25

Usage:
    uv run python experiments/question-generation/voi-validation/test_magnitude_prompts.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from monorepo root before importing llm_forecasting
_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import numpy as np
from scipy import stats
import litellm

from llm_forecasting.voi import (
    RHO_SHARED_DRIVERS_WITH_WARNING_PROMPT,
    RHO_MAGNITUDE_PROMPT,
    RHO_MAGNITUDE_CALIBRATED_PROMPT,
    RHO_MAGNITUDE_DISCOUNT_PROMPT,
    RHO_MAGNITUDE_ANCHORED_PROMPT,
    linear_voi_from_rho,
    entropy_voi_from_rho,
)

# Paths
VOI_DIR = Path(__file__).parent
RESULTS_DIR = VOI_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CURATED_PAIRS_FILE = VOI_DIR / "curated_pairs_nontrivial.json"
VALIDATION_RESULTS_FILE = RESULTS_DIR / "voi_validation_nontrivial.json"

# Model for testing (use Sonnet for consistency with previous experiments)
TEST_MODEL = "anthropic/claude-sonnet-4-20250514"


def load_data() -> tuple[list[dict], dict]:
    """Load curated pairs and validation results."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)

    with open(VALIDATION_RESULTS_FILE) as f:
        validation = json.load(f)

    return curated["curated_pairs"], validation


async def estimate_direction(
    question_a: str,
    question_b: str,
    model: str = TEST_MODEL,
) -> tuple[str, str, dict]:
    """Estimate direction using shared_drivers_with_warning prompt (fixed)."""
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": RHO_SHARED_DRIVERS_WITH_WARNING_PROMPT.format(
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


async def estimate_magnitude(
    question_a: str,
    question_b: str,
    direction_word: str,
    magnitude_prompt: str,
    model: str = TEST_MODEL,
) -> tuple[float, str]:
    """Estimate magnitude using a specific prompt template."""
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": magnitude_prompt.format(
                    question_a=question_a,
                    question_b=question_b,
                    direction=direction_word,
                )
            }],
            max_tokens=300,
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

        # Handle discount prompt which has different output format
        if "market_magnitude" in result:
            magnitude = float(result.get("market_magnitude", 0.3))
        else:
            magnitude = float(result.get("magnitude", 0.3))

        reasoning = result.get("reasoning", "")

        return min(1.0, max(0.0, magnitude)), reasoning

    except Exception as e:
        return 0.3, f"Error: {e}"


def direction_to_sign(direction: str) -> int:
    """Convert direction string to sign for ρ computation."""
    if direction == "more_likely":
        return 1
    elif direction == "less_likely":
        return -1
    return 0


def sign(x: float) -> int:
    """Get sign of a number."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


async def test_magnitude_variant(
    pairs: list[dict],
    validation_pairs: list[dict],
    magnitude_prompt: str,
    variant_name: str,
) -> dict:
    """Test a single magnitude prompt variant on all pairs.

    Uses fixed direction estimation (shared_drivers_with_warning).
    """
    print(f"\n{'='*60}")
    print(f"Testing: {variant_name}")
    print(f"{'='*60}")

    # Build validation lookup
    validation_lookup = {
        (p["question_a"], p["question_b"]): p
        for p in validation_pairs
    }

    results = []

    for i, pair in enumerate(pairs):
        question_a = pair["question_a"]
        question_b = pair["question_b"]
        ground_truth_rho = pair["rho"]
        category = pair["classification"]["category"]

        # Get validation data
        val_data = validation_lookup.get((question_a, question_b))
        if not val_data:
            print(f"  [{i+1}/{len(pairs)}] SKIP - no validation data")
            continue

        actual_shift = val_data["actual_shift"]
        p_before = val_data["p_before"]

        # Step 1: Estimate direction (fixed prompt)
        direction, dir_reasoning, _ = await estimate_direction(question_a, question_b)

        if direction == "error":
            print(f"  [{i+1}/{len(pairs)}] ERROR - direction failed")
            continue

        # Convert direction to sign and word
        predicted_sign = direction_to_sign(direction)
        actual_sign = sign(ground_truth_rho)

        if direction == "more_likely":
            direction_word = "more likely"
        elif direction == "less_likely":
            direction_word = "less likely"
        else:
            # Independent - rho = 0
            predicted_rho = 0.0
            mag_reasoning = "Independent (direction=no_effect)"
            magnitude = 0.0
            results.append({
                "question_a": question_a,
                "question_b": question_b,
                "category": category,
                "ground_truth_rho": ground_truth_rho,
                "predicted_rho": predicted_rho,
                "predicted_sign": predicted_sign,
                "actual_sign": actual_sign,
                "direction_correct": predicted_sign == actual_sign,
                "magnitude": magnitude,
                "mag_error": abs(predicted_rho) - abs(ground_truth_rho),
                "rho_error": predicted_rho - ground_truth_rho,
                "rho_abs_error": abs(predicted_rho - ground_truth_rho),
                "p_before": p_before,
                "actual_shift": actual_shift,
                "reasoning": mag_reasoning,
            })
            status = "✓" if predicted_sign == actual_sign else "✗"
            print(f"  [{i+1}/{len(pairs)}] {status} {category[:15]:15s} | ρ GT={ground_truth_rho:+.2f} LLM={predicted_rho:+.2f} | (indep)")
            await asyncio.sleep(0.1)
            continue

        # Step 2: Estimate magnitude (variable prompt)
        magnitude, mag_reasoning = await estimate_magnitude(
            question_a, question_b, direction_word, magnitude_prompt
        )

        # Combine sign and magnitude
        predicted_rho = predicted_sign * magnitude

        results.append({
            "question_a": question_a,
            "question_b": question_b,
            "category": category,
            "ground_truth_rho": ground_truth_rho,
            "predicted_rho": predicted_rho,
            "predicted_sign": predicted_sign,
            "actual_sign": actual_sign,
            "direction_correct": predicted_sign == actual_sign,
            "magnitude": magnitude,
            "mag_error": magnitude - abs(ground_truth_rho),
            "rho_error": predicted_rho - ground_truth_rho,
            "rho_abs_error": abs(predicted_rho - ground_truth_rho),
            "p_before": p_before,
            "actual_shift": actual_shift,
            "reasoning": mag_reasoning,
        })

        status = "✓" if predicted_sign == actual_sign else "✗"
        print(f"  [{i+1}/{len(pairs)}] {status} {category[:15]:15s} | ρ GT={ground_truth_rho:+.2f} LLM={predicted_rho:+.2f} (|m|={magnitude:.2f})")

        await asyncio.sleep(0.1)

    if not results:
        return {"variant": variant_name, "error": "No results"}

    # Compute VOI for all pairs
    for r in results:
        p_b = 0.5  # Symmetric assumption (matches validation)
        r["gt_entropy_voi"] = entropy_voi_from_rho(r["ground_truth_rho"], r["p_before"], p_b)
        r["llm_entropy_voi"] = entropy_voi_from_rho(r["predicted_rho"], r["p_before"], p_b)
        r["gt_linear_voi"] = linear_voi_from_rho(r["ground_truth_rho"], r["p_before"], p_b)
        r["llm_linear_voi"] = linear_voi_from_rho(r["predicted_rho"], r["p_before"], p_b)

    # Compute metrics
    actual_shifts = np.array([r["actual_shift"] for r in results])
    llm_entropy_voi = np.array([r["llm_entropy_voi"] for r in results])
    gt_entropy_voi = np.array([r["gt_entropy_voi"] for r in results])
    llm_rho = np.array([r["predicted_rho"] for r in results])
    gt_rho = np.array([r["ground_truth_rho"] for r in results])
    magnitudes = np.array([r["magnitude"] for r in results])
    gt_magnitudes = np.abs(gt_rho)
    rho_errors = np.array([r["rho_abs_error"] for r in results])
    direction_correct = np.array([r["direction_correct"] for r in results])

    # Correlations
    voi_r, voi_p = stats.pearsonr(llm_entropy_voi, actual_shifts)
    gt_voi_r, _ = stats.pearsonr(gt_entropy_voi, actual_shifts)
    rho_r, _ = stats.pearsonr(llm_rho, gt_rho)

    # Magnitude metrics
    mag_mae = np.mean(np.abs(magnitudes - gt_magnitudes))
    mag_mean_error = np.mean(magnitudes - gt_magnitudes)  # Positive = overpredicting

    # Direction accuracy
    dir_accuracy = np.mean(direction_correct)

    # By category
    by_category = {}
    categories = set(r["category"] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_shifts = np.array([r["actual_shift"] for r in cat_results])
        cat_llm_voi = np.array([r["llm_entropy_voi"] for r in cat_results])
        cat_gt_voi = np.array([r["gt_entropy_voi"] for r in cat_results])
        cat_mags = np.array([r["magnitude"] for r in cat_results])
        cat_gt_mags = np.abs(np.array([r["ground_truth_rho"] for r in cat_results]))

        cat_voi_r = np.nan
        cat_gt_voi_r = np.nan
        if len(cat_results) >= 3:
            try:
                cat_voi_r, _ = stats.pearsonr(cat_llm_voi, cat_shifts)
                cat_gt_voi_r, _ = stats.pearsonr(cat_gt_voi, cat_shifts)
            except:
                pass

        by_category[cat] = {
            "n": len(cat_results),
            "voi_r": cat_voi_r,
            "gt_voi_r": cat_gt_voi_r,
            "mag_mae": np.mean(np.abs(cat_mags - cat_gt_mags)),
            "mag_mean_error": np.mean(cat_mags - cat_gt_mags),
            "direction_accuracy": np.mean([r["direction_correct"] for r in cat_results]),
        }

    print(f"\n  VOI r = {voi_r:.3f} (GT r = {gt_voi_r:.3f})")
    print(f"  Direction accuracy: {dir_accuracy:.1%}")
    print(f"  Magnitude MAE: {mag_mae:.3f} (mean error: {mag_mean_error:+.3f})")
    print(f"  ρ MAE: {np.mean(rho_errors):.3f}")

    return {
        "variant": variant_name,
        "voi_r": voi_r,
        "voi_p": voi_p,
        "gt_voi_r": gt_voi_r,
        "rho_r": rho_r,
        "rho_mae": np.mean(rho_errors),
        "mag_mae": mag_mae,
        "mag_mean_error": mag_mean_error,
        "direction_accuracy": dir_accuracy,
        "n": len(results),
        "by_category": by_category,
        "results": results,
    }


async def main():
    print("=" * 70)
    print("MAGNITUDE CALIBRATION PROMPT EXPERIMENT (Phase 4)")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {TEST_MODEL}")
    print(f"Direction prompt: shared_drivers_with_warning (fixed)")

    pairs, validation = load_data()
    validation_pairs = validation["pairs"]
    print(f"\nLoaded {len(pairs)} pairs, {len(validation_pairs)} validation pairs")

    # Count by category
    categories = {}
    for p in pairs:
        cat = p["classification"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\nBy category:")
    for cat, n in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")

    # Define magnitude prompt variants to test
    magnitude_variants = {
        "baseline": RHO_MAGNITUDE_PROMPT,
        "calibrated": RHO_MAGNITUDE_CALIBRATED_PROMPT,
        "discount": RHO_MAGNITUDE_DISCOUNT_PROMPT,
        "anchored": RHO_MAGNITUDE_ANCHORED_PROMPT,
    }

    # Run tests
    all_results = {}
    for variant_name, magnitude_prompt in magnitude_variants.items():
        result = await test_magnitude_variant(
            pairs, validation_pairs, magnitude_prompt, variant_name
        )
        all_results[variant_name] = result

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:15s} {:>8s} {:>8s} {:>8s} {:>10s} {:>12s}".format(
        "Variant", "VOI r", "ρ MAE", "|m| MAE", "Dir Acc", "Mag Error"
    ))
    print("-" * 65)

    for variant_name, result in all_results.items():
        if "error" in result:
            print(f"{variant_name:15s} ERROR")
            continue
        print("{:15s} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.1%} {:>+11.3f}".format(
            variant_name,
            result["voi_r"],
            result["rho_mae"],
            result["mag_mae"],
            result["direction_accuracy"],
            result["mag_mean_error"],
        ))

    # Identify best variant
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if valid_results:
        best_voi = max(valid_results.items(), key=lambda x: x[1]["voi_r"])
        best_rho = min(valid_results.items(), key=lambda x: x[1]["rho_mae"])

        print(f"\nBest VOI r: {best_voi[0]} (r={best_voi[1]['voi_r']:.3f})")
        print(f"Best ρ MAE: {best_rho[0]} (MAE={best_rho[1]['rho_mae']:.3f})")

    # VOI by category for each variant
    print("\n" + "-" * 70)
    print("VOI r BY CATEGORY")
    print("-" * 70)

    all_cats = set()
    for result in valid_results.values():
        all_cats.update(result["by_category"].keys())
    all_cats = sorted(all_cats)

    header = "{:25s}".format("Category")
    for variant_name in magnitude_variants.keys():
        header += f" {variant_name[:10]:>10s}"
    print(header)
    print("-" * (25 + 11 * len(magnitude_variants)))

    for cat in all_cats:
        row = f"{cat[:24]:25s}"
        for variant_name in magnitude_variants.keys():
            if variant_name not in valid_results:
                row += "        -- "
                continue
            cat_stats = valid_results[variant_name]["by_category"].get(cat, {})
            if cat_stats and not np.isnan(cat_stats.get("voi_r", np.nan)):
                row += f" {cat_stats['voi_r']:>9.3f} "
            else:
                row += "        -- "
        print(row)

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": TEST_MODEL,
            "direction_prompt": "shared_drivers_with_warning",
            "n_pairs": len(pairs),
        },
        "summary": {
            variant_name: {
                "voi_r": result.get("voi_r"),
                "voi_p": result.get("voi_p"),
                "gt_voi_r": result.get("gt_voi_r"),
                "rho_mae": result.get("rho_mae"),
                "mag_mae": result.get("mag_mae"),
                "mag_mean_error": result.get("mag_mean_error"),
                "direction_accuracy": result.get("direction_accuracy"),
                "by_category": result.get("by_category"),
            }
            for variant_name, result in all_results.items()
        },
        "best": {
            "voi_r": best_voi[0] if valid_results else None,
            "rho_mae": best_rho[0] if valid_results else None,
        },
        "full_results": {
            variant_name: result.get("results", [])
            for variant_name, result in all_results.items()
        },
    }

    output_path = RESULTS_DIR / "magnitude_prompt_experiment.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print(f"\nSaved to {output_path}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    baseline_voi_r = valid_results.get("baseline", {}).get("voi_r", 0)
    best_voi_r = best_voi[1]["voi_r"] if valid_results else 0

    if best_voi_r > 0.35:
        print(f"\n✅ SUCCESS: {best_voi[0]} achieves VOI r = {best_voi_r:.3f}")
        print("   Strong VOI correlation — can proceed with this prompt")
    elif best_voi_r > 0.25:
        print(f"\n⚠️ PARTIAL SUCCESS: {best_voi[0]} achieves VOI r = {best_voi_r:.3f}")
        print("   Modest improvement — may need further refinement")
    elif best_voi_r > baseline_voi_r:
        print(f"\n📈 MARGINAL: {best_voi[0]} improves to r = {best_voi_r:.3f} (from {baseline_voi_r:.3f})")
        print("   Some improvement but still weak")
    else:
        print(f"\n❌ NO IMPROVEMENT: Best r = {best_voi_r:.3f}, baseline was {baseline_voi_r:.3f}")
        print("   Magnitude calibration prompts don't help — problem may be fundamental")

    # Compare to Phase 3 baseline
    print(f"\nComparison to Phase 3:")
    print(f"  Phase 3 market-aware: r = -0.049 (this experiment's direction prompt)")
    print(f"  Phase 1 baseline:     r =  0.138 (original two-step)")
    print(f"  Ground truth:         r =  0.653")


if __name__ == "__main__":
    asyncio.run(main())
