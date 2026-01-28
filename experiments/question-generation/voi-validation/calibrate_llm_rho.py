"""LLM ρ Calibration Experiment.

Tests whether VOI computed from LLM-estimated ρ predicts actual market shifts,
compared to the r=0.653 baseline using ground-truth market ρ.

Key question: What is the practical ceiling for LLM-based VOI?

Uses the existing 34 non-trivial Polymarket pairs (where we have ground-truth ρ
and actual_shift) as a calibration dataset.

Results: experiments/question-generation/voi-validation/results/llm_rho_calibration.json
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

from scipy import stats
import numpy as np

from llm_forecasting.voi import (
    estimate_rho_two_step,
    linear_voi_from_rho,
    entropy_voi_from_rho,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
CURATED_PAIRS_FILE = SCRIPT_DIR / "curated_pairs_nontrivial.json"
VALIDATION_RESULTS_FILE = SCRIPT_DIR / "results" / "voi_validation_nontrivial.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "llm_rho_calibration.json"

# Model to use for LLM estimation
MODEL = "anthropic/claude-sonnet-4-20250514"


def load_data() -> tuple[list[dict], dict]:
    """Load curated pairs and validation results."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)

    with open(VALIDATION_RESULTS_FILE) as f:
        validation = json.load(f)

    return curated["curated_pairs"], validation


async def estimate_llm_rho_for_pairs(pairs: list[dict]) -> list[tuple[float, str]]:
    """Estimate ρ using LLM for all pairs sequentially."""
    print(f"Estimating ρ for {len(pairs)} pairs using {MODEL}...")

    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['question_a'][:50]}...")
        rho, reasoning = await estimate_rho_two_step(
            pair["question_a"],
            pair["question_b"],
            model=MODEL,
        )
        results.append((rho, reasoning))

    return results


def compute_voi_metrics(
    pairs: list[dict],
    llm_results: list[tuple[float, str]],
    validation_pairs: list[dict],
) -> list[dict]:
    """Compute VOI metrics for each pair using both ground-truth and LLM ρ."""
    results = []

    # Build lookup for validation data
    validation_lookup = {
        (p["question_a"], p["question_b"]): p
        for p in validation_pairs
    }

    for pair, (llm_rho, llm_reasoning) in zip(pairs, llm_results):
        # Get validation data
        key = (pair["question_a"], pair["question_b"])
        val_data = validation_lookup.get(key)
        if not val_data:
            print(f"Warning: No validation data for {pair['question_a'][:50]}...")
            continue

        ground_truth_rho = pair["rho"]
        actual_shift = val_data["actual_shift"]
        p_before = val_data["p_before"]  # Prior probability of the "other" market

        # For VOI computation, we need:
        # - p_a: the scenario probability (the "other" market that shifts)
        # - p_b: the signal probability (the market that resolves)
        # Since one market resolved (hit >90% or <10%), p_b ≈ 0.95 or 0.05
        # We use p_before as p_a

        # Assume p_b = 0.5 for VOI computation (symmetric assumption)
        # This matches how VOI was computed in the validation
        p_b = 0.5

        # Compute VOI using ground-truth ρ
        gt_linear_voi = linear_voi_from_rho(ground_truth_rho, p_before, p_b)
        gt_entropy_voi = entropy_voi_from_rho(ground_truth_rho, p_before, p_b)

        # Compute VOI using LLM ρ
        llm_linear_voi = linear_voi_from_rho(llm_rho, p_before, p_b)
        llm_entropy_voi = entropy_voi_from_rho(llm_rho, p_before, p_b)

        results.append({
            "question_a": pair["question_a"],
            "question_b": pair["question_b"],
            "classification": pair["classification"]["category"],
            "ground_truth_rho": ground_truth_rho,
            "llm_rho": llm_rho,
            "llm_reasoning": llm_reasoning,
            "rho_error": llm_rho - ground_truth_rho,
            "rho_abs_error": abs(llm_rho - ground_truth_rho),
            "rho_direction_match": (
                (ground_truth_rho > 0 and llm_rho > 0) or
                (ground_truth_rho < 0 and llm_rho < 0) or
                (abs(ground_truth_rho) < 0.05 and abs(llm_rho) < 0.05)
            ),
            "p_before": p_before,
            "actual_shift": actual_shift,
            "gt_linear_voi": gt_linear_voi,
            "gt_entropy_voi": gt_entropy_voi,
            "llm_linear_voi": llm_linear_voi,
            "llm_entropy_voi": llm_entropy_voi,
        })

    return results


def compute_correlations(results: list[dict]) -> dict:
    """Compute correlation metrics."""
    if not results:
        return {}

    # Extract arrays
    actual_shifts = np.array([r["actual_shift"] for r in results])
    gt_linear_voi = np.array([r["gt_linear_voi"] for r in results])
    gt_entropy_voi = np.array([r["gt_entropy_voi"] for r in results])
    llm_linear_voi = np.array([r["llm_linear_voi"] for r in results])
    llm_entropy_voi = np.array([r["llm_entropy_voi"] for r in results])
    gt_rho = np.array([r["ground_truth_rho"] for r in results])
    llm_rho = np.array([r["llm_rho"] for r in results])

    # Compute correlations
    correlations = {}

    # Ground-truth VOI vs actual shift (baseline - should match ~0.653)
    r, p = stats.pearsonr(gt_entropy_voi, actual_shifts)
    correlations["ground_truth_entropy_voi_vs_shift"] = {"r": r, "p": p}

    r, p = stats.pearsonr(gt_linear_voi, actual_shifts)
    correlations["ground_truth_linear_voi_vs_shift"] = {"r": r, "p": p}

    # LLM VOI vs actual shift (the main test)
    r, p = stats.pearsonr(llm_entropy_voi, actual_shifts)
    correlations["llm_entropy_voi_vs_shift"] = {"r": r, "p": p}

    r, p = stats.pearsonr(llm_linear_voi, actual_shifts)
    correlations["llm_linear_voi_vs_shift"] = {"r": r, "p": p}

    # LLM ρ vs ground-truth ρ (estimation accuracy)
    r, p = stats.pearsonr(llm_rho, gt_rho)
    correlations["llm_rho_vs_ground_truth_rho"] = {"r": r, "p": p}

    # Also compute Spearman for robustness
    rho, p = stats.spearmanr(llm_rho, gt_rho)
    correlations["llm_rho_vs_ground_truth_rho_spearman"] = {"rho": rho, "p": p}

    return correlations


def compute_rho_estimation_metrics(results: list[dict]) -> dict:
    """Compute ρ estimation accuracy metrics."""
    if not results:
        return {}

    rho_errors = [r["rho_error"] for r in results]
    rho_abs_errors = [r["rho_abs_error"] for r in results]
    direction_matches = [r["rho_direction_match"] for r in results]

    # By category
    by_category = {}
    categories = set(r["classification"] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r["classification"] == cat]
        cat_errors = [r["rho_abs_error"] for r in cat_results]
        cat_directions = [r["rho_direction_match"] for r in cat_results]
        by_category[cat] = {
            "n": len(cat_results),
            "mae": np.mean(cat_errors),
            "direction_accuracy": np.mean(cat_directions),
        }

    return {
        "mae": np.mean(rho_abs_errors),
        "rmse": np.sqrt(np.mean(np.array(rho_errors) ** 2)),
        "mean_error": np.mean(rho_errors),  # Bias
        "direction_accuracy": np.mean(direction_matches),
        "n_correct_direction": sum(direction_matches),
        "n_total": len(direction_matches),
        "by_category": by_category,
    }


def identify_failure_cases(results: list[dict], threshold: float = 0.5) -> list[dict]:
    """Identify pairs where LLM ρ differs significantly from ground truth."""
    failures = []
    for r in results:
        if r["rho_abs_error"] > threshold:
            failures.append({
                "question_a": r["question_a"],
                "question_b": r["question_b"],
                "classification": r["classification"],
                "ground_truth_rho": r["ground_truth_rho"],
                "llm_rho": r["llm_rho"],
                "error": r["rho_error"],
                "llm_reasoning": r["llm_reasoning"],
            })
    return sorted(failures, key=lambda x: -abs(x["error"]))


async def main():
    """Run the LLM ρ calibration experiment."""
    print("=" * 60)
    print("LLM ρ Calibration Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    pairs, validation = load_data()
    validation_pairs = validation["pairs"]

    print(f"  Curated pairs: {len(pairs)}")
    print(f"  Validation pairs: {len(validation_pairs)}")

    # Estimate LLM ρ for all pairs
    llm_results = await estimate_llm_rho_for_pairs(pairs)

    # Compute VOI metrics
    print("\nComputing VOI metrics...")
    results = compute_voi_metrics(pairs, llm_results, validation_pairs)
    print(f"  Computed metrics for {len(results)} pairs")

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(results)

    # Print key results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- VOI vs Actual Shift Correlations ---")
    print(f"Ground-truth entropy VOI: r = {correlations['ground_truth_entropy_voi_vs_shift']['r']:.3f} (baseline, expect ~0.653)")
    print(f"Ground-truth linear VOI:  r = {correlations['ground_truth_linear_voi_vs_shift']['r']:.3f}")
    print(f"LLM entropy VOI:          r = {correlations['llm_entropy_voi_vs_shift']['r']:.3f} (p = {correlations['llm_entropy_voi_vs_shift']['p']:.4f})")
    print(f"LLM linear VOI:           r = {correlations['llm_linear_voi_vs_shift']['r']:.3f} (p = {correlations['llm_linear_voi_vs_shift']['p']:.4f})")

    print("\n--- ρ Estimation Accuracy ---")
    print(f"LLM ρ vs ground-truth ρ:  r = {correlations['llm_rho_vs_ground_truth_rho']['r']:.3f}")

    # Compute estimation metrics
    rho_metrics = compute_rho_estimation_metrics(results)
    print(f"MAE:                      {rho_metrics['mae']:.3f}")
    print(f"RMSE:                     {rho_metrics['rmse']:.3f}")
    print(f"Mean error (bias):        {rho_metrics['mean_error']:.3f}")
    print(f"Direction accuracy:       {rho_metrics['direction_accuracy']:.1%} ({rho_metrics['n_correct_direction']}/{rho_metrics['n_total']})")

    print("\n--- By Category ---")
    for cat, metrics in sorted(rho_metrics["by_category"].items()):
        print(f"  {cat}: n={metrics['n']}, MAE={metrics['mae']:.3f}, direction={metrics['direction_accuracy']:.1%}")

    # Identify failure cases
    failures = identify_failure_cases(results, threshold=0.5)
    print(f"\n--- Failure Cases (|error| > 0.5): {len(failures)} ---")
    for f in failures[:5]:  # Show top 5
        print(f"\n  Q_A: {f['question_a'][:60]}...")
        print(f"  Q_B: {f['question_b'][:60]}...")
        print(f"  Category: {f['classification']}")
        print(f"  Ground-truth ρ: {f['ground_truth_rho']:.3f}, LLM ρ: {f['llm_rho']:.3f}, Error: {f['error']:.3f}")

    # Compute gap
    gt_r = correlations['ground_truth_entropy_voi_vs_shift']['r']
    llm_r = correlations['llm_entropy_voi_vs_shift']['r']
    gap = gt_r - llm_r

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Practical ceiling gap: {gap:.3f} (ground-truth r={gt_r:.3f} vs LLM r={llm_r:.3f})")
    print(f"LLM VOI retains {llm_r/gt_r:.1%} of ground-truth VOI predictive power")

    # Save results
    output = {
        "metadata": {
            "n_pairs": len(results),
            "model": MODEL,
            "method": "two_step_batch",
            "generated_at": datetime.now().isoformat(),
        },
        "correlations": correlations,
        "rho_estimation": rho_metrics,
        "summary": {
            "ground_truth_entropy_voi_r": gt_r,
            "llm_entropy_voi_r": llm_r,
            "practical_ceiling_gap": gap,
            "retained_predictive_power": llm_r / gt_r if gt_r > 0 else 0,
        },
        "failure_cases": failures,
        "pairs": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
