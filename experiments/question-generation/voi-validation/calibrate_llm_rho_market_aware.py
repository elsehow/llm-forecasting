"""LLM ρ Calibration with Market-Aware Prompt.

Phase 3 of the LLM ρ calibration experiment. Tests whether the improved direction
accuracy from shared_drivers_with_warning prompt (77% on mutually_exclusive, up from 54%)
translates to improved VOI correlation with actual market shifts.

Hypothesis: Direction accuracy improvement will translate to VOI improvement.

Key question: Does VOI computed from market-aware LLM ρ correlate better with
actual shifts than baseline (r=0.138)?

Decision thresholds:
- r > 0.45: Ship market-aware prompt for production use
- r = 0.25-0.45: Use with caveats; document trade-offs
- r < 0.25: Don't ship; diagnose and iterate

Results: experiments/question-generation/voi-validation/results/llm_rho_market_aware.json
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
    estimate_rho_market_aware,
    linear_voi_from_rho,
    entropy_voi_from_rho,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
CURATED_PAIRS_FILE = SCRIPT_DIR / "curated_pairs_nontrivial.json"
VALIDATION_RESULTS_FILE = SCRIPT_DIR / "results" / "voi_validation_nontrivial.json"
BASELINE_RESULTS_FILE = SCRIPT_DIR / "results" / "llm_rho_calibration.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "llm_rho_market_aware.json"

# Model to use for LLM estimation (same as baseline for fair comparison)
MODEL = "anthropic/claude-sonnet-4-20250514"


def load_data() -> tuple[list[dict], dict, dict]:
    """Load curated pairs, validation results, and baseline results."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)

    with open(VALIDATION_RESULTS_FILE) as f:
        validation = json.load(f)

    with open(BASELINE_RESULTS_FILE) as f:
        baseline = json.load(f)

    return curated["curated_pairs"], validation, baseline


async def estimate_llm_rho_for_pairs(pairs: list[dict]) -> list[tuple[float, str]]:
    """Estimate ρ using market-aware LLM prompt for all pairs sequentially."""
    print(f"Estimating ρ for {len(pairs)} pairs using market-aware prompt ({MODEL})...")

    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['question_a'][:50]}...")
        rho, reasoning = await estimate_rho_market_aware(
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
    """Compute VOI metrics for each pair using both ground-truth and market-aware LLM ρ."""
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
        p_before = val_data["p_before"]

        # Assume p_b = 0.5 for VOI computation (symmetric assumption)
        p_b = 0.5

        # Compute VOI using ground-truth ρ
        gt_linear_voi = linear_voi_from_rho(ground_truth_rho, p_before, p_b)
        gt_entropy_voi = entropy_voi_from_rho(ground_truth_rho, p_before, p_b)

        # Compute VOI using market-aware LLM ρ
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
            "mae": float(np.mean(cat_errors)),
            "direction_accuracy": float(np.mean(cat_directions)),
        }

    return {
        "mae": float(np.mean(rho_abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(rho_errors) ** 2))),
        "mean_error": float(np.mean(rho_errors)),  # Bias
        "direction_accuracy": float(np.mean(direction_matches)),
        "n_correct_direction": sum(direction_matches),
        "n_total": len(direction_matches),
        "by_category": by_category,
    }


def compute_voi_correlation_by_category(results: list[dict]) -> dict:
    """Compute VOI vs actual_shift correlation by category."""
    by_category = {}
    categories = set(r["classification"] for r in results)

    for cat in categories:
        cat_results = [r for r in results if r["classification"] == cat]
        if len(cat_results) < 3:  # Need at least 3 for meaningful correlation
            by_category[cat] = {
                "n": len(cat_results),
                "llm_voi_r": None,
                "gt_voi_r": None,
            }
            continue

        actual_shifts = np.array([r["actual_shift"] for r in cat_results])
        llm_voi = np.array([r["llm_entropy_voi"] for r in cat_results])
        gt_voi = np.array([r["gt_entropy_voi"] for r in cat_results])

        try:
            llm_r, _ = stats.pearsonr(llm_voi, actual_shifts)
            gt_r, _ = stats.pearsonr(gt_voi, actual_shifts)
            by_category[cat] = {
                "n": len(cat_results),
                "llm_voi_r": float(llm_r),
                "gt_voi_r": float(gt_r),
            }
        except Exception:
            by_category[cat] = {
                "n": len(cat_results),
                "llm_voi_r": None,
                "gt_voi_r": None,
            }

    return by_category


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
    """Run the market-aware LLM ρ calibration experiment."""
    print("=" * 70)
    print("LLM ρ Calibration with Market-Aware Prompt (Phase 3)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    pairs, validation, baseline = load_data()
    validation_pairs = validation["pairs"]

    print(f"  Curated pairs: {len(pairs)}")
    print(f"  Validation pairs: {len(validation_pairs)}")
    print(f"  Baseline VOI r: {baseline['correlations']['llm_entropy_voi_vs_shift']['r']:.3f}")

    # Estimate LLM ρ for all pairs using market-aware prompt
    llm_results = await estimate_llm_rho_for_pairs(pairs)

    # Compute VOI metrics
    print("\nComputing VOI metrics...")
    results = compute_voi_metrics(pairs, llm_results, validation_pairs)
    print(f"  Computed metrics for {len(results)} pairs")

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(results)

    # Compute estimation metrics
    rho_metrics = compute_rho_estimation_metrics(results)

    # Compute VOI correlation by category
    voi_by_category = compute_voi_correlation_by_category(results)

    # Print key results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Get baseline values for comparison
    baseline_llm_r = baseline['correlations']['llm_entropy_voi_vs_shift']['r']
    baseline_dir_acc = baseline['rho_estimation']['direction_accuracy']

    print("\n--- VOI vs Actual Shift Correlations ---")
    gt_r = correlations['ground_truth_entropy_voi_vs_shift']['r']
    llm_r = correlations['llm_entropy_voi_vs_shift']['r']

    print(f"Ground-truth entropy VOI:   r = {gt_r:.3f} (baseline)")
    print(f"Baseline LLM entropy VOI:   r = {baseline_llm_r:.3f}")
    print(f"Market-aware LLM VOI:       r = {llm_r:.3f} (p = {correlations['llm_entropy_voi_vs_shift']['p']:.4f})")

    improvement = llm_r - baseline_llm_r
    print(f"\nImprovement over baseline:  {improvement:+.3f}")

    print("\n--- ρ Estimation Accuracy ---")
    print(f"LLM ρ vs ground-truth ρ:    r = {correlations['llm_rho_vs_ground_truth_rho']['r']:.3f}")
    print(f"MAE:                        {rho_metrics['mae']:.3f}")
    print(f"RMSE:                       {rho_metrics['rmse']:.3f}")
    print(f"Mean error (bias):          {rho_metrics['mean_error']:.3f}")
    print(f"Direction accuracy:         {rho_metrics['direction_accuracy']:.1%} ({rho_metrics['n_correct_direction']}/{rho_metrics['n_total']})")
    print(f"  (baseline was {baseline_dir_acc:.1%})")

    print("\n--- Direction Accuracy by Category ---")
    for cat, metrics in sorted(rho_metrics["by_category"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {cat}: n={metrics['n']}, MAE={metrics['mae']:.3f}, direction={metrics['direction_accuracy']:.1%}")

    print("\n--- VOI Correlation by Category ---")
    for cat, metrics in sorted(voi_by_category.items(), key=lambda x: -x[1]["n"]):
        llm_voi_r = f"{metrics['llm_voi_r']:.3f}" if metrics['llm_voi_r'] is not None else "N/A"
        gt_voi_r = f"{metrics['gt_voi_r']:.3f}" if metrics['gt_voi_r'] is not None else "N/A"
        print(f"  {cat}: n={metrics['n']}, LLM_VOI_r={llm_voi_r}, GT_VOI_r={gt_voi_r}")

    # Identify failure cases
    failures = identify_failure_cases(results, threshold=0.5)
    print(f"\n--- Failure Cases (|error| > 0.5): {len(failures)} ---")
    for f in failures[:5]:  # Show top 5
        print(f"\n  Q_A: {f['question_a'][:60]}...")
        print(f"  Q_B: {f['question_b'][:60]}...")
        print(f"  Category: {f['classification']}")
        print(f"  Ground-truth ρ: {f['ground_truth_rho']:.3f}, LLM ρ: {f['llm_rho']:.3f}, Error: {f['error']:.3f}")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    if llm_r > 0.45:
        decision = "SHIP"
        interpretation = "Direction was the main bottleneck. Market-aware prompt is sufficient."
    elif llm_r > 0.25:
        decision = "USE_WITH_CAVEATS"
        interpretation = "Partial improvement. Direction helps but magnitude matters too."
    elif llm_r > baseline_llm_r + 0.05:
        decision = "MARGINAL_IMPROVEMENT"
        interpretation = "Improvement over baseline but still weak. Consider magnitude fixes."
    else:
        decision = "DONT_SHIP"
        interpretation = "Direction improvement didn't translate to VOI. Diagnose further."

    print(f"\nDecision: {decision}")
    print(f"Interpretation: {interpretation}")
    print(f"\nKey metrics:")
    print(f"  - Market-aware VOI r: {llm_r:.3f}")
    print(f"  - Baseline VOI r: {baseline_llm_r:.3f}")
    print(f"  - Ground-truth VOI r: {gt_r:.3f}")
    print(f"  - Improvement: {improvement:+.3f}")
    print(f"  - Retained power: {llm_r/gt_r:.1%} (was {baseline_llm_r/gt_r:.1%})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n| Condition | VOI r | Direction Acc |")
    print(f"|-----------|-------|---------------|")
    print(f"| Ground-truth ρ | {gt_r:.3f} | 100% |")
    print(f"| Baseline LLM ρ | {baseline_llm_r:.3f} | {baseline_dir_acc:.1%} |")
    print(f"| Market-aware LLM ρ | {llm_r:.3f} | {rho_metrics['direction_accuracy']:.1%} |")

    # Save results
    output = {
        "metadata": {
            "n_pairs": len(results),
            "model": MODEL,
            "method": "market_aware_two_step",
            "prompt": "shared_drivers_with_warning",
            "generated_at": datetime.now().isoformat(),
        },
        "correlations": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                            for kk, vv in v.items()}
                        for k, v in correlations.items()},
        "rho_estimation": rho_metrics,
        "voi_by_category": voi_by_category,
        "summary": {
            "ground_truth_entropy_voi_r": float(gt_r),
            "baseline_llm_entropy_voi_r": float(baseline_llm_r),
            "market_aware_llm_entropy_voi_r": float(llm_r),
            "improvement_over_baseline": float(improvement),
            "retained_predictive_power": float(llm_r / gt_r) if gt_r > 0 else 0,
            "baseline_retained_power": float(baseline_llm_r / gt_r) if gt_r > 0 else 0,
        },
        "decision": {
            "outcome": decision,
            "interpretation": interpretation,
            "thresholds": {
                "ship": "> 0.45",
                "use_with_caveats": "0.25 - 0.45",
                "dont_ship": "< 0.25",
            },
        },
        "comparison_with_baseline": {
            "baseline_direction_accuracy": float(baseline_dir_acc),
            "market_aware_direction_accuracy": float(rho_metrics['direction_accuracy']),
            "direction_accuracy_improvement": float(rho_metrics['direction_accuracy'] - baseline_dir_acc),
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
