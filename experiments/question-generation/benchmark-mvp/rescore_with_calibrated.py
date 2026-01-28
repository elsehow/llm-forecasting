#!/usr/bin/env python3
"""
Rescore existing benchmark cruxes with calibrated ρ estimator.

Uses estimate_rho_market_aware() (Phase 4 calibrated prompt) instead of the
baseline estimate_rho() to compute VOI for existing cruxes.

Key insight: We don't need to regenerate cruxes - just re-estimate ρ for each
(ultimate, crux) pair. This isolates the ρ estimation variable and is faster.

Phase 4 experiment showed:
- Baseline magnitude: VOI r = -0.049 (overpredicts by +0.208)
- Calibrated magnitude: VOI r = 0.355 (overpredicts by only +0.024)

Usage:
    uv run python experiments/question-generation/benchmark-mvp/rescore_with_calibrated.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

# Load .env from monorepo root
_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

from llm_forecasting.voi import (
    estimate_rho_market_aware,
    linear_voi_from_rho,
    entropy_voi_from_rho,
    entropy_voi_normalized_from_rho,
)

# Paths
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
INPUT_FILE = RESULTS_DIR / "benchmark_results.json"
OUTPUT_FILE = RESULTS_DIR / "benchmark_results_calibrated.json"

# Model for ρ estimation (same as Phase 4 calibration experiment for consistency)
MODEL = "anthropic/claude-sonnet-4-20250514"


def load_benchmark_results() -> dict:
    """Load existing benchmark results."""
    with open(INPUT_FILE) as f:
        return json.load(f)


async def rescore_cruxes(data: dict) -> dict:
    """Rescore all cruxes using calibrated ρ estimator."""
    results = data.get("results", [])
    total_pairs = sum(
        len(r.get("crux_scores", []))
        for r in results
        if "error" not in r
    )

    print(f"Rescoring {total_pairs} (ultimate, crux) pairs with calibrated estimator...", flush=True)
    print(f"Using model: {MODEL}", flush=True)

    # Collect all pairs for processing
    pairs_to_score = []
    for r_idx, result in enumerate(results):
        if "error" in result:
            continue
        ultimate = result["ultimate"]
        for c_idx, crux_score in enumerate(result.get("crux_scores", [])):
            crux = crux_score["crux"]
            pairs_to_score.append((r_idx, c_idx, ultimate, crux))

    # Score each pair sequentially (could batch later)
    new_rhos = {}
    for i, (r_idx, c_idx, ultimate, crux) in enumerate(pairs_to_score):
        print(f"  [{i+1}/{len(pairs_to_score)}] {ultimate[:40]}... × {crux[:30]}...", flush=True)
        rho, reasoning = await estimate_rho_market_aware(ultimate, crux, model=MODEL)
        new_rhos[(r_idx, c_idx)] = (rho, reasoning)

    # Update results with new ρ and recompute VOI
    rescored_results = []
    all_linear_vois = []
    all_entropy_vois = []
    all_entropy_vois_norm = []

    for r_idx, result in enumerate(results):
        if "error" in result:
            rescored_results.append(result)
            continue

        new_result = {**result}
        new_crux_scores = []

        for c_idx, cs in enumerate(result.get("crux_scores", [])):
            new_cs = {**cs}

            # Store old ρ for comparison
            new_cs["rho_estimated_baseline"] = cs.get("rho_estimated", 0)
            new_cs["rho_reasoning_baseline"] = cs.get("rho_reasoning", "")

            # Get new calibrated ρ
            rho_calibrated, reasoning_calibrated = new_rhos.get(
                (r_idx, c_idx), (0.0, "Not scored")
            )
            new_cs["rho_estimated"] = rho_calibrated
            new_cs["rho_reasoning"] = reasoning_calibrated
            new_cs["rho_method"] = "market_aware_calibrated"

            # Get base rates from conditionals
            conditionals = cs.get("conditionals", {})
            if "error" in conditionals:
                new_cs["voi_calibrated"] = 0.0
                new_cs["voi_linear_calibrated"] = 0.0
                new_cs["voi_entropy_calibrated"] = 0.0
                new_cs["voi_entropy_normalized_calibrated"] = 0.0
            else:
                p_ultimate = conditionals.get("p_ultimate", 0.5)
                p_crux = conditionals.get("p_crux", 0.5)

                # Recompute VOI using calibrated ρ
                voi_linear = linear_voi_from_rho(rho_calibrated, p_ultimate, p_crux)
                voi_entropy = entropy_voi_from_rho(rho_calibrated, p_ultimate, p_crux)
                voi_entropy_norm = entropy_voi_normalized_from_rho(
                    rho_calibrated, p_ultimate, p_crux
                )

                new_cs["voi_calibrated"] = voi_linear
                new_cs["voi_linear_calibrated"] = voi_linear
                new_cs["voi_entropy_calibrated"] = voi_entropy
                new_cs["voi_entropy_normalized_calibrated"] = voi_entropy_norm

                # Also update main VOI fields to use calibrated values
                new_cs["voi"] = voi_linear
                new_cs["voi_linear"] = voi_linear
                new_cs["voi_entropy"] = voi_entropy
                new_cs["voi_entropy_normalized"] = voi_entropy_norm

                all_linear_vois.append(voi_linear)
                all_entropy_vois.append(voi_entropy)
                all_entropy_vois_norm.append(voi_entropy_norm)

            new_crux_scores.append(new_cs)

        new_result["crux_scores"] = new_crux_scores
        rescored_results.append(new_result)

    # Update metadata
    new_metadata = {
        **data.get("metadata", {}),
        "rescored_timestamp": datetime.now().isoformat(),
        "rho_method": "market_aware_calibrated",
        "rho_model": MODEL,
        "original_timestamp": data.get("metadata", {}).get("timestamp"),
    }

    # Update statistics
    new_statistics = {
        **data.get("statistics", {}),
        "mean_voi_linear": float(np.mean(all_linear_vois)) if all_linear_vois else 0,
        "median_voi_linear": float(np.median(all_linear_vois)) if all_linear_vois else 0,
        "max_voi_linear": float(np.max(all_linear_vois)) if all_linear_vois else 0,
        "mean_voi_entropy": float(np.mean(all_entropy_vois)) if all_entropy_vois else 0,
        "median_voi_entropy": float(np.median(all_entropy_vois)) if all_entropy_vois else 0,
        "max_voi_entropy": float(np.max(all_entropy_vois)) if all_entropy_vois else 0,
        "mean_voi_entropy_normalized": float(np.mean(all_entropy_vois_norm)) if all_entropy_vois_norm else 0,
        "median_voi_entropy_normalized": float(np.median(all_entropy_vois_norm)) if all_entropy_vois_norm else 0,
        "max_voi_entropy_normalized": float(np.max(all_entropy_vois_norm)) if all_entropy_vois_norm else 0,
        # Keep backwards compat
        "mean_voi": float(np.mean(all_linear_vois)) if all_linear_vois else 0,
        "median_voi": float(np.median(all_linear_vois)) if all_linear_vois else 0,
        "max_voi": float(np.max(all_linear_vois)) if all_linear_vois else 0,
    }

    return {
        "metadata": new_metadata,
        "statistics": new_statistics,
        "results": rescored_results,
    }


def compare_statistics(original: dict, rescored: dict):
    """Print comparison of original vs rescored statistics."""
    orig_stats = original.get("statistics", {})
    new_stats = rescored.get("statistics", {})

    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Calibrated ρ Estimator")
    print("=" * 70)

    metrics = [
        ("Mean Linear VOI", "mean_voi_linear"),
        ("Median Linear VOI", "median_voi_linear"),
        ("Max Linear VOI", "max_voi_linear"),
        ("Mean Entropy VOI", "mean_voi_entropy"),
        ("Median Entropy VOI", "median_voi_entropy"),
    ]

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Calibrated':>12} {'Change':>12}")
    print("-" * 63)
    for label, key in metrics:
        orig = orig_stats.get(key, 0)
        new = new_stats.get(key, 0)
        change = new - orig
        print(f"{label:<25} {orig:>12.4f} {new:>12.4f} {change:>+12.4f}")


def analyze_rho_changes(rescored: dict):
    """Analyze how ρ estimates changed between methods."""
    results = rescored.get("results", [])

    rho_baseline = []
    rho_calibrated = []
    direction_flips = 0
    total = 0

    for result in results:
        if "error" in result:
            continue
        for cs in result.get("crux_scores", []):
            baseline = cs.get("rho_estimated_baseline", 0)
            calibrated = cs.get("rho_estimated", 0)
            rho_baseline.append(baseline)
            rho_calibrated.append(calibrated)

            # Check for direction flip
            if (baseline > 0.05 and calibrated < -0.05) or (baseline < -0.05 and calibrated > 0.05):
                direction_flips += 1
            total += 1

    print("\n" + "-" * 70)
    print("ρ ESTIMATION CHANGES")
    print("-" * 70)

    print(f"\nMean ρ:")
    print(f"  Baseline:   {np.mean(rho_baseline):+.3f}")
    print(f"  Calibrated: {np.mean(rho_calibrated):+.3f}")

    print(f"\nMean |ρ| (magnitude):")
    print(f"  Baseline:   {np.mean(np.abs(rho_baseline)):.3f}")
    print(f"  Calibrated: {np.mean(np.abs(rho_calibrated)):.3f}")
    print(f"  (Calibrated prompt should reduce magnitude overprediction)")

    print(f"\nDirection flips: {direction_flips}/{total} ({direction_flips/total:.1%})")
    print("  (Cases where sign changed between estimators)")


async def main():
    print("=" * 70, flush=True)
    print("RESCORE BENCHMARK WITH CALIBRATED ρ ESTIMATOR", flush=True)
    print("=" * 70, flush=True)

    # Load original results
    print(f"\nLoading benchmark results from {INPUT_FILE}...", flush=True)
    original = load_benchmark_results()
    print(f"  Original timestamp: {original.get('metadata', {}).get('timestamp')}", flush=True)
    print(f"  Total cruxes: {original.get('metadata', {}).get('total_cruxes')}", flush=True)

    # Rescore with calibrated estimator
    rescored = await rescore_cruxes(original)

    # Compare statistics
    compare_statistics(original, rescored)

    # Analyze ρ changes
    analyze_rho_changes(rescored)

    # Save rescored results
    print(f"\nSaving rescored results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(rescored, f, indent=2)

    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print("\nNext step: Run validation with --results calibrated:")
    print("  uv run python experiments/question-generation/paper-trading/validate_q4.py --results calibrated")


if __name__ == "__main__":
    asyncio.run(main())
