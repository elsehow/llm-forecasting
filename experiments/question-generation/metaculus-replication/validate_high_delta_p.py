#!/usr/bin/env python3
"""
Stratified validation of LLM conditional estimation by observed ΔP.

Hypothesis: Signal is stronger in pairs where X actually moved.
Low-ΔP pairs are noise (already priced in, unrelated, measurement window missed).

Method:
- Stratify existing llm_estimations.json by |ΔP| threshold
- For each stratum, compute r(|estimated shift|, |observed ΔP|) and direction accuracy
- Compare across thresholds

Success criteria:
- r > 0.30 and direction > 60% in high-ΔP strata validates "LLM works when ground truth is clean"
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "llm_estimations.json"

THRESHOLDS = [0, 0.01, 0.02, 0.05]  # |ΔP| thresholds


def compute_direction_accuracy(results: list) -> tuple[int, int]:
    """Compute direction accuracy for results with |ΔP| > 1%."""
    correct = 0
    total = 0
    for r in results:
        if abs(r["x_delta_p"]) < 0.01:
            continue

        llm_direction = 1 if r["p_x_given_q_yes"] > r["p_x_given_q_no"] else -1
        q_resolution = r.get("q_resolution", 1)
        expected_dp_sign = llm_direction if q_resolution == 1 else -llm_direction
        actual_dp_sign = 1 if r["x_delta_p"] > 0 else -1

        total += 1
        if expected_dp_sign == actual_dp_sign:
            correct += 1

    return correct, total


def analyze_stratum(results: list, threshold: float) -> dict:
    """Analyze a stratum of results with |ΔP| >= threshold."""
    filtered = [r for r in results if abs(r["x_delta_p"]) >= threshold]

    if len(filtered) < 5:
        return {"n": len(filtered), "r": None, "direction": None, "too_small": True}

    estimated_shifts = np.array([r["estimated_shift"] for r in filtered])
    observed_dps = np.array([abs(r["x_delta_p"]) for r in filtered])

    r, p_value = stats.spearmanr(estimated_shifts, observed_dps)
    correct, total = compute_direction_accuracy(filtered)
    direction = correct / total if total > 0 else None

    return {
        "n": len(filtered),
        "r": r,
        "p_value": p_value,
        "direction": direction,
        "direction_correct": correct,
        "direction_total": total,
        "too_small": False,
        "mean_estimated_shift": float(np.mean(estimated_shifts)),
        "mean_observed_dp": float(np.mean(observed_dps)),
    }


def print_sample_pairs(results: list, threshold: float, n: int = 3):
    """Print sample pairs at given threshold level."""
    filtered = [r for r in results if abs(r["x_delta_p"]) >= threshold]
    sorted_pairs = sorted(filtered, key=lambda x: -abs(x["x_delta_p"]))[:n]

    for i, r in enumerate(sorted_pairs):
        print(f"\n    [{i+1}] Q: {r['q_title'][:55]}...")
        print(f"        X: {r['x_title'][:55]}...")
        print(f"        Est. shift: {r['estimated_shift']:.2f}, Obs. |ΔP|: {abs(r['x_delta_p']):.2f}")
        llm_dir = "+" if r["p_x_given_q_yes"] > r["p_x_given_q_no"] else "-"
        dp_dir = "+" if r["x_delta_p"] > 0 else "-"
        print(f"        LLM direction: {llm_dir}, Actual ΔP: {dp_dir}")


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r["llm_error"] is None]
    print(f"Total valid results: {len(results)}")

    # Main stratified analysis
    print(f"\n{'='*70}")
    print("STRATIFIED ANALYSIS BY |ΔP| THRESHOLD")
    print(f"{'='*70}")
    print(f"\n{'Threshold':<12} {'n':<8} {'r':<12} {'Direction':<15} {'Interpretation'}")
    print("-" * 70)

    strata = {}
    for threshold in THRESHOLDS:
        result = analyze_stratum(results, threshold)
        strata[threshold] = result

        thresh_str = f"{threshold:.0%}"
        if result["too_small"]:
            print(f"|ΔP| >= {thresh_str:<5} {result['n']:<8} {'N/A':<12} {'N/A':<15} Too few samples")
        else:
            r_str = f"{result['r']:.3f}"
            dir_str = f"{result['direction']:.1%} ({result['direction_correct']}/{result['direction_total']})"

            # Interpretation
            if result["r"] >= 0.30 and result["direction"] >= 0.60:
                interp = "STRONG"
            elif result["r"] >= 0.20 and result["direction"] >= 0.55:
                interp = "Moderate"
            else:
                interp = "Weak"

            print(f"|ΔP| >= {thresh_str:<5} {result['n']:<8} {r_str:<12} {dir_str:<15} {interp}")

    # Show trend
    print(f"\n{'='*70}")
    print("TREND ANALYSIS")
    print(f"{'='*70}")

    r_values = [strata[t]["r"] for t in THRESHOLDS if not strata[t]["too_small"]]
    dir_values = [strata[t]["direction"] for t in THRESHOLDS if not strata[t]["too_small"] and strata[t]["direction"]]

    if len(r_values) >= 2:
        r_trend = r_values[-1] - r_values[0]
        print(f"\n  r trend (all -> highest threshold): {r_trend:+.3f}")
        if r_trend > 0.05:
            print("  -> Signal INCREASES with |ΔP|: Validates 'LLM works when ground truth is clean'")
        elif r_trend < -0.05:
            print("  -> Signal DECREASES with |ΔP|: Unexpected, investigate further")
        else:
            print("  -> Signal FLAT: LLM estimation quality independent of ground truth magnitude")

    if len(dir_values) >= 2:
        dir_trend = dir_values[-1] - dir_values[0]
        print(f"\n  Direction trend: {dir_trend:+.1%}")

    # Sample pairs at different levels
    print(f"\n{'='*70}")
    print("SAMPLE PAIRS BY THRESHOLD")
    print(f"{'='*70}")

    for threshold in [0.05, 0.02]:
        n_pairs = len([r for r in results if abs(r["x_delta_p"]) >= threshold])
        if n_pairs >= 3:
            print(f"\n  |ΔP| >= {threshold:.0%} (n={n_pairs}):")
            print_sample_pairs(results, threshold)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY FOR PAPER")
    print(f"{'='*70}")

    high_dp = strata.get(0.02, strata.get(0.01))
    if high_dp and not high_dp["too_small"]:
        print(f"\n  High-movement subset (|ΔP| >= 2%):")
        print(f"    n = {high_dp['n']}")
        print(f"    r = {high_dp['r']:.3f}")
        print(f"    Direction = {high_dp['direction']:.1%}")

        if high_dp["r"] >= 0.30 and high_dp["direction"] >= 0.60:
            print("\n  CONCLUSION: Validates LLM conditional estimation")
            print("  -> Use high-movement subset for paper claims")
            print("  -> Low-movement pairs are noise, not LLM failure")
        elif high_dp["r"] >= 0.20:
            print("\n  CONCLUSION: Moderate support for LLM estimation")
            print("  -> Signal improves with cleaner ground truth")
            print("  -> Combine with CivBench for stronger validation")
        else:
            print("\n  CONCLUSION: Limited support")
            print("  -> LLM estimation doesn't improve much with cleaner data")
            print("  -> Lean on CivBench for primary validation")


if __name__ == "__main__":
    main()
