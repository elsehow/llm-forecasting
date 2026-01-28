#!/usr/bin/env python3
"""
Validate entropy VOI with LLM-estimated P(Q) against observed |ΔP|.

Compares three approaches:
1. Magnitude: |P(X|Q=Y) - P(X|Q=N)| (baseline from Q2)
2. Entropy VOI with P(Q)=0.5 (current approach)
3. Entropy VOI with LLM P(Q) (new approach)

Success criteria:
- Entropy VOI (LLM P(Q)) beats magnitude at ≥1% ΔP threshold (r > 0.340)
- If not, at least improves over P(Q)=0.5 (r > 0.112)

Usage:
    uv run python validate_with_p_q.py                    # Use full data
    uv run python validate_with_p_q.py --pilot            # Use pilot data
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from llm_forecasting.voi import entropy_voi, linear_voi

# Configuration
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "llm_estimations_with_p_q.json"
INPUT_FILE_PILOT = DATA_DIR / "llm_estimations_with_p_q_pilot.json"

# Baseline numbers from Q2 validation
BASELINE_MAGNITUDE_R = 0.340  # At ≥1% ΔP threshold
BASELINE_ENTROPY_R = 0.112  # At ≥1% ΔP threshold with P(Q)=0.5


def compute_voi_metrics(result: dict) -> dict:
    """Compute all VOI metrics for a single pair.

    Args:
        result: Dict with p_x_given_q_yes, p_x_given_q_no, x_prob_before, p_q_llm

    Returns:
        Dict with magnitude, entropy_voi_baseline, entropy_voi_llm
    """
    p_x = result["x_prob_before"]
    p_x_given_q_yes = result["p_x_given_q_yes"]
    p_x_given_q_no = result["p_x_given_q_no"]
    p_q_llm = result.get("p_q_llm", 0.5)

    # Clamp values to valid ranges
    p_x = max(0.01, min(0.99, p_x))
    p_x_given_q_yes = max(0.01, min(0.99, p_x_given_q_yes))
    p_x_given_q_no = max(0.01, min(0.99, p_x_given_q_no))
    p_q_llm = max(0.01, min(0.99, p_q_llm))

    # Magnitude (baseline from Q2)
    magnitude = abs(p_x_given_q_yes - p_x_given_q_no)

    # Entropy VOI with P(Q)=0.5 (current approach)
    entropy_baseline = entropy_voi(
        p_x=p_x,
        p_q=0.5,
        p_x_given_q_yes=p_x_given_q_yes,
        p_x_given_q_no=p_x_given_q_no,
    )

    # Entropy VOI with LLM P(Q) (new approach)
    entropy_llm = entropy_voi(
        p_x=p_x,
        p_q=p_q_llm,
        p_x_given_q_yes=p_x_given_q_yes,
        p_x_given_q_no=p_x_given_q_no,
    )

    # Also compute linear VOI for comparison
    linear_baseline = linear_voi(
        p_x=p_x,
        p_q=0.5,
        p_x_given_q_yes=p_x_given_q_yes,
        p_x_given_q_no=p_x_given_q_no,
    )

    linear_llm = linear_voi(
        p_x=p_x,
        p_q=p_q_llm,
        p_x_given_q_yes=p_x_given_q_yes,
        p_x_given_q_no=p_x_given_q_no,
    )

    return {
        "magnitude": magnitude,
        "entropy_voi_baseline": entropy_baseline,
        "entropy_voi_llm": entropy_llm,
        "linear_voi_baseline": linear_baseline,
        "linear_voi_llm": linear_llm,
    }


def validate_at_threshold(results: list[dict], min_delta_p: float) -> dict:
    """Validate VOI metrics at a given ΔP threshold.

    Args:
        results: List of result dicts with VOI metrics and x_delta_p
        min_delta_p: Minimum |ΔP| to include

    Returns:
        Dict with correlation stats for each metric
    """
    # Filter by threshold
    filtered = [r for r in results if abs(r["x_delta_p"]) >= min_delta_p]
    n = len(filtered)

    if n < 10:
        return {
            "n": n,
            "status": "insufficient_data",
        }

    # Extract arrays
    abs_delta_p = np.array([abs(r["x_delta_p"]) for r in filtered])
    magnitude = np.array([r["magnitude"] for r in filtered])
    entropy_baseline = np.array([r["entropy_voi_baseline"] for r in filtered])
    entropy_llm = np.array([r["entropy_voi_llm"] for r in filtered])
    linear_baseline = np.array([r["linear_voi_baseline"] for r in filtered])
    linear_llm = np.array([r["linear_voi_llm"] for r in filtered])

    # Compute Spearman correlations
    r_magnitude, p_magnitude = stats.spearmanr(magnitude, abs_delta_p)
    r_entropy_baseline, p_entropy_baseline = stats.spearmanr(entropy_baseline, abs_delta_p)
    r_entropy_llm, p_entropy_llm = stats.spearmanr(entropy_llm, abs_delta_p)
    r_linear_baseline, p_linear_baseline = stats.spearmanr(linear_baseline, abs_delta_p)
    r_linear_llm, p_linear_llm = stats.spearmanr(linear_llm, abs_delta_p)

    return {
        "n": n,
        "status": "ok",
        "magnitude": {
            "r": float(r_magnitude),
            "p": float(p_magnitude),
        },
        "entropy_voi_baseline": {
            "r": float(r_entropy_baseline),
            "p": float(p_entropy_baseline),
        },
        "entropy_voi_llm": {
            "r": float(r_entropy_llm),
            "p": float(p_entropy_llm),
        },
        "linear_voi_baseline": {
            "r": float(r_linear_baseline),
            "p": float(p_linear_baseline),
        },
        "linear_voi_llm": {
            "r": float(r_linear_llm),
            "p": float(p_linear_llm),
        },
    }


def print_comparison_table(thresholds: dict[str, dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("VOI CORRELATION WITH |ΔP| (Spearman r)")
    print("=" * 90)

    # Header
    print(f"{'Threshold':<15} {'n':<8} {'Magnitude':<12} {'Entropy(0.5)':<14} {'Entropy(LLM)':<14} {'Linear(0.5)':<12} {'Linear(LLM)':<12}")
    print("-" * 90)

    for thresh_name, result in thresholds.items():
        n = result["n"]
        if result["status"] != "ok":
            print(f"{thresh_name:<15} {n:<8} {'N/A':<12} {'N/A':<14} {'N/A':<14} {'N/A':<12} {'N/A':<12}")
            continue

        mag_r = result["magnitude"]["r"]
        ent_base_r = result["entropy_voi_baseline"]["r"]
        ent_llm_r = result["entropy_voi_llm"]["r"]
        lin_base_r = result["linear_voi_baseline"]["r"]
        lin_llm_r = result["linear_voi_llm"]["r"]

        # Highlight best entropy result
        ent_llm_str = f"{ent_llm_r:+.3f}"
        if ent_llm_r > ent_base_r:
            ent_llm_str += " *"  # Improved over baseline

        print(f"{thresh_name:<15} {n:<8} {mag_r:+.3f}       {ent_base_r:+.3f}         {ent_llm_str:<14} {lin_base_r:+.3f}       {lin_llm_r:+.3f}")

    print("-" * 90)
    print("* = Improved over P(Q)=0.5 baseline")


def main():
    parser = argparse.ArgumentParser(description="Validate entropy VOI with LLM P(Q)")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use pilot data (100 pairs) instead of full data",
    )
    args = parser.parse_args()

    # Choose input file
    input_file = INPUT_FILE_PILOT if args.pilot else INPUT_FILE

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Run estimate_p_q.py first to generate P(Q) estimates.")
        return

    # Load data
    with open(input_file) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r.get("llm_error") is None and r.get("p_q_error") is None]
    print(f"Loaded {len(results)} valid pairs")
    print(f"Source: {input_file}")

    # Print P(Q) calibration summary
    metadata = data.get("metadata", {})
    calib = metadata.get("calibration", {})
    print(f"\nP(Q) Estimation Calibration:")
    print(f"  Mean P(Q): {calib.get('p_q_mean', 'N/A'):.3f}")
    print(f"  r(P(Q), resolution): {calib.get('r_p_q_vs_resolution', 'N/A')}")

    # Compute VOI metrics for each pair
    for r in results:
        metrics = compute_voi_metrics(r)
        r.update(metrics)

    # Validate at multiple thresholds
    thresholds = {
        "All": 0.0,
        ">=1% ΔP": 0.01,
        ">=2% ΔP": 0.02,
        ">=5% ΔP": 0.05,
    }

    threshold_results = {}
    for name, thresh in thresholds.items():
        threshold_results[name] = validate_at_threshold(results, thresh)

    # Print comparison table
    print_comparison_table(threshold_results)

    # Assessment
    print("\n" + "=" * 90)
    print("ASSESSMENT")
    print("=" * 90)

    # Check key threshold (≥1% ΔP)
    key_result = threshold_results.get(">=1% ΔP", {})
    if key_result.get("status") == "ok":
        mag_r = key_result["magnitude"]["r"]
        ent_base_r = key_result["entropy_voi_baseline"]["r"]
        ent_llm_r = key_result["entropy_voi_llm"]["r"]

        print(f"\nAt ≥1% ΔP threshold (n={key_result['n']}):")
        print(f"  Magnitude correlation: r = {mag_r:.3f} (baseline to beat: {BASELINE_MAGNITUDE_R:.3f})")
        print(f"  Entropy VOI (P(Q)=0.5): r = {ent_base_r:.3f} (Q2 result: {BASELINE_ENTROPY_R:.3f})")
        print(f"  Entropy VOI (LLM P(Q)): r = {ent_llm_r:.3f}")

        improvement = ent_llm_r - ent_base_r
        print(f"\n  Improvement from LLM P(Q): {improvement:+.3f}")

        if ent_llm_r > BASELINE_MAGNITUDE_R:
            print("\n  WIN: Entropy VOI with LLM P(Q) beats magnitude!")
            print("  -> LLM P(Q) estimation is the missing ingredient")
        elif ent_llm_r > ent_base_r:
            print("\n  PARTIAL: Entropy VOI improved but still loses to magnitude")
            print(f"  -> P(Q) helps, but not enough (gap: {BASELINE_MAGNITUDE_R - ent_llm_r:.3f})")
            print("  -> Consider: entropy non-linearity at extremes may still dominate")
        else:
            print("\n  NULL: No improvement from LLM P(Q)")
            print("  -> P(Q) assumption is not the problem")
            print("  -> Consider: entropy formula itself may be unsuitable")

    # P(Q) quality check
    print("\n" + "-" * 50)
    print("P(Q) Quality Check:")

    if calib.get("r_p_q_vs_resolution"):
        r_pq_res = calib["r_p_q_vs_resolution"]
        if r_pq_res > 0.3:
            print(f"  P(Q) estimates correlate well with resolutions (r={r_pq_res:.3f})")
            print("  -> P(Q) estimation is reasonable quality")
        elif r_pq_res > 0.1:
            print(f"  P(Q) estimates weakly correlate with resolutions (r={r_pq_res:.3f})")
            print("  -> P(Q) estimation has signal but is noisy")
        else:
            print(f"  P(Q) estimates don't correlate with resolutions (r={r_pq_res:.3f})")
            print("  -> WARNING: P(Q) estimation may be poor quality")
            print("  -> Results may be unreliable due to bad P(Q) estimates")


if __name__ == "__main__":
    main()
