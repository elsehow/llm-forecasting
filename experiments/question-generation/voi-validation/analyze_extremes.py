#!/usr/bin/env python3
"""
Analyze VOI robustness at probability extremes.

Three experiments:
1. Threshold sweep: test robustness across 5-95%, 10-90%, 15-85%, 20-80%
2. Linear vs Entropy at extremes: which VOI metric works better at p<0.10 or p>0.90?
3. Ceiling effect: are actual shifts bounded by max possible shift?

Usage:
    uv run python experiments/question-generation/voi-validation/analyze_extremes.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from llm_forecasting.voi import entropy_voi_from_rho, linear_voi_from_rho

RESULTS_DIR = Path(__file__).parent / "results"
SMART_RESULTS = RESULTS_DIR / "voi_validation_smart.json"


def load_data() -> list[dict]:
    """Load the N=74 smart-curated pairs."""
    with open(SMART_RESULTS) as f:
        data = json.load(f)
    return data["results"]


def filter_by_threshold(
    pairs: list[dict], lower: float, upper: float
) -> list[dict]:
    """Filter pairs where p_before is within [lower, upper]."""
    return [p for p in pairs if lower <= p["p_before"] <= upper]


def compute_voi_values(pairs: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Compute linear VOI, entropy VOI, and actual shifts for pairs.

    Assumes p_a (signal probability) = 0.5 since we don't have it in the data.
    Uses p_before as p_b (target probability).
    """
    linear_vois = []
    entropy_vois = []
    actual_shifts = []

    for p in pairs:
        rho = p["rho"]
        p_b = p["p_before"]  # probability of target question
        p_a = 0.5  # assume signal question was 50/50

        linear_voi = linear_voi_from_rho(rho, p_a, p_b)
        entropy_voi = entropy_voi_from_rho(rho, p_a, p_b)

        linear_vois.append(linear_voi)
        entropy_vois.append(entropy_voi)
        actual_shifts.append(p["actual_shift"])

    return linear_vois, entropy_vois, actual_shifts


def experiment_1_threshold_sweep(pairs: list[dict]) -> None:
    """Test robustness across different probability thresholds."""
    print("\nQuestion: Is r=0.653 robust across different probability thresholds?")
    print("Method: Filter pairs at different thresholds, compute correlations\n")

    thresholds = [
        (0.05, 0.95, "5-95%"),
        (0.10, 0.90, "10-90%"),
        (0.15, 0.85, "15-85%"),
        (0.20, 0.80, "20-80%"),
    ]

    print(f"{'Threshold':<12} {'N':>4}  {'r (entropy)':>12} {'r (linear)':>12} {'p-value':>10}")
    print("-" * 56)

    for lower, upper, label in thresholds:
        filtered = filter_by_threshold(pairs, lower, upper)
        n = len(filtered)

        if n < 3:
            print(f"{label:<12} {n:>4}  {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            continue

        linear_vois, entropy_vois, actual_shifts = compute_voi_values(filtered)

        # Compute correlations
        r_entropy, p_entropy = stats.pearsonr(entropy_vois, actual_shifts)
        r_linear, p_linear = stats.pearsonr(linear_vois, actual_shifts)

        # Use the more significant p-value for display
        p_val = min(p_entropy, p_linear)
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"

        print(f"{label:<12} {n:>4}  {r_entropy:>12.3f} {r_linear:>12.3f} {p_str:>10}")

    print("\nInterpretation:")
    print("- If r values are similar across thresholds, the result is robust")
    print("- If r drops sharply at tighter thresholds, 10-90% may be optimal")


def experiment_2_linear_vs_entropy_extremes(pairs: list[dict]) -> None:
    """Compare linear vs entropy VOI at extreme probabilities."""
    print("\nQuestion: Does linear VOI outperform entropy VOI at extreme probabilities?")
    print("Method: Filter to p<0.10 or p>0.90, compare correlations\n")

    # Filter to extremes
    extreme_pairs = [
        p for p in pairs
        if p["p_before"] < 0.10 or p["p_before"] > 0.90
    ]

    n = len(extreme_pairs)
    print(f"At extremes (p<0.10 or p>0.90):")
    print(f"  N pairs: {n}")

    if n < 3:
        print("  Insufficient data for correlation analysis")
        return

    linear_vois, entropy_vois, actual_shifts = compute_voi_values(extreme_pairs)

    r_linear, p_linear = stats.pearsonr(linear_vois, actual_shifts)
    r_entropy, p_entropy = stats.pearsonr(entropy_vois, actual_shifts)

    p_linear_str = f"{p_linear:.4f}" if p_linear >= 0.0001 else "<0.0001"
    p_entropy_str = f"{p_entropy:.4f}" if p_entropy >= 0.0001 else "<0.0001"

    print(f"  Linear VOI vs shift:  r = {r_linear:.3f}, p = {p_linear_str}")
    print(f"  Entropy VOI vs shift: r = {r_entropy:.3f}, p = {p_entropy_str}")

    if r_linear > r_entropy:
        print(f"  Winner: Linear VOI (Δr = {r_linear - r_entropy:.3f})")
    elif r_entropy > r_linear:
        print(f"  Winner: Entropy VOI (Δr = {r_entropy - r_linear:.3f})")
    else:
        print("  Winner: Tie")

    # Also show breakdown by extreme direction
    low_pairs = [p for p in pairs if p["p_before"] < 0.10]
    high_pairs = [p for p in pairs if p["p_before"] > 0.90]

    print(f"\n  Breakdown:")
    print(f"    Low (p<0.10): {len(low_pairs)} pairs")
    print(f"    High (p>0.90): {len(high_pairs)} pairs")


def experiment_3_ceiling_effect(pairs: list[dict]) -> None:
    """Analyze whether shifts are bounded by max possible shift."""
    print("\nQuestion: Are extreme-probability pairs failing because shifts are mathematically bounded?")
    print("Method: Compare max_possible_shift with actual_shift\n")

    max_possible_shifts = []
    actual_shifts = []
    p_befores = []

    for p in pairs:
        p_before = p["p_before"]
        max_shift = min(p_before, 1 - p_before)

        max_possible_shifts.append(max_shift)
        actual_shifts.append(p["actual_shift"])
        p_befores.append(p_before)

    # Compute correlation
    r, p_val = stats.pearsonr(max_possible_shifts, actual_shifts)
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"

    print(f"Correlation (max_possible_shift vs actual_shift):")
    print(f"  r = {r:.3f}, p = {p_str}")

    # Check ceiling saturation at extremes
    extreme_pairs = [
        (mp, actual, pb)
        for mp, actual, pb in zip(max_possible_shifts, actual_shifts, p_befores)
        if pb < 0.10 or pb > 0.90
    ]

    if extreme_pairs:
        saturation_ratios = [actual / mp if mp > 0 else 0 for mp, actual, _ in extreme_pairs]
        mean_saturation = np.mean(saturation_ratios)
        print(f"\nAt extremes (p<0.10 or p>0.90):")
        print(f"  N pairs: {len(extreme_pairs)}")
        print(f"  Mean saturation (actual/max_possible): {mean_saturation:.1%}")
        print(f"  Interpretation: If saturation is high (>50%), ceiling effect is real")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by whether extreme or not
    colors = ["red" if pb < 0.10 or pb > 0.90 else "blue" for pb in p_befores]

    ax.scatter(max_possible_shifts, actual_shifts, c=colors, alpha=0.6, edgecolors="white", s=60)

    # Add diagonal line (ceiling)
    max_val = max(max(max_possible_shifts), max(actual_shifts))
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Ceiling (actual = max)")

    ax.set_xlabel("Max Possible Shift (min(p, 1-p))")
    ax.set_ylabel("Actual Shift")
    ax.set_title(f"Ceiling Effect Analysis (r={r:.3f})")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Extreme (p<0.10 or p>0.90)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Non-extreme"),
        Line2D([0], [0], linestyle="--", color="k", label="Ceiling line"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Save plot
    output_path = RESULTS_DIR / "ceiling_effect.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nPlot saved: {output_path}")
    print("\nInterpretation:")
    print("- Red points near the diagonal = ceiling effect (shift bounded by math)")
    print("- Blue points below diagonal = normal variation")
    print("- If ceiling effect is strong, null result at extremes is NOT VOI failure")


def main():
    pairs = load_data()
    print(f"Loaded {len(pairs)} pairs from {SMART_RESULTS.name}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: THRESHOLD SWEEP")
    print("=" * 70)
    experiment_1_threshold_sweep(pairs)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LINEAR VS ENTROPY AT EXTREMES")
    print("=" * 70)
    experiment_2_linear_vs_entropy_extremes(pairs)

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CEILING EFFECT ANALYSIS")
    print("=" * 70)
    experiment_3_ceiling_effect(pairs)


if __name__ == "__main__":
    main()
