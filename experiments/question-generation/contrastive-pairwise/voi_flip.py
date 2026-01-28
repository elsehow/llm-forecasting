#!/usr/bin/env python3
"""
VOI Flip Experiment.

Tests whether negating VOI scores within-domain yields positive correlation with |return|.

The within-earnings VOI experiment showed r=-0.15 at stock-day level.
The contrarian pairwise flip showed r=+0.162 by negating Bradley-Terry scores.

Question: Does the same pattern hold for raw VOI? If LLM intuition is systematically
inverted within-domain, low-VOI cruxes should predict higher |return|.
"""

import json
from pathlib import Path

import pandas as pd
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent / "data"
RUSSELL_DATA_DIR = Path(__file__).parent.parent.parent / "russell-2000-crux" / "data"


def load_data():
    """Load cruxes with VOI and returns."""
    cruxes = pd.read_parquet(RUSSELL_DATA_DIR / "cruxes_with_voi.parquet")
    returns = pd.read_parquet(RUSSELL_DATA_DIR / "stock_returns.parquet")
    return cruxes, returns


def run_experiment():
    """Run the VOI flip experiment at crux level."""
    print("=" * 70)
    print("VOI FLIP EXPERIMENT")
    print("=" * 70)
    print("\nHypothesis: If LLM intuition is inverted, low VOI → high |return|")

    # Load data
    print("\n[1/3] Loading data...")
    cruxes, returns = load_data()

    # Filter returns to earnings days only (matching within-earnings experiment)
    earnings_returns = returns[returns["is_earnings_day"]].copy()
    earnings_returns["abs_return"] = earnings_returns["return"].abs()

    print(f"      Cruxes: {len(cruxes)}")
    print(f"      Earnings days: {len(earnings_returns)}")

    # Merge cruxes with returns at crux level
    print("\n[2/3] Merging at crux level...")
    merged = cruxes.merge(
        earnings_returns[["ticker", "date", "abs_return"]],
        on=["ticker", "date"],
        how="inner"
    )
    print(f"      Merged observations: {len(merged)}")
    print(f"      Stock-days: {merged.groupby(['ticker', 'date']).ngroups}")

    # Compute correlations
    print("\n[3/3] Computing correlations...")

    # Original VOI
    r_original, p_original = stats.pearsonr(merged["linear_voi"], merged["abs_return"])
    rho_original, _ = stats.spearmanr(merged["linear_voi"], merged["abs_return"])

    # Flipped VOI (negate)
    merged["flipped_voi"] = -merged["linear_voi"]
    r_flipped, p_flipped = stats.pearsonr(merged["flipped_voi"], merged["abs_return"])
    rho_flipped, _ = stats.spearmanr(merged["flipped_voi"], merged["abs_return"])

    # Results
    print("\n" + "=" * 70)
    print("RESULTS (Crux Level)")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Original VOI':<15} {'Flipped VOI':<15}")
    print("-" * 50)
    print(f"{'Pearson r':<20} {r_original:>12.4f} {r_flipped:>15.4f}")
    print(f"{'p-value':<20} {p_original:>12.4f} {p_flipped:>15.4f}")
    print(f"{'Spearman ρ':<20} {rho_original:>12.4f} {rho_flipped:>15.4f}")

    # Stock-day level (for comparison with original experiment)
    print("\n" + "-" * 50)
    print("Stock-Day Level (aggregated, for comparison)")
    print("-" * 50)

    stockday = merged.groupby(["ticker", "date"]).agg({
        "linear_voi": "mean",
        "abs_return": "first"  # Same for all cruxes in a stock-day
    }).reset_index()

    r_agg, p_agg = stats.pearsonr(stockday["linear_voi"], stockday["abs_return"])
    r_agg_flip, p_agg_flip = stats.pearsonr(-stockday["linear_voi"], stockday["abs_return"])

    print(f"{'Pearson r (orig)':<20} {r_agg:>12.4f}")
    print(f"{'Pearson r (flip)':<20} {r_agg_flip:>12.4f}")
    print(f"{'p-value':<20} {p_agg:>12.4f}")
    print(f"{'n stock-days':<20} {len(stockday):>12}")

    # Interpretation
    print("\n" + "-" * 50)

    if r_flipped > 0.10 and p_flipped < 0.05:
        interpretation = "STRONG SUCCESS"
        explanation = "Flipped VOI achieves r>0.1 with p<0.05"
    elif r_flipped > 0 and p_flipped < 0.10:
        interpretation = "WEAK SUCCESS"
        explanation = "Flipped VOI is directionally correct"
    elif abs(r_flipped) < 0.05:
        interpretation = "NULL"
        explanation = "VOI has no signal even when flipped"
    else:
        interpretation = "STILL INVERTED"
        explanation = "Flipping doesn't help"

    print(f"Interpretation: {interpretation}")
    print(f"  {explanation}")

    # Compare with pairwise flip
    print("\n" + "-" * 50)
    print("Comparison with Contrarian Pairwise Flip")
    print("-" * 50)
    print(f"{'Method':<25} {'r':<10} {'p':<10}")
    print(f"{'Pairwise BT flip':<25} {'+0.162':<10} {'0.015':<10}")
    print(f"{'VOI flip (crux)':<25} {r_flipped:<10.3f} {p_flipped:<10.4f}")
    print(f"{'VOI flip (stock-day)':<25} {r_agg_flip:<10.3f} {p_agg_flip:<10.4f}")

    # Save results
    results = {
        "experiment": "voi_flip",
        "n_crux_observations": len(merged),
        "n_stockdays": len(stockday),
        "crux_level": {
            "original": {"pearson_r": float(r_original), "pearson_p": float(p_original), "spearman_rho": float(rho_original)},
            "flipped": {"pearson_r": float(r_flipped), "pearson_p": float(p_flipped), "spearman_rho": float(rho_flipped)},
        },
        "stockday_level": {
            "original": {"pearson_r": float(r_agg), "pearson_p": float(p_agg)},
            "flipped": {"pearson_r": float(r_agg_flip), "pearson_p": float(p_agg_flip)},
        },
        "interpretation": interpretation,
        "explanation": explanation,
    }

    output_path = DATA_DIR / "voi_flip_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
