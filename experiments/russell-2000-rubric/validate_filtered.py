"""Validate rubric-filtered cruxes vs unfiltered.

Compares:
- Unfiltered (v2): All cruxes → r = -0.24
- Filtered (v3): Only timely cruxes → r = ?

Success criteria:
- Filtered VOI positively correlates with |return| (r > 0.1)
- Earnings days have higher filtered-VOI than non-earnings days
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path(__file__).parent / "data"
V2_DATA_DIR = Path(__file__).parent.parent / "russell-2000-crux" / "data"


def compute_voi_return_correlation(cruxes_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """Compute correlation between max VOI per stock-day and |return|."""
    # Get max VOI per stock-day
    max_voi = cruxes_df.groupby(["ticker", "date"]).agg({
        "linear_voi": "max",
        "entropy_voi": "max",
    }).reset_index()

    # Join with returns
    analysis_df = max_voi.merge(
        returns_df[["ticker", "date", "return"]],
        on=["ticker", "date"],
        how="inner",
    )
    analysis_df["abs_return"] = analysis_df["return"].abs()

    if len(analysis_df) < 5:
        return {"error": f"Too few observations: {len(analysis_df)}"}

    # Correlation tests
    r_linear, p_linear = pearsonr(analysis_df["linear_voi"], analysis_df["abs_return"])
    rho_linear, _ = spearmanr(analysis_df["linear_voi"], analysis_df["abs_return"])

    return {
        "n_observations": len(analysis_df),
        "pearson_r": r_linear,
        "pearson_p": p_linear,
        "spearman_rho": rho_linear,
        "analysis_df": analysis_df,
    }


def compare_earnings_vs_non_earnings_voi(cruxes_df: pd.DataFrame) -> dict:
    """Compare mean VOI between earnings and non-earnings days."""
    # Get max VOI per stock-day
    max_voi = cruxes_df.groupby(["ticker", "date"]).agg({
        "linear_voi": "max",
        "is_earnings_day": "first",
    }).reset_index()

    earnings_voi = max_voi[max_voi["is_earnings_day"]]["linear_voi"]
    non_earnings_voi = max_voi[~max_voi["is_earnings_day"]]["linear_voi"]

    if len(earnings_voi) < 2 or len(non_earnings_voi) < 2:
        return {"error": "Insufficient data"}

    t_stat, t_p = ttest_ind(earnings_voi, non_earnings_voi)

    return {
        "earnings_mean_voi": earnings_voi.mean(),
        "earnings_n": len(earnings_voi),
        "non_earnings_mean_voi": non_earnings_voi.mean(),
        "non_earnings_n": len(non_earnings_voi),
        "t_test_p": t_p,
        "earnings_higher": earnings_voi.mean() > non_earnings_voi.mean(),
    }


def create_comparison_plot(
    unfiltered_results: dict,
    filtered_results: dict,
    save_path: Path,
):
    """Create side-by-side scatter plots comparing filtered vs unfiltered."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = [
        (unfiltered_results, "Unfiltered (v2)", axes[0]),
        (filtered_results, "Filtered (v3 - timely only)", axes[1]),
    ]

    for results, title, ax in datasets:
        if "error" in results or "analysis_df" not in results:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(title)
            continue

        df = results["analysis_df"]
        ax.scatter(df["linear_voi"], df["abs_return"], alpha=0.6, s=50)
        ax.set_xlabel("Linear VOI")
        ax.set_ylabel("|Return|")

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df["linear_voi"], df["abs_return"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df["linear_voi"].min(), df["linear_voi"].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        r = results["pearson_r"]
        p_val = results["pearson_p"]
        ax.set_title(f"{title}\nr={r:.3f}, p={p_val:.3f}, n={results['n_observations']}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    # Load data
    evaluated_path = DATA_DIR / "cruxes_evaluated.parquet"
    returns_path = V2_DATA_DIR / "stock_returns.parquet"

    if not evaluated_path.exists():
        print("Run apply_timeliness.py first!")
        return

    cruxes_df = pd.read_parquet(evaluated_path)
    returns_df = pd.read_parquet(returns_path)

    print("=" * 60)
    print("RUSSELL 2000 RUBRIC FILTERING VALIDATION (v3)")
    print("=" * 60)

    # Split into filtered (timely) and unfiltered
    filtered_df = cruxes_df[cruxes_df["timely"]]
    print(f"\nCrux counts:")
    print(f"  Unfiltered: {len(cruxes_df)}")
    print(f"  Filtered (timely): {len(filtered_df)} ({len(filtered_df)/len(cruxes_df)*100:.1f}%)")

    # Test 1: VOI vs |return| correlation
    print("\n=== Test 1: VOI vs |Return| Correlation ===")
    unfiltered_results = compute_voi_return_correlation(cruxes_df, returns_df)
    filtered_results = compute_voi_return_correlation(filtered_df, returns_df)

    print(f"\nUnfiltered (v2):")
    if "error" not in unfiltered_results:
        print(f"  n = {unfiltered_results['n_observations']}")
        print(f"  Pearson r = {unfiltered_results['pearson_r']:.4f}")
        print(f"  Pearson p = {unfiltered_results['pearson_p']:.4f}")
    else:
        print(f"  Error: {unfiltered_results['error']}")

    print(f"\nFiltered (v3 - timely only):")
    if "error" not in filtered_results:
        print(f"  n = {filtered_results['n_observations']}")
        print(f"  Pearson r = {filtered_results['pearson_r']:.4f}")
        print(f"  Pearson p = {filtered_results['pearson_p']:.4f}")

        delta_r = filtered_results["pearson_r"] - unfiltered_results.get("pearson_r", 0)
        print(f"\n  Δr = {delta_r:+.4f}")
    else:
        print(f"  Error: {filtered_results['error']}")

    # Test 2: Earnings vs non-earnings VOI comparison
    print("\n=== Test 2: Earnings vs Non-Earnings VOI ===")
    unfiltered_earnings = compare_earnings_vs_non_earnings_voi(cruxes_df)
    filtered_earnings = compare_earnings_vs_non_earnings_voi(filtered_df)

    print(f"\nUnfiltered (v2):")
    if "error" not in unfiltered_earnings:
        print(f"  Earnings mean VOI: {unfiltered_earnings['earnings_mean_voi']:.4f} (n={unfiltered_earnings['earnings_n']})")
        print(f"  Non-earnings mean VOI: {unfiltered_earnings['non_earnings_mean_voi']:.4f} (n={unfiltered_earnings['non_earnings_n']})")
        print(f"  Earnings higher: {'YES' if unfiltered_earnings['earnings_higher'] else 'NO'}")
    else:
        print(f"  Error: {unfiltered_earnings['error']}")

    print(f"\nFiltered (v3 - timely only):")
    if "error" not in filtered_earnings:
        print(f"  Earnings mean VOI: {filtered_earnings['earnings_mean_voi']:.4f} (n={filtered_earnings['earnings_n']})")
        print(f"  Non-earnings mean VOI: {filtered_earnings['non_earnings_mean_voi']:.4f} (n={filtered_earnings['non_earnings_n']})")
        print(f"  Earnings higher: {'YES' if filtered_earnings['earnings_higher'] else 'NO'}")
    else:
        print(f"  Error: {filtered_earnings['error']}")

    # Create comparison plot
    if "error" not in unfiltered_results or "error" not in filtered_results:
        create_comparison_plot(
            unfiltered_results,
            filtered_results,
            DATA_DIR / "filtered_validation.png",
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_correlation = (
        "error" not in filtered_results and
        filtered_results["pearson_r"] > 0.1
    )
    success_earnings_higher = (
        "error" not in filtered_earnings and
        filtered_earnings["earnings_higher"]
    )

    print(f"\n1. Filtered VOI correlates with |return| (r > 0.1): "
          f"{'PASS' if success_correlation else 'FAIL'}")
    print(f"2. Earnings have higher filtered VOI: "
          f"{'PASS' if success_earnings_higher else 'FAIL'}")

    overall = success_correlation and success_earnings_higher
    print(f"\nOverall: {'PASS - Rubric filtering works!' if overall else 'FAIL - Need different approach'}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"\n{'Metric':<35} {'Unfiltered (v2)':<18} {'Filtered (v3)':<18}")
    print("-" * 71)

    if "error" not in unfiltered_results and "error" not in filtered_results:
        print(f"{'VOI vs |return| r':<35} {unfiltered_results['pearson_r']:<18.4f} {filtered_results['pearson_r']:<18.4f}")

    if "error" not in unfiltered_earnings and "error" not in filtered_earnings:
        e_higher_unf = 'YES' if unfiltered_earnings['earnings_higher'] else 'NO'
        e_higher_flt = 'YES' if filtered_earnings['earnings_higher'] else 'NO'
        print(f"{'Earnings VOI > Non-Earnings':<35} {e_higher_unf:<18} {e_higher_flt:<18}")

    print(f"{'N cruxes':<35} {len(cruxes_df):<18} {len(filtered_df):<18}")

    # Save results
    results = {
        "unfiltered": {
            "n_cruxes": len(cruxes_df),
            "voi_correlation": {k: v for k, v in unfiltered_results.items() if k != "analysis_df"} if "error" not in unfiltered_results else unfiltered_results,
            "earnings_comparison": unfiltered_earnings,
        },
        "filtered": {
            "n_cruxes": len(filtered_df),
            "voi_correlation": {k: v for k, v in filtered_results.items() if k != "analysis_df"} if "error" not in filtered_results else filtered_results,
            "earnings_comparison": filtered_earnings,
        },
        "summary": {
            "success_correlation": success_correlation,
            "success_earnings_higher": success_earnings_higher,
            "overall_pass": overall,
        }
    }

    with open(DATA_DIR / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved results to {DATA_DIR / 'validation_results.json'}")


if __name__ == "__main__":
    main()
