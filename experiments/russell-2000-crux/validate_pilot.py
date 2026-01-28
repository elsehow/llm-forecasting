"""Validate Phase 0: Earnings Pilot.

Tests:
1. Do earnings days have higher |return| than non-earnings days? (baseline)
2. Does model generate earnings-related cruxes on earnings days?
3. Does VOI correlate with |return| on earnings days?
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"


def check_earnings_crux(crux: str, ticker: str) -> bool:
    """Check if a crux mentions earnings."""
    earnings_keywords = [
        "earnings", "quarter", "q1", "q2", "q3", "q4",
        "revenue", "profit", "eps", "guidance",
        "beat", "miss", "expectations", "analyst",
        "financial results", "report"
    ]
    crux_lower = crux.lower()
    return any(kw in crux_lower for kw in earnings_keywords)


def analyze_crux_relevance(cruxes_df: pd.DataFrame) -> dict:
    """Check if model generates earnings-related cruxes on earnings days.

    Success criterion: >80% of earnings days have earnings crux in top 3.
    """
    # Group by stock-day
    stock_days = cruxes_df.groupby(["ticker", "date"])

    n_with_earnings_crux = 0
    n_total = 0

    for (ticker, date), group in stock_days:
        n_total += 1

        # Check if any crux mentions earnings
        has_earnings_crux = any(
            check_earnings_crux(crux, ticker)
            for crux in group["crux"]
        )

        if has_earnings_crux:
            n_with_earnings_crux += 1

    hit_rate = n_with_earnings_crux / n_total if n_total > 0 else 0

    return {
        "n_stock_days": n_total,
        "n_with_earnings_crux": n_with_earnings_crux,
        "earnings_crux_hit_rate": hit_rate,
        "success": hit_rate >= 0.8,
    }


def analyze_voi_return_correlation(
    cruxes_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> dict:
    """Check if VOI correlates with |return| on earnings days.

    Primary test: max VOI per stock-day vs |return|
    """
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
    r_entropy, p_entropy = pearsonr(analysis_df["entropy_voi"], analysis_df["abs_return"])
    rho_linear, _ = spearmanr(analysis_df["linear_voi"], analysis_df["abs_return"])
    rho_entropy, _ = spearmanr(analysis_df["entropy_voi"], analysis_df["abs_return"])

    return {
        "n_observations": len(analysis_df),
        "linear_voi": {
            "pearson_r": r_linear,
            "pearson_p": p_linear,
            "spearman_rho": rho_linear,
        },
        "entropy_voi": {
            "pearson_r": r_entropy,
            "pearson_p": p_entropy,
            "spearman_rho": rho_entropy,
        },
        "analysis_df": analysis_df,
    }


def analyze_earnings_vs_non_earnings(returns_df: pd.DataFrame) -> dict:
    """Compare |return| on earnings days vs non-earnings days.

    Baseline check: earnings days should have higher volatility.
    """
    earnings_returns = returns_df[returns_df["is_earnings_day"]]["return"].abs()
    non_earnings_returns = returns_df[~returns_df["is_earnings_day"]]["return"].abs()

    if len(earnings_returns) < 5 or len(non_earnings_returns) < 5:
        return {"error": "Insufficient data"}

    t_stat, t_p = ttest_ind(earnings_returns, non_earnings_returns)

    return {
        "earnings_days": {
            "n": len(earnings_returns),
            "mean_abs_return": earnings_returns.mean(),
            "std_abs_return": earnings_returns.std(),
        },
        "non_earnings_days": {
            "n": len(non_earnings_returns),
            "mean_abs_return": non_earnings_returns.mean(),
            "std_abs_return": non_earnings_returns.std(),
        },
        "t_test": {
            "t_statistic": t_stat,
            "p_value": t_p,
            "earnings_higher": earnings_returns.mean() > non_earnings_returns.mean(),
        }
    }


def create_validation_plot(
    voi_results: dict,
    save_path: Path,
):
    """Create scatter plot of VOI vs |return|."""
    if "error" in voi_results or "analysis_df" not in voi_results:
        print("Cannot create plot - insufficient data")
        return

    df = voi_results["analysis_df"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (voi_col, title) in enumerate([
        ("linear_voi", "Linear VOI"),
        ("entropy_voi", "Entropy VOI"),
    ]):
        ax = axes[i]
        ax.scatter(df[voi_col], df["abs_return"], alpha=0.6, s=50)
        ax.set_xlabel(title)
        ax.set_ylabel("|Return|")

        # Add trend line
        z = np.polyfit(df[voi_col], df["abs_return"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[voi_col].min(), df[voi_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        r = voi_results[voi_col.replace("_voi", "_voi")]["pearson_r"]
        p_val = voi_results[voi_col.replace("_voi", "_voi")]["pearson_p"]
        ax.set_title(f"{title} vs |Return|\nr={r:.3f}, p={p_val:.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    # Load data
    cruxes_path = DATA_DIR / "cruxes_with_voi.parquet"
    returns_path = DATA_DIR / "stock_returns.parquet"

    if not cruxes_path.exists():
        print("Run compute_voi.py first!")
        return

    cruxes_df = pd.read_parquet(cruxes_path)
    returns_df = pd.read_parquet(returns_path)

    print("=" * 60)
    print("PHASE 0 VALIDATION: Earnings Pilot")
    print("=" * 60)

    # Test 1: Baseline - earnings days should have higher |return|
    print("\n=== Test 1: Earnings vs Non-Earnings Baseline ===")
    baseline_results = analyze_earnings_vs_non_earnings(returns_df)
    if "error" not in baseline_results:
        print(f"Earnings days: n={baseline_results['earnings_days']['n']}, "
              f"mean |return|={baseline_results['earnings_days']['mean_abs_return']:.4f}")
        print(f"Non-earnings days: n={baseline_results['non_earnings_days']['n']}, "
              f"mean |return|={baseline_results['non_earnings_days']['mean_abs_return']:.4f}")
        print(f"T-test p-value: {baseline_results['t_test']['p_value']:.4f}")
        print(f"Earnings higher: {'YES' if baseline_results['t_test']['earnings_higher'] else 'NO'}")
    else:
        print(f"Error: {baseline_results['error']}")

    # Test 2: Crux relevance - model should identify earnings
    print("\n=== Test 2: Earnings Crux Identification ===")
    relevance_results = analyze_crux_relevance(cruxes_df)
    print(f"Stock-days analyzed: {relevance_results['n_stock_days']}")
    print(f"With earnings-related crux: {relevance_results['n_with_earnings_crux']}")
    print(f"Hit rate: {relevance_results['earnings_crux_hit_rate']:.1%}")
    print(f"Success (>=80%): {'YES' if relevance_results['success'] else 'NO'}")

    # Test 3: VOI correlation with |return|
    print("\n=== Test 3: VOI vs |Return| Correlation ===")
    voi_results = analyze_voi_return_correlation(cruxes_df, returns_df)
    if "error" not in voi_results:
        print(f"Observations: {voi_results['n_observations']}")
        print(f"\nLinear VOI:")
        print(f"  Pearson r = {voi_results['linear_voi']['pearson_r']:.4f}")
        print(f"  Pearson p = {voi_results['linear_voi']['pearson_p']:.4f}")
        print(f"  Spearman ρ = {voi_results['linear_voi']['spearman_rho']:.4f}")
        print(f"\nEntropy VOI:")
        print(f"  Pearson r = {voi_results['entropy_voi']['pearson_r']:.4f}")
        print(f"  Pearson p = {voi_results['entropy_voi']['pearson_p']:.4f}")
        print(f"  Spearman ρ = {voi_results['entropy_voi']['spearman_rho']:.4f}")

        # Create plot
        create_validation_plot(voi_results, DATA_DIR / "pilot_validation.png")
    else:
        print(f"Error: {voi_results['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_baseline = (
        "error" not in baseline_results and
        baseline_results["t_test"]["earnings_higher"]
    )
    success_relevance = relevance_results["success"]
    success_voi = (
        "error" not in voi_results and
        voi_results["linear_voi"]["pearson_r"] > 0.15
    )

    print(f"\n1. Baseline (earnings > non-earnings): {'PASS' if success_baseline else 'FAIL'}")
    print(f"2. Crux relevance (>80% hit rate): {'PASS' if success_relevance else 'FAIL'}")
    print(f"3. VOI correlation (r > 0.15): {'PASS' if success_voi else 'FAIL'}")

    overall = success_baseline and success_relevance and success_voi
    print(f"\nOverall: {'PASS - Ready for Phase 1' if overall else 'FAIL - Review results'}")

    # Save results
    results = {
        "baseline": baseline_results if "error" not in baseline_results else {"error": baseline_results["error"]},
        "crux_relevance": relevance_results,
        "voi_correlation": {k: v for k, v in voi_results.items() if k != "analysis_df"} if "error" not in voi_results else {"error": voi_results["error"]},
        "summary": {
            "success_baseline": success_baseline,
            "success_relevance": success_relevance,
            "success_voi": success_voi,
            "overall_pass": overall,
        }
    }

    with open(DATA_DIR / "pilot_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved results to pilot_validation_results.json")


if __name__ == "__main__":
    main()
