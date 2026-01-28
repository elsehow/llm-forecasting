"""Validate VOI predictions on held-out test data.

Primary test: VOI (from train) should correlate with |return| on test Fed days.
Null baseline: VOI should NOT correlate with returns on non-Fed days.

Tests both Linear VOI and Entropy VOI to compare their predictive validity.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"


def compute_avg_abs_returns(
    returns_df: pd.DataFrame,
    dates: list[str] | None = None,
) -> pd.Series:
    """Compute average absolute return per ticker.

    Args:
        returns_df: DataFrame with ticker, date, return columns
        dates: Optional filter to specific dates

    Returns:
        Series mapping ticker to average |return|
    """
    if dates is not None:
        returns_df = returns_df[returns_df["date"].isin(dates)]

    return returns_df.groupby("ticker")["return"].apply(lambda x: np.abs(x).mean())


def validate_voi_metric(
    voi_df: pd.DataFrame,
    test_returns: pd.DataFrame,
    non_fed_returns: pd.DataFrame,
    metric_name: str,
    test_dates: list[str],
) -> dict:
    """Validate a single VOI metric.

    Args:
        voi_df: DataFrame with ticker and VOI columns
        test_returns: Returns on Fed days
        non_fed_returns: Returns on non-Fed days
        metric_name: Which VOI column to test ('linear_voi' or 'entropy_voi')
        test_dates: List of test period Fed dates

    Returns:
        Dict with validation results
    """
    # Compute avg |return| on test Fed days
    fed_abs_returns = compute_avg_abs_returns(test_returns, test_dates)

    # Compute avg |return| on non-Fed days (null baseline)
    non_fed_abs_returns = compute_avg_abs_returns(non_fed_returns)

    # Join with VOI scores
    analysis_df = voi_df[["ticker", metric_name]].copy()
    analysis_df["fed_abs_return"] = analysis_df["ticker"].map(fed_abs_returns)
    analysis_df["non_fed_abs_return"] = analysis_df["ticker"].map(non_fed_abs_returns)
    analysis_df = analysis_df.dropna()

    if len(analysis_df) < 20:
        return {"error": f"Too few stocks with complete data: {len(analysis_df)}"}

    # Primary validation: VOI vs Fed-day |returns|
    r_fed, p_fed = pearsonr(analysis_df[metric_name], analysis_df["fed_abs_return"])
    rho_fed, p_rho_fed = spearmanr(analysis_df[metric_name], analysis_df["fed_abs_return"])

    # Null baseline: VOI vs non-Fed-day |returns|
    r_null, p_null = pearsonr(analysis_df[metric_name], analysis_df["non_fed_abs_return"])
    rho_null, p_rho_null = spearmanr(analysis_df[metric_name], analysis_df["non_fed_abs_return"])

    return {
        "metric": metric_name,
        "n_stocks": len(analysis_df),
        "primary_test": {
            "description": f"Correlation between {metric_name} (train) and |return| (test Fed days)",
            "pearson_r": r_fed,
            "pearson_p": p_fed,
            "spearman_rho": rho_fed,
            "spearman_p": p_rho_fed,
            "success": r_fed > 0.2 and p_fed < 0.05,
        },
        "null_baseline": {
            "description": f"Correlation between {metric_name} (train) and |return| (non-Fed days)",
            "pearson_r": r_null,
            "pearson_p": p_null,
            "spearman_rho": rho_null,
            "spearman_p": p_rho_null,
            "success": abs(r_null) < 0.1,  # Should be near zero
        },
        "analysis_df": analysis_df,
    }


def tertile_analysis(
    voi_df: pd.DataFrame,
    test_returns: pd.DataFrame,
    metric_name: str,
    test_dates: list[str],
) -> dict:
    """Compare high-VOI vs low-VOI stocks on Fed days.

    Divides stocks into tertiles by VOI and compares average |returns|.
    """
    # Filter to test dates
    test_returns = test_returns[test_returns["date"].isin(test_dates)].copy()

    # Join VOI scores
    test_returns = test_returns.merge(voi_df[["ticker", metric_name]], on="ticker")
    test_returns = test_returns.dropna(subset=[metric_name])

    # Create tertiles
    test_returns["voi_tertile"] = pd.qcut(
        test_returns[metric_name],
        q=3,
        labels=["low", "medium", "high"],
    )

    # Compare |returns| by tertile
    tertile_stats = test_returns.groupby("voi_tertile").agg({
        "return": [
            ("mean_abs_return", lambda x: np.abs(x).mean()),
            ("std_abs_return", lambda x: np.abs(x).std()),
            ("n_obs", "count"),
        ]
    })
    tertile_stats.columns = tertile_stats.columns.get_level_values(1)

    # T-test: high VOI vs low VOI
    high_voi_returns = test_returns[test_returns["voi_tertile"] == "high"]["return"].abs()
    low_voi_returns = test_returns[test_returns["voi_tertile"] == "low"]["return"].abs()
    t_stat, t_p = ttest_ind(high_voi_returns, low_voi_returns)

    return {
        "metric": metric_name,
        "tertile_stats": tertile_stats.to_dict(),
        "high_vs_low_ttest": {
            "t_statistic": t_stat,
            "p_value": t_p,
            "high_mean": high_voi_returns.mean(),
            "low_mean": low_voi_returns.mean(),
            "effect_size": (high_voi_returns.mean() - low_voi_returns.mean()) / low_voi_returns.std(),
        }
    }


def create_validation_plots(
    linear_results: dict,
    entropy_results: dict,
    save_path: Path,
):
    """Create visualization of validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for i, (results, name) in enumerate([
        (linear_results, "Linear VOI"),
        (entropy_results, "Entropy VOI"),
    ]):
        df = results["analysis_df"]

        # Top row: VOI vs Fed-day returns
        ax = axes[0, i]
        ax.scatter(df[results["metric"]], df["fed_abs_return"], alpha=0.5, s=20)
        ax.set_xlabel(name)
        ax.set_ylabel("Avg |Return| on Test Fed Days")
        ax.set_title(f"{name} vs Fed-Day Returns\nr={results['primary_test']['pearson_r']:.3f}, p={results['primary_test']['pearson_p']:.3f}")

        # Add trend line
        z = np.polyfit(df[results["metric"]], df["fed_abs_return"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[results["metric"]].min(), df[results["metric"]].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Bottom row: VOI vs non-Fed-day returns (should be flat)
        ax = axes[1, i]
        ax.scatter(df[results["metric"]], df["non_fed_abs_return"], alpha=0.5, s=20, color="gray")
        ax.set_xlabel(name)
        ax.set_ylabel("Avg |Return| on Non-Fed Days")
        ax.set_title(f"{name} vs Non-Fed-Day Returns (Null)\nr={results['null_baseline']['pearson_r']:.3f}, p={results['null_baseline']['pearson_p']:.3f}")

        # Add trend line
        z = np.polyfit(df[results["metric"]], df["non_fed_abs_return"], 1)
        p = np.poly1d(z)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved validation plot to {save_path}")


def main():
    # Load data
    fed_path = DATA_DIR / "fed_meetings.json"
    voi_path = DATA_DIR / "voi_scores.parquet"
    fed_returns_path = DATA_DIR / "fed_returns.parquet"
    non_fed_returns_path = DATA_DIR / "non_fed_returns.parquet"

    for path in [fed_path, voi_path, fed_returns_path, non_fed_returns_path]:
        if not path.exists():
            print(f"Missing {path.name}. Run previous scripts first!")
            return

    with open(fed_path) as f:
        fed_data = json.load(f)

    voi_df = pd.read_parquet(voi_path)
    fed_returns = pd.read_parquet(fed_returns_path)
    non_fed_returns = pd.read_parquet(non_fed_returns_path)

    test_dates = [m["date"] for m in fed_data["test"]]
    print(f"Test period: {len(test_dates)} Fed meetings")
    print(f"Test date range: {min(test_dates)} to {max(test_dates)}")

    # Filter to test period
    test_returns = fed_returns[fed_returns["date"].isin(test_dates)]
    print(f"Test period observations: {len(test_returns)}")

    # ============================================================
    # VALIDATION: LINEAR VOI
    # ============================================================
    print("\n" + "=" * 60)
    print("VALIDATION: LINEAR VOI")
    print("=" * 60)

    linear_results = validate_voi_metric(
        voi_df, test_returns, non_fed_returns, "linear_voi", test_dates
    )

    print(f"\nPrimary Test (Fed Days):")
    print(f"  Pearson r = {linear_results['primary_test']['pearson_r']:.4f}")
    print(f"  Pearson p = {linear_results['primary_test']['pearson_p']:.4f}")
    print(f"  Spearman ρ = {linear_results['primary_test']['spearman_rho']:.4f}")
    print(f"  SUCCESS: {'✓' if linear_results['primary_test']['success'] else '✗'}")

    print(f"\nNull Baseline (Non-Fed Days):")
    print(f"  Pearson r = {linear_results['null_baseline']['pearson_r']:.4f}")
    print(f"  Pearson p = {linear_results['null_baseline']['pearson_p']:.4f}")
    print(f"  SUCCESS: {'✓' if linear_results['null_baseline']['success'] else '✗'}")

    # Tertile analysis
    linear_tertile = tertile_analysis(voi_df, fed_returns, "linear_voi", test_dates)
    print(f"\nTertile Analysis:")
    print(f"  High VOI avg |return|: {linear_tertile['high_vs_low_ttest']['high_mean']:.4f}")
    print(f"  Low VOI avg |return|: {linear_tertile['high_vs_low_ttest']['low_mean']:.4f}")
    print(f"  T-test p-value: {linear_tertile['high_vs_low_ttest']['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {linear_tertile['high_vs_low_ttest']['effect_size']:.3f}")

    # ============================================================
    # VALIDATION: ENTROPY VOI
    # ============================================================
    print("\n" + "=" * 60)
    print("VALIDATION: ENTROPY VOI")
    print("=" * 60)

    entropy_results = validate_voi_metric(
        voi_df, test_returns, non_fed_returns, "entropy_voi", test_dates
    )

    print(f"\nPrimary Test (Fed Days):")
    print(f"  Pearson r = {entropy_results['primary_test']['pearson_r']:.4f}")
    print(f"  Pearson p = {entropy_results['primary_test']['pearson_p']:.4f}")
    print(f"  Spearman ρ = {entropy_results['primary_test']['spearman_rho']:.4f}")
    print(f"  SUCCESS: {'✓' if entropy_results['primary_test']['success'] else '✗'}")

    print(f"\nNull Baseline (Non-Fed Days):")
    print(f"  Pearson r = {entropy_results['null_baseline']['pearson_r']:.4f}")
    print(f"  Pearson p = {entropy_results['null_baseline']['pearson_p']:.4f}")
    print(f"  SUCCESS: {'✓' if entropy_results['null_baseline']['success'] else '✗'}")

    # Tertile analysis
    entropy_tertile = tertile_analysis(voi_df, fed_returns, "entropy_voi", test_dates)
    print(f"\nTertile Analysis:")
    print(f"  High VOI avg |return|: {entropy_tertile['high_vs_low_ttest']['high_mean']:.4f}")
    print(f"  Low VOI avg |return|: {entropy_tertile['high_vs_low_ttest']['low_mean']:.4f}")
    print(f"  T-test p-value: {entropy_tertile['high_vs_low_ttest']['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {entropy_tertile['high_vs_low_ttest']['effect_size']:.3f}")

    # ============================================================
    # COMPARISON: LINEAR VS ENTROPY VOI
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON: LINEAR VOI VS ENTROPY VOI")
    print("=" * 60)

    print(f"\nPredictive validity (r with test Fed-day |returns|):")
    print(f"  Linear VOI:  r = {linear_results['primary_test']['pearson_r']:.4f}")
    print(f"  Entropy VOI: r = {entropy_results['primary_test']['pearson_r']:.4f}")

    better = "Linear" if linear_results['primary_test']['pearson_r'] > entropy_results['primary_test']['pearson_r'] else "Entropy"
    print(f"\n  → {better} VOI has better predictive validity")

    print(f"\nNull discrimination (should be ~0 on non-Fed days):")
    print(f"  Linear VOI:  |r| = {abs(linear_results['null_baseline']['pearson_r']):.4f}")
    print(f"  Entropy VOI: |r| = {abs(entropy_results['null_baseline']['pearson_r']):.4f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    validation_results = {
        "metadata": {
            "test_period": fed_data["metadata"]["test_period"],
            "n_test_meetings": len(test_dates),
            "n_stocks": linear_results["n_stocks"],
        },
        "linear_voi": {
            "primary_test": {k: v for k, v in linear_results["primary_test"].items() if k != "analysis_df"},
            "null_baseline": {k: v for k, v in linear_results["null_baseline"].items() if k != "analysis_df"},
            "tertile_analysis": {
                "high_mean": linear_tertile["high_vs_low_ttest"]["high_mean"],
                "low_mean": linear_tertile["high_vs_low_ttest"]["low_mean"],
                "t_test_p": linear_tertile["high_vs_low_ttest"]["p_value"],
                "effect_size": linear_tertile["high_vs_low_ttest"]["effect_size"],
            }
        },
        "entropy_voi": {
            "primary_test": {k: v for k, v in entropy_results["primary_test"].items() if k != "analysis_df"},
            "null_baseline": {k: v for k, v in entropy_results["null_baseline"].items() if k != "analysis_df"},
            "tertile_analysis": {
                "high_mean": entropy_tertile["high_vs_low_ttest"]["high_mean"],
                "low_mean": entropy_tertile["high_vs_low_ttest"]["low_mean"],
                "t_test_p": entropy_tertile["high_vs_low_ttest"]["p_value"],
                "effect_size": entropy_tertile["high_vs_low_ttest"]["effect_size"],
            }
        },
        "success_criteria": {
            "primary_r_threshold": 0.2,
            "null_r_threshold": 0.1,
            "linear_voi_passed": linear_results["primary_test"]["success"] and linear_results["null_baseline"]["success"],
            "entropy_voi_passed": entropy_results["primary_test"]["success"] and entropy_results["null_baseline"]["success"],
        }
    }

    with open(DATA_DIR / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2, default=float)

    print(f"\nSaved validation_results.json")

    # Create plots
    create_validation_plots(
        linear_results,
        entropy_results,
        DATA_DIR / "validation_plots.png"
    )

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    linear_passed = linear_results["primary_test"]["success"] and linear_results["null_baseline"]["success"]
    entropy_passed = entropy_results["primary_test"]["success"] and entropy_results["null_baseline"]["success"]

    print(f"\nLinear VOI validation: {'PASSED ✓' if linear_passed else 'FAILED ✗'}")
    print(f"Entropy VOI validation: {'PASSED ✓' if entropy_passed else 'FAILED ✗'}")

    if linear_passed or entropy_passed:
        print("\n→ VOI successfully predicts which stocks move more on Fed days")
        print("  (trained on 2015-2022, validated on 2023-2025)")
    else:
        print("\n→ VOI did not meet success criteria")
        print("  Consider: more stocks, different time periods, or sector-specific analysis")


if __name__ == "__main__":
    main()
