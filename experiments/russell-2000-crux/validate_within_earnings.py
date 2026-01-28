"""Validate VOI predictive power within earnings events only.

Test: Does VOI predict |return| magnitude when restricting to earnings days?

This is the non-trivial claim. If r ~ 0 within earnings, VOI is just detecting
"earnings exist" (trivial). If r > 0.2, VOI has predictive value beyond
category identification.

Hypothesis: If VOI has predictive value beyond category identification,
we should see r > 0.2 when restricting to earnings days.

Current baseline: r = -0.153 (n=27, underpowered)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate VOI predictive power within earnings")
    parser.add_argument(
        "--input",
        type=str,
        default="cruxes_with_voi.parquet",
        help="Input cruxes file with VOI (default: cruxes_with_voi.parquet)"
    )
    return parser.parse_args()


def load_data(cruxes_input: str = "cruxes_with_voi.parquet"):
    """Load cruxes with VOI and stock returns.

    Args:
        cruxes_input: Filename for cruxes parquet file
    """
    if not cruxes_input.endswith(".parquet"):
        cruxes_input += ".parquet"

    cruxes_path = DATA_DIR / cruxes_input
    returns_path = DATA_DIR / "stock_returns.parquet"

    if not cruxes_path.exists():
        raise FileNotFoundError(f"Cruxes file not found: {cruxes_path}\nRun compute_voi.py first!")
    if not returns_path.exists():
        raise FileNotFoundError("Run fetch_stock_data.py first!")

    cruxes_df = pd.read_parquet(cruxes_path)
    returns_df = pd.read_parquet(returns_path)

    return cruxes_df, returns_df


def validate_within_earnings(
    cruxes_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    aggregation: str = "mean",
) -> dict:
    """Test: Does VOI predict |return| within earnings events only?

    Args:
        cruxes_df: DataFrame with ticker, date, linear_voi, entropy_voi
        returns_df: DataFrame with ticker, date, return, is_earnings_day
        aggregation: How to aggregate cruxes per stock-day ('mean', 'max', 'median')

    Returns:
        Dict with correlation results and interpretation.
    """
    # Filter to earnings days only
    earnings_returns = returns_df[returns_df["is_earnings_day"]].copy()

    if len(earnings_returns) == 0:
        return {"error": "No earnings days found in returns data"}

    # Aggregate VOI per stock-day
    agg_func = {"mean": "mean", "max": "max", "median": "median"}[aggregation]

    voi_by_day = cruxes_df.groupby(["ticker", "date"]).agg({
        "linear_voi": agg_func,
        "entropy_voi": agg_func,
        "rho": "mean",  # Always mean for rho
    }).reset_index()

    # Merge with returns (inner join - only keep stock-days in both)
    merged = voi_by_day.merge(
        earnings_returns[["ticker", "date", "return"]],
        on=["ticker", "date"],
        how="inner"
    )

    if len(merged) < 5:
        return {"error": f"Too few observations ({len(merged)}) for correlation analysis"}

    merged["abs_return"] = merged["return"].abs()

    # Test correlations
    r_linear, p_linear = stats.pearsonr(merged["linear_voi"], merged["abs_return"])
    r_entropy, p_entropy = stats.pearsonr(merged["entropy_voi"], merged["abs_return"])
    r_rho, p_rho = stats.pearsonr(merged["rho"].abs(), merged["abs_return"])

    # Spearman (rank correlation) as robustness check
    rho_linear, _ = stats.spearmanr(merged["linear_voi"], merged["abs_return"])
    rho_entropy, _ = stats.spearmanr(merged["entropy_voi"], merged["abs_return"])

    # Interpret results
    def interpret(r, p, n):
        """Interpret correlation result."""
        if p >= 0.05:
            sig = "not significant"
        elif p < 0.01:
            sig = "highly significant"
        else:
            sig = "significant"

        if r > 0.3:
            strength = "strong positive"
        elif r > 0.2:
            strength = "moderate positive"
        elif r > 0.1:
            strength = "weak positive"
        elif r > -0.1:
            strength = "negligible"
        elif r > -0.2:
            strength = "weak negative"
        elif r > -0.3:
            strength = "moderate negative"
        else:
            strength = "strong negative"

        return f"{strength} ({sig})"

    results = {
        "n_observations": len(merged),
        "aggregation": aggregation,
        "linear_voi": {
            "pearson_r": float(r_linear),
            "pearson_p": float(p_linear),
            "spearman_rho": float(rho_linear),
            "interpretation": interpret(r_linear, p_linear, len(merged)),
        },
        "entropy_voi": {
            "pearson_r": float(r_entropy),
            "pearson_p": float(p_entropy),
            "spearman_rho": float(rho_entropy),
            "interpretation": interpret(r_entropy, p_entropy, len(merged)),
        },
        "abs_rho": {
            "pearson_r": float(r_rho),
            "pearson_p": float(p_rho),
        },
        "summary_stats": {
            "mean_abs_return": float(merged["abs_return"].mean()),
            "std_abs_return": float(merged["abs_return"].std()),
            "mean_linear_voi": float(merged["linear_voi"].mean()),
            "mean_entropy_voi": float(merged["entropy_voi"].mean()),
        },
        "merged_df": merged,  # For plotting
    }

    return results


def create_validation_plot(results: dict, save_path: Path):
    """Create scatter plot of VOI vs |return| for within-earnings validation."""
    if "error" in results or "merged_df" not in results:
        print("Cannot create plot - insufficient data")
        return

    df = results["merged_df"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (voi_col, title) in enumerate([
        ("linear_voi", "Linear VOI"),
        ("entropy_voi", "Entropy VOI"),
    ]):
        ax = axes[i]
        ax.scatter(df[voi_col], df["abs_return"], alpha=0.6, s=50, c="steelblue")
        ax.set_xlabel(title)
        ax.set_ylabel("|Return| on Earnings Day")

        # Add trend line
        z = np.polyfit(df[voi_col], df["abs_return"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[voi_col].min(), df[voi_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        r = results[voi_col]["pearson_r"]
        p_val = results[voi_col]["pearson_p"]
        ax.set_title(f"{title} vs |Return| (Within Earnings)\nn={len(df)}, r={r:.3f}, p={p_val:.4f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def power_analysis(target_r: float = 0.2, alpha: float = 0.05, power: float = 0.8) -> int:
    """Calculate required sample size for detecting correlation."""
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Fisher z transformation
    z_r = 0.5 * np.log((1 + target_r) / (1 - target_r))

    n = ((z_alpha + z_beta) / z_r) ** 2 + 3
    return int(np.ceil(n))


def main():
    args = parse_args()

    print("=" * 60)
    print("WITHIN-EARNINGS VOI VALIDATION")
    print("=" * 60)
    print(f"\nInput: {args.input}")
    print("\nQuestion: Does VOI predict |return| WITHIN earnings events?")
    print("If r ~ 0, VOI just detects 'earnings exist' (trivial)")
    print("If r > 0.2, VOI has predictive value beyond category identification")

    # Load data
    try:
        cruxes_df, returns_df = load_data(args.input)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    n_earnings = returns_df["is_earnings_day"].sum()
    n_cruxes = len(cruxes_df)
    print(f"\nData loaded:")
    print(f"  Total cruxes: {n_cruxes}")
    print(f"  Earnings days in returns: {n_earnings}")

    # Run validation with different aggregations
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for agg in ["mean", "max"]:
        print(f"\n--- Aggregation: {agg} VOI per stock-day ---")
        results = validate_within_earnings(cruxes_df, returns_df, aggregation=agg)

        if "error" in results:
            print(f"Error: {results['error']}")
            continue

        print(f"Observations: {results['n_observations']}")
        print(f"\nLinear VOI:")
        print(f"  Pearson r = {results['linear_voi']['pearson_r']:.4f}")
        print(f"  p-value   = {results['linear_voi']['pearson_p']:.4f}")
        print(f"  Spearman  = {results['linear_voi']['spearman_rho']:.4f}")
        print(f"  Interpretation: {results['linear_voi']['interpretation']}")

        print(f"\nEntropy VOI:")
        print(f"  Pearson r = {results['entropy_voi']['pearson_r']:.4f}")
        print(f"  p-value   = {results['entropy_voi']['pearson_p']:.4f}")
        print(f"  Spearman  = {results['entropy_voi']['spearman_rho']:.4f}")
        print(f"  Interpretation: {results['entropy_voi']['interpretation']}")

        # Create plot for mean aggregation
        if agg == "mean":
            create_validation_plot(results, DATA_DIR / "within_earnings_validation.png")

    # Power analysis
    print("\n" + "=" * 60)
    print("POWER ANALYSIS")
    print("=" * 60)
    for target_r in [0.15, 0.2, 0.25, 0.3]:
        n_required = power_analysis(target_r)
        print(f"To detect r={target_r} (alpha=0.05, power=0.8): n={n_required}")

    # Success criteria evaluation
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Use mean aggregation for final assessment
    results = validate_within_earnings(cruxes_df, returns_df, aggregation="mean")
    if "error" in results:
        print(f"Cannot evaluate: {results['error']}")
        return

    r = results["linear_voi"]["pearson_r"]
    p = results["linear_voi"]["pearson_p"]
    n = results["n_observations"]

    if r > 0.2 and p < 0.05:
        verdict = "SUCCESS: VOI predicts which earnings matter more"
        emoji = "+"
    elif r > 0.1 and p < 0.1:
        verdict = "WEAK: Some signal, may need more data or better cruxes"
        emoji = "~"
    elif abs(r) < 0.1:
        verdict = "FAIL: VOI does not predict within-earnings magnitude"
        emoji = "-"
    else:
        verdict = "ANOMALOUS: Negative correlation - investigate further"
        emoji = "?"

    print(f"\nWith n={n}, r={r:.3f}, p={p:.4f}:")
    print(f"  [{emoji}] {verdict}")

    # Save results
    results_to_save = {k: v for k, v in results.items() if k != "merged_df"}
    results_to_save["verdict"] = verdict

    with open(DATA_DIR / "within_earnings_validation_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, default=float)

    print(f"\nSaved results to within_earnings_validation_results.json")

    return results


if __name__ == "__main__":
    main()
