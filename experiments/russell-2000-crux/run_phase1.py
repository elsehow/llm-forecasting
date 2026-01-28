"""Phase 1: Add non-earnings days to test VOI variance hypothesis.

The pilot showed that all earnings cruxes have similarly high VOI (no variance).
This phase samples non-earnings days where cruxes should be lower quality,
then tests if max-VOI correlates with |return| across all days.

Hypothesis:
- Earnings days: high-VOI cruxes, high |return|
- Non-earnings days: low-VOI cruxes, low |return|
"""

import asyncio
import json
import random
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt

from config import PILOT_N_CRUXES
from generate_cruxes import generate_cruxes_batch
from compute_voi import compute_voi_batch

DATA_DIR = Path(__file__).parent / "data"

# Sample size for non-earnings days (roughly 2x earnings to get variance)
N_NON_EARNINGS_SAMPLE = 50


def sample_non_earnings_days(returns_df: pd.DataFrame, n_sample: int = N_NON_EARNINGS_SAMPLE) -> pd.DataFrame:
    """Sample non-earnings days stratified by stock."""
    non_earnings = returns_df[~returns_df["is_earnings_day"]].copy()

    # Sample roughly equally from each stock
    stocks = non_earnings["ticker"].unique()
    samples_per_stock = max(1, n_sample // len(stocks))

    sampled = []
    for ticker in stocks:
        ticker_days = non_earnings[non_earnings["ticker"] == ticker]
        n_take = min(samples_per_stock, len(ticker_days))
        sampled.append(ticker_days.sample(n=n_take, random_state=42))

    result = pd.concat(sampled)

    # If we need more, sample randomly from remaining
    if len(result) < n_sample:
        remaining = non_earnings[~non_earnings.index.isin(result.index)]
        extra = remaining.sample(n=min(n_sample - len(result), len(remaining)), random_state=42)
        result = pd.concat([result, extra])

    return result


def build_non_earnings_stock_days(sample_df: pd.DataFrame, universe: list[dict]) -> list[dict]:
    """Build stock-day list for non-earnings days."""
    company_info = {u["ticker"]: u for u in universe}

    stock_days = []
    for _, row in sample_df.iterrows():
        ticker = row["ticker"]
        info = company_info.get(ticker, {})

        stock_days.append({
            "ticker": ticker,
            "date": row["date"],
            "company_name": info.get("company_name", ticker),
            "sector": info.get("sector", "Unknown"),
            "context": "",  # No special context for non-earnings days
        })

    return stock_days


async def main():
    print("=" * 60)
    print("PHASE 1: Non-Earnings Days Sampling")
    print("=" * 60)

    # Load existing data
    returns_df = pd.read_parquet(DATA_DIR / "stock_returns.parquet")
    with open(DATA_DIR / "stock_universe.json") as f:
        universe = json.load(f)

    # Load pilot cruxes with VOI
    pilot_cruxes = pd.read_parquet(DATA_DIR / "cruxes_with_voi.parquet")

    print(f"\nExisting pilot data:")
    print(f"  Earnings stock-days: {pilot_cruxes.groupby(['ticker', 'date']).ngroups}")
    print(f"  Total cruxes: {len(pilot_cruxes)}")

    # Sample non-earnings days
    print(f"\nSampling {N_NON_EARNINGS_SAMPLE} non-earnings days...")
    sample_df = sample_non_earnings_days(returns_df, N_NON_EARNINGS_SAMPLE)
    print(f"  Sampled {len(sample_df)} stock-days")
    print(f"  From {sample_df['ticker'].nunique()} stocks")

    # Check if we already have non-earnings cruxes
    non_earnings_path = DATA_DIR / "cruxes_non_earnings.parquet"
    if non_earnings_path.exists():
        print(f"\nLoading existing non-earnings cruxes...")
        non_earnings_cruxes = pd.read_parquet(non_earnings_path)
    else:
        # Build stock-day list
        stock_days = build_non_earnings_stock_days(sample_df, universe)

        print(f"\n=== Generating Cruxes for Non-Earnings Days ===")
        print(f"Stock-days: {len(stock_days)}")

        # Generate cruxes
        results = await generate_cruxes_batch(stock_days, PILOT_N_CRUXES)

        # Flatten
        flat_results = []
        for r in results:
            for i, crux in enumerate(r["cruxes"]):
                flat_results.append({
                    "ticker": r["ticker"],
                    "date": r["date"],
                    "company_name": r["company_name"],
                    "sector": r["sector"],
                    "context": r["context"],
                    "crux_index": i,
                    "crux": crux,
                })

        non_earnings_cruxes = pd.DataFrame(flat_results)
        non_earnings_cruxes.to_parquet(DATA_DIR / "cruxes_non_earnings_raw.parquet", index=False)
        print(f"  Generated {len(non_earnings_cruxes)} cruxes")

        # Compute VOI
        print(f"\n=== Computing VOI for Non-Earnings Cruxes ===")
        non_earnings_cruxes = await compute_voi_batch(non_earnings_cruxes)
        non_earnings_cruxes.to_parquet(non_earnings_path, index=False)
        print(f"  Computed VOI for {len(non_earnings_cruxes)} cruxes")

    print(f"\nNon-earnings cruxes with VOI: {len(non_earnings_cruxes)}")

    # Combine with pilot data
    print(f"\n=== Combining Earnings and Non-Earnings Data ===")

    # Add is_earnings flag
    pilot_cruxes["is_earnings_day"] = True
    non_earnings_cruxes["is_earnings_day"] = False

    all_cruxes = pd.concat([pilot_cruxes, non_earnings_cruxes], ignore_index=True)
    print(f"Total cruxes: {len(all_cruxes)}")
    print(f"  Earnings: {len(pilot_cruxes)}")
    print(f"  Non-earnings: {len(non_earnings_cruxes)}")

    # Compute max VOI per stock-day
    max_voi = all_cruxes.groupby(["ticker", "date", "is_earnings_day"]).agg({
        "linear_voi": "max",
        "entropy_voi": "max",
    }).reset_index()

    # Join with returns
    analysis_df = max_voi.merge(
        returns_df[["ticker", "date", "return"]],
        on=["ticker", "date"],
        how="inner"
    )
    analysis_df["abs_return"] = analysis_df["return"].abs()

    print(f"\nAnalysis dataset: {len(analysis_df)} stock-days")
    print(f"  Earnings: {analysis_df['is_earnings_day'].sum()}")
    print(f"  Non-earnings: {(~analysis_df['is_earnings_day']).sum()}")

    # === VALIDATION TESTS ===
    print("\n" + "=" * 60)
    print("PHASE 1 VALIDATION")
    print("=" * 60)

    earnings_df = analysis_df[analysis_df["is_earnings_day"]]
    non_earnings_df = analysis_df[~analysis_df["is_earnings_day"]]

    # Test 1: Do earnings days have higher max-VOI?
    print("\n=== Test 1: Earnings vs Non-Earnings VOI ===")
    t_voi, p_voi = ttest_ind(earnings_df["linear_voi"], non_earnings_df["linear_voi"])
    print(f"Earnings mean VOI: {earnings_df['linear_voi'].mean():.4f}")
    print(f"Non-earnings mean VOI: {non_earnings_df['linear_voi'].mean():.4f}")
    print(f"T-test p-value: {p_voi:.4f}")
    print(f"Earnings VOI higher: {'YES' if earnings_df['linear_voi'].mean() > non_earnings_df['linear_voi'].mean() else 'NO'}")

    # Test 2: Does max-VOI correlate with |return|?
    print("\n=== Test 2: VOI vs |Return| Correlation (All Days) ===")
    r_linear, p_linear = pearsonr(analysis_df["linear_voi"], analysis_df["abs_return"])
    r_entropy, p_entropy = pearsonr(analysis_df["entropy_voi"], analysis_df["abs_return"])
    rho_linear, _ = spearmanr(analysis_df["linear_voi"], analysis_df["abs_return"])

    print(f"Observations: {len(analysis_df)}")
    print(f"\nLinear VOI:")
    print(f"  Pearson r = {r_linear:.4f} (p = {p_linear:.4f})")
    print(f"  Spearman ρ = {rho_linear:.4f}")
    print(f"\nEntropy VOI:")
    print(f"  Pearson r = {r_entropy:.4f} (p = {p_entropy:.4f})")

    # Test 3: Tertile analysis
    print("\n=== Test 3: Tertile Analysis ===")
    try:
        analysis_df["voi_tertile"] = pd.qcut(analysis_df["linear_voi"], q=3, labels=False, duplicates="drop")
        # Map to labels based on actual number of bins
        n_bins = analysis_df["voi_tertile"].nunique()
        if n_bins == 3:
            label_map = {0: "Low", 1: "Medium", 2: "High"}
        elif n_bins == 2:
            label_map = {0: "Low", 1: "High"}
        else:
            label_map = {i: f"Bin_{i}" for i in range(n_bins)}
        analysis_df["voi_tertile"] = analysis_df["voi_tertile"].map(label_map)
        tertile_means = analysis_df.groupby("voi_tertile")["abs_return"].agg(["mean", "std", "count"])
        print(tertile_means.to_string())
    except Exception as e:
        print(f"Could not compute tertiles: {e}")
        tertile_means = None

    # Test 4: Verify earnings days still have higher |return|
    print("\n=== Test 4: Earnings vs Non-Earnings |Return| ===")
    t_ret, p_ret = ttest_ind(earnings_df["abs_return"], non_earnings_df["abs_return"])
    print(f"Earnings mean |return|: {earnings_df['abs_return'].mean():.4f}")
    print(f"Non-earnings mean |return|: {non_earnings_df['abs_return'].mean():.4f}")
    print(f"T-test p-value: {p_ret:.4f}")

    # === PLOTS ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: VOI distribution by day type
    ax = axes[0, 0]
    ax.hist(earnings_df["linear_voi"], bins=15, alpha=0.7, label="Earnings", color="blue")
    ax.hist(non_earnings_df["linear_voi"], bins=15, alpha=0.7, label="Non-earnings", color="orange")
    ax.set_xlabel("Max Linear VOI")
    ax.set_ylabel("Count")
    ax.set_title("VOI Distribution by Day Type")
    ax.legend()

    # Plot 2: |Return| distribution by day type
    ax = axes[0, 1]
    ax.hist(earnings_df["abs_return"], bins=15, alpha=0.7, label="Earnings", color="blue")
    ax.hist(non_earnings_df["abs_return"], bins=15, alpha=0.7, label="Non-earnings", color="orange")
    ax.set_xlabel("|Return|")
    ax.set_ylabel("Count")
    ax.set_title("|Return| Distribution by Day Type")
    ax.legend()

    # Plot 3: VOI vs |Return| scatter
    ax = axes[1, 0]
    ax.scatter(earnings_df["linear_voi"], earnings_df["abs_return"],
               alpha=0.7, label="Earnings", color="blue", s=50)
    ax.scatter(non_earnings_df["linear_voi"], non_earnings_df["abs_return"],
               alpha=0.7, label="Non-earnings", color="orange", s=50)
    ax.set_xlabel("Max Linear VOI")
    ax.set_ylabel("|Return|")
    ax.set_title(f"VOI vs |Return| (r={r_linear:.3f}, p={p_linear:.3f})")
    ax.legend()

    # Add trend line
    z = np.polyfit(analysis_df["linear_voi"], analysis_df["abs_return"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(analysis_df["linear_voi"].min(), analysis_df["linear_voi"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Plot 4: Tertile means (or VOI by day type if tertiles failed)
    ax = axes[1, 1]
    if tertile_means is not None and len(tertile_means) >= 2:
        tertile_order = [t for t in ["Low", "Medium", "High"] if t in tertile_means.index]
        means = [tertile_means.loc[t, "mean"] for t in tertile_order]
        stds = [tertile_means.loc[t, "std"] for t in tertile_order]
        colors = ["lightblue", "skyblue", "steelblue"][:len(tertile_order)]
        ax.bar(tertile_order, means, yerr=stds, capsize=5, color=colors)
        ax.set_xlabel("VOI Tertile")
        ax.set_ylabel("Mean |Return|")
        ax.set_title("Mean |Return| by VOI Tertile")
    else:
        # Fallback: show mean VOI by day type
        day_types = ["Earnings", "Non-Earnings"]
        voi_means = [earnings_df["linear_voi"].mean(), non_earnings_df["linear_voi"].mean()]
        ax.bar(day_types, voi_means, color=["blue", "orange"])
        ax.set_xlabel("Day Type")
        ax.set_ylabel("Mean VOI")
        ax.set_title("Mean VOI by Day Type")
        means = None

    plt.tight_layout()
    plt.savefig(DATA_DIR / "phase1_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to phase1_validation.png")

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_voi_diff = earnings_df["linear_voi"].mean() > non_earnings_df["linear_voi"].mean() and p_voi < 0.1
    success_correlation = r_linear > 0.1 and p_linear < 0.1
    if means is not None and len(means) >= 2:
        success_tertile = means[-1] > means[0]  # High > Low
    else:
        success_tertile = False

    print(f"\n1. Earnings VOI > Non-Earnings VOI: {'PASS' if success_voi_diff else 'FAIL'}")
    print(f"2. VOI correlates with |return|: {'PASS' if success_correlation else 'FAIL'}")
    print(f"3. High-VOI tertile > Low-VOI tertile: {'PASS' if success_tertile else 'FAIL'}")

    overall = success_voi_diff and success_correlation and success_tertile
    print(f"\nOverall: {'PASS - VOI captures information' if overall else 'FAIL - Review methodology'}")

    # Save results
    results = {
        "voi_comparison": {
            "earnings_mean_voi": float(earnings_df["linear_voi"].mean()),
            "non_earnings_mean_voi": float(non_earnings_df["linear_voi"].mean()),
            "t_test_p": float(p_voi),
            "earnings_higher": bool(earnings_df["linear_voi"].mean() > non_earnings_df["linear_voi"].mean()),
        },
        "voi_return_correlation": {
            "n_observations": len(analysis_df),
            "pearson_r": float(r_linear),
            "pearson_p": float(p_linear),
            "spearman_rho": float(rho_linear),
        },
        "tertile_analysis": {
            "low_mean": float(means[0]) if means is not None and len(means) > 0 else None,
            "medium_mean": float(means[1]) if means is not None and len(means) > 1 else None,
            "high_mean": float(means[-1]) if means is not None and len(means) > 0 else None,
            "high_vs_low_ratio": float(means[-1] / means[0]) if means is not None and len(means) >= 2 and means[0] > 0 else None,
        },
        "summary": {
            "success_voi_diff": success_voi_diff,
            "success_correlation": success_correlation,
            "success_tertile": success_tertile,
            "overall_pass": overall,
        }
    }

    with open(DATA_DIR / "phase1_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else float(x))
    print(f"\nSaved results to phase1_validation_results.json")

    # Save combined cruxes
    all_cruxes.to_parquet(DATA_DIR / "cruxes_all.parquet", index=False)
    print(f"Saved combined cruxes to cruxes_all.parquet")


if __name__ == "__main__":
    asyncio.run(main())
