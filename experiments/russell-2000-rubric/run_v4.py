"""v4: Scale up to n=200+ filtered stock-days for statistical power.

Sample 600 additional non-earnings stock-days, generate cruxes, compute VOI,
apply timeliness filtering, combine with v3 data, and re-run validation.

Expected outcome:
- ~180 new filtered stock-days (30% pass rate for non-earnings)
- ~221 total filtered stock-days (41 existing + 180 new)
- p<0.05 for VOI-return correlation if r remains ~0.20
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt

# Add parent to path for imports from russell-2000-crux
sys.path.insert(0, str(Path(__file__).parent.parent / "russell-2000-crux"))
from generate_cruxes import generate_cruxes_batch
from compute_voi import compute_voi_batch
from config import CRUX_GENERATION_MODEL, RHO_ESTIMATION_MODEL

sys.path.insert(0, str(Path(__file__).parent))
from timeliness import evaluate_timeliness_batch

# Paths
DATA_DIR = Path(__file__).parent / "data"
CRUX_DATA_DIR = Path(__file__).parent.parent / "russell-2000-crux" / "data"

# Config for v4
N_ADDITIONAL_STOCK_DAYS = 600  # Sample this many new non-earnings days
N_CRUXES_PER_DAY = 5


def sample_additional_stock_days(
    returns_df: pd.DataFrame,
    existing_cruxes_df: pd.DataFrame,
    universe: list[dict],
    n_samples: int = N_ADDITIONAL_STOCK_DAYS,
    seed: int = 42,
) -> list[dict]:
    """Sample additional non-earnings stock-days not already in existing cruxes.

    Stratified by stock to ensure representation across all tickers.
    """
    np.random.seed(seed)

    # Build lookup for company info
    company_info = {u["ticker"]: u for u in universe}

    # Get already-used stock-days
    existing_keys = set(
        existing_cruxes_df["ticker"] + "_" + existing_cruxes_df["date"]
    )

    # Filter returns to non-earnings, not already used
    returns_df = returns_df.copy()
    returns_df["key"] = returns_df["ticker"] + "_" + returns_df["date"]
    available_df = returns_df[
        ~returns_df["is_earnings_day"] &
        ~returns_df["key"].isin(existing_keys)
    ]

    print(f"Available non-earnings stock-days: {len(available_df)}")

    # Stratified sampling by ticker
    tickers = available_df["ticker"].unique()
    samples_per_ticker = n_samples // len(tickers)
    extra_samples = n_samples % len(tickers)

    sampled_rows = []
    for i, ticker in enumerate(tickers):
        ticker_df = available_df[available_df["ticker"] == ticker]
        n_sample = samples_per_ticker + (1 if i < extra_samples else 0)
        n_sample = min(n_sample, len(ticker_df))  # Can't sample more than available

        if n_sample > 0:
            sampled = ticker_df.sample(n=n_sample, random_state=seed + i)
            sampled_rows.append(sampled)

    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    print(f"Sampled {len(sampled_df)} stock-days from {len(tickers)} tickers")

    # Convert to list of dicts for crux generation
    stock_days = []
    for _, row in sampled_df.iterrows():
        ticker = row["ticker"]
        info = company_info.get(ticker, {})
        stock_days.append({
            "ticker": ticker,
            "date": row["date"],
            "company_name": info.get("company_name", ticker),
            "sector": info.get("sector", "Unknown"),
            "context": "",  # Non-earnings, no special context
        })

    return stock_days


def create_v4_validation_plot(
    unfiltered_results: dict,
    filtered_results: dict,
    save_path: Path,
):
    """Create validation scatter plots for v4."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = [
        (unfiltered_results, "Unfiltered (all cruxes)", axes[0]),
        (filtered_results, "Filtered (timely only)", axes[1]),
    ]

    for results, title, ax in datasets:
        if "error" in results or "analysis_df" not in results:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(title)
            continue

        df = results["analysis_df"]
        ax.scatter(df["linear_voi"], df["abs_return"], alpha=0.5, s=30)
        ax.set_xlabel("Max Linear VOI")
        ax.set_ylabel("|Return|")

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df["linear_voi"], df["abs_return"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df["linear_voi"].min(), df["linear_voi"].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        r = results["pearson_r"]
        p_val = results["pearson_p"]
        ax.set_title(f"{title}\nr={r:.3f}, p={p_val:.4f}, n={results['n_observations']}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


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
    rho_linear, p_spearman = spearmanr(analysis_df["linear_voi"], analysis_df["abs_return"])

    return {
        "n_observations": len(analysis_df),
        "pearson_r": r_linear,
        "pearson_p": p_linear,
        "spearman_rho": rho_linear,
        "spearman_p": p_spearman,
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
        "t_stat": t_stat,
        "t_test_p": t_p,
        "earnings_higher": earnings_voi.mean() > non_earnings_voi.mean(),
    }


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUSSELL 2000 VOI VALIDATION - v4 SCALE-UP")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # Step 1: Load existing data
    # =========================================================================
    print("\n=== Step 1: Load Existing Data ===")

    returns_df = pd.read_parquet(CRUX_DATA_DIR / "stock_returns.parquet")
    existing_cruxes_df = pd.read_parquet(CRUX_DATA_DIR / "cruxes_all.parquet")
    v3_evaluated_df = pd.read_parquet(DATA_DIR / "cruxes_evaluated.parquet")

    with open(CRUX_DATA_DIR / "stock_universe.json") as f:
        universe = json.load(f)

    print(f"Returns data: {len(returns_df)} stock-days")
    print(f"Existing cruxes (v3): {len(existing_cruxes_df)} cruxes across {existing_cruxes_df.groupby(['ticker', 'date']).ngroups} stock-days")
    print(f"V3 filtered stock-days: {v3_evaluated_df[v3_evaluated_df['timely']].groupby(['ticker', 'date']).ngroups}")

    # =========================================================================
    # Step 2: Sample additional stock-days
    # =========================================================================
    print(f"\n=== Step 2: Sample {N_ADDITIONAL_STOCK_DAYS} Additional Stock-Days ===")

    new_stock_days = sample_additional_stock_days(
        returns_df,
        existing_cruxes_df,
        universe,
        n_samples=N_ADDITIONAL_STOCK_DAYS,
    )

    # =========================================================================
    # Step 3: Generate cruxes
    # =========================================================================
    print(f"\n=== Step 3: Generate Cruxes ===")
    print(f"Stock-days to process: {len(new_stock_days)}")
    print(f"Cruxes per stock-day: {N_CRUXES_PER_DAY}")
    print(f"Expected total cruxes: {len(new_stock_days) * N_CRUXES_PER_DAY}")
    print(f"Model: {CRUX_GENERATION_MODEL}")

    crux_results = await generate_cruxes_batch(new_stock_days, n_cruxes=N_CRUXES_PER_DAY)

    # Flatten to DataFrame
    flat_cruxes = []
    for r in crux_results:
        for i, crux in enumerate(r["cruxes"]):
            flat_cruxes.append({
                "ticker": r["ticker"],
                "date": r["date"],
                "company_name": r["company_name"],
                "sector": r["sector"],
                "context": r["context"],
                "crux_index": i,
                "crux": crux,
                "is_earnings_day": False,  # All new samples are non-earnings
            })

    new_cruxes_df = pd.DataFrame(flat_cruxes)
    print(f"\nGenerated {len(new_cruxes_df)} cruxes")

    # =========================================================================
    # Step 4: Compute VOI
    # =========================================================================
    print(f"\n=== Step 4: Compute VOI ===")
    print(f"Cruxes to process: {len(new_cruxes_df)}")
    print(f"Model: {RHO_ESTIMATION_MODEL}")

    new_cruxes_with_voi = await compute_voi_batch(new_cruxes_df)

    print(f"\nVOI Summary:")
    print(f"  Linear VOI: mean={new_cruxes_with_voi['linear_voi'].mean():.4f}, max={new_cruxes_with_voi['linear_voi'].max():.4f}")
    print(f"  Rho: mean={new_cruxes_with_voi['rho'].mean():.4f}, std={new_cruxes_with_voi['rho'].std():.4f}")

    # =========================================================================
    # Step 5: Apply timeliness filter
    # =========================================================================
    print(f"\n=== Step 5: Apply Timeliness Filter ===")

    cruxes_list = new_cruxes_with_voi.to_dict("records")
    timeliness_results = await evaluate_timeliness_batch(cruxes_list)
    new_cruxes_evaluated = pd.DataFrame(timeliness_results)

    n_timely = new_cruxes_evaluated["timely"].sum()
    print(f"Timely cruxes: {n_timely} / {len(new_cruxes_evaluated)} ({n_timely/len(new_cruxes_evaluated)*100:.1f}%)")

    # Stock-day level pass rates
    new_stock_day_pass = new_cruxes_evaluated.groupby(["ticker", "date"]).agg({
        "timely": lambda x: any(x)
    }).reset_index()
    n_sd_pass = new_stock_day_pass["timely"].sum()
    print(f"Stock-days with at least one timely crux: {n_sd_pass} / {len(new_stock_day_pass)} ({n_sd_pass/len(new_stock_day_pass)*100:.1f}%)")

    # Save new cruxes
    new_cruxes_evaluated.to_parquet(DATA_DIR / "cruxes_v4_new.parquet", index=False)
    print(f"\nSaved new cruxes to cruxes_v4_new.parquet")

    # =========================================================================
    # Step 6: Combine with v3 data
    # =========================================================================
    print(f"\n=== Step 6: Combine with V3 Data ===")

    # Ensure columns match
    common_cols = list(set(v3_evaluated_df.columns) & set(new_cruxes_evaluated.columns))
    all_cruxes_df = pd.concat([
        v3_evaluated_df[common_cols],
        new_cruxes_evaluated[common_cols],
    ], ignore_index=True)

    print(f"Combined dataset: {len(all_cruxes_df)} cruxes")
    print(f"  From v3: {len(v3_evaluated_df)}")
    print(f"  From v4 (new): {len(new_cruxes_evaluated)}")

    # Save combined
    all_cruxes_df.to_parquet(DATA_DIR / "cruxes_all_v4.parquet", index=False)
    print(f"Saved combined cruxes to cruxes_all_v4.parquet")

    # =========================================================================
    # Step 7: Run validation
    # =========================================================================
    print(f"\n=== Step 7: Validation ===")
    print("=" * 70)

    # Split into filtered (timely) and unfiltered
    filtered_df = all_cruxes_df[all_cruxes_df["timely"]]

    # Count stock-days
    all_stock_days = all_cruxes_df.groupby(["ticker", "date"]).ngroups
    filtered_stock_days = filtered_df.groupby(["ticker", "date"]).ngroups

    print(f"\nCrux counts:")
    print(f"  Unfiltered: {len(all_cruxes_df)} cruxes, {all_stock_days} stock-days")
    print(f"  Filtered (timely): {len(filtered_df)} cruxes, {filtered_stock_days} stock-days")

    # Test 1: VOI vs |return| correlation
    print("\n--- Test 1: VOI vs |Return| Correlation ---")

    unfiltered_results = compute_voi_return_correlation(all_cruxes_df, returns_df)
    filtered_results = compute_voi_return_correlation(filtered_df, returns_df)

    print(f"\nUnfiltered:")
    if "error" not in unfiltered_results:
        print(f"  n = {unfiltered_results['n_observations']}")
        print(f"  Pearson r = {unfiltered_results['pearson_r']:.4f}")
        print(f"  Pearson p = {unfiltered_results['pearson_p']:.4f}")
        print(f"  Spearman ρ = {unfiltered_results['spearman_rho']:.4f}")

    print(f"\nFiltered (timely only):")
    if "error" not in filtered_results:
        print(f"  n = {filtered_results['n_observations']}")
        print(f"  Pearson r = {filtered_results['pearson_r']:.4f}")
        print(f"  Pearson p = {filtered_results['pearson_p']:.4f}")
        print(f"  Spearman ρ = {filtered_results['spearman_rho']:.4f}")

        # Check significance
        significant = filtered_results["pearson_p"] < 0.05
        print(f"\n  >>> p < 0.05: {'YES ✓' if significant else 'NO'}")

    # Test 2: Earnings vs non-earnings VOI
    print("\n--- Test 2: Earnings vs Non-Earnings VOI ---")

    filtered_earnings = compare_earnings_vs_non_earnings_voi(filtered_df)
    if "error" not in filtered_earnings:
        print(f"  Earnings mean VOI: {filtered_earnings['earnings_mean_voi']:.4f} (n={filtered_earnings['earnings_n']})")
        print(f"  Non-earnings mean VOI: {filtered_earnings['non_earnings_mean_voi']:.4f} (n={filtered_earnings['non_earnings_n']})")
        print(f"  Earnings higher: {'YES ✓' if filtered_earnings['earnings_higher'] else 'NO'}")
        print(f"  t-test p: {filtered_earnings['t_test_p']:.4f}")

    # Create validation plot
    create_v4_validation_plot(
        unfiltered_results,
        filtered_results,
        DATA_DIR / "validation_v4.png",
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_n = filtered_stock_days >= 200
    success_correlation = (
        "error" not in filtered_results and
        filtered_results["pearson_r"] > 0.1
    )
    success_significance = (
        "error" not in filtered_results and
        filtered_results["pearson_p"] < 0.05
    )
    success_earnings = (
        "error" not in filtered_earnings and
        filtered_earnings["earnings_higher"]
    )

    print(f"\n1. Filtered n >= 200: {'PASS ✓' if success_n else 'FAIL'} (n={filtered_stock_days})")
    print(f"2. Filtered r > 0.1: {'PASS ✓' if success_correlation else 'FAIL'} (r={filtered_results.get('pearson_r', 'N/A'):.4f})")
    print(f"3. p < 0.05: {'PASS ✓' if success_significance else 'FAIL'} (p={filtered_results.get('pearson_p', 'N/A'):.4f})")
    print(f"4. Earnings VOI > Non-Earnings: {'PASS ✓' if success_earnings else 'FAIL'}")

    overall = success_n and success_correlation and success_significance and success_earnings
    print(f"\nOverall: {'PASS - Scale-up successful!' if overall else 'NEEDS INVESTIGATION'}")

    # Comparison with v3
    print("\n" + "=" * 70)
    print("COMPARISON: v3 vs v4")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'v3':<15} {'v4':<15}")
    print("-" * 60)
    print(f"{'Filtered n (stock-days)':<30} {41:<15} {filtered_stock_days:<15}")
    if "error" not in filtered_results:
        print(f"{'Pearson r':<30} {0.20:<15.4f} {filtered_results['pearson_r']:<15.4f}")
        print(f"{'p-value':<30} {0.22:<15.4f} {filtered_results['pearson_p']:<15.4f}")

    # Save results
    results = {
        "run_time": datetime.now().isoformat(),
        "config": {
            "n_additional_stock_days": N_ADDITIONAL_STOCK_DAYS,
            "n_cruxes_per_day": N_CRUXES_PER_DAY,
            "crux_model": CRUX_GENERATION_MODEL,
            "voi_model": RHO_ESTIMATION_MODEL,
        },
        "counts": {
            "new_cruxes_generated": len(new_cruxes_evaluated),
            "new_cruxes_timely": int(new_cruxes_evaluated["timely"].sum()),
            "new_stock_days_passing": int(n_sd_pass),
            "total_cruxes": len(all_cruxes_df),
            "total_filtered_cruxes": len(filtered_df),
            "total_stock_days": all_stock_days,
            "total_filtered_stock_days": filtered_stock_days,
        },
        "validation": {
            "unfiltered": {k: v for k, v in unfiltered_results.items() if k != "analysis_df"} if "error" not in unfiltered_results else unfiltered_results,
            "filtered": {k: v for k, v in filtered_results.items() if k != "analysis_df"} if "error" not in filtered_results else filtered_results,
            "earnings_comparison": filtered_earnings,
        },
        "success_criteria": {
            "n_over_200": success_n,
            "r_over_0.1": success_correlation,
            "p_under_0.05": success_significance,
            "earnings_higher": success_earnings,
            "overall": overall,
        },
    }

    with open(DATA_DIR / "validation_v4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved results to validation_v4_results.json")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
