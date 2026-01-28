"""Investigate why VOI validation failed.

Hypothesis: VOI is picking up general volatility, not Fed-specific sensitivity.
High-rho stocks may just be high-volatility stocks that move more on ALL days.

Alternative approach: Test if VOI predicts Fed-day excess volatility
(|return_fed| - |return_non_fed|) rather than absolute volatility.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path(__file__).parent / "data"


def main():
    # Load data
    voi_df = pd.read_parquet(DATA_DIR / "voi_scores.parquet")
    fed_returns = pd.read_parquet(DATA_DIR / "fed_returns.parquet")
    non_fed_returns = pd.read_parquet(DATA_DIR / "non_fed_returns.parquet")

    with open(DATA_DIR / "fed_meetings.json") as f:
        fed_data = json.load(f)

    test_dates = [m["date"] for m in fed_data["test"]]

    # Compute average |return| by ticker for both periods
    test_fed = fed_returns[fed_returns["date"].isin(test_dates)]

    fed_vol = test_fed.groupby("ticker")["return"].apply(lambda x: np.abs(x).mean())
    non_fed_vol = non_fed_returns.groupby("ticker")["return"].apply(lambda x: np.abs(x).mean())

    # Join
    analysis = voi_df[["ticker", "linear_voi", "abs_rho"]].copy()
    analysis["fed_vol"] = analysis["ticker"].map(fed_vol)
    analysis["non_fed_vol"] = analysis["ticker"].map(non_fed_vol)
    analysis = analysis.dropna()

    # Compute excess volatility: how much MORE volatile on Fed days
    analysis["excess_vol"] = analysis["fed_vol"] - analysis["non_fed_vol"]
    analysis["vol_ratio"] = analysis["fed_vol"] / analysis["non_fed_vol"]

    print("=" * 60)
    print("CONFOUND ANALYSIS: VOI vs General Volatility")
    print("=" * 60)

    # 1. Does VOI correlate with general (non-Fed) volatility?
    r_voi_nonfed, p = pearsonr(analysis["linear_voi"], analysis["non_fed_vol"])
    print(f"\nVOI vs Non-Fed Volatility: r = {r_voi_nonfed:.4f} (p = {p:.4f})")
    print("→ If high, VOI is confounded with general volatility")

    # 2. Does VOI predict EXCESS Fed volatility?
    r_voi_excess, p = pearsonr(analysis["linear_voi"], analysis["excess_vol"])
    print(f"\nVOI vs Excess Fed Volatility: r = {r_voi_excess:.4f} (p = {p:.4f})")
    print("→ This is the true test: Fed-specific sensitivity")

    # 3. What if we residualize?
    # Regress Fed vol on non-Fed vol, predict residual with VOI
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(analysis["non_fed_vol"], analysis["fed_vol"])
    analysis["fed_vol_residual"] = analysis["fed_vol"] - (slope * analysis["non_fed_vol"] + intercept)

    r_voi_residual, p = pearsonr(analysis["linear_voi"], analysis["fed_vol_residual"])
    print(f"\nVOI vs Fed Vol Residual (controlling for non-Fed vol): r = {r_voi_residual:.4f} (p = {p:.4f})")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nAverage Fed-day volatility: {analysis['fed_vol'].mean():.4f}")
    print(f"Average non-Fed-day volatility: {analysis['non_fed_vol'].mean():.4f}")
    print(f"Excess Fed volatility: {analysis['excess_vol'].mean():.4f}")

    # Is there even a Fed day effect on average?
    t_fed = analysis["fed_vol"].mean()
    t_nonfed = analysis["non_fed_vol"].mean()
    print(f"\nFed day effect (cross-stock): {(t_fed/t_nonfed - 1)*100:.1f}% higher volatility")

    print("\n" + "=" * 60)
    print("ALTERNATIVE VALIDATION: Direction prediction")
    print("=" * 60)

    # Maybe VOI should predict DIRECTION rather than magnitude
    # i.e., high-rho stocks should move in consistent direction with Fed
    train_dates = [m["date"] for m in fed_data["train"]]
    train_outcomes = {m["date"]: m["outcome"] for m in fed_data["train"]}

    # For each stock, compute sign alignment on test data
    test_outcomes = {m["date"]: m["outcome"] for m in fed_data["test"]}

    alignment_scores = {}
    for ticker in analysis["ticker"]:
        ticker_data = fed_returns[
            (fed_returns["ticker"] == ticker) &
            (fed_returns["date"].isin(test_dates))
        ]
        if len(ticker_data) < 5:
            continue

        # For each test meeting, check if stock moved in expected direction
        # based on rho sign from training
        rho = voi_df[voi_df["ticker"] == ticker]["rho"].values[0]

        correct = 0
        total = 0
        for _, row in ticker_data.iterrows():
            outcome = test_outcomes.get(row["date"], 0)
            if outcome == 0:  # Hold meetings uninformative
                continue
            ret = row["return"]
            # If rho > 0 and Fed raised, expect positive return
            # If rho > 0 and Fed cut, expect negative return
            expected_sign = np.sign(rho * outcome)
            actual_sign = np.sign(ret)
            if expected_sign == actual_sign:
                correct += 1
            total += 1

        if total > 0:
            alignment_scores[ticker] = correct / total

    if alignment_scores:
        alignment_df = pd.DataFrame([
            {"ticker": k, "alignment": v} for k, v in alignment_scores.items()
        ])
        alignment_df = alignment_df.merge(voi_df[["ticker", "linear_voi", "abs_rho"]], on="ticker")

        print(f"\nComputed direction alignment for {len(alignment_df)} stocks")
        print(f"Average alignment: {alignment_df['alignment'].mean():.3f} (0.5 = chance)")

        # Does VOI predict alignment?
        r_align, p = pearsonr(alignment_df["linear_voi"], alignment_df["alignment"])
        print(f"\nVOI vs Direction Alignment: r = {r_align:.4f} (p = {p:.4f})")

        # Does |rho| predict alignment?
        r_rho_align, p = pearsonr(alignment_df["abs_rho"], alignment_df["alignment"])
        print(f"|ρ| vs Direction Alignment: r = {r_rho_align:.4f} (p = {p:.4f})")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("""
Likely explanation for validation failure:

1. VOI is highly correlated with general stock volatility
   → High-VOI stocks move more on ALL days, not just Fed days

2. The Fed-day "signal" is noisy at the individual stock level
   → Stock returns are driven by many factors beyond Fed decisions
   → N=17 test meetings is small for individual stock analysis

3. The analogy to Polymarket may not hold:
   - Polymarket: discrete event A affects discrete event B
   - This setting: categorical Fed decision affects continuous stock returns
   - The joint distribution structure is fundamentally different

Possible improvements:
1. Use sector/factor portfolios instead of individual stocks
2. Look at intraday returns around announcement time only
3. Control for market beta (excess return vs market)
4. Use different time windows (2-day, weekly returns)
""")


if __name__ == "__main__":
    main()
