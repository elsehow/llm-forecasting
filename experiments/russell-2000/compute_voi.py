"""Compute ρ and VOI for each stock from Fed day returns.

Computes correlation between stock returns and Fed outcomes on training data,
then converts to both Linear VOI and Entropy VOI for comparison.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "llm-forecasting" / "src"))

from llm_forecasting.voi import (
    linear_voi_from_rho,
    entropy_voi_from_rho,
    entropy_voi_normalized_from_rho,
    compare_voi_metrics_from_rho,
)

DATA_DIR = Path(__file__).parent / "data"


def compute_stock_rho(
    returns_df: pd.DataFrame,
    fed_outcomes: dict[str, int],  # date -> outcome
    ticker: str,
) -> tuple[float, float, int]:
    """Compute Pearson correlation between stock returns and Fed outcomes.

    Args:
        returns_df: DataFrame with ticker, date, return columns
        fed_outcomes: Mapping of date to outcome (+1, 0, -1)
        ticker: Stock symbol

    Returns:
        (rho, p_value, n_observations)
    """
    # Filter to this ticker
    ticker_returns = returns_df[returns_df["ticker"] == ticker].copy()

    # Join with outcomes
    ticker_returns["outcome"] = ticker_returns["date"].map(fed_outcomes)
    ticker_returns = ticker_returns.dropna(subset=["outcome"])

    if len(ticker_returns) < 10:
        return np.nan, np.nan, len(ticker_returns)

    # Pearson correlation
    rho, p_value = pearsonr(ticker_returns["return"], ticker_returns["outcome"])

    return rho, p_value, len(ticker_returns)


def compute_voi_scores(
    rho_scores: dict[str, float],
    p_stock_up: float = 0.5,  # Prior: stock equally likely to go up or down
    p_fed_change: float = 0.35,  # ~35% of meetings have rate changes
) -> pd.DataFrame:
    """Compute VOI from rho for each stock.

    In this context:
    - p_a = P(stock goes up) ≈ 0.5 (we're predicting stock direction)
    - p_b = P(Fed changes rate) - the signal probability
    - rho = correlation between stock return and Fed outcome

    Returns DataFrame with linear_voi, entropy_voi, entropy_voi_normalized.
    """
    results = []

    for ticker, rho in rho_scores.items():
        if np.isnan(rho):
            results.append({
                "ticker": ticker,
                "rho": np.nan,
                "linear_voi": np.nan,
                "entropy_voi": np.nan,
                "entropy_voi_normalized": np.nan,
            })
            continue

        # Compute all VOI metrics
        metrics = compare_voi_metrics_from_rho(rho, p_stock_up, p_fed_change)

        results.append({
            "ticker": ticker,
            "rho": rho,
            "abs_rho": abs(rho),
            "linear_voi": metrics["linear_voi"],
            "entropy_voi": metrics["entropy_voi"],
            "entropy_voi_normalized": metrics["entropy_voi_normalized"],
        })

    return pd.DataFrame(results)


def main():
    # Load data
    fed_path = DATA_DIR / "fed_meetings.json"
    returns_path = DATA_DIR / "fed_returns.parquet"

    if not fed_path.exists() or not returns_path.exists():
        print("Run fetch_fed_data.py and fetch_stock_returns.py first!")
        return

    with open(fed_path) as f:
        fed_data = json.load(f)

    returns_df = pd.read_parquet(returns_path)

    # Build outcome mapping for TRAIN period only
    train_outcomes = {m["date"]: m["outcome"] for m in fed_data["train"]}
    train_dates = set(train_outcomes.keys())

    # Filter returns to train period
    train_returns = returns_df[returns_df["date"].isin(train_dates)]
    print(f"Train period: {len(train_dates)} Fed meetings")
    print(f"Train observations: {len(train_returns)}")

    # Compute rho for each stock on TRAIN data
    tickers = train_returns["ticker"].unique()
    print(f"\nComputing ρ for {len(tickers)} stocks...")

    rho_results = {}
    p_values = {}
    n_obs = {}

    for ticker in tickers:
        rho, p, n = compute_stock_rho(train_returns, train_outcomes, ticker)
        rho_results[ticker] = rho
        p_values[ticker] = p
        n_obs[ticker] = n

    # Estimate empirical probabilities from train data
    outcomes = list(train_outcomes.values())
    p_raise = outcomes.count(1) / len(outcomes)
    p_hold = outcomes.count(0) / len(outcomes)
    p_cut = outcomes.count(-1) / len(outcomes)
    p_fed_change = p_raise + p_cut  # Any non-hold

    print(f"\nTrain period Fed outcome distribution:")
    print(f"  P(raise) = {p_raise:.3f}")
    print(f"  P(hold) = {p_hold:.3f}")
    print(f"  P(cut) = {p_cut:.3f}")
    print(f"  P(change) = {p_fed_change:.3f}")

    # Compute VOI scores
    voi_df = compute_voi_scores(rho_results, p_stock_up=0.5, p_fed_change=p_fed_change)
    voi_df["p_value"] = voi_df["ticker"].map(p_values)
    voi_df["n_obs"] = voi_df["ticker"].map(n_obs)

    # Sort by linear VOI
    voi_df = voi_df.sort_values("linear_voi", ascending=False)

    # Display top/bottom stocks
    print("\n=== Top 20 by Linear VOI (most Fed-sensitive) ===")
    print(voi_df.head(20).to_string(index=False))

    print("\n=== Bottom 20 by Linear VOI (least Fed-sensitive) ===")
    print(voi_df.tail(20).to_string(index=False))

    # Summary statistics
    print("\n=== VOI Distribution ===")
    print(f"Linear VOI:")
    print(f"  Mean: {voi_df['linear_voi'].mean():.4f}")
    print(f"  Std: {voi_df['linear_voi'].std():.4f}")
    print(f"  Min: {voi_df['linear_voi'].min():.4f}")
    print(f"  Max: {voi_df['linear_voi'].max():.4f}")

    print(f"\nEntropy VOI:")
    print(f"  Mean: {voi_df['entropy_voi'].mean():.4f}")
    print(f"  Std: {voi_df['entropy_voi'].std():.4f}")
    print(f"  Min: {voi_df['entropy_voi'].min():.4f}")
    print(f"  Max: {voi_df['entropy_voi'].max():.4f}")

    # Correlation between linear and entropy VOI
    valid_voi = voi_df.dropna()
    if len(valid_voi) > 10:
        corr, _ = pearsonr(valid_voi["linear_voi"], valid_voi["entropy_voi"])
        print(f"\nCorrelation between linear and entropy VOI: {corr:.3f}")

    # Check for expected patterns
    # Financials should have higher |rho| than average
    financials = ["HBAN", "CFG", "RF", "ZION", "KEY", "FHN", "SNV", "WTFC", "WAL", "PACW"]
    fin_voi = voi_df[voi_df["ticker"].isin(financials)]
    if len(fin_voi) > 0:
        print(f"\n=== Sanity Check: Financials ===")
        print(f"Financial sector avg |ρ|: {fin_voi['abs_rho'].mean():.4f}")
        print(f"All stocks avg |ρ|: {voi_df['abs_rho'].mean():.4f}")

    # Save results
    voi_df.to_parquet(DATA_DIR / "voi_scores.parquet", index=False)

    # Also save as JSON for readability
    results_json = {
        "metadata": {
            "train_period": fed_data["metadata"]["train_period"],
            "n_stocks": len(voi_df),
            "n_train_meetings": len(train_dates),
            "p_fed_change": p_fed_change,
        },
        "voi_summary": {
            "linear_voi_mean": voi_df["linear_voi"].mean(),
            "linear_voi_std": voi_df["linear_voi"].std(),
            "entropy_voi_mean": voi_df["entropy_voi"].mean(),
            "entropy_voi_std": voi_df["entropy_voi"].std(),
        },
        "top_20_linear_voi": voi_df.head(20).to_dict(orient="records"),
        "bottom_20_linear_voi": voi_df.tail(20).to_dict(orient="records"),
    }

    with open(DATA_DIR / "voi_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=float)

    print(f"\nSaved voi_scores.parquet and voi_results.json")


if __name__ == "__main__":
    main()
