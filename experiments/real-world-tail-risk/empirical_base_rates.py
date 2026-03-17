"""
Empirical base rates: what percentile were the observed outcomes?

For each asset × horizon, compute the empirical distribution of historical
returns at that horizon length, then rank the actual post-cutoff return.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import timedelta

# ─── Asset definitions ──────────────────────────────────────────────────────

ASSETS = {
    # ─── Tier 1: used in LLM evaluation ─────────────────────────────────
    "Oil (WTI)": {
        "ticker": "CL=F",
        "cutoff": "2026-02-13",
        "cutoff_price": 62.89,
        "history_start": "1986-01-01",
        "tier": 1,
        "horizons": {
            "H1": {"days": 7, "actual_price": 66.39},
            "H2": {"days": 14, "actual_price": 67.02},
            "H3": {"days": 21, "actual_price": 90.90},
        },
    },
    "Bitcoin": {
        "ticker": "BTC-USD",
        "cutoff": "2025-09-01",
        "cutoff_price": 109250.59,
        "history_start": "2014-01-01",
        "tier": 1,
        "horizons": {
            "H1": {"days": 7, "actual_price": 112071.43},
            "H2": {"days": 14, "actual_price": 115444.88},
            "H3": {"days": 30, "actual_price": 118648.93},
            "H4": {"days": 61, "actual_price": 110064.02},
            "H5": {"days": 91, "actual_price": 86321.57},
            "H6": {"days": 122, "actual_price": 88731.98},
            "H7": {"days": 181, "actual_price": 65738.10},
        },
    },
    "Silver": {
        "ticker": "SI=F",
        "cutoff": "2025-12-01",
        "cutoff_price": 58.42,
        "history_start": "1968-01-01",
        "tier": 1,
        "horizons": {
            "H1": {"days": 7, "actual_price": 57.78},
            "H2": {"days": 14, "actual_price": 62.94},
            "H3": {"days": 30, "actual_price": 70.13},
            "H4": {"days": 63, "actual_price": 76.78},
            "H5": {"days": 91, "actual_price": 88.28},
        },
    },
    "Natural Gas": {
        "ticker": "NG=F",
        "cutoff": "2026-02-13",
        "cutoff_price": 3.243,
        "history_start": "1997-01-01",
        "tier": 1,
        "horizons": {
            "H1": {"days": 7, "actual_price": 3.047},
            "H2": {"days": 14, "actual_price": 2.859},
            "H3": {"days": 21, "actual_price": 3.186},
        },
    },
    # ─── Tier 2: base rates only (not in LLM evaluation) ────────────────
    "Gold": {
        "ticker": "GC=F",
        "cutoff": "2025-10-01",
        "cutoff_price": 3867.50,
        "history_start": "1975-01-01",
        "tier": 2,
        "horizons": {
            "H1": {"days": 7, "actual_price": 4043.30},
            "H2": {"days": 14, "actual_price": 4176.90},
            "H3": {"days": 30, "actual_price": 3982.20},
            "H4": {"days": 63, "actual_price": 4199.30},
            "H5": {"days": 91, "actual_price": 4325.60},
            "H6": {"days": 122, "actual_price": 4713.90},
        },
    },
    "Wheat": {
        "ticker": "ZW=F",
        "cutoff": "2026-02-13",
        "cutoff_price": 548.75,
        "history_start": "1970-01-01",
        "tier": 2,
        "horizons": {
            "H1": {"days": 7, "actual_price": 573.50},
            "H2": {"days": 14, "actual_price": 591.25},
            "H3": {"days": 21, "actual_price": 611.25},
        },
    },
    "Copper": {
        "ticker": "HG=F",
        "cutoff": "2025-11-01",
        "cutoff_price": 5.07,
        "history_start": "1970-01-01",
        "tier": 2,
        "horizons": {
            "H1": {"days": 7, "actual_price": 4.94},
            "H2": {"days": 14, "actual_price": 5.05},
            "H3": {"days": 30, "actual_price": 5.22},
            "H4": {"days": 63, "actual_price": 5.64},
            "H5": {"days": 91, "actual_price": 5.90},
        },
    },
}


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily closing prices from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        prices = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"].iloc[:, 0]
    else:
        prices = df["Close"]
    return prices.dropna()


def compute_nonoverlapping_returns(prices: pd.Series, horizon_days: int) -> np.ndarray:
    """
    Compute non-overlapping returns over horizon_days calendar days.

    Uses calendar-day spacing (not trading days) to match how the evaluation
    horizons are defined. For each window start, finds the nearest available
    trading day at start + horizon_days.
    """
    dates = prices.index
    returns = []
    i = 0
    while i < len(dates):
        start_date = dates[i]
        target_end = start_date + timedelta(days=horizon_days)
        # Find nearest trading day to target_end
        candidates = dates[(dates >= target_end - timedelta(days=3)) &
                          (dates <= target_end + timedelta(days=3))]
        if len(candidates) == 0:
            i += 1
            continue
        end_date = candidates[np.argmin(np.abs(candidates - target_end))]
        start_price = prices[start_date]
        end_price = prices[end_date]
        if start_price > 0:
            returns.append((end_price / start_price - 1) * 100)  # percentage
        # Jump to next non-overlapping window
        next_start = end_date + timedelta(days=1)
        next_indices = dates[dates >= next_start]
        if len(next_indices) == 0:
            break
        i = dates.get_loc(next_indices[0])
    return np.array(returns)


def compute_trend_slope(prices: pd.Series, end_date, lookback_days: int = 60) -> float:
    """Compute the percentage return over the lookback period ending at end_date."""
    start_date = end_date - timedelta(days=lookback_days)
    window = prices[(prices.index >= start_date) & (prices.index <= end_date)]
    if len(window) < 2:
        return np.nan
    return (window.iloc[-1] / window.iloc[0] - 1) * 100


def compute_trend_conditioned_returns(
    prices: pd.Series, horizon_days: int, ref_slope: float, tolerance: float = 5.0
) -> np.ndarray:
    """
    Compute non-overlapping returns, filtered to windows where the prior
    60-day trend slope is within ±tolerance of ref_slope.
    """
    dates = prices.index
    returns = []
    i = 0
    while i < len(dates):
        start_date = dates[i]
        target_end = start_date + timedelta(days=horizon_days)
        candidates = dates[(dates >= target_end - timedelta(days=3)) &
                          (dates <= target_end + timedelta(days=3))]
        if len(candidates) == 0:
            i += 1
            continue
        end_date = candidates[np.argmin(np.abs(candidates - target_end))]
        # Check trend condition
        slope = compute_trend_slope(prices, start_date)
        if not np.isnan(slope) and abs(slope - ref_slope) <= tolerance:
            start_price = prices[start_date]
            end_price = prices[end_date]
            if start_price > 0:
                returns.append((end_price / start_price - 1) * 100)
        # Jump to next non-overlapping window
        next_start = end_date + timedelta(days=1)
        next_indices = dates[dates >= next_start]
        if len(next_indices) == 0:
            break
        i = dates.get_loc(next_indices[0])
    return np.array(returns)


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """What percentile is `value` in the empirical distribution?"""
    return np.mean(distribution <= value) * 100


def main():
    results = []

    for asset_name, cfg in ASSETS.items():
        print(f"\n{'='*60}")
        print(f"  {asset_name} ({cfg['ticker']})")
        print(f"{'='*60}")

        # Download prices up to cutoff (don't include post-cutoff)
        prices = download_prices(cfg["ticker"], cfg["history_start"], cfg["cutoff"])
        print(f"  History: {prices.index[0].date()} to {prices.index[-1].date()} "
              f"({len(prices)} trading days)")

        # Pre-cutoff trend slope (60-day)
        ref_slope = compute_trend_slope(prices, prices.index[-1])
        print(f"  Pre-cutoff 60-day trend: {ref_slope:+.1f}%")

        for hz_name, hz in cfg["horizons"].items():
            actual_return = (hz["actual_price"] / cfg["cutoff_price"] - 1) * 100

            # Unconditional
            uncond = compute_nonoverlapping_returns(prices, hz["days"])
            if len(uncond) == 0:
                print(f"  {hz_name}: no windows!")
                continue
            uncond_pctile = percentile_rank(actual_return, uncond)

            # Trend-conditioned
            trend_cond = compute_trend_conditioned_returns(
                prices, hz["days"], ref_slope, tolerance=5.0
            )
            trend_pctile = percentile_rank(actual_return, trend_cond) if len(trend_cond) > 5 else None

            row = {
                "asset": asset_name,
                "tier": cfg.get("tier", 1),
                "horizon": hz_name,
                "days": hz["days"],
                "actual_return_pct": round(actual_return, 1),
                "n_uncond_windows": len(uncond),
                "uncond_percentile": round(uncond_pctile, 1),
                "uncond_mean": round(np.mean(uncond), 1),
                "uncond_std": round(np.std(uncond), 1),
                "ref_slope": round(ref_slope, 1),
                "n_trend_windows": len(trend_cond),
                "trend_percentile": round(trend_pctile, 1) if trend_pctile is not None else None,
            }
            results.append(row)

            print(f"\n  {hz_name} ({hz['days']}d): actual return = {actual_return:+.1f}%")
            print(f"    Unconditional: p{uncond_pctile:.1f} "
                  f"(N={len(uncond)}, mean={np.mean(uncond):+.1f}%, std={np.std(uncond):.1f}%)")
            if trend_pctile is not None:
                print(f"    Trend-conditioned: p{trend_pctile:.1f} "
                      f"(N={len(trend_cond)}, ref slope={ref_slope:+.1f}% ±5pp)")
            else:
                print(f"    Trend-conditioned: insufficient windows "
                      f"(N={len(trend_cond)}, need >5)")

    # ─── Summary table ───────────────────────────────────────────────────

    print(f"\n\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Asset':<14} {'Hz':<4} {'Days':<5} {'Actual ret':>10} "
          f"{'N':>5} {'Uncond pctile':>14} {'N(trend)':>9} {'Trend pctile':>13}")
    print("-" * 80)
    for r in results:
        tp = f"p{r['trend_percentile']}" if r['trend_percentile'] is not None else "n/a"
        print(f"{r['asset']:<14} {r['horizon']:<4} {r['days']:<5} "
              f"{r['actual_return_pct']:>+9.1f}% "
              f"{r['n_uncond_windows']:>5} "
              f"{'p' + str(r['uncond_percentile']):>14} "
              f"{r['n_trend_windows']:>9} "
              f"{tp:>13}")

    # Save results
    out_path = Path(__file__).parent / "empirical_base_rates.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
