"""
Options-implied comparison (Level 1): compare LLM-implied volatility
to market implied volatility.

For each asset × horizon:
1. Compute LLM-implied vol from each model's stated p10/p90
2. Get market implied vol (OVX for oil, realized vol as lower bound for BTC/silver)
3. Compare: are LLMs' intervals narrower than the market's?

Key caveat: options give risk-neutral distributions, which overweight tails
(investors pay a premium for crash protection). If LLMs are narrower than
even the risk-neutral distribution, they're more overconfident than a market
specifically designed to price downside risk.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import timedelta
from scipy.stats import norm

# ─── Constants ───────────────────────────────────────────────────────────────

Z_90 = norm.ppf(0.90)   # 1.2816
Z_99 = norm.ppf(0.99)   # 2.3263
TRADING_DAYS_PER_YEAR = 252


def annualized_vol_from_quantiles(p10: float, p90: float, S: float, t_years: float) -> float:
    """
    Back out annualized implied vol from symmetric quantiles.
    Under lognormal: ln(p90/p10) = 2 * z_0.9 * σ * sqrt(t)
    """
    if p10 <= 0 or p90 <= 0 or p10 >= p90:
        return np.nan
    return np.log(p90 / p10) / (2 * Z_90 * np.sqrt(t_years))


def lognormal_quantiles(S: float, sigma_ann: float, t_years: float):
    """
    Compute risk-neutral lognormal quantiles given annualized vol.
    Returns dict of quantile label -> price.
    """
    sigma_t = sigma_ann * np.sqrt(t_years)
    drift = -0.5 * sigma_ann**2 * t_years  # risk-neutral drift
    quantiles = {}
    for label, z in [("p1", norm.ppf(0.01)), ("p5", norm.ppf(0.05)),
                      ("p10", norm.ppf(0.10)), ("p50", 0.0),
                      ("p90", norm.ppf(0.90)), ("p95", norm.ppf(0.95)),
                      ("p99", norm.ppf(0.99))]:
        quantiles[label] = S * np.exp(drift + z * sigma_t)
    return quantiles


def compute_realized_vol(prices: pd.Series, end_date: str, lookback_days: int = 30) -> float:
    """Compute annualized realized vol from daily log returns over lookback period."""
    end_dt = pd.Timestamp(end_date)
    start_dt = end_dt - timedelta(days=int(lookback_days * 1.5))  # extra margin for trading days
    window = prices[(prices.index >= start_dt) & (prices.index <= end_dt)]
    if len(window) < 10:
        return np.nan
    log_returns = np.log(window / window.shift(1)).dropna()
    return log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily closing prices."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        prices = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"].iloc[:, 0]
    else:
        prices = df["Close"]
    return prices.dropna()


def download_vol_index(ticker: str, date: str) -> float:
    """Get the vol index value at a specific date."""
    start = (pd.Timestamp(date) - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(date) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return np.nan
    if isinstance(df.columns, pd.MultiIndex):
        prices = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"].iloc[:, 0]
    else:
        prices = df["Close"]
    near = prices[prices.index <= pd.Timestamp(date)]
    if near.empty:
        return np.nan
    return near.iloc[-1]


def main():
    # Load LLM forecasts
    forecasts_path = Path(__file__).parent / "results" / "forecasts.json"
    with open(forecasts_path) as f:
        forecasts = json.load(f)

    # ─── Get market implied vols ─────────────────────────────────────────

    print("Fetching market implied volatilities...\n")

    # Oil: use OVX (CBOE Crude Oil Volatility Index)
    ovx_at_cutoff = download_vol_index("^OVX", "2026-02-13")
    print(f"OVX at oil cutoff (2026-02-13): {ovx_at_cutoff:.1f}%")

    # BTC: no vol index on Yahoo; use realized vol as LOWER BOUND
    btc_prices = download_prices("BTC-USD", "2025-01-01", "2025-09-02")
    btc_realized_vol = compute_realized_vol(btc_prices, "2025-09-01", lookback_days=30)
    print(f"BTC 30-day realized vol at cutoff (2025-09-01): {btc_realized_vol*100:.1f}%")
    print(f"  (Note: implied vol is typically 10-30% higher than realized for BTC)")

    # Silver: no vol index; use realized vol as LOWER BOUND
    slv_prices = download_prices("SI=F", "2025-01-01", "2025-12-02")
    slv_realized_vol = compute_realized_vol(slv_prices, "2025-12-01", lookback_days=30)
    print(f"Silver 30-day realized vol at cutoff (2025-12-01): {slv_realized_vol*100:.1f}%")

    market_vols = {
        "oil": {"vol": ovx_at_cutoff / 100, "source": "OVX (implied)", "is_implied": True},
        "bitcoin": {"vol": btc_realized_vol, "source": "30d realized (lower bound)", "is_implied": False},
        "silver": {"vol": slv_realized_vol, "source": "30d realized (lower bound)", "is_implied": False},
    }

    # ─── Compute LLM-implied vols and compare ───────────────────────────

    assets = {
        "wti_crude": {"cutoff_price": 62.89, "cutoff": "2026-02-13"},
        "bitcoin": {"cutoff_price": 109250.59, "cutoff": "2025-09-01"},
        "silver": {"cutoff_price": 58.42, "cutoff": "2025-12-01"},
    }

    market_vols_by_id = {
        "wti_crude": market_vols["oil"],
        "bitcoin": market_vols["bitcoin"],
        "silver": market_vols["silver"],
    }

    # Group forecasts by asset and horizon
    results = []
    for asset_id in ["wti_crude", "bitcoin", "silver"]:
        asset_forecasts = [f for f in forecasts if f["asset_id"] == asset_id]
        horizons = sorted(set(f["horizon"] for f in asset_forecasts))
        S = assets[asset_id]["cutoff_price"]
        market_vol = market_vols_by_id[asset_id]["vol"]
        vol_source = market_vols_by_id[asset_id]["source"]

        for hz in horizons:
            hz_forecasts = [f for f in asset_forecasts if f["horizon"] == hz]
            days = hz_forecasts[0]["days_from_cutoff"]
            t_years = days / TRADING_DAYS_PER_YEAR
            actual = hz_forecasts[0]["ground_truth"]

            # Market-implied quantiles
            mkt_q = lognormal_quantiles(S, market_vol, t_years)

            # Per-model LLM-implied vol
            model_vols = []
            for fc in hz_forecasts:
                p = fc["percentiles"]
                llm_vol = annualized_vol_from_quantiles(p["p10"], p["p90"], S, t_years)
                model_name = fc["model"].split("/")[-1]
                model_vols.append({
                    "model": model_name,
                    "llm_vol": llm_vol,
                    "p10": p["p10"],
                    "p50": p["p50"],
                    "p90": p["p90"],
                    "p99": p.get("p99"),
                })

            avg_llm_vol = np.nanmean([m["llm_vol"] for m in model_vols])

            row = {
                "asset": asset_id,
                "horizon": hz,
                "days": days,
                "cutoff_price": S,
                "actual": actual,
                "actual_return": (actual / S - 1) * 100,
                "market_vol_ann": round(market_vol * 100, 1),
                "vol_source": vol_source,
                "market_p10": round(mkt_q["p10"], 2),
                "market_p50": round(mkt_q["p50"], 2),
                "market_p90": round(mkt_q["p90"], 2),
                "market_p99": round(mkt_q["p99"], 2),
                "avg_llm_vol_ann": round(avg_llm_vol * 100, 1),
                "vol_ratio": round(avg_llm_vol / market_vol, 2) if market_vol > 0 else None,
                "models": model_vols,
            }
            results.append(row)

    # ─── Print results ───────────────────────────────────────────────────

    print(f"\n\n{'='*90}")
    print("  VOLATILITY COMPARISON: LLM-implied vs Market-implied")
    print(f"{'='*90}")
    print(f"\n{'Asset':<10} {'Hz':<4} {'Days':<5} {'Mkt vol':>8} {'LLM vol':>8} "
          f"{'Ratio':>6} {'Mkt p10':>10} {'LLM p10':>10} {'Mkt p90':>10} {'LLM p90':>10} "
          f"{'Actual':>12}")
    print("-" * 100)

    for r in results:
        avg_p10 = np.mean([m["p10"] for m in r["models"]])
        avg_p90 = np.mean([m["p90"] for m in r["models"]])
        print(f"{r['asset']:<10} {r['horizon']:<4} {r['days']:<5} "
              f"{r['market_vol_ann']:>7.1f}% {r['avg_llm_vol_ann']:>7.1f}% "
              f"{r['vol_ratio']:>5.2f}x "
              f"{r['market_p10']:>10.2f} {avg_p10:>10.2f} "
              f"{r['market_p90']:>10.2f} {avg_p90:>10.2f} "
              f"{r['actual']:>12.2f}")

    # ─── Per-asset summary ───────────────────────────────────────────────

    for asset_id in ["wti_crude", "bitcoin", "silver"]:
        asset_results = [r for r in results if r["asset"] == asset_id]
        print(f"\n\n{'='*70}")
        print(f"  {asset_id.upper()} — {market_vols_by_id[asset_id]['source']}")
        print(f"{'='*70}")

        for r in asset_results:
            print(f"\n  {r['horizon']} ({r['days']}d): actual = ${r['actual']:,.2f} "
                  f"(return = {r['actual_return']:+.1f}%)")
            print(f"    Market vol: {r['market_vol_ann']:.1f}% ann → "
                  f"p10=${r['market_p10']:,.2f}  p50=${r['market_p50']:,.2f}  "
                  f"p90=${r['market_p90']:,.2f}  p99=${r['market_p99']:,.2f}")

            for m in r["models"]:
                print(f"    {m['model']:<30} vol={m['llm_vol']*100:5.1f}%  "
                      f"p10=${m['p10']:>10,.2f}  p90=${m['p90']:>10,.2f}  "
                      f"ratio={m['llm_vol']/market_vols_by_id[asset_id]['vol']:.2f}x")

    # ─── Save ────────────────────────────────────────────────────────────

    # Simplify for JSON serialization
    for r in results:
        for m in r["models"]:
            m["llm_vol"] = round(m["llm_vol"] * 100, 1) if not np.isnan(m["llm_vol"]) else None
    out_path = Path(__file__).parent / "options_implied_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
