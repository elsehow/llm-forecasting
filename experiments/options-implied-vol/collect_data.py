#!/usr/bin/env python3
"""
Download price/vol data and generate questions for the options-implied-vol study.

Downloads:
- Price histories from Yahoo Finance (10 assets)
- Vol index histories from CBOE (10 indices)

Generates weekly cutoff dates from Sep 2025 to Mar 2026, with H1/H2/H3 horizons.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/collect_data.py
"""

import json
import io
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"
VOL_DIR = DATA_DIR / "vol_indices"

ASSETS = {
    "sp500":     {"name": "S&P 500",          "unit": "index points", "price_ticker": "^GSPC", "vol_ticker": "VIX"},
    "nasdaq100": {"name": "Nasdaq-100",        "unit": "index points", "price_ticker": "^NDX",  "vol_ticker": "VXN"},
    "russell":   {"name": "Russell 2000",      "unit": "index points", "price_ticker": "^RUT",  "vol_ticker": "RVX"},
    "em":        {"name": "Emerging Markets (EEM)", "unit": "USD",     "price_ticker": "EEM",   "vol_ticker": "VXEEM"},
    "oil":       {"name": "WTI Crude Oil",     "unit": "$/barrel",    "price_ticker": "CL=F",  "vol_ticker": "OVX"},
    "gold":      {"name": "Gold",              "unit": "$/troy oz",   "price_ticker": "GC=F",  "vol_ticker": "GVZ"},
    "treasury":  {"name": "20+ Year Treasury (TLT)", "unit": "USD",   "price_ticker": "TLT",   "vol_ticker": "VXTLT"},
    "apple":     {"name": "Apple",             "unit": "USD",         "price_ticker": "AAPL",  "vol_ticker": "VXAPL"},
    "amazon":    {"name": "Amazon",            "unit": "USD",         "price_ticker": "AMZN",  "vol_ticker": "VXAZN"},
    "alphabet":  {"name": "Alphabet (Google)", "unit": "USD",         "price_ticker": "GOOGL", "vol_ticker": "VXGOG"},
}

HORIZONS = {"H1": 7, "H2": 14, "H3": 21}

# Weekly cutoffs: every Monday from Sep 2025 through early Mar 2026
CUTOFF_START = "2025-09-01"
CUTOFF_END = "2026-03-10"

# Price history: start early enough to have ~200 trading days before earliest cutoff
PRICE_HISTORY_START = "2024-06-01"
PRICE_HISTORY_END = "2026-04-15"  # past latest horizon


def generate_cutoff_mondays(start: str, end: str) -> list[str]:
    """Generate Monday dates between start and end."""
    d = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    # Advance to first Monday
    while d.weekday() != 0:
        d += timedelta(days=1)
    mondays = []
    while d <= end_d:
        mondays.append(str(d.date()))
        d += timedelta(days=7)
    return mondays


def download_prices():
    """Download price histories from Yahoo Finance."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    for asset_id, cfg in ASSETS.items():
        ticker = cfg["price_ticker"]
        out_path = PRICES_DIR / f"{asset_id}.csv"
        print(f"  Downloading {ticker} → {out_path.name}...", end=" ")
        df = yf.download(ticker, start=PRICE_HISTORY_START, end=PRICE_HISTORY_END,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna()
        df.index.name = "Date"
        df.to_csv(out_path)
        print(f"{len(df)} rows")


def download_vol_indices():
    """Download vol index histories from CBOE."""
    VOL_DIR.mkdir(parents=True, exist_ok=True)
    for asset_id, cfg in ASSETS.items():
        vol_ticker = cfg["vol_ticker"]
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{vol_ticker}_History.csv"
        out_path = VOL_DIR / f"{vol_ticker}.csv"
        print(f"  Downloading {vol_ticker} → {out_path.name}...", end=" ")
        resp = requests.get(url)
        resp.raise_for_status()

        # Parse CBOE CSV (handles both OHLC and single-value formats)
        raw_df = pd.read_csv(io.StringIO(resp.text))
        date_col = raw_df.columns[0]
        raw_df[date_col] = pd.to_datetime(raw_df[date_col])
        raw_df = raw_df.set_index(date_col)
        raw_df.index.name = "Date"

        if "CLOSE" in raw_df.columns:
            close = raw_df["CLOSE"]
        else:
            close = raw_df.iloc[:, 0]

        out_df = pd.DataFrame({"Close": close})
        out_df.to_csv(out_path)
        print(f"{len(out_df)} rows")


def nearest_trading_day(df: pd.DataFrame, target: str) -> tuple[str, float] | tuple[None, None]:
    """Find nearest trading day within ±5 days of target."""
    target_dt = pd.Timestamp(target)
    window_start = target_dt - timedelta(days=5)
    window_end = target_dt + timedelta(days=5)
    mask = (df.index >= window_start) & (df.index <= window_end)
    candidates = df[mask]
    if candidates.empty:
        return None, None
    diffs = abs(candidates.index - target_dt)
    idx = diffs.argmin()
    row = candidates.iloc[idx]
    close_val = row["Close"]
    if hasattr(close_val, "iloc"):
        close_val = close_val.iloc[0]
    return str(row.name.date()), float(close_val)


def generate_questions():
    """Generate question JSON from downloaded data."""
    cutoff_mondays = generate_cutoff_mondays(CUTOFF_START, CUTOFF_END)
    print(f"\n{len(cutoff_mondays)} cutoff dates: {cutoff_mondays[0]} to {cutoff_mondays[-1]}")

    all_questions = []
    skipped = 0

    for asset_id, cfg in ASSETS.items():
        # Load price data
        price_df = pd.read_csv(PRICES_DIR / f"{asset_id}.csv", parse_dates=["Date"])
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)
        price_df = price_df.set_index("Date")

        # Load vol index
        vol_df = pd.read_csv(VOL_DIR / f"{cfg['vol_ticker']}.csv", parse_dates=["Date"])
        vol_df = vol_df.set_index("Date")

        print(f"\n{cfg['name']} ({asset_id}):")

        for cutoff_monday in cutoff_mondays:
            cutoff_date, cutoff_price = nearest_trading_day(price_df, cutoff_monday)
            if cutoff_price is None:
                skipped += 1
                continue

            # Get vol index at cutoff
            vol_date, vol_value = nearest_trading_day(vol_df, cutoff_date)
            if vol_value is None:
                skipped += 1
                continue

            for hz_label, days in HORIZONS.items():
                target = pd.Timestamp(cutoff_date) + timedelta(days=days)
                actual_date, actual_price = nearest_trading_day(price_df, str(target.date()))
                # Ground truth may not be available yet for recent cutoffs
                ground_truth = actual_price  # None if not yet resolved

                all_questions.append({
                    "asset_id": asset_id,
                    "asset_name": cfg["name"],
                    "unit": cfg["unit"],
                    "price_ticker": cfg["price_ticker"],
                    "vol_ticker": cfg["vol_ticker"],
                    "horizon": hz_label,
                    "target_date": str(target.date()),
                    "actual_date": actual_date,
                    "days_from_cutoff": (pd.Timestamp(actual_date) - pd.Timestamp(cutoff_date)).days if actual_date else days,
                    "ground_truth": ground_truth,
                    "cutoff_date": cutoff_date,
                    "cutoff_price": cutoff_price,
                    "vol_index_at_cutoff": vol_value,
                })

        n = sum(1 for q in all_questions if q["asset_id"] == asset_id)
        print(f"  {n} questions generated")

    # Save
    out_path = DATA_DIR / "questions.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "description": "LLM distributional forecasts vs options-implied volatility",
            "study_params": {
                "cutoff_dates": len(cutoff_mondays),
                "assets": len(ASSETS),
                "horizons": list(HORIZONS.keys()),
                "horizon_days": list(HORIZONS.values()),
            },
            "questions": all_questions,
        }, f, indent=2)

    resolved = sum(1 for q in all_questions if q["ground_truth"] is not None)
    print(f"\nSaved {len(all_questions)} questions ({resolved} with ground truth, {skipped} skipped)")
    print(f"  → {out_path}")


def main():
    print("=== Downloading price histories ===")
    download_prices()

    print("\n=== Downloading vol indices ===")
    download_vol_indices()

    print("\n=== Generating questions ===")
    generate_questions()


if __name__ == "__main__":
    main()
