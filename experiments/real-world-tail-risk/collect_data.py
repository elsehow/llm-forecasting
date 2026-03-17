#!/usr/bin/env python3
"""
Download price histories for Tier 1 real-world tail-risk assets and build question set.

Tier 1 assets (all disruptions after Aug 2025, the latest model training cutoff):
  - WTI Crude Oil: gentle downtrend → +78% spike from Iran/Hormuz (Feb 28 2026)
  - Bitcoin: bull trend → -52% crash from ATH (Oct 2025 onward)
  - Silver: 144% rally → flash crash (Jan-Mar 2026)
  - Henry Hub Natural Gas: normalizing → +55% Hormuz supply shock (Mar 2026)

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/real-world-tail-risk/collect_data.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"

# Asset configurations
# cutoff = last date of data the model sees (before the disruption)
# history_start = start of price history provided to the model
ASSETS = {
    "wti_crude": {
        "ticker": "CL=F",
        "name": "WTI Crude Oil",
        "unit": "$/barrel",
        "history_start": "2024-01-01",
        "cutoff": "2026-02-14",
        "horizons": {
            "H1": "2026-02-21",
            "H2": "2026-02-28",
            "H3": "2026-03-07",
        },
    },
    "bitcoin": {
        "ticker": "BTC-USD",
        "name": "Bitcoin",
        "unit": "USD",
        "history_start": "2024-01-01",
        "cutoff": "2025-09-01",
        "horizons": {
            "H1": "2025-09-08",
            "H2": "2025-09-15",
            "H3": "2025-10-01",
            "H4": "2025-11-01",
            "H5": "2025-12-01",
            "H6": "2026-01-01",
            "H7": "2026-03-01",
        },
    },
    "silver": {
        "ticker": "SI=F",
        "name": "Silver",
        "unit": "$/troy oz",
        "history_start": "2024-01-01",
        "cutoff": "2025-12-01",
        "horizons": {
            "H1": "2025-12-08",
            "H2": "2025-12-15",
            "H3": "2026-01-01",
            "H4": "2026-02-01",
            "H5": "2026-03-01",
        },
    },
    "natural_gas": {
        "ticker": "NG=F",
        "name": "Henry Hub Natural Gas",
        "unit": "$/MMBtu",
        "history_start": "2024-01-01",
        "cutoff": "2026-02-14",
        "horizons": {
            "H1": "2026-02-21",
            "H2": "2026-02-28",
            "H3": "2026-03-07",
        },
    },
    "wheat": {
        "ticker": "ZW=F",
        "name": "CBOT Wheat",
        "unit": "¢/bushel",
        "history_start": "2024-01-01",
        "cutoff": "2026-02-14",
        "horizons": {
            "H1": "2026-02-21",
            "H2": "2026-02-28",
            "H3": "2026-03-07",
        },
    },
}


def nearest_trading_day(df: pd.DataFrame, target: str) -> tuple[str, float]:
    """Find the nearest trading day to target date and return (date_str, close_price).

    Searches within a 7-day window around the target date.
    """
    target_dt = pd.Timestamp(target)
    # Search window: 5 days before to 5 days after
    window_start = target_dt - timedelta(days=5)
    window_end = target_dt + timedelta(days=5)
    mask = (df.index >= window_start) & (df.index <= window_end)
    candidates = df[mask]
    if candidates.empty:
        return None, None
    # Find closest by absolute date difference
    diffs = abs(candidates.index - target_dt)
    idx = diffs.argmin()
    row = candidates.iloc[idx]
    close_val = row["Close"]
    if hasattr(close_val, "iloc"):
        close_val = close_val.iloc[0]
    return str(row.name.date()), float(close_val)


def download_asset(asset_id: str, config: dict) -> dict | None:
    """Download price history and compute ground truth for an asset."""
    ticker = config["ticker"]
    print(f"\nDownloading {config['name']} ({ticker})...")

    # Download through today to get ground truth
    df = yf.download(ticker, start=config["history_start"], end="2026-03-13",
                     progress=False, auto_adjust=True)

    if df.empty:
        print(f"  ERROR: No data for {ticker}")
        return None

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  Got {len(df)} daily prices from {df.index[0].date()} to {df.index[-1].date()}")

    # Save clean price history (just Date,Close)
    csv_path = PRICES_DIR / f"{asset_id}.csv"
    clean = pd.DataFrame({"Date": df.index.strftime("%Y-%m-%d"), "Close": df["Close"].values})
    clean.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")

    # Get cutoff price
    cutoff_date, cutoff_price = nearest_trading_day(df, config["cutoff"])
    if cutoff_price is None:
        print(f"  ERROR: No data near cutoff {config['cutoff']}")
        return None
    print(f"  Cutoff: {cutoff_date} @ {cutoff_price:.2f}")

    # Build history (up to cutoff) for the prompt
    cutoff_dt = pd.Timestamp(cutoff_date)
    history = df[df.index <= cutoff_dt].copy()
    print(f"  History for model: {len(history)} days")

    # Get ground truth for each horizon
    questions = []
    for horizon_label, target_date in config["horizons"].items():
        actual_date, actual_price = nearest_trading_day(df, target_date)
        if actual_price is None:
            print(f"  WARNING: No ground truth for {horizon_label} ({target_date})")
            continue

        # Compute days from cutoff
        days_ahead = (pd.Timestamp(actual_date) - cutoff_dt).days

        questions.append({
            "asset_id": asset_id,
            "asset_name": config["name"],
            "unit": config["unit"],
            "horizon": horizon_label,
            "target_date": target_date,
            "actual_date": actual_date,
            "days_from_cutoff": days_ahead,
            "ground_truth": actual_price,
            "cutoff_date": cutoff_date,
            "cutoff_price": cutoff_price,
        })
        pct_change = (actual_price - cutoff_price) / cutoff_price * 100
        print(f"  {horizon_label}: {actual_date} @ {actual_price:.2f} "
              f"({pct_change:+.1f}% from cutoff)")

    return {
        "asset_id": asset_id,
        "config": config,
        "cutoff_date": cutoff_date,
        "cutoff_price": cutoff_price,
        "history_rows": len(history),
        "questions": questions,
    }


def main():
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    all_questions = []
    asset_summaries = []

    for asset_id, config in ASSETS.items():
        result = download_asset(asset_id, config)
        if result:
            all_questions.extend(result["questions"])
            asset_summaries.append({
                "asset_id": asset_id,
                "name": config["name"],
                "ticker": config["ticker"],
                "cutoff_date": result["cutoff_date"],
                "cutoff_price": result["cutoff_price"],
                "history_rows": result["history_rows"],
                "num_horizons": len(result["questions"]),
            })

    # Save questions
    questions_path = DATA_DIR / "questions.json"
    with open(questions_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "assets": asset_summaries,
            "questions": all_questions,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(all_questions)} questions across {len(asset_summaries)} assets")
    print(f"Questions file: {questions_path}")
    for s in asset_summaries:
        print(f"  {s['name']}: {s['num_horizons']} horizons, "
              f"cutoff {s['cutoff_date']} @ {s['cutoff_price']:.2f}")


if __name__ == "__main__":
    main()
