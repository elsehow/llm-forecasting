#!/usr/bin/env python3
"""
Download price data and generate questions for JKM and TTF (Hormuz experiment).

Appends to the existing questions.json so run_models.py can process them
alongside the original 10 assets.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/collect_hormuz_assets.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"

HORMUZ_ASSETS = {
    "jkm": {
        "name": "Asian LNG spot price (Japan Korea Marker)",
        "unit": "$/MMBtu",
        "price_ticker": "JKM=F",
        "vol_ticker": None,  # No CBOE vol index
    },
    "ttf": {
        "name": "European natural gas (Dutch TTF)",
        "unit": "€/MWh",
        "price_ticker": "TTF=F",
        "vol_ticker": None,
    },
}

HORIZONS = {"H1": 7, "H2": 14, "H3": 21}

CUTOFF_START = "2025-09-01"
CUTOFF_END = "2026-03-10"

PRICE_HISTORY_START = "2024-06-01"
PRICE_HISTORY_END = "2026-04-15"


def generate_cutoff_mondays(start: str, end: str) -> list[str]:
    d = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    while d.weekday() != 0:
        d += timedelta(days=1)
    mondays = []
    while d <= end_d:
        mondays.append(str(d.date()))
        d += timedelta(days=7)
    return mondays


def nearest_trading_day(df, target):
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


def main():
    print("=" * 60)
    print("  Collecting Hormuz assets: JKM + TTF")
    print("=" * 60)

    # Download prices
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    price_dfs = {}
    for asset_id, cfg in HORMUZ_ASSETS.items():
        ticker = cfg["price_ticker"]
        out_path = PRICES_DIR / f"{asset_id}.csv"
        print(f"\n  Downloading {ticker} → {out_path.name}...", end=" ")
        df = yf.download(ticker, start=PRICE_HISTORY_START, end=PRICE_HISTORY_END,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna()
        df.index.name = "Date"
        df.to_csv(out_path)
        print(f"{len(df)} rows")
        price_dfs[asset_id] = df

    # Generate questions
    cutoff_mondays = generate_cutoff_mondays(CUTOFF_START, CUTOFF_END)
    print(f"\n{len(cutoff_mondays)} cutoff dates: {cutoff_mondays[0]} to {cutoff_mondays[-1]}")

    new_questions = []
    skipped = 0

    for asset_id, cfg in HORMUZ_ASSETS.items():
        price_df = price_dfs[asset_id]
        print(f"\n{cfg['name']} ({asset_id}):")

        for cutoff_monday in cutoff_mondays:
            cutoff_date, cutoff_price = nearest_trading_day(price_df, cutoff_monday)
            if cutoff_price is None:
                skipped += 1
                continue

            for hz_label, days in HORIZONS.items():
                target = pd.Timestamp(cutoff_date) + timedelta(days=days)
                actual_date, actual_price = nearest_trading_day(price_df, str(target.date()))

                new_questions.append({
                    "asset_id": asset_id,
                    "asset_name": cfg["name"],
                    "unit": cfg["unit"],
                    "price_ticker": cfg["price_ticker"],
                    "vol_ticker": None,
                    "horizon": hz_label,
                    "target_date": str(target.date()),
                    "actual_date": actual_date,
                    "days_from_cutoff": (pd.Timestamp(actual_date) - pd.Timestamp(cutoff_date)).days if actual_date else days,
                    "ground_truth": actual_price,
                    "cutoff_date": cutoff_date,
                    "cutoff_price": cutoff_price,
                    "vol_index_at_cutoff": None,
                })

        n = sum(1 for q in new_questions if q["asset_id"] == asset_id)
        print(f"  {n} questions generated")

    print(f"\nTotal new questions: {len(new_questions)} ({skipped} skipped)")

    # Load existing questions and append
    questions_path = DATA_DIR / "questions.json"
    with open(questions_path) as f:
        data = json.load(f)

    existing_ids = {q["asset_id"] for q in data["questions"]}
    # Remove any prior JKM/TTF questions (in case of re-run)
    data["questions"] = [q for q in data["questions"]
                         if q["asset_id"] not in HORMUZ_ASSETS]

    data["questions"].extend(new_questions)
    data["study_params"]["assets"] = len(
        set(q["asset_id"] for q in data["questions"])
    )

    with open(questions_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nAppended to {questions_path}")
    print(f"Total questions in file: {len(data['questions'])}")

    # Summary of ground truths around crisis
    print(f"\n{'=' * 60}")
    print("  Crisis-period ground truths")
    print(f"{'=' * 60}")
    for asset_id in HORMUZ_ASSETS:
        crisis_qs = [q for q in new_questions
                     if q["asset_id"] == asset_id
                     and q["cutoff_date"] >= "2026-02-23"
                     and q["horizon"] == "H1"]
        for q in crisis_qs:
            if q["ground_truth"] and q["cutoff_price"]:
                ret = (q["ground_truth"] - q["cutoff_price"]) / q["cutoff_price"] * 100
                print(f"  {asset_id} @ {q['cutoff_date']}: "
                      f"${q['cutoff_price']:.2f} → ${q['ground_truth']:.2f} ({ret:+.1f}%)")


if __name__ == "__main__":
    main()
