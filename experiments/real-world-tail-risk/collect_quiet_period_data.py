#!/usr/bin/env python3
"""
Generate question set for quiet-period WTI windows.

Purpose: establish baseline LLM-implied vol vs OVX during calm markets,
so the Hormuz case becomes "what structural narrowness costs you" rather
than "one time LLMs were wrong."

Windows are post all training cutoffs (latest: Aug 2025 Opus 4.6) and
pre-crisis (Feb 28 2026). Same horizons as crisis analysis: +7d, +14d, +21d.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/real-world-tail-risk/collect_quiet_period_data.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"

# Quiet-period windows — all WTI, same horizons as crisis
WINDOWS = [
    {"id": "sep", "cutoff": "2025-09-15", "label": "Sep 2025"},
    {"id": "oct", "cutoff": "2025-10-20", "label": "Oct 2025"},
    {"id": "nov", "cutoff": "2025-11-24", "label": "Nov 2025"},
    {"id": "jan", "cutoff": "2026-01-12", "label": "Jan 2026"},
]

HORIZONS = {"H1": 7, "H2": 14, "H3": 21}


def nearest_trading_day(df: pd.DataFrame, target: str) -> tuple[str, float]:
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


def main():
    # Load WTI price history
    csv_path = PRICES_DIR / "wti_crude.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.set_index("Date")

    all_questions = []
    asset_summaries = []

    for window in WINDOWS:
        cutoff_date, cutoff_price = nearest_trading_day(df, window["cutoff"])
        if cutoff_price is None:
            print(f"WARNING: No data near cutoff {window['cutoff']}")
            continue

        print(f"\n{window['label']}: cutoff {cutoff_date} @ ${cutoff_price:.2f}")

        questions = []
        for hz_label, days in HORIZONS.items():
            target = pd.Timestamp(cutoff_date) + timedelta(days=days)
            actual_date, actual_price = nearest_trading_day(df, str(target.date()))
            if actual_price is None:
                print(f"  WARNING: No ground truth for {hz_label}")
                continue

            days_actual = (pd.Timestamp(actual_date) - pd.Timestamp(cutoff_date)).days
            pct = (actual_price - cutoff_price) / cutoff_price * 100

            questions.append({
                "asset_id": f"wti_crude_quiet_{window['id']}",
                "asset_name": "WTI Crude Oil",
                "unit": "$/barrel",
                "horizon": hz_label,
                "target_date": str(target.date()),
                "actual_date": actual_date,
                "days_from_cutoff": days_actual,
                "ground_truth": actual_price,
                "cutoff_date": cutoff_date,
                "cutoff_price": cutoff_price,
                "window_label": window["label"],
            })
            print(f"  {hz_label} (+{days_actual}d): {actual_date} @ ${actual_price:.2f} ({pct:+.1f}%)")

        all_questions.extend(questions)
        asset_summaries.append({
            "asset_id": f"wti_crude_quiet_{window['id']}",
            "name": f"WTI Crude Oil ({window['label']})",
            "ticker": "CL=F",
            "cutoff_date": cutoff_date,
            "cutoff_price": cutoff_price,
            "num_horizons": len(questions),
        })

    # Save
    out_path = DATA_DIR / "quiet_period_questions.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "description": "Quiet-period WTI windows for baseline vol comparison",
            "assets": asset_summaries,
            "questions": all_questions,
        }, f, indent=2)

    print(f"\nSaved {len(all_questions)} questions across {len(asset_summaries)} windows")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
