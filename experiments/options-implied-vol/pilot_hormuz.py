#!/usr/bin/env python3
"""
Pilot test: can models produce parseable distributional forecasts for JKM and TTF?

Tests 3 models × 2 assets × 1 cutoff date (Feb 23, 2026) × 3 horizons.
Uses the same prompt format and parser as the main pipeline.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/pilot_hormuz.py
"""

import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from litellm import acompletion

PILOT_MODELS = [
    "openai/gpt-4.1-2025-04-14",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-3.5-turbo-0125",  # weakest model — stress test
]

PILOT_ASSETS = {
    "jkm": {
        "name": "Asian LNG spot price (Japan Korea Marker)",
        "unit": "$/MMBtu",
        "ticker": "JKM=F",
    },
    "ttf": {
        "name": "European natural gas (Dutch TTF)",
        "unit": "€/MWh",
        "ticker": "TTF=F",
    },
}

CUTOFF_DATE = "2026-02-23"
HORIZONS = {"H1": 7, "H2": 14, "H3": 21}
PRICE_HISTORY_DAYS = 200


def build_prompt(asset_name, unit, price_history_csv, questions):
    num_q = len(questions)
    questions_text = "\n".join(
        f"Q{i+1} ({q['target_date']}): p1=__, p5=__, p10=__, p25=__, p50=__, p75=__, p90=__, p95=__, p99=__"
        for i, q in enumerate(questions)
    )
    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing historical price data for a financial asset. Make predictions based on the price history below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable estimates, but always return numeric percentile values.

## Asset: {asset_name} ({unit})

## Price History (daily closing prices)
{price_history_csv}

## Questions

What will the closing price of {asset_name} be on the following dates?

{questions_text}

You may analyze the data, but you MUST end your response with percentile estimates in this exact format:
<<<PERCENTILES>>>
Q1: p1=2, p5=4, p10=5, p25=10, p50=15, p75=20, p90=25, p95=28, p99=35
Q2: p1=50, p5=80, p10=100, p25=200, p50=300, p75=400, p90=500, p95=600, p99=800
<<<END>>>

For each question, provide one line with percentile estimates for all {num_q} questions, in order.
- p1 means you estimate there's a 1% chance the true value is below this number
- p5 means you estimate there's a 5% chance the true value is below this number
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number
- p95 means you estimate there's a 95% chance the true value is below this number
- p99 means you estimate there's a 99% chance the true value is below this number"""


def parse_percentiles(response, num_questions):
    if response is None:
        return [None] * num_questions
    match = re.search(r"<<<PERCENTILES>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions
    block = match.group(1).strip()
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    results = []
    for line in lines[:num_questions]:
        pcts = {}
        for m in re.finditer(r"p(\d+)\s*=\s*([\d.,]+)", line):
            key = f"p{m.group(1)}"
            val = m.group(2).replace(",", "")
            try:
                pcts[key] = float(val)
            except ValueError:
                pass
        results.append(pcts if len(pcts) >= 5 else None)
    while len(results) < num_questions:
        results.append(None)
    return results


async def run_one(model, asset_id, asset_cfg, price_csv, questions):
    prompt = build_prompt(asset_cfg["name"], asset_cfg["unit"], price_csv, questions)
    model_name = model.split("/")[-1]

    try:
        resp = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content
        parsed = parse_percentiles(text, len(questions))
        n_ok = sum(1 for p in parsed if p is not None)
        print(f"  ✓ {model_name:>35} × {asset_id:<4}: {n_ok}/{len(questions)} parsed")
        if n_ok > 0:
            # Show the parsed percentiles for the first question
            first = next(p for p in parsed if p is not None)
            print(f"    Sample Q1: {first}")
        return {"model": model_name, "asset": asset_id, "parsed": parsed, "raw": text}
    except Exception as e:
        print(f"  ✗ {model_name:>35} × {asset_id:<4}: ERROR — {e}")
        return {"model": model_name, "asset": asset_id, "parsed": [None]*len(questions), "error": str(e)}


async def main():
    print("=" * 60)
    print("  PILOT: JKM + TTF parse test")
    print("=" * 60)

    # Download price data
    print("\nDownloading price data...")
    cutoff_dt = pd.Timestamp(CUTOFF_DATE)
    start_dt = cutoff_dt - timedelta(days=int(PRICE_HISTORY_DAYS * 1.5))

    price_csvs = {}
    for asset_id, cfg in PILOT_ASSETS.items():
        df = yf.download(cfg["ticker"], start=str(start_dt.date()),
                         end=str((cutoff_dt + timedelta(days=1)).date()),
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna().tail(PRICE_HISTORY_DAYS)
        csv = df.to_csv()
        price_csvs[asset_id] = csv
        last_price = df["Close"].iloc[-1]
        print(f"  {asset_id}: {len(df)} days, last price: {last_price:.2f} {cfg['unit']}")

    # Build questions
    questions = []
    for hz_name, days in HORIZONS.items():
        target = cutoff_dt + timedelta(days=days)
        questions.append({
            "horizon": hz_name,
            "target_date": str(target.date()),
            "days_from_cutoff": days,
        })

    # Run models
    print(f"\nRunning {len(PILOT_MODELS)} models × {len(PILOT_ASSETS)} assets...")
    print(f"Cutoff: {CUTOFF_DATE}, Horizons: H1/H2/H3\n")

    tasks = []
    for model in PILOT_MODELS:
        for asset_id, cfg in PILOT_ASSETS.items():
            tasks.append(run_one(model, asset_id, cfg, price_csvs[asset_id], questions))

    results = await asyncio.gather(*tasks)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    total = 0
    ok = 0
    for r in results:
        for p in r["parsed"]:
            total += 1
            if p is not None:
                ok += 1

    print(f"\n  Parse rate: {ok}/{total} ({ok/total*100:.0f}%)")
    print()

    for asset_id in PILOT_ASSETS:
        asset_results = [r for r in results if r["asset"] == asset_id]
        asset_ok = sum(1 for r in asset_results for p in r["parsed"] if p is not None)
        asset_total = sum(len(r["parsed"]) for r in asset_results)
        print(f"  {asset_id}: {asset_ok}/{asset_total} ({asset_ok/asset_total*100:.0f}%)")

    for model in PILOT_MODELS:
        model_name = model.split("/")[-1]
        model_results = [r for r in results if r["model"] == model_name]
        model_ok = sum(1 for r in model_results for p in r["parsed"] if p is not None)
        model_total = sum(len(r["parsed"]) for r in model_results)
        print(f"  {model_name}: {model_ok}/{model_total} ({model_ok/model_total*100:.0f}%)")

    # Check distribution widths — are they reasonable?
    print(f"\n  Distribution widths (p90 - p10) for parsed forecasts:")
    for r in results:
        for i, p in enumerate(r["parsed"]):
            if p and "p10" in p and "p90" in p:
                width = p["p90"] - p["p10"]
                pct_width = width / p.get("p50", 1) * 100 if p.get("p50") else 0
                print(f"    {r['model']:>35} × {r['asset']:<4} Q{i+1}: "
                      f"p10={p['p10']:.2f}  p50={p['p50']:.2f}  p90={p['p90']:.2f}  "
                      f"width={width:.2f} ({pct_width:.1f}% of p50)")


if __name__ == "__main__":
    asyncio.run(main())
