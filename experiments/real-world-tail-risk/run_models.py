#!/usr/bin/env python3
"""
Run LLM models on real-world tail-risk forecasting questions.

Mirrors the CivBench trajectory condition protocol:
- Shows price history up to a cutoff date
- Asks p10/p25/p50/p75/p90 at multiple future horizons
- Parses structured <<<PERCENTILES>>> responses
- Saves results in the same JSON schema as CivBench

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/real-world-tail-risk/run_models.py --dry-run
    uv run python experiments/real-world-tail-risk/run_models.py
    uv run python experiments/real-world-tail-risk/run_models.py --models openai/gpt-5-2025-08-07
    uv run python experiments/real-world-tail-risk/run_models.py --assets bitcoin silver
"""

import argparse
import asyncio
import json
import re
import time
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load API keys from .env files (try both repos)
load_dotenv(Path(__file__).parent.parent.parent / ".env")
load_dotenv(Path("/Users/elsehow/Projects/civbench/.env"))

from litellm import acompletion

# Force unbuffered output
print = partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS = [
    "openai/gpt-4.1-2025-04-14",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5-2025-08-07",
    "google/gemini-3-pro-preview",
    "anthropic/claude-opus-4-6",
]

# Models that don't support temperature parameter
NO_TEMPERATURE = {"gpt-5", "gpt-5-mini", "o1", "o3", "o3-mini", "o3-pro", "o4-mini"}

# LiteLLM prefix normalization
PREFIX_MAP = {"google": "gemini"}


def litellm_model_id(model_id: str) -> str:
    """Normalize model ID for LiteLLM."""
    if "/" in model_id:
        prefix, name = model_id.split("/", 1)
        prefix = PREFIX_MAP.get(prefix, prefix)
        return f"{prefix}/{name}"
    return model_id


def supports_temperature(model_id: str) -> bool:
    """Check if model supports temperature parameter."""
    name = model_id.split("/")[-1].rsplit("-202", 1)[0]
    return name not in NO_TEMPERATURE


def build_prompt(asset_name: str, unit: str, price_history_csv: str,
                 questions: list[dict]) -> str:
    """Build a trajectory-format prompt for a single asset.

    Mirrors the CivBench trajectory prompt: historical data + questions grouped
    by time series with horizons ascending.
    """
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


def parse_percentiles(response: str, num_questions: int) -> list[dict | None]:
    """Parse percentile estimates from model response.

    Same parsing logic as CivBench: extract from <<<PERCENTILES>>>...<<<END>>> block.
    """
    results = []
    match = re.search(r"<<<PERCENTILES>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    lines = [l.strip() for l in block.split("\n") if l.strip()]

    for line in lines:
        if "p10" not in line.lower():
            continue
        line = re.sub(r"^Q\d+[:\s]*", "", line)
        pct = {}
        for key in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            m = re.search(rf"(?<!\d){key}\s*=\s*([-\d.,]+)", line)
            if m:
                pct[key] = float(m.group(1).replace(",", ""))
        if len(pct) >= 5:
            results.append(pct)
        else:
            results.append(None)

    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


def load_price_history(asset_id: str, cutoff_date: str) -> str:
    """Load price history CSV up to cutoff date, formatted for prompt."""
    csv_path = DATA_DIR / "prices" / f"{asset_id}.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)
    df = df[df.index <= pd.Timestamp(cutoff_date)]

    # Format as Date,Close
    lines = ["Date,Close"]
    for date, row in df.iterrows():
        close = row["Close"] if "Close" in row.index else row.iloc[0]
        lines.append(f"{date.strftime('%Y-%m-%d')},{float(close):.2f}")
    return "\n".join(lines)


async def call_model(model_id: str, prompt: str) -> str:
    """Call a model via LiteLLM."""
    # Reasoning models (GPT-5, o-series) need more tokens for thinking
    base_name = model_id.split("/")[-1].rsplit("-202", 1)[0]
    is_reasoning = base_name in {"gpt-5", "gpt-5-mini", "o1", "o3", "o3-mini", "o3-pro", "o4-mini"}
    max_tok = 16384 if is_reasoning else 4096

    kwargs = {
        "model": litellm_model_id(model_id),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
    }
    if supports_temperature(model_id):
        kwargs["temperature"] = 0.0

    response = await acompletion(**kwargs)
    return response.choices[0].message.content


async def run_experiment(models: list[str], assets: list[str] | None,
                          dry_run: bool = False):
    """Run all models on all assets."""
    # Load questions
    with open(DATA_DIR / "questions.json") as f:
        data = json.load(f)

    # Group questions by asset
    by_asset = {}
    for q in data["questions"]:
        by_asset.setdefault(q["asset_id"], []).append(q)

    # Filter assets if specified
    if assets:
        by_asset = {k: v for k, v in by_asset.items() if k in assets}

    all_results = []
    total_calls = len(models) * len(by_asset)
    call_num = 0

    for model_id in models:
        model_short = model_id.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        for asset_id, questions in sorted(by_asset.items()):
            call_num += 1
            asset_name = questions[0]["asset_name"]
            unit = questions[0]["unit"]
            cutoff_date = questions[0]["cutoff_date"]

            # Load price history
            price_csv = load_price_history(asset_id, cutoff_date)

            # Build prompt
            prompt = build_prompt(asset_name, unit, price_csv, questions)

            print(f"\n  [{call_num}/{total_calls}] {model_short} × {asset_name}: "
                  f"{len(questions)} horizons, ~{len(prompt)//4} tokens")

            if dry_run:
                print(f"    [DRY RUN] Skipping API call")
                for q in questions:
                    print(f"    {q['horizon']}: {q['target_date']} → truth={q['ground_truth']:.2f}")
                continue

            start = time.monotonic()
            try:
                response = await call_model(model_id, prompt)
                latency = (time.monotonic() - start) * 1000
                print(f"    Response in {latency:.0f}ms")

                percentiles = parse_percentiles(response, len(questions))

                for q, pcts in zip(questions, percentiles):
                    truth = q["ground_truth"]
                    result = {
                        "model": model_id,
                        "asset_id": asset_id,
                        "asset_name": asset_name,
                        "unit": unit,
                        "horizon": q["horizon"],
                        "target_date": q["target_date"],
                        "actual_date": q["actual_date"],
                        "days_from_cutoff": q["days_from_cutoff"],
                        "cutoff_date": cutoff_date,
                        "cutoff_price": q["cutoff_price"],
                        "ground_truth": truth,
                        "percentiles": pcts,
                        "raw_response_length": len(response),
                    }
                    all_results.append(result)

                    if pcts:
                        p50 = pcts.get("p50", "?")
                        p10 = pcts.get("p10", "?")
                        p90 = pcts.get("p90", "?")
                        if isinstance(p50, (int, float)) and truth:
                            ratio = f"{p50/truth:.2f}x"
                        else:
                            ratio = "?"
                        print(f"    {q['horizon']}: truth={truth:<10.2f} "
                              f"p10={p10:<10} p50={p50:<10}({ratio}) p90={p90:<10}")
                    else:
                        print(f"    {q['horizon']}: PARSE FAILED")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Real-world tail-risk forecasting experiment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Asset IDs to test (default: all)")
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.models, args.assets, dry_run=args.dry_run))

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "forecasts.json"

        # Load existing and merge (dedup by model + asset + horizon)
        existing = []
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"\nLoaded {len(existing)} existing results")

        new_keys = {(r["model"], r["asset_id"], r["horizon"]) for r in results}
        merged = [r for r in existing
                  if (r["model"], r["asset_id"], r["horizon"]) not in new_keys]
        merged.extend(results)

        with open(out_file, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\nResults saved to {out_file} ({len(merged)} total)")

        # Summary
        from collections import Counter
        counts = Counter((r["model"].split("/")[-1], r["asset_id"]) for r in merged)
        print("\nResult counts:")
        for (model, asset), count in sorted(counts.items()):
            print(f"  {model:<35} {asset:<20} {count} horizons")


if __name__ == "__main__":
    main()
