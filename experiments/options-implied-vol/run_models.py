#!/usr/bin/env python3
"""
Run LLM models on options-implied-vol questions with concurrency and resume.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/run_models.py --dry-run
    uv run python experiments/options-implied-vol/run_models.py
    uv run python experiments/options-implied-vol/run_models.py --assets sp500 oil --concurrency 5
"""

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
load_dotenv(Path("/Users/elsehow/Projects/civbench/.env"))

from litellm import acompletion

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

NO_TEMPERATURE = {"gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o3-mini", "o3-pro", "o4-mini"}
PREFIX_MAP = {"google": "gemini"}

# Number of price history days to include in prompt
PRICE_HISTORY_DAYS = 200


def litellm_model_id(model_id: str) -> str:
    if "/" in model_id:
        prefix, name = model_id.split("/", 1)
        prefix = PREFIX_MAP.get(prefix, prefix)
        return f"{prefix}/{name}"
    return model_id


def supports_temperature(model_id: str) -> bool:
    name = model_id.split("/")[-1].rsplit("-202", 1)[0]
    return name not in NO_TEMPERATURE


def build_prompt(asset_name: str, unit: str, price_history_csv: str,
                 questions: list[dict]) -> str:
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


def parse_percentiles(response: str | None, num_questions: int) -> list[dict | None]:
    if response is None:
        return [None] * num_questions
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
    """Load ~200 trading days of price history ending at cutoff."""
    csv_path = DATA_DIR / "prices" / f"{asset_id}.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.set_index("Date")
    df = df[df.index <= pd.Timestamp(cutoff_date)]
    df = df.tail(PRICE_HISTORY_DAYS)

    lines = ["Date,Close"]
    for date, row in df.iterrows():
        close = row["Close"] if "Close" in row.index else row.iloc[0]
        lines.append(f"{date.strftime('%Y-%m-%d')},{float(close):.2f}")
    return "\n".join(lines)


async def call_model(model_id: str, prompt: str) -> str:
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


async def call_with_retry(model_id: str, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return await call_model(model_id, prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * (1 + random.random())
            print(f"    Retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
            await asyncio.sleep(wait)


async def run_experiment(models: list[str], questions_file: Path, output_file: Path,
                          concurrency: int, assets_filter: list[str] | None,
                          dry_run: bool):
    # Load questions
    with open(questions_file) as f:
        data = json.load(f)

    # Group questions by (asset_id, cutoff_date)
    groups = {}
    for q in data["questions"]:
        if assets_filter and q["asset_id"] not in assets_filter:
            continue
        key = (q["asset_id"], q["cutoff_date"])
        groups.setdefault(key, []).append(q)

    # Load existing results for resume
    existing = []
    completed_keys = set()
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        completed_keys = {(r["model"], r["asset_id"], r["cutoff_date"]) for r in existing}
        print(f"Loaded {len(existing)} existing results, {len(completed_keys)} completed groups")

    # Build task list
    tasks = []
    for model in models:
        for (asset_id, cutoff_date), qs in sorted(groups.items()):
            if (model, asset_id, cutoff_date) in completed_keys:
                continue
            tasks.append((model, asset_id, cutoff_date, qs))

    total = len(tasks)
    skipped = len(models) * len(groups) - total
    print(f"\n{total} tasks to run ({skipped} already completed)")
    print(f"Models: {[m.split('/')[-1] for m in models]}")
    print(f"Assets: {len(groups)} (asset, cutoff) groups")
    print(f"Concurrency: {concurrency}")

    if dry_run:
        for model, asset_id, cutoff_date, qs in tasks[:5]:
            print(f"  [DRY RUN] {model.split('/')[-1]} × {asset_id} @ {cutoff_date}")
        if total > 5:
            print(f"  ... and {total - 5} more")
        return []

    # Run with semaphore
    sem = asyncio.Semaphore(concurrency)
    counter = {"done": 0, "errors": 0, "saved": 0}
    lock = asyncio.Lock()

    def _save_results(new_results: list):
        """Incrementally save results to disk (called under lock)."""
        existing = []
        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)
        new_keys = {(r["model"], r["asset_id"], r["cutoff_date"], r["horizon"]) for r in new_results}
        merged = [r for r in existing
                  if (r["model"], r["asset_id"], r["cutoff_date"], r["horizon"]) not in new_keys]
        merged.extend(new_results)
        with open(output_file, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        counter["saved"] += len(new_results)

    async def process_one(model, asset_id, cutoff_date, qs):
        asset_name = qs[0]["asset_name"]
        unit = qs[0]["unit"]

        price_csv = load_price_history(asset_id, cutoff_date)
        prompt = build_prompt(asset_name, unit, price_csv, qs)

        async with sem:
            start = time.monotonic()
            try:
                response = await call_with_retry(model, prompt)
                latency = (time.monotonic() - start) * 1000
            except Exception as e:
                counter["errors"] += 1
                counter["done"] += 1
                print(f"  [{counter['done']}/{total}] ERROR {model.split('/')[-1]} × {asset_id} @ {cutoff_date}: {e}")
                return

        percentiles = parse_percentiles(response, len(qs))
        results = []
        for q, pcts in zip(qs, percentiles):
            results.append({
                "model": model,
                "asset_id": asset_id,
                "asset_name": asset_name,
                "unit": unit,
                "horizon": q["horizon"],
                "target_date": q["target_date"],
                "actual_date": q["actual_date"],
                "days_from_cutoff": q["days_from_cutoff"],
                "cutoff_date": cutoff_date,
                "cutoff_price": q["cutoff_price"],
                "vol_index_at_cutoff": q["vol_index_at_cutoff"],
                "ground_truth": q["ground_truth"],
                "percentiles": pcts,
                "raw_response_length": len(response) if response else 0,
            })

        async with lock:
            _save_results(results)
            counter["done"] += 1

        model_short = model.split("/")[-1]
        status = "OK" if all(r["percentiles"] for r in results) else "PARSE_FAIL"
        p50s = [r["percentiles"]["p50"] for r in results if r["percentiles"] and "p50" in r["percentiles"]]
        p50_str = ", ".join(f"{p:.0f}" for p in p50s) if p50s else "?"
        print(f"  [{counter['done']}/{total}] {model_short:>30} × {asset_id:<10} @ {cutoff_date} "
              f"{status} {latency:.0f}ms p50=[{p50_str}]")

    await asyncio.gather(*[process_one(m, a, c, qs) for m, a, c, qs in tasks])

    print(f"\nDone: {counter['done']} tasks, {counter['errors']} errors, {counter['saved']} results saved")


def main():
    parser = argparse.ArgumentParser(description="Options-implied-vol experiment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--questions-file", default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    questions_file = Path(args.questions_file) if args.questions_file else DATA_DIR / "questions.json"
    output_file = Path(args.output_file) if args.output_file else RESULTS_DIR / "forecasts.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_experiment(
        args.models, questions_file, output_file,
        args.concurrency, args.assets, args.dry_run
    ))

    # Print final summary
    if output_file.exists():
        with open(output_file) as f:
            all_data = json.load(f)
        print(f"\nTotal in {output_file}: {len(all_data)} results")
        counts = Counter(r["model"].split("/")[-1] for r in all_data)
        for model, count in sorted(counts.items()):
            print(f"  {model:<35} {count} forecasts")


if __name__ == "__main__":
    main()
