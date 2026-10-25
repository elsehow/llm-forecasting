#!/usr/bin/env python3
"""
GenSME (Generate-Scenario Mixture Elicitation) on financial assets.

Adapts the CivBench GenSME approach to financial forecasting: models generate
2-3 market scenarios, assign probability weights, and provide conditional
percentile forecasts under each scenario. Mixtures are reconstructed externally.

Compares GenSME CRPS vs monolithic CRPS vs historical volatility baseline.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/run_gensme.py --dry-run
    uv run python experiments/options-implied-vol/run_gensme.py
    uv run python experiments/options-implied-vol/run_gensme.py --models openai/gpt-4.1-2025-04-14
"""

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
load_dotenv(Path("/Users/elsehow/Projects/civbench/.env"))

from litellm import acompletion

print = partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

# Same 5 intervention models as CivBench GenSME
MODELS = [
    "openai/gpt-4.1-2025-04-14",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5-2025-08-07",
    "google/gemini-3-pro-preview",
    "anthropic/claude-opus-4-6",
]

NO_TEMPERATURE = {"gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o3-mini", "o3-pro", "o4-mini"}
PREFIX_MAP = {"google": "gemini"}

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


# ── GenSME Prompt ───────────────────────────────────────────────────────────

def build_gensme_prompt(asset_name: str, unit: str, price_history_csv: str,
                         questions: list[dict]) -> str:
    """Build a GenSME prompt for financial assets.

    The model generates 2-3 market scenarios, assigns probability weights at
    each horizon, and provides conditional percentile forecasts under each
    scenario. External reconstruction computes the mixture distribution.
    """
    num_q = len(questions)
    questions_text = "\n".join(
        f"Q{i+1} ({q['target_date']}): What will the closing price be?"
        for i, q in enumerate(questions)
    )

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing historical price data for a financial asset.

Your task is to:
1. Generate 2-3 plausible market scenarios that could unfold over the forecast horizons
2. Assign probability weights to each scenario at each horizon
3. Provide conditional price forecasts (percentiles) under each scenario

Think carefully about what could happen: continued trends, reversals, volatility spikes, drawdowns, rallies. At least one scenario should involve a significant move against the current trend.

## Asset: {asset_name} ({unit})

## Price History (daily closing prices)
{price_history_csv}

## Questions

{questions_text}

You may analyze the data before your output. You MUST end your response with a block in this exact format:

<<<GENSME>>>
SCENARIOS: A=Continued trend, B=Correction/drawdown, C=Sideways consolidation
Q1: wA=60, wB=25, wC=15, A_p10=240, A_p25=243, A_p50=248, A_p75=253, A_p90=258, B_p10=220, B_p25=225, B_p50=232, B_p75=238, B_p90=242, C_p10=235, C_p25=238, C_p50=242, C_p75=246, C_p90=250
Q2: wA=55, wB=30, wC=15, A_p10=245, A_p25=250, A_p50=258, A_p75=266, A_p90=275, B_p10=210, B_p25=218, B_p50=228, B_p75=236, B_p90=242, C_p10=232, C_p25=237, C_p50=243, C_p75=249, C_p90=255
Q3: wA=50, wB=35, wC=15, ...
<<<END>>>

Rules:
- Weights for each question must sum to 100 (e.g., wA + wB + wC = 100)
- Use 2-3 scenarios (labeled A, B, C)
- All values are actual predicted prices in {unit} (not percentages)
- Provide one line per question, {num_q} questions total
- Percentiles: p10 means 10% chance the price is below this value, etc."""


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_gensme(response: str | None, num_questions: int) -> list[dict | None]:
    """Parse GenSME output from financial forecasting response.

    Returns list of dicts with keys:
    - scenario_labels: list of labels
    - weights: dict of label -> raw weight
    - weights_normalized: dict of label -> normalized weight (sum to 1)
    - conditionals: dict of label -> {p10, p25, p50, p75, p90}
    - mixture: {p10, p25, p50, p75, p90} (externally mixed)
    """
    if response is None:
        return [None] * num_questions

    match = re.search(r"<<<GENSME>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    results = []
    current_scenarios = []

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Scenario definition line
        if line.upper().startswith("SCENARIOS:"):
            scenario_str = line.split(":", 1)[1].strip()
            current_scenarios = []
            for part in scenario_str.split(","):
                part = part.strip()
                if "=" in part:
                    label = part.split("=")[0].strip()
                    current_scenarios.append(label)
            continue

        # Question line
        q_match = re.match(r"Q\d+:\s*(.*)", line)
        if q_match and current_scenarios:
            content = q_match.group(1)

            # Parse weights
            weights = {}
            for label in current_scenarios:
                w_match = re.search(rf"w{label}\s*=\s*([\d.]+)", content)
                if w_match:
                    weights[label] = float(w_match.group(1))

            # Parse conditional percentiles
            conditionals = {}
            for label in current_scenarios:
                pcts = {}
                for pk in ["p10", "p25", "p50", "p75", "p90"]:
                    m = re.search(rf"{label}_{pk}\s*=\s*([-\d.,]+)", content)
                    if m:
                        pcts[pk] = float(m.group(1).replace(",", ""))
                if len(pcts) == 5:
                    conditionals[label] = pcts

            # Validate: need at least 2 scenarios with weights and conditionals
            if len(weights) >= 2 and len(conditionals) >= 2:
                total_w = sum(weights.values())
                if total_w > 0:
                    norm_weights = {k: v / total_w for k, v in weights.items()}
                else:
                    norm_weights = {k: 1.0 / len(weights) for k in weights}

                # Compute mixture percentiles
                mixture = {}
                for pk in ["p10", "p25", "p50", "p75", "p90"]:
                    val = 0.0
                    for label in conditionals:
                        if label in norm_weights:
                            val += norm_weights[label] * conditionals[label][pk]
                    mixture[pk] = val

                results.append({
                    "scenario_labels": current_scenarios,
                    "weights": weights,
                    "weights_normalized": norm_weights,
                    "conditionals": conditionals,
                    "mixture": mixture,
                    "n_scenarios": len(current_scenarios),
                })
            else:
                results.append(None)

    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


# ── CRPS ────────────────────────────────────────────────────────────────────

def crps_quantile(quantiles: dict, ground_truth: float) -> float:
    """Compute CRPS from quantile forecasts using pinball loss."""
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    keys = ["p10", "p25", "p50", "p75", "p90"]
    total = 0.0
    for tau, key in zip(taus, keys):
        q = quantiles[key]
        diff = ground_truth - q
        if diff >= 0:
            total += tau * diff
        else:
            total += (tau - 1) * diff
    return total * 2 / len(taus)


# ── Data loading ────────────────────────────────────────────────────────────

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


# ── Model calls ─────────────────────────────────────────────────────────────

async def call_model(model_id: str, prompt: str) -> str:
    base_name = model_id.split("/")[-1].rsplit("-202", 1)[0]
    is_reasoning = base_name in {"gpt-5", "gpt-5-mini", "o1", "o3", "o3-mini", "o3-pro", "o4-mini"}
    max_tok = 16384 if is_reasoning else 8192  # GenSME needs more tokens than monolithic

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


# ── Main experiment ─────────────────────────────────────────────────────────

async def run_experiment(models: list[str], questions_file: Path, output_file: Path,
                          concurrency: int, assets_filter: list[str] | None,
                          dry_run: bool):
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
    print(f"Assets: {len(set(a for a, _ in groups))} assets, {len(groups)} (asset, cutoff) groups")
    print(f"Concurrency: {concurrency}")

    if dry_run:
        for model, asset_id, cutoff_date, qs in tasks[:5]:
            print(f"  [DRY RUN] {model.split('/')[-1]} × {asset_id} @ {cutoff_date} ({len(qs)} questions)")
        if total > 5:
            print(f"  ... and {total - 5} more")
        return

    # Run with semaphore
    sem = asyncio.Semaphore(concurrency)
    counter = {"done": 0, "errors": 0, "saved": 0}
    lock = asyncio.Lock()

    def _save_results(new_results: list):
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
        prompt = build_gensme_prompt(asset_name, unit, price_csv, qs)

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

        gensme_results = parse_gensme(response, len(qs))
        results = []

        for q, gs in zip(qs, gensme_results):
            result = {
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
                "raw_response_length": len(response) if response else 0,
            }

            if gs is not None:
                result.update({
                    "gensme_parsed": True,
                    "n_scenarios": gs["n_scenarios"],
                    "scenario_labels": gs["scenario_labels"],
                    "weights": gs["weights"],
                    "conditionals": gs["conditionals"],
                    "mixture_percentiles": gs["mixture"],
                })
                if q["ground_truth"] is not None:
                    result["mixture_crps"] = crps_quantile(gs["mixture"], q["ground_truth"])
                else:
                    result["mixture_crps"] = None  # unresolved question
            else:
                result.update({
                    "gensme_parsed": False,
                    "n_scenarios": None,
                    "scenario_labels": None,
                    "weights": None,
                    "conditionals": None,
                    "mixture_percentiles": None,
                    "mixture_crps": None,
                })

            results.append(result)

        async with lock:
            _save_results(results)
            counter["done"] += 1

        model_short = model.split("/")[-1]
        parsed = sum(1 for r in results if r["gensme_parsed"])
        crps_vals = [r["mixture_crps"] for r in results if r["mixture_crps"] is not None]
        crps_str = f"CRPS={np.mean(crps_vals):.1f}" if crps_vals else "NO_PARSE"
        print(f"  [{counter['done']}/{total}] {model_short:>30} × {asset_id:<10} @ {cutoff_date} "
              f"parsed={parsed}/{len(results)} {crps_str} {latency:.0f}ms")

    await asyncio.gather(*[process_one(m, a, c, qs) for m, a, c, qs in tasks])

    print(f"\nDone: {counter['done']} tasks, {counter['errors']} errors, {counter['saved']} results saved")


def print_summary(output_file: Path):
    """Print summary comparison of GenSME vs monolithic vs hist-vol baseline."""
    if not output_file.exists():
        return

    with open(output_file) as f:
        gensme_data = json.load(f)

    # Load monolithic results for comparison
    mono_file = RESULTS_DIR / "forecasts.json"
    mono_data = []
    if mono_file.exists():
        with open(mono_file) as f:
            mono_data = json.load(f)

    print(f"\n{'='*70}")
    print("GENSME vs MONOLITHIC COMPARISON")
    print(f"{'='*70}")

    # Parse rate
    total = len(gensme_data)
    parsed = sum(1 for r in gensme_data if r["gensme_parsed"])
    print(f"\nParse rate: {parsed}/{total} ({100*parsed/total:.1f}%)")

    # Avg scenarios
    n_scenarios = [r["n_scenarios"] for r in gensme_data if r["n_scenarios"] is not None]
    if n_scenarios:
        print(f"Avg scenarios per question: {np.mean(n_scenarios):.1f}")

    # Per-model CRPS comparison
    print(f"\n{'Model':<20} {'GenSME CRPS':>12} {'Mono CRPS':>12} {'Δ':>8} {'N':>5}")
    print("-" * 60)

    # Build monolithic lookup
    mono_lookup = {}
    for r in mono_data:
        if r.get("percentiles") is not None:
            key = (r["model"], r["asset_id"], r["cutoff_date"], r["horizon"])
            pcts = r["percentiles"]
            if all(k in pcts for k in ["p10", "p25", "p50", "p75", "p90"]) and r["ground_truth"] is not None:
                mono_lookup[key] = crps_quantile(
                    {k: pcts[k] for k in ["p10", "p25", "p50", "p75", "p90"]},
                    r["ground_truth"]
                )

    by_model = defaultdict(lambda: {"gensme": [], "mono": []})
    for r in gensme_data:
        if r["mixture_crps"] is not None:
            model = r["model"]
            by_model[model]["gensme"].append(r["mixture_crps"])
            key = (r["model"], r["asset_id"], r["cutoff_date"], r["horizon"])
            if key in mono_lookup:
                by_model[model]["mono"].append(mono_lookup[key])

    for model in sorted(by_model):
        g = by_model[model]["gensme"]
        m = by_model[model]["mono"]
        model_short = model.split("/")[-1]
        if g and m:
            g_mean = np.mean(g)
            m_mean = np.mean(m[:len(g)])  # Match count
            delta = (g_mean - m_mean) / m_mean * 100
            print(f"{model_short:<20} {g_mean:>11.2f} {m_mean:>11.2f} {delta:>+7.1f}% {len(g):>5}")

    # Per-horizon
    print(f"\n{'Horizon':>8} {'GenSME':>10} {'Mono':>10} {'Δ':>8}")
    print("-" * 40)
    for h in ["H1", "H2", "H3"]:
        g_vals = [r["mixture_crps"] for r in gensme_data
                  if r["mixture_crps"] is not None and r["horizon"] == h]
        m_vals = []
        for r in gensme_data:
            if r["mixture_crps"] is not None and r["horizon"] == h:
                key = (r["model"], r["asset_id"], r["cutoff_date"], r["horizon"])
                if key in mono_lookup:
                    m_vals.append(mono_lookup[key])
        if g_vals and m_vals:
            g_mean = np.mean(g_vals)
            m_mean = np.mean(m_vals)
            delta = (g_mean - m_mean) / m_mean * 100
            print(f"  {h:<6} {g_mean:>9.2f} {m_mean:>9.2f} {delta:>+7.1f}%")

    # Compare to hist-vol baseline
    # (hist-vol CRPS from the paper: 39.79 pooled)
    all_gensme_crps = [r["mixture_crps"] for r in gensme_data if r["mixture_crps"] is not None]
    if all_gensme_crps:
        gensme_pooled = np.mean(all_gensme_crps)
        print(f"\nPooled GenSME CRPS: {gensme_pooled:.2f}")
        print(f"Hist-vol baseline CRPS: 39.79 (from main results)")
        print(f"Best monolithic LLM CRPS: 41.22 (o4-mini)")
        if gensme_pooled < 39.79:
            print(f"GenSME BEATS hist-vol baseline by {(39.79 - gensme_pooled)/39.79*100:.1f}%")
        else:
            print(f"GenSME loses to hist-vol by {(gensme_pooled - 39.79)/39.79*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="GenSME on financial assets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--questions-file", default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    questions_file = Path(args.questions_file) if args.questions_file else DATA_DIR / "questions.json"
    output_file = Path(args.output_file) if args.output_file else RESULTS_DIR / "gensme_forecasts.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_experiment(
        args.models, questions_file, output_file,
        args.concurrency, args.assets, args.dry_run
    ))

    print_summary(output_file)


if __name__ == "__main__":
    main()
