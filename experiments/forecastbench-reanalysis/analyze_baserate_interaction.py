"""
Base-rate × source interaction analysis.

Question: Does the anti-g pattern on yfinance intensify at low base rates?
Is this interaction absent on non-volatile sources (Wikipedia, ACLED)?

Approach: For each question, compute the "crowd forecast" (mean across all models)
as a proxy for expected base rate. Bin questions by this crowd forecast. Within
each (source, base-rate bin), compute ECI × Brier correlation.
"""

import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy import stats

# ── ECI scores (same as main analysis) ──
ECI_BY_FB_NAME = {
    "claude-2.1": 112.0,
    "claude-3-haiku-20240307": 118.0,
    "claude-3-opus-20240229": 130.0,
    "claude-3-5-sonnet-20240620": 133.0,
    "claude-3-5-sonnet-20241022": 135.0,
    "claude-3-7-sonnet-20250219": 140.0,
    "claude-haiku-4-5-20251001": 140.0,
    "claude-sonnet-4-20250514": 145.0,
    "claude-opus-4-20250514": 147.0,
    "claude-sonnet-4-5-20250929": 146.6,
    "claude-opus-4-1-20250805": 149.0,
    "claude-opus-4-5-20251101": 150.1,
    "claude-opus-4-6": 152.0,
    "gpt-3.5-turbo-0125": 110.0,
    "gpt-4-0613": 120.0,
    "gpt-4-turbo-2024-04-09": 128.0,
    "gpt-4o": 132.0,
    "gpt-4o-2024-05-13": 132.0,
    "gpt-4.1-2025-04-14": 137.4,
    "gpt-4.5-preview-2025-02-27": 140.0,
    "gpt-5-nano-2025-08-07": 138.0,
    "gpt-5-mini-2025-08-07": 144.3,
    "gpt-5-2025-08-07": 150.0,
    "gpt-5.1-2025-11-13": 149.7,
    "gpt-5.2-2025-12-11": 151.0,
    "o3-mini-2025-01-31": 140.0,
    "o3-2025-04-16": 146.6,
    "o4-mini-2025-04-16": 142.0,
    "gemini-1.5-flash": 122.0,
    "gemini-1.5-pro": 128.0,
    "gemini-2.0-flash-lite-001": 125.0,
    "gemini-2.5-flash": 145.0,
    "gemini-2.5-flash-preview-04-17": 145.0,
    "gemini-2.5-pro": 147.0,
    "gemini-2.5-pro-exp-03-25": 147.0,
    "gemini-2.5-pro-preview-03-25": 147.0,
    "gemini-3-flash-preview": 148.0,
    "gemini-3-pro-preview": 154.2,
    "grok-beta": 125.0,
    "grok-4-0709": 150.0,
    "grok-4-fast-reasoning": 150.0,
    "grok-4-fast-non-reasoning": 145.0,
    "grok-4-1-fast-reasoning": 151.0,
    "grok-4-1-fast-non-reasoning": 146.0,
    "deepseek-r1": 138.0,
    "deepseek-v3": 135.0,
    "deepseek-v3.1": 137.0,
    "llama-2-70b-chat-hf": 105.0,
    "llama-3-70b-chat-hf": 118.0,
    "llama-3-8b-chat-hf": 108.0,
    "llama-3.2-3b-instruct-turbo": 105.0,
    "llama-3.3-70b-instruct-turbo": 125.0,
    "llama-4-maverick-17b-128e-instruct-fp8": 132.0,
    "llama-4-scout-17b-16e-instruct": 128.0,
    "meta-llama-3.1-405b-instruct-turbo": 130.0,
    "mistral-large-2407": 128.0,
    "mistral-large-2411": 130.0,
    "mistral-large-latest": 130.0,
    "magistral-medium-2506": 132.0,
    "mixtral-8x22b-instruct-v0.1": 118.0,
    "mixtral-8x7b-instruct-v0.1": 110.0,
    "kimi-k2-instruct": 140.0,
    "kimi-k2-instruct-0905": 140.0,
    "kimi-k2-thinking": 142.0,
    "glm-4.5-air-fp8": 125.0,
    "glm-4.6": 128.0,
    "glm-4.7": 130.0,
    "qwq-32b-preview": 130.0,
    "qwen1.5-110b-chat": 118.0,
    "qwen2.5-72b-instruct-turbo": 128.0,
    "qwen3-235b-a22b-fp8-tput": 138.0,
    "qwen3-235b-a22b-thinking-2507": 140.0,
}


def match_eci(model_name):
    name = model_name.split(" (")[0].strip().lower()
    for key, eci in ECI_BY_FB_NAME.items():
        if key == name:
            return eci
    return None


def brier_score(forecasts, outcomes):
    return np.mean((np.array(forecasts) - np.array(outcomes)) ** 2)


def load_all_forecasts(base_dir):
    """Load all resolved zero-shot forecasts, keyed by (model, question_key)."""
    # question_key = (source, id, resolution_date, forecast_due_date)
    # Returns: model_data[model_name] = list of entries
    #          question_forecasts[question_key] = list of (model, forecast)
    model_data = defaultdict(list)
    question_forecasts = defaultdict(list)

    for date_dir in sorted(os.listdir(base_dir)):
        dp = os.path.join(base_dir, date_dir)
        if not os.path.isdir(dp):
            continue

        for fname in sorted(os.listdir(dp)):
            if not fname.endswith(".json"):
                continue
            if any(
                x in fname
                for x in [
                    "Always-",
                    "Random",
                    "Imputed",
                    "Naive",
                    "Baselines",
                    "external.",
                ]
            ):
                continue
            if "_zero_shot.json" not in fname:
                continue

            fp = os.path.join(dp, fname)
            with open(fp) as fh:
                data = json.load(fh)

            model_name = data.get("model", "")
            eci = match_eci(model_name)
            if eci is None:
                continue

            for fc in data["forecasts"]:
                if not fc.get("resolved"):
                    continue
                if fc.get("imputed"):
                    continue
                resolved_to = fc["resolved_to"]
                if resolved_to not in (0.0, 1.0):
                    continue
                forecast = fc["forecast"]
                if forecast is None:
                    continue

                qkey = (
                    fc["source"],
                    str(fc["id"]),
                    fc["resolution_date"],
                    data["forecast_due_date"],
                )
                entry = {
                    "forecast": forecast,
                    "outcome": resolved_to,
                    "source": fc["source"],
                    "model": model_name,
                    "eci": eci,
                    "qkey": qkey,
                }
                model_data[model_name].append(entry)
                question_forecasts[qkey].append((model_name, forecast))

    return dict(model_data), dict(question_forecasts)


if __name__ == "__main__":
    BASE = "/Users/elsehow/Projects/forecastbench-datasets/processed_forecast_sets"

    print("Loading forecasts...")
    model_data, question_forecasts = load_all_forecasts(BASE)
    print(f"Loaded {len(model_data)} models, {len(question_forecasts)} unique questions\n")

    # ── Step 1: Compute crowd forecast (mean across models) for each question ──
    question_crowd = {}
    question_source = {}
    question_outcome = {}
    for qkey, forecasts in question_forecasts.items():
        if len(forecasts) < 5:  # need enough models for a meaningful crowd
            continue
        question_crowd[qkey] = np.mean([f for _, f in forecasts])
        question_source[qkey] = qkey[0]
        # All models see the same outcome for the same question
        # Get it from any model's data
        question_outcome[qkey] = None

    # Get outcomes from model_data
    for model_name, entries in model_data.items():
        for e in entries:
            if e["qkey"] in question_crowd:
                question_outcome[e["qkey"]] = e["outcome"]

    print(f"Questions with crowd forecasts: {len(question_crowd)}")

    # ── Step 2: Bin questions by crowd forecast (base-rate proxy) ──
    base_rate_bins = [
        (0.0, 0.20, "rare (0-20%)"),
        (0.20, 0.40, "low (20-40%)"),
        (0.40, 0.60, "mid (40-60%)"),
        (0.60, 0.80, "high (60-80%)"),
        (0.80, 1.01, "very likely (80-100%)"),
    ]

    sources_of_interest = [
        "yfinance",
        "fred",
        "dbnomics",
        "acled",
        "wikipedia",
        "polymarket",
        "manifold",
        "metaculus",
    ]

    # ── Step 3: Within each (source, base-rate bin), compute ECI × Brier ──
    print(f"\n{'='*100}")
    print("BASE-RATE × SOURCE INTERACTION: ECI × Brier")
    print(f"{'='*100}")

    # Collect per-model Brier within each (source, bin)
    for source in sources_of_interest:
        print(f"\n{'─'*80}")
        print(f"  {source.upper()}")
        print(f"{'─'*80}")

        for br_lo, br_hi, br_label in base_rate_bins:
            # Find questions in this (source, bin)
            bin_qkeys = set()
            for qkey, crowd in question_crowd.items():
                if question_source[qkey] == source and br_lo <= crowd < br_hi:
                    bin_qkeys.add(qkey)

            if len(bin_qkeys) < 10:
                print(f"    {br_label}: {len(bin_qkeys)} questions (skip)")
                continue

            # Compute actual resolution rate for this bin
            outcomes = [
                question_outcome[qk] for qk in bin_qkeys if question_outcome[qk] is not None
            ]
            actual_rate = np.mean(outcomes) if outcomes else float("nan")

            # Per-model Brier within this bin
            model_briers = {}
            model_ecis = {}
            model_counts = defaultdict(int)

            for model_name, entries in model_data.items():
                eci = match_eci(model_name)
                if eci is None:
                    continue
                bin_entries = [e for e in entries if e["qkey"] in bin_qkeys]
                if len(bin_entries) < 5:
                    continue
                forecasts = [e["forecast"] for e in bin_entries]
                outcomes_m = [e["outcome"] for e in bin_entries]
                model_briers[model_name] = brier_score(forecasts, outcomes_m)
                model_ecis[model_name] = eci
                model_counts[model_name] = len(bin_entries)

            if len(model_briers) < 10:
                print(
                    f"    {br_label}: {len(bin_qkeys)} Qs, {len(model_briers)} models (skip)"
                )
                continue

            ecis = [model_ecis[m] for m in model_briers]
            briers = [model_briers[m] for m in model_briers]
            rho, p = stats.spearmanr(ecis, briers)
            sig = (
                "***"
                if p < 0.001
                else "**" if p < 0.01 else "*" if p < 0.05 else ""
            )

            mean_brier = np.mean(briers)
            print(
                f"    {br_label}: {len(bin_qkeys):4d} Qs, actual_rate={actual_rate:.2f}, "
                f"N={len(model_briers)} models, mean Brier={mean_brier:.4f}, "
                f"ρ(ECI,Brier)={rho:+.3f} p={p:.4f} {sig}"
            )

    # ── Step 4: Summary table — just the key comparison ──
    print(f"\n{'='*100}")
    print("SUMMARY: ECI × Brier by base-rate bin for key sources")
    print(f"{'='*100}")
    print(
        f"{'Source':<14s} {'rare(0-20)':<18s} {'low(20-40)':<18s} {'mid(40-60)':<18s} {'high(60-80)':<18s} {'likely(80+)':<18s}"
    )
    print("-" * 100)

    for source in sources_of_interest:
        row = f"{source:<14s}"
        for br_lo, br_hi, br_label in base_rate_bins:
            bin_qkeys = set()
            for qkey, crowd in question_crowd.items():
                if question_source[qkey] == source and br_lo <= crowd < br_hi:
                    bin_qkeys.add(qkey)

            if len(bin_qkeys) < 10:
                row += f"{'—':^18s}"
                continue

            model_briers = {}
            model_ecis = {}
            for model_name, entries in model_data.items():
                eci = match_eci(model_name)
                if eci is None:
                    continue
                bin_entries = [e for e in entries if e["qkey"] in bin_qkeys]
                if len(bin_entries) < 5:
                    continue
                forecasts = [e["forecast"] for e in bin_entries]
                outcomes_m = [e["outcome"] for e in bin_entries]
                model_briers[model_name] = brier_score(forecasts, outcomes_m)
                model_ecis[model_name] = eci

            if len(model_briers) < 10:
                row += f"{'—':^18s}"
                continue

            ecis = [model_ecis[m] for m in model_briers]
            briers = [model_briers[m] for m in model_briers]
            rho, p = stats.spearmanr(ecis, briers)
            sig = (
                "***"
                if p < 0.001
                else "**" if p < 0.01 else "*" if p < 0.05 else ""
            )
            row += f"  {rho:+.2f} {sig:<4s} (N={len(model_briers):2d})"

        print(row)
