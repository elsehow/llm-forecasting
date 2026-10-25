#!/usr/bin/env python3
"""
Pinball loss × ECI decomposition: calm vs Hormuz crisis (oil only)
===================================================================
Tests whether the per-quantile g-loading changes during a crash event,
analogous to CivBench's crash-world vs growth-world split.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/plot_pinball_eci_hormuz.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

ALL_QUANTILES = {
    "p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
    "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99,
}

ECI_MAP = {
    "claude-3-haiku-20240307": 98,
    "gpt-3.5-turbo-0125": 100,
    "Mixtral-8x7B-Instruct-v0.1": 89,
    "Llama-3.3-70B-Instruct-Turbo": 117,
    "mistral-large-2407": 118,
    "mistral-large-2411": 125,
    "mistral-large-latest": 130,
    "gpt-4o": 132,
    "gemini-2.0-flash-lite-001": 110,
    "gemini-2.5-flash": 135,
    "gemini-2.5-pro": 140,
    "DeepSeek-V3": 128,
    "DeepSeek-V3.1": 135,
    "o3-mini-2025-01-31": 140,
    "gpt-4.1-2025-04-14": 137,
    "claude-sonnet-4-20250514": 140,
    "claude-haiku-4-5-20251001": 138,
    "o4-mini-2025-04-16": 145,
    "gpt-5-nano-2025-08-07": 120,
    "gpt-5-mini-2025-08-07": 142,
    "claude-sonnet-4-5-20250929": 148,
    "o3-2025-04-16": 149,
    "gpt-5-2025-08-07": 150,
    "gpt-5.1-2025-11-13": 152,
    "gemini-3-pro-preview": 154,
    "claude-opus-4-5-20251101": 155,
    "grok-4-fast-non-reasoning": 140,
    "grok-4-fast-reasoning": 148,
    "grok-4-0709": 151,
    "grok-4-1-fast-non-reasoning": 145,
    "grok-4-1-fast-reasoning": 152,
    "claude-opus-4-6": 157,
}

CRISIS_CUTOFF = "2026-02-23"

PLOT_QUANTILES = ["p10", "p25", "p50", "p75", "p90"]


def pinball_loss(predicted_quantile, actual, tau):
    if actual >= predicted_quantile:
        return tau * (actual - predicted_quantile)
    else:
        return (1 - tau) * (predicted_quantile - actual)


def main():
    with open(RESULTS_DIR / "forecasts.json") as f:
        forecasts = json.load(f)

    valid = [r for r in forecasts
             if r.get("percentiles") and r.get("ground_truth") is not None]

    # Filter to included models
    MIN_VALID = 300
    models = sorted(set(r["model"] for r in valid))
    model_counts = {m: sum(1 for r in valid if r["model"] == m) for m in models}
    included = [m for m in models if model_counts[m] >= MIN_VALID]

    # Oil only
    oil = [r for r in valid if r["asset_id"] == "oil" and r["model"] in included]
    oil_with_eci = [r for r in oil if r["model"].split("/")[-1] in ECI_MAP]

    calm = [r for r in oil_with_eci if r["cutoff_date"] < CRISIS_CUTOFF]
    crisis = [r for r in oil_with_eci if r["cutoff_date"] >= CRISIS_CUTOFF]

    print(f"Oil forecasts: {len(calm)} calm, {len(crisis)} crisis")
    print(f"Calm cutoffs: {len(set(r['cutoff_date'] for r in calm))}")
    print(f"Crisis cutoffs: {len(set(r['cutoff_date'] for r in crisis))}")

    models_with_eci = sorted(set(r["model"] for r in oil_with_eci))

    # ── Per-quantile pinball × ECI, calm vs crisis ──
    for regime_label, regime_data in [("CALM", calm), ("CRISIS", crisis)]:
        print(f"\n{'='*60}")
        print(f"  {regime_label}: Per-quantile pinball × ECI (oil, all horizons pooled)")
        print(f"{'='*60}")

        # Compute per-model, per-quantile mean pinball loss (normalized by price)
        losses = defaultdict(lambda: defaultdict(list))
        for r in regime_data:
            model = r["model"]
            gt = r["ground_truth"]
            price = r["cutoff_price"]
            pcts = r["percentiles"]
            for qkey in PLOT_QUANTILES:
                tau = ALL_QUANTILES[qkey]
                if qkey in pcts:
                    loss = pinball_loss(pcts[qkey], gt, tau)
                    losses[model][qkey].append(loss / price)

        print(f"\n  {'Quantile':<10} {'ρ':>8} {'p':>10} {'N':>4} {'sig':>5}")
        print(f"  {'-'*40}")

        for qkey in PLOT_QUANTILES:
            model_means = []
            ecis = []
            for model in models_with_eci:
                model_name = model.split("/")[-1]
                vals = losses[model][qkey]
                if len(vals) >= 3:  # lower threshold for crisis (only 3 cutoffs × 3 horizons)
                    model_means.append(np.mean(vals))
                    ecis.append(ECI_MAP[model_name])

            if len(model_means) >= 5:
                rho, p = spearmanr(ecis, model_means)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {qkey:<10} {rho:>+8.3f} {p:>10.4f} {len(model_means):>4} {sig:>5}")
            else:
                print(f"  {qkey:<10} {'N/A':>8} (too few models)")

    # ── Per-horizon breakdown for crisis ──
    print(f"\n{'='*60}")
    print(f"  CRISIS: Per-quantile pinball × ECI by horizon")
    print(f"{'='*60}")

    for hz in ["H1", "H2", "H3"]:
        hz_data = [r for r in crisis if r["horizon"] == hz]
        if not hz_data:
            continue

        losses = defaultdict(lambda: defaultdict(list))
        for r in hz_data:
            model = r["model"]
            gt = r["ground_truth"]
            price = r["cutoff_price"]
            pcts = r["percentiles"]
            for qkey in PLOT_QUANTILES:
                tau = ALL_QUANTILES[qkey]
                if qkey in pcts:
                    loss = pinball_loss(pcts[qkey], gt, tau)
                    losses[model][qkey].append(loss / price)

        print(f"\n  {hz} ({len(hz_data)} forecasts):")
        print(f"  {'Quantile':<10} {'ρ':>8} {'p':>10} {'N':>4} {'sig':>5}")
        print(f"  {'-'*40}")

        for qkey in PLOT_QUANTILES:
            model_means = []
            ecis = []
            for model in models_with_eci:
                model_name = model.split("/")[-1]
                vals = losses[model][qkey]
                if vals:
                    model_means.append(np.mean(vals))
                    ecis.append(ECI_MAP[model_name])

            if len(model_means) >= 5:
                rho, p = spearmanr(ecis, model_means)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {qkey:<10} {rho:>+8.3f} {p:>10.4f} {len(model_means):>4} {sig:>5}")

    # ── Delta analysis: how much does each model's pinball loss CHANGE? ──
    print(f"\n{'='*60}")
    print(f"  DELTA: Change in pinball loss (crisis - calm) × ECI")
    print(f"{'='*60}")
    print(f"  Positive delta = model got worse in crisis")

    calm_losses = defaultdict(lambda: defaultdict(list))
    crisis_losses = defaultdict(lambda: defaultdict(list))

    for r in calm:
        model = r["model"]
        gt = r["ground_truth"]
        price = r["cutoff_price"]
        pcts = r["percentiles"]
        for qkey in PLOT_QUANTILES:
            tau = ALL_QUANTILES[qkey]
            if qkey in pcts:
                loss = pinball_loss(pcts[qkey], gt, tau)
                calm_losses[model][qkey].append(loss / price)

    for r in crisis:
        model = r["model"]
        gt = r["ground_truth"]
        price = r["cutoff_price"]
        pcts = r["percentiles"]
        for qkey in PLOT_QUANTILES:
            tau = ALL_QUANTILES[qkey]
            if qkey in pcts:
                loss = pinball_loss(pcts[qkey], gt, tau)
                crisis_losses[model][qkey].append(loss / price)

    print(f"\n  {'Quantile':<10} {'ρ':>8} {'p':>10} {'N':>4} {'sig':>5}  {'interp'}")
    print(f"  {'-'*65}")

    for qkey in PLOT_QUANTILES:
        deltas = []
        ecis = []
        for model in models_with_eci:
            model_name = model.split("/")[-1]
            calm_vals = calm_losses[model][qkey]
            crisis_vals = crisis_losses[model][qkey]
            if len(calm_vals) >= 3 and len(crisis_vals) >= 3:
                delta = np.mean(crisis_vals) - np.mean(calm_vals)
                deltas.append(delta)
                ecis.append(ECI_MAP[model_name])

        if len(deltas) >= 5:
            rho, p = spearmanr(ecis, deltas)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if rho > 0:
                interp = "capable models degrade MORE in crisis"
            else:
                interp = "capable models degrade LESS in crisis"
            print(f"  {qkey:<10} {rho:>+8.3f} {p:>10.4f} {len(deltas):>4} {sig:>5}  {interp}")


if __name__ == "__main__":
    main()
