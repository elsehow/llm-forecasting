#!/usr/bin/env python3
"""
Plot: Per-quantile pinball loss × ECI for financial data
=========================================================
Mirrors CivBench's pinball decomposition. For each quantile level,
computes mean pinball loss per model, then Spearman ρ(pinball, ECI).

Output: fri-vault/_artifacts/static/realworld_pinball_eci_by_quantile.png

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/plot_pinball_eci_by_quantile.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static")

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

# Quantiles to plot (subset for readability, matching CivBench)
PLOT_QUANTILES = {
    "p10": {"tau": 0.10, "color": "#b2182b", "marker": "s"},
    "p25": {"tau": 0.25, "color": "#d6604d", "marker": "v"},
    "p50": {"tau": 0.50, "color": "#666666", "marker": "D"},
    "p75": {"tau": 0.75, "color": "#4393c3", "marker": "^"},
    "p90": {"tau": 0.90, "color": "#2166ac", "marker": "o"},
}

HORIZONS = ["H1", "H2", "H3"]
HORIZON_DAYS = {"H1": 7, "H2": 14, "H3": 21}


def pinball_loss(predicted_quantile, actual, tau):
    """Pinball (quantile) loss for a single observation."""
    if actual >= predicted_quantile:
        return tau * (actual - predicted_quantile)
    else:
        return (1 - tau) * (predicted_quantile - actual)


def main():
    with open(RESULTS_DIR / "forecasts.json") as f:
        forecasts = json.load(f)

    valid = [r for r in forecasts
             if r.get("percentiles") and r.get("ground_truth") is not None]

    # Exclude models with too few valid results
    MIN_VALID = 300
    models = sorted(set(r["model"] for r in valid))
    model_counts = {m: sum(1 for r in valid if r["model"] == m) for m in models}
    included_models = [m for m in models if model_counts[m] >= MIN_VALID]
    valid = [r for r in valid if r["model"] in included_models]

    # Filter to models with ECI
    included_with_eci = [m for m in included_models
                         if m.split("/")[-1] in ECI_MAP]
    print(f"Models with ECI: {len(included_with_eci)} / {len(included_models)}")

    # Compute per-model, per-horizon, per-quantile pinball loss
    # Normalize pinball loss by cutoff price to make cross-asset comparable
    losses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # losses[model][horizon][quantile_key] = [normalized pinball losses]

    for r in valid:
        model = r["model"]
        if model not in included_with_eci:
            continue
        hz = r["horizon"]
        gt = r["ground_truth"]
        price = r["cutoff_price"]
        pcts = r["percentiles"]

        for qkey, tau in ALL_QUANTILES.items():
            if qkey in pcts:
                loss = pinball_loss(pcts[qkey], gt, tau)
                # Normalize by cutoff price so losses are comparable across assets
                losses[model][hz][qkey].append(loss / price)

    # For each horizon × quantile, compute Spearman ρ(mean_pinball, ECI)
    print(f"\n{'Horizon':<8} {'Quantile':<10} {'ρ':>8} {'p':>10} {'N':>4}")
    print("-" * 45)

    results = {}  # results[qkey] = {"rho": [...], "p": [...]}
    for qkey in PLOT_QUANTILES:
        results[qkey] = {"rho": [], "p": []}

    for hz in HORIZONS:
        for qkey in PLOT_QUANTILES:
            model_means = []
            ecis = []
            for model in included_with_eci:
                model_name = model.split("/")[-1]
                vals = losses[model][hz][qkey]
                if len(vals) >= 50:
                    model_means.append(np.mean(vals))
                    ecis.append(ECI_MAP[model_name])

            if len(model_means) >= 5:
                rho, p = spearmanr(ecis, model_means)
                results[qkey]["rho"].append(rho)
                results[qkey]["p"].append(p)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"{hz:<8} {qkey:<10} {rho:>+8.3f} {p:>10.4f} {len(model_means):>4} {sig}")
            else:
                results[qkey]["rho"].append(None)
                results[qkey]["p"].append(None)
                print(f"{hz:<8} {qkey:<10} {'N/A':>8} {'N/A':>10}")

    # Also compute pooled (all horizons)
    print(f"\n{'Pooled':<8} {'Quantile':<10} {'ρ':>8} {'p':>10} {'N':>4}")
    print("-" * 45)
    for qkey in PLOT_QUANTILES:
        model_means = []
        ecis = []
        for model in included_with_eci:
            model_name = model.split("/")[-1]
            all_vals = []
            for hz in HORIZONS:
                all_vals.extend(losses[model][hz][qkey])
            if len(all_vals) >= 50:
                model_means.append(np.mean(all_vals))
                ecis.append(ECI_MAP[model_name])
        if len(model_means) >= 5:
            rho, p = spearmanr(ecis, model_means)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{'All':<8} {qkey:<10} {rho:>+8.3f} {p:>10.4f} {len(model_means):>4} {sig}")

    # ── Plot ──
    horizon_days = [HORIZON_DAYS[h] for h in HORIZONS]

    fig, ax = plt.subplots(figsize=(8, 5))

    for qkey, cfg in PLOT_QUANTILES.items():
        rhos = results[qkey]["rho"]
        ps = results[qkey]["p"]

        # Filter Nones
        valid_idx = [i for i, r in enumerate(rhos) if r is not None]
        if not valid_idx:
            continue

        x = [horizon_days[i] for i in valid_idx]
        y = [rhos[i] for i in valid_idx]
        p_vals = [ps[i] for i in valid_idx]

        label = f"{qkey} (τ={cfg['tau']:.2f})"
        ax.plot(x, y, '-', color=cfg["color"], linewidth=1.8, alpha=0.85, zorder=2)

        for xi, yi, pi in zip(x, y, p_vals):
            if pi < 0.05:
                ax.plot(xi, yi, cfg["marker"], color=cfg["color"],
                        markersize=8, zorder=4)
            else:
                ax.plot(xi, yi, cfg["marker"], color=cfg["color"],
                        markersize=8, markerfacecolor='white',
                        markeredgewidth=1.5, zorder=4)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

    ax.axhspan(-1, 0, alpha=0.03, color='#2166ac')
    ax.axhspan(0, 1, alpha=0.03, color='#b2182b')
    ax.text(20.5, -0.55, 'pro-g', fontsize=9, color='#2166ac',
            ha='right', style='italic', alpha=0.7)
    ax.text(20.5, 0.55, 'anti-g', fontsize=9, color='#b2182b',
            ha='right', style='italic', alpha=0.7)

    ax.set_xlabel('Forecast horizon (days ahead)', fontsize=11)
    ax.set_ylabel('Spearman ρ (quantile pinball loss × ECI)', fontsize=11)
    ax.set_xticks(horizon_days)
    ax.set_xticklabels([f'{h}\n(+{d}d)' for h, d in zip(HORIZONS, horizon_days)],
                        fontsize=9)
    ax.set_ylim(-0.65, 0.65)
    ax.set_xlim(4, 24)

    handles = []
    for qkey, cfg in PLOT_QUANTILES.items():
        h = mlines.Line2D([], [], color=cfg["color"], marker=cfg["marker"],
                           linewidth=1.8, markersize=7,
                           label=f"{qkey} (τ={cfg['tau']:.2f})")
        handles.append(h)
    handles.append(mlines.Line2D([], [], color='gray', marker='o', linewidth=0,
                                  markersize=7, label='p < 0.05'))
    handles.append(mlines.Line2D([], [], color='gray', marker='o', linewidth=0,
                                  markersize=7, markerfacecolor='white',
                                  markeredgewidth=1.5, label='n.s.'))
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "realworld_pinball_eci_by_quantile.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
