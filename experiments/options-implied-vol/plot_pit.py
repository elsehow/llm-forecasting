#!/usr/bin/env python3
"""
Plot PIT (probability integral transform) histograms for the options-implied-vol experiment.

Left panel: by asset class (pooled across models)
Right panel: by model (pooled across assets), viridis gradient sorted by peakedness

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/plot_pit.py
"""

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static")


def compute_pit(recs):
    """Compute PIT values from forecast records."""
    pits = []
    for rec in recs:
        gt = rec["ground_truth"]
        qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        keys = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
        qvals = [
            (q, rec["percentiles"][k])
            for q, k in zip(qs, keys)
            if k in rec["percentiles"]
        ]
        if len(qvals) < 5:
            continue
        qs_arr = np.array([q for q, v in qvals])
        vals_arr = np.array([v for q, v in qvals])
        if gt <= vals_arr[0]:
            pit = qs_arr[0] / 2
        elif gt >= vals_arr[-1]:
            pit = 1 - (1 - qs_arr[-1]) / 2
        else:
            pit = np.interp(gt, vals_arr, qs_arr)
        pits.append(pit)
    return np.array(pits)


def main():
    with open(RESULTS_DIR / "forecasts.json") as f:
        data = json.load(f)

    valid = [
        r
        for r in data
        if r.get("percentiles") and r.get("ground_truth") is not None
    ]
    print(f"Loaded {len(valid)} valid forecasts")

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Asset groupings
    groups = {
        "Equity indices": ["sp500", "nasdaq100", "russell", "em"],
        "Commodities": ["oil", "gold"],
        "Single stocks": ["apple", "amazon", "alphabet"],
        "Bonds": ["treasury"],
    }
    group_colors = {
        "Equity indices": "#2166AC",
        "Commodities": "#B2182B",
        "Single stocks": "#762A83",
        "Bonds": "#1B7837",
    }

    # Sort models by peakedness (mid-concentration) for gradient ordering
    models = sorted(set(r["model"] for r in valid))
    model_peak = {}
    for model in models:
        pits = compute_pit([r for r in valid if r["model"] == model])
        if len(pits) > 0:
            model_peak[model] = np.mean((pits >= 0.3) & (pits <= 0.7))
    models_sorted = sorted(model_peak.keys(), key=lambda m: model_peak[m])
    cmap = plt.get_cmap("viridis", len(models_sorted))

    print(f"Models: {len(models_sorted)}")
    print("\nPer-model mid-concentration (0.3-0.7):")
    for m in models_sorted:
        print(f"  {m.split('/')[-1]:<40} {model_peak[m]:.1%} (ideal 40%)")

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Left panel: by asset class
    for group_name, asset_ids in groups.items():
        group_recs = [r for r in valid if r["asset_id"] in asset_ids]
        pits = compute_pit(group_recs)
        counts, _ = np.histogram(pits, bins=bins)
        freqs = counts / counts.sum()
        ax1.plot(
            bin_centers, freqs, marker="o", markersize=5, linewidth=2.2, alpha=0.9,
            color=group_colors[group_name], label=group_name,
        )

    ax1.axhline(0.10, color="#666666", linestyle="--", linewidth=1, alpha=0.6,
                label="Well-calibrated")
    ax1.set_xlabel("PIT value", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("By asset class", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.20)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.95, edgecolor="#cccccc")
    ax1.grid(axis="y", alpha=0.12)

    # Right panel: by model, gradient, no legend, no markers
    for i, model in enumerate(models_sorted):
        model_recs = [r for r in valid if r["model"] == model]
        pits = compute_pit(model_recs)
        counts, _ = np.histogram(pits, bins=bins)
        freqs = counts / counts.sum()
        ax2.plot(bin_centers, freqs, linewidth=1.6, alpha=0.75, color=cmap(i))

    ax2.axhline(0.10, color="#666666", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xlabel("PIT value", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("By model", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.20)
    ax2.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    out = OUTPUT_DIR / "width_vs_shape.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
