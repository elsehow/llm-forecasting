"""
Plot: Spearman rho(ECI, Brier) for yfinance questions by horizon.
Recreated from analyze_fb_calibration.py data.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/forecastbench-reanalysis/plot_fb_yfinance_horizon.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from scipy import stats

# ── ECI scores — full list from analyze_fb_calibration.py ──
ECI_BY_FB_NAME = {
    "claude-2.1": 112.0, "claude-3-haiku-20240307": 118.0,
    "claude-3-opus-20240229": 130.0, "claude-3-5-sonnet-20240620": 133.0,
    "claude-3-5-sonnet-20241022": 135.0, "claude-3-7-sonnet-20250219": 140.0,
    "claude-haiku-4-5-20251001": 140.0, "claude-sonnet-4-20250514": 145.0,
    "claude-opus-4-20250514": 147.0, "claude-sonnet-4-5-20250929": 146.6,
    "claude-opus-4-1-20250805": 149.0, "claude-opus-4-5-20251101": 150.1,
    "claude-opus-4-6": 152.0,
    "gpt-3.5-turbo-0125": 110.0, "gpt-4-0613": 120.0,
    "gpt-4-turbo-2024-04-09": 128.0, "gpt-4o": 132.0,
    "gpt-4o-2024-05-13": 132.0, "gpt-4.1-2025-04-14": 137.4,
    "gpt-4.5-preview-2025-02-27": 140.0, "gpt-5-nano-2025-08-07": 138.0,
    "gpt-5-mini-2025-08-07": 144.3, "gpt-5-2025-08-07": 150.0,
    "gpt-5.1-2025-11-13": 149.7, "gpt-5.2-2025-12-11": 151.0,
    "o3-mini-2025-01-31": 140.0, "o3-2025-04-16": 146.6, "o4-mini-2025-04-16": 142.0,
    "gemini-1.5-flash": 122.0, "gemini-1.5-pro": 128.0,
    "gemini-2.0-flash-lite-001": 125.0, "gemini-2.5-flash": 145.0,
    "gemini-2.5-pro": 147.0, "gemini-3-pro-preview": 154.2,
    "deepseek-r1": 138.0, "deepseek-v3": 135.0, "deepseek-v3.1": 137.0,
    "llama-2-70b-chat-hf": 105.0, "llama-3-70b-chat-hf": 118.0,
    "llama-3-8b-chat-hf": 108.0, "llama-3.3-70b-instruct-turbo": 125.0,
    "mistral-large-latest": 130.0, "mixtral-8x22b-instruct-v0.1": 118.0,
    "mixtral-8x7b-instruct-v0.1": 110.0, "qwen1.5-110b-chat": 118.0,
}

OUT_PATH = "/Users/elsehow/Projects/fri-vault/_artifacts/static/fb_yfinance_horizon_flip.png"


def match_eci(model_name):
    """Match a ForecastBench model name to an ECI score (case-insensitive)."""
    name = model_name.split(" (")[0].strip().lower()
    for key, eci in ECI_BY_FB_NAME.items():
        if key == name:
            return eci
    return None


def brier_score(forecasts, outcomes):
    return np.mean([(f - o) ** 2 for f, o in zip(forecasts, outcomes)])


def load_all_forecasts(base_dir, condition="zero_shot"):
    """Load all resolved forecasts. Matches analyze_fb_calibration.py format."""
    model_data = defaultdict(list)
    for date_dir in sorted(os.listdir(base_dir)):
        dp = os.path.join(base_dir, date_dir)
        if not os.path.isdir(dp):
            continue
        for fname in sorted(os.listdir(dp)):
            if not fname.endswith(".json"):
                continue
            if any(x in fname for x in ["Always-", "Random", "Imputed", "Crowd"]):
                continue
            if condition and f"_{condition}.json" not in fname:
                continue
            fp = os.path.join(dp, fname)
            try:
                with open(fp) as fh:
                    data = json.load(fh)
            except Exception:
                continue

            model_name = data.get("model", "")
            if not model_name:
                continue
            try:
                due_date = datetime.fromisoformat(data["forecast_due_date"])
            except Exception:
                continue

            for fc in data.get("forecasts", []):
                if not fc.get("resolved"):
                    continue
                if fc.get("imputed"):
                    continue
                resolved_to = fc.get("resolved_to")
                if resolved_to not in (0.0, 1.0):
                    continue
                forecast = fc.get("forecast")
                if forecast is None:
                    continue
                try:
                    res_date = datetime.fromisoformat(fc["resolution_date"])
                    horizon_days = (res_date - due_date).days
                except Exception:
                    horizon_days = None

                model_data[model_name].append({
                    "forecast": float(forecast),
                    "outcome": float(resolved_to),
                    "source": fc.get("source", ""),
                    "horizon_days": horizon_days,
                })
    return dict(model_data)


# ── Main ──
BASE = "/Users/elsehow/Projects/forecastbench-datasets/processed_forecast_sets"
print("Loading forecasts...")
model_data = load_all_forecasts(BASE, condition="zero_shot")
print(f"Loaded {len(model_data)} models")

horizon_bins = [
    (0, 10, "1wk"),
    (25, 35, "1mo"),
    (80, 100, "3mo"),
    (170, 190, "6mo"),
    (350, 380, "12mo"),
]

results = []
for h_lo, h_hi, h_label in horizon_bins:
    h_results = []
    for model_name, entries in model_data.items():
        eci = match_eci(model_name)
        if eci is None:
            continue
        yf = [e for e in entries if e["source"] == "yfinance" and h_lo <= (e["horizon_days"] or 0) <= h_hi]
        if len(yf) < 20:
            continue
        forecasts = [e["forecast"] for e in yf]
        outcomes = [e["outcome"] for e in yf]
        h_results.append({"eci": eci, "brier": brier_score(forecasts, outcomes)})

    if len(h_results) >= 5:
        ecis = [r["eci"] for r in h_results]
        briers = [r["brier"] for r in h_results]
        rho, p = stats.spearmanr(ecis, briers)
        results.append({"label": h_label, "rho": rho, "p": p, "n": len(h_results)})
        print(f"  {h_label}: ρ={rho:+.3f}, p={p:.4f}, N={len(h_results)}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(7, 4.5))

labels = [r["label"] for r in results]
rhos = [r["rho"] for r in results]
ps = [r["p"] for r in results]
x = np.arange(len(results))

colors = ["#2d6a8f" if r < 0 else "#b84a2a" for r in rhos]
bars = ax.bar(x, rhos, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

ax.axhline(y=0, color="black", linewidth=0.8)

# Labels with p-values
n_tests = len(results)
for i, (r, p) in enumerate(zip(rhos, ps)):
    corrected_p = min(p * n_tests, 1.0)
    sig = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else ""
    label = f"ρ={r:+.2f}\n(p={corrected_p:.2f}){sig}"
    y_offset = 0.02 if r > 0 else -0.02
    va = "bottom" if r > 0 else "top"
    ax.text(i, r + y_offset, label, ha="center", va=va, fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_xlabel("Forecast horizon", fontsize=12)
ax.set_ylabel("Spearman ρ(ECI, Brier)", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

# Shade
ax.axhspan(0, ax.get_ylim()[1], alpha=0.04, color="#b84a2a")
ax.axhspan(ax.get_ylim()[0], 0, alpha=0.04, color="#2d6a8f")
ax.text(0.02, 0.98, "Anti-g (capability hurts)", transform=ax.transAxes,
        fontsize=8, fontstyle="italic", color="#b84a2a", va="top")
ax.text(0.02, 0.02, "Pro-g (capability helps)", transform=ax.transAxes,
        fontsize=8, fontstyle="italic", color="#2d6a8f", va="bottom")

fig.text(0.5, 0.01,
         f"Bonferroni-corrected for {n_tests} tests. yfinance questions only.",
         ha="center", fontsize=8, color="gray", fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved to {OUT_PATH}")
