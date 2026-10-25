#!/usr/bin/env python3
"""Trend chart: ECI vs discrimination on the certified skill class.
Directly answers Q1 (is there a trend?) and annotates Q2 (FB transfer | ECI)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad")
CORPUS = json.load(open(S / "lab_classes.json"))
CACHE = json.load(open(S / "lab_eval_cache.json"))
OUT = S / "lab_eci_trend.png"

MODELS = [
    ("deepinfra/deepseek-ai/DeepSeek-V3",   "DeepSeek-V3",      132.96),
    ("openai/gpt-4.1-2025-04-14",           "GPT-4.1",          137.43),
    ("openai/gpt-5-nano-2025-08-07",        "GPT-5 Nano",       139.88),
    ("anthropic/claude-haiku-4-5-20251001", "Haiku 4.5",        140.66),
    ("openai/o3-mini-2025-01-31",           "o3-mini",          141.11),
    ("openai/gpt-5-mini-2025-08-07",        "GPT-5 Mini",       144.32),
    ("openai/o4-mini-2025-04-16",           "o4-mini",          145.06),
    ("anthropic/claude-sonnet-4-5-20250929","Sonnet 4.5",       146.61),
    ("openai/o3-2025-04-16",                "o3",               146.62),
    ("xai/grok-4-0709",                     "Grok 4",           147.07),
    ("openai/gpt-5.1-2025-11-13",           "GPT-5.1",          149.74),
    ("openai/gpt-5-2025-08-07",             "GPT-5",            150.00),
]
skill_qs = CORPUS["skill"]

pts = []
for mid, disp, eci in MODELS:
    preds, tgt = [], []
    for q in skill_qs:
        p = CACHE.get(f"{mid}|skill|{q['idx']}")
        if p is not None:
            preds.append(p); tgt.append(q["pi"])
    if len(preds) < 100:
        continue
    pts.append((eci, float(np.corrcoef(preds, tgt)[0, 1]), disp))

eci = np.array([p[0] for p in pts])
disc = np.array([p[1] for p in pts])
r = stats.pearsonr(eci, disc)
rho = stats.spearmanr(eci, disc)
b1, b0 = np.polyfit(eci, disc, 1)

BLUE, RED, INK, MUTE, GRID = "#2a78d6", "#e34948", "#20242b", "#5b6069", "#dcdad2"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.edgecolor": GRID})
fig, ax = plt.subplots(figsize=(10, 6.6))

xs = np.array([eci.min() - 1, eci.max() + 1])
ax.plot(xs, b0 + b1 * xs, color=BLUE, lw=2, zorder=2, alpha=0.9)
ax.scatter(eci, disc, s=90, color=BLUE, edgecolor="white", lw=1.5, zorder=4)

# label points, nudging to avoid overlap
for x, y, name in pts:
    dy = 10
    if name in ("o3", "Sonnet 4.5"):     # near-duplicate ECI ~146.6
        dy = -16 if name == "o3" else 12
    ax.annotate(name, (x, y), xytext=(6, dy), textcoords="offset points",
                fontsize=9.5, color=INK)

# highlight the two near-zero anomalies and the top
for x, y, name in pts:
    if y < 0.1:
        ax.scatter([x], [y], s=90, facecolor="none", edgecolor=RED, lw=2, zorder=5)

ax.axhline(0, color=MUTE, lw=1, ls=":")
ax.set_xlabel("Epoch Capabilities Index (ECI)")
ax.set_ylabel("Discrimination on certified skill class  —  corr(forecast, ground-truth π)")
ax.set_ylim(-0.08, 0.85)
ax.set_title("Q1: forecasting skill on the certified instrument rises with capability",
             fontsize=13, loc="left", pad=12, color=INK)
ax.grid(color=GRID, lw=0.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

box = (f"Q1  TREND\n"
       f"  Pearson r = {r.statistic:+.2f}  (p={r.pvalue:.3f})\n"
       f"  Spearman ρ = {rho.statistic:+.2f}  (p={rho.pvalue:.2f})\n"
       f"  n = {len(pts)}   (Gemini 2.5 Pro dropped: n=44)\n\n"
       f"Q2  TRANSFER TO FORECASTBENCH | ECI\n"
       f"  partial r = +0.08 (FB overall, p=.81)\n"
       f"            = +0.23 (FB dataset, p=.48)\n"
       f"  vs −0.57 on FreeCiv's hedging templates\n"
       f"  → artifact removed; no proxy, but no anti-signal")
ax.annotate(box, (0.015, 0.985), xycoords="axes fraction", va="top", ha="left",
            fontsize=9.2, color=INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f7f6f2", ec=GRID, lw=1))
ax.annotate("red rings: near-zero discrimination —\ncan't tell a growing outbreak from a fizzling one\n(o3-mini is the surprise: high ECI, reasoning model)",
            (0.985, 0.03), xycoords="axes fraction", va="bottom", ha="right",
            fontsize=8.5, color=RED)

fig.text(0.5, 0.005,
         "Every model is badly overconfident on the tail (no one beats climatology); "
         "'discrimination' isolates ranking skill, the only axis that shows signal.",
         ha="center", fontsize=8.5, color=MUTE)
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT, dpi=150, facecolor="white")
print(f"-> {OUT}")
