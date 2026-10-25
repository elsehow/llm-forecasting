#!/usr/bin/env python3
"""Chart the models-vs-certified-instrument run, live from the eval cache.
Re-runnable: reads whatever models are complete so far.

Two panels:
  L. DISCRIMINATION — corr(model P, ground-truth pi) on the SKILL class:
     can the model RANK which outbreaks take off? (the certified signal is
     recoverable at CV-R2=0.50, so >0 means the model accesses it)
  R. CALIBRATION — each model's mean forecast vs the true tail rate (0.086):
     bars to the right of the true-rate line = tail overconfidence.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad")
CORPUS = json.load(open(S / "lab_classes.json"))
CACHE = json.load(open(S / "lab_eval_cache.json"))
OUT = S / "lab_certified_instrument.png"

# (litellm id, display, eci) — same slate/order as lab_eval
MODELS = [
    ("deepinfra/deepseek-ai/DeepSeek-V3",   "DeepSeek-V3",      132.96),
    ("openai/gpt-4.1-2025-04-14",           "GPT-4.1",          137.43),
    ("openai/gpt-5-nano-2025-08-07",        "GPT-5 Nano",       139.88),
    ("anthropic/claude-haiku-4-5-20251001", "Claude Haiku 4.5", 140.66),
    ("openai/o3-mini-2025-01-31",           "o3-mini",          141.11),
    ("openai/gpt-5-mini-2025-08-07",        "GPT-5 Mini",       144.32),
    ("openai/o4-mini-2025-04-16",           "o4-mini",          145.06),
    ("google/gemini-2.5-pro",               "Gemini 2.5 Pro",   146.06),
    ("anthropic/claude-sonnet-4-5-20250929","Claude Son. 4.5",  146.61),
    ("openai/o3-2025-04-16",                "o3",               146.62),
    ("xai/grok-4-0709",                     "Grok 4",           147.07),
    ("openai/gpt-5.1-2025-11-13",           "GPT-5.1",          149.74),
    ("openai/gpt-5-2025-08-07",             "GPT-5",            150.00),
    ("anthropic/claude-opus-4-5-20251101",  "Claude Opus 4.5",  150.05),
]

skill_qs = CORPUS["skill"]
pi = np.array([q["pi"] for q in skill_qs])
TRUE_RATE = float(pi.mean())

rows = []
for mid, disp, eci in MODELS:
    preds, tgt = [], []
    for q in skill_qs:
        p = CACHE.get(f"{mid}|skill|{q['idx']}")
        if p is not None:
            preds.append(p); tgt.append(q["pi"])
    if len(preds) < 100:          # only fully(ish)-complete models
        continue
    preds, tgt = np.array(preds), np.array(tgt)
    rows.append(dict(disp=disp, eci=eci, n=len(preds),
                     corr=float(np.corrcoef(preds, tgt)[0, 1]),
                     meanp=float(preds.mean())))

if not rows:
    print("no complete models yet"); raise SystemExit

# palette (validated earlier this session)
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#b7b3a9"
INK, MUTE, GRID = "#20242b", "#5b6069", "#dcdad2"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.edgecolor": GRID, "axes.linewidth": 1.0,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 0.55 * len(rows) + 2.4))

# ---- L: discrimination, sorted ascending so best is on top ----
rL = sorted(rows, key=lambda r: r["corr"])
y = np.arange(len(rL))
corrs = [r["corr"] for r in rL]
cols = [BLUE if c > 0.08 else (RED if c < -0.08 else NEUTRAL) for c in corrs]
axL.barh(y, corrs, color=cols, height=0.66, zorder=3)
axL.axvline(0, color=MUTE, lw=1.2, zorder=2)
axL.set_yticks(y); axL.set_yticklabels([f"{r['disp']}  ·ECI {r['eci']:.0f}" for r in rL])
for yi, c in zip(y, corrs):
    axL.annotate(f"{c:+.2f}", (c, yi), xytext=(4 if c >= 0 else -4, 0),
                 textcoords="offset points", va="center",
                 ha="left" if c >= 0 else "right", fontsize=10, color=INK)
axL.set_xlim(min(-0.15, min(corrs) - 0.12), max(0.85, max(corrs) + 0.12))
axL.set_xlabel("corr(forecast, ground-truth π)  —  ability to RANK which outbreaks take off")
axL.set_title("Discrimination on the certified skill class", fontsize=12, loc="left", pad=10)
axL.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    axL.spines[s].set_visible(False)

# ---- R: calibration, same model order as L ----
meanps = [r["meanp"] for r in rL]
axR.barh(y, meanps, color=NEUTRAL, height=0.66, zorder=3)
axR.axvline(TRUE_RATE, color=RED, lw=2.0, zorder=4)
axR.annotate(f"true tail rate\nπ̄ = {TRUE_RATE:.3f}", (TRUE_RATE, len(rL) - 0.4),
             xytext=(8, 0), textcoords="offset points", va="top", fontsize=9.5,
             color=RED)
axR.set_yticks(y); axR.set_yticklabels([])
for yi, mp in zip(y, meanps):
    axR.annotate(f"{mp:.2f}  ({mp/TRUE_RATE:.0f}×)", (mp, yi), xytext=(4, 0),
                 textcoords="offset points", va="center", ha="left",
                 fontsize=10, color=INK)
axR.set_xlim(0, min(1.0, max(meanps) + 0.18))
axR.set_xlabel("mean forecast  —  bars right of the red line = tail overconfidence")
axR.set_title("Calibration on the certified skill class", fontsize=12, loc="left", pad=10)
axR.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    axR.spines[s].set_visible(False)

done = len(rows)
fig.suptitle(f"Frontier models on a lab-certified forecasting instrument  "
             f"({done}/{len(MODELS)} models scored vs per-instance ground-truth π)",
             fontsize=13.5, y=0.995, x=0.5, ha="center", color=INK)
fig.text(0.5, 0.008,
         "Skill class n=110 · resolvable (CV-R² 0.50) · headroom 2× · "
         "signal is recoverable from the shown curve — so corr>0 = the model reads it, "
         "and mean forecast ≫ π̄ = the model can't calibrate the tail",
         ha="center", fontsize=9, color=MUTE)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(OUT, dpi=150, facecolor="white")
print(f"-> {OUT}  ({done} models)")
