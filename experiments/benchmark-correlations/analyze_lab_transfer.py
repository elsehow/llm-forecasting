#!/usr/bin/env python3
"""Jason's extrapolation test, re-asked on the CERTIFIED lab instrument.

Reads the eval cache (robust to partial runs), scores each model's
discrimination on the skill class = corr(forecast, ground-truth pi), then:

  Q1 (trend):   corr(ECI, lab discrimination)
  Q2 (transfer): partial corr(lab discrimination, ForecastBench | ECI)
                 -- does sim skill predict FB *beyond* what ECI already says?
                 This is the exact test that came back -0.57 on FreeCiv's
                 flawed hedging templates; here on a certified instrument.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

S = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad")
CORPUS = json.load(open(S / "lab_classes.json"))
CACHE = json.load(open(S / "lab_eval_cache.json"))

# (litellm id, display, eci, fb_overall, fb_dataset) — FB/ECI from the
# multi_benchmark_correlation.py DATA table.
MODELS = [
    ("deepinfra/deepseek-ai/DeepSeek-V3",   "DeepSeek-V3",      132.96, 61.9, 59.1),
    ("openai/gpt-4.1-2025-04-14",           "GPT-4.1",          137.43, 63.3, 60.1),
    ("openai/gpt-5-nano-2025-08-07",        "GPT-5 Nano",       139.88, 60.4, 57.6),
    ("anthropic/claude-haiku-4-5-20251001", "Claude Haiku 4.5", 140.66, 62.8, 59.4),
    ("openai/o3-mini-2025-01-31",           "o3-mini",          141.11, 62.3, 58.9),
    ("openai/gpt-5-mini-2025-08-07",        "GPT-5 Mini",       144.32, 62.9, 60.8),
    ("openai/o4-mini-2025-04-16",           "o4-mini",          145.06, 63.8, 60.7),
    ("google/gemini-2.5-pro",               "Gemini 2.5 Pro",   146.06, 63.8, 61.0),
    ("anthropic/claude-sonnet-4-5-20250929","Claude Son. 4.5",  146.61, 64.3, 60.7),
    ("openai/o3-2025-04-16",                "o3",               146.62, 64.8, 60.4),
    ("xai/grok-4-0709",                     "Grok 4",           147.07, 63.3, 60.4),
    ("openai/gpt-5.1-2025-11-13",           "GPT-5.1",          149.74, 59.0, 58.7),
    ("openai/gpt-5-2025-08-07",             "GPT-5",            150.00, 62.5, 61.8),
    ("anthropic/claude-opus-4-5-20251101",  "Claude Opus 4.5",  150.05, 61.4, 60.5),
]

skill_qs = CORPUS["skill"]


def partial_corr(x, y, z):
    """corr(x, y | z) via residuals of linear regression on z."""
    x, y, z = map(np.asarray, (x, y, z))
    Z = np.c_[np.ones_like(z), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = stats.pearsonr(rx, ry)
    return r.statistic, r.pvalue


rows = []
for mid, disp, eci, fbo, fbd in MODELS:
    preds, tgt = [], []
    for q in skill_qs:
        p = CACHE.get(f"{mid}|skill|{q['idx']}")
        if p is not None:
            preds.append(p); tgt.append(q["pi"])
    if len(preds) < 100:
        continue
    preds, tgt = np.array(preds), np.array(tgt)
    rows.append(dict(disp=disp, eci=eci, fbo=fbo, fbd=fbd, n=len(preds),
                     disc=float(np.corrcoef(preds, tgt)[0, 1]),
                     meanp=float(preds.mean())))

print(f"\n{len(rows)}/{len(MODELS)} models scored\n")
print(f"{'model':18s} {'ECI':>6s} {'disc':>6s} {'FB_ov':>6s} {'FB_ds':>6s} {'meanP':>6s}")
for r in sorted(rows, key=lambda r: r["eci"]):
    print(f"{r['disp']:18s} {r['eci']:6.2f} {r['disc']:+6.2f} {r['fbo']:6.1f} {r['fbd']:6.1f} {r['meanp']:6.2f}")

if len(rows) < 5:
    print("\n(need >=5 models for correlations; waiting on the run)")
    raise SystemExit

eci = np.array([r["eci"] for r in rows])
disc = np.array([r["disc"] for r in rows])
fbo = np.array([r["fbo"] for r in rows])
fbd = np.array([r["fbd"] for r in rows])

print("\n" + "=" * 64)
print("Q1  TREND — does lab discrimination rise with ECI?")
print("=" * 64)
r = stats.pearsonr(eci, disc)
rs = stats.spearmanr(eci, disc)
print(f"  corr(ECI, lab discrimination)  Pearson r={r.statistic:+.2f} (p={r.pvalue:.3f})")
print(f"                                 Spearman ρ={rs.statistic:+.2f} (p={rs.pvalue:.3f})  n={len(rows)}")

print("\n" + "=" * 64)
print("Q2  TRANSFER — does lab skill predict ForecastBench, controlling ECI?")
print("     (Jason's test; came back -0.57 on FreeCiv hedging templates)")
print("=" * 64)
for name, fb in (("FB overall", fbo), ("FB dataset-only", fbd)):
    r0 = stats.pearsonr(disc, fb)
    rp, pp = partial_corr(disc, fb, eci)
    r_fb_eci = stats.pearsonr(fb, eci).statistic
    print(f"  [{name}]")
    print(f"    raw   corr(lab disc, FB)          r={r0.statistic:+.2f} (p={r0.pvalue:.3f})")
    print(f"    corr(FB, ECI)                     r={r_fb_eci:+.2f}")
    print(f"    PARTIAL corr(lab disc, FB | ECI)  r={rp:+.2f} (p={pp:.3f})")
    print()
