#!/usr/bin/env python3
"""Abaluck test on the Starsim pandemic world: does Starsim skill predict
ForecastBench conditional on ECI? Plus the three template gates
(reliability, beats-trivial, style-independence), per question kind."""
import json
import numpy as np
from scipy import stats

from pandemic_world.scenarios import sample_scenarios, build_corpus

CACHE = "/Users/elsehow/Projects/forecastbench-sim/worlds/pandemic/.response_cache.json"
# name -> (litellm id, fb_dataset, eci)
MODELS = {
    "GPT-4.1":        ("openai/gpt-4.1-2025-04-14", 60.1, 137.43),
    "GPT-5":          ("openai/gpt-5-2025-08-07", 61.8, 150.00),
    "GPT-5 Mini":     ("openai/gpt-5-mini-2025-08-07", 60.8, 144.32),
    "GPT-5 Nano":     ("openai/gpt-5-nano-2025-08-07", 57.6, 139.88),
    "GPT-5.1":        ("openai/gpt-5.1-2025-11-13", 58.7, 149.74),
    "o3":             ("openai/o3-2025-04-16", 60.4, 146.62),
    "o3-mini":        ("openai/o3-mini-2025-01-31", 58.9, 141.11),
    "o4-mini":        ("openai/o4-mini-2025-04-16", 60.7, 145.06),
    "Gemini 2.5 Pro": ("google/gemini-2.5-pro", 61.0, 146.06),
    "DeepSeek V3":    ("deepinfra/deepseek-ai/DeepSeek-V3", 59.1, 132.96),
    "Grok 4":         ("xai/grok-4-0709", 60.4, 147.07),
}

corpus = build_corpus(sample_scenarios(30, 11))
cache = json.load(open(CACHE))
names = list(MODELS)
fb = np.array([MODELS[n][1] for n in names])
eci = np.array([MODELS[n][2] for n in names])

def preds_for(kind):
    """questions of a kind -> list of (gt, {model: p})"""
    out = []
    for c in corpus:
        if kind != "all" and c["kind"] != kind:
            continue
        ps = {}
        for n in names:
            p = cache.get(f"{MODELS[n][0]}|{c['question_id']}")
            if p is not None:
                ps[n] = p
        out.append((float(c["ground_truth"]), ps))
    return out

def skill_dec(qs):
    sk, dec = [], []
    for n in names:
        bs = [(ps[n] - gt) ** 2 for gt, ps in qs if n in ps]
        ds = [abs(ps[n] - 0.5) for gt, ps in qs if n in ps]
        sk.append(-np.mean(bs)); dec.append(np.mean(ds))
    return np.array(sk), np.array(dec)

def partial(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)

def loo(Xcols, y):
    n = len(y); preds = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        X1 = np.column_stack([np.ones(m.sum())] + [x[m] for x in Xcols])
        beta, *_ = np.linalg.lstsq(X1, y[m], rcond=None)
        preds[i] = np.concatenate([[1], [x[i] for x in Xcols]]) @ beta
    return 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print(f"n models = {len(names)} (no Claude x3, no Gemini 3 Pro)")
for kind in ("unconditional", "conditional", "all"):
    qs = preds_for(kind)
    sk, dec = skill_dec(qs)
    a, _ = skill_dec(qs[0::2]); b, _ = skill_dec(qs[1::2])
    rh = stats.spearmanr(a, b).statistic
    rel = 2 * rh / (1 + rh) if rh > -1 else np.nan
    br = np.mean([gt for gt, _ in qs])
    trivial = br * (1 - br)  # flat base-rate forecast
    pr, pp = partial(sk, fb, eci)
    biv = stats.pearsonr(sk, fb)
    r_eci = stats.pearsonr(sk, eci).statistic
    ds_r = stats.pearsonr(dec, sk).statistic
    print(f"\n=== {kind} (nq={len(qs)}, base rate={br:.2f}) ===")
    print(f"  gates: reliability={rel:.2f}  best-model Brier={-max(sk):.3f} vs trivial {trivial:.3f} vs 0.5-flat 0.25  dec->skill={ds_r:+.2f}")
    print(f"  corr with ECI: {r_eci:+.2f}   bivariate with FB-ds: {biv.statistic:+.2f} (p={biv.pvalue:.3f})")
    print(f"  PARTIAL with FB-ds | ECI: {pr:+.2f} (p={pp:.3f})")
    print(f"  LOO CV-R2: ECI only {loo([eci], fb):+.3f} | ECI+starsim {loo([eci, sk], fb):+.3f} | starsim only {loo([sk], fb):+.3f}")

# ranking table for eyeballing
qs = preds_for("all")
sk, dec = skill_dec(qs)
print("\nmodel                skill(-Brier)  dec    FBds   ECI")
for i in np.argsort(-sk):
    print(f"{names[i]:20s} {-sk[i]:.4f}      {dec[i]:.2f}  {fb[i]:5.1f}  {eci[i]:6.2f}")
