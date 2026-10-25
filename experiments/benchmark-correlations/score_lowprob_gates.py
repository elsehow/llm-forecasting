#!/usr/bin/env python3
"""Score the PRE-REGISTERED low-prob gates (fixed before data collection):
  G1 reliability >= 0.5 (split-half Spearman-Brown of model ranking)
  G2 best-model Brier <= 0.9 x flat-class-base-rate Brier on same instances
  G3 |corr(decisiveness, skill)| <= 0.35
Then: transfer of the surviving composite to FB-dataset conditional on ECI."""
import json
import numpy as np
from scipy import stats

RES = "/Users/elsehow/Projects/forecastbench-sim/data/results/lowprob_gates_eval.json"
FB_ECI = {
    "openai/gpt-4.1-2025-04-14": (60.1, 137.43), "openai/gpt-5-2025-08-07": (61.8, 150.00),
    "openai/gpt-5-mini-2025-08-07": (60.8, 144.32), "openai/gpt-5-nano-2025-08-07": (57.6, 139.88),
    "openai/gpt-5.1-2025-11-13": (58.7, 149.74), "openai/o3-2025-04-16": (60.4, 146.62),
    "openai/o3-mini-2025-01-31": (58.9, 141.11), "openai/o4-mini-2025-04-16": (60.7, 145.06),
    "anthropic/claude-sonnet-4-5-20250929": (60.7, 146.61),
    "anthropic/claude-haiku-4-5-20251001": (59.4, 140.66),
    "google/gemini-2.5-pro": (61.0, 146.06), "deepinfra/deepseek-ai/DeepSeek-V3": (59.1, 132.96),
    "xai/grok-4-0709": (60.4, 147.07),
}
d = json.load(open(RES))
qs = d["questions"]
models = [m for m in FB_ECI if any(m in q["predictions"] for q in qs[:50])]
fb = np.array([FB_ECI[m][0] for m in models])
eci = np.array([FB_ECI[m][1] for m in models])

TREE_QS = {q["question_id"]: q for q in json.load(open(
    "/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/lowprob-tree/questions_all.json"))["questions"]}

def key(q):
    t = TREE_QS[q["question_id"]]
    p = t["parameters"]
    return (q["template_id"], p.get("target", ""), t.get("horizon", "?"))

def pred(q, m):
    v = q["predictions"].get(m)
    if isinstance(v, dict):
        v = v.get("probability")
    return v

classes = {}
for q in qs:
    classes.setdefault(key(q), []).append(q)

def skill_dec(sub):
    sk, dec = [], []
    for m in models:
        bs = [(pred(q, m) - float(q["ground_truth"])) ** 2 for q in sub if pred(q, m) is not None]
        ds = [abs(pred(q, m) - 0.5) for q in sub if pred(q, m) is not None]
        sk.append(-np.mean(bs) if bs else np.nan)
        dec.append(np.mean(ds) if ds else np.nan)
    return np.array(sk), np.array(dec)

def partial(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)

rows, passing = [], []
for k, sub in sorted(classes.items(), key=lambda kv: -len(kv[1])):
    if len(sub) < 20:
        continue
    br = np.mean([float(q["ground_truth"]) for q in sub])
    cbr = TREE_QS[sub[0]["question_id"]]["parameters"]["class_base_rate"]
    sk, dec = skill_dec(sub)
    a, _ = skill_dec(sub[0::2]); b, _ = skill_dec(sub[1::2])
    rh = stats.spearmanr(a, b).statistic
    rel = 2 * rh / (1 + rh) if rh > -1 else np.nan
    trivial = np.mean([(cbr - float(q["ground_truth"])) ** 2 for q in sub])
    best = -max(sk)
    ds_r = stats.pearsonr(dec, sk).statistic
    g1, g2, g3 = rel >= 0.5, best <= 0.9 * trivial, abs(ds_r) <= 0.35
    ok = g1 and g2 and g3
    if ok:
        passing.append((k, sk))
    rows.append((k, len(sub), cbr, rel, best, trivial, ds_r, g1, g2, g3, ok))

print(f"{'class':58s} {'n':>4s} {'cbr':>5s} {'rel':>6s} {'best':>6s} {'triv':>6s} {'d->s':>6s} gates")
for k, n, cbr, rel, best, triv, ds_r, g1, g2, g3, ok in rows:
    name = f"{k[0]}|{k[1].split('=')[-1][:24]}|{k[2]}"
    gate = "".join("✓" if g else "×" for g in (g1, g2, g3)) + ("  PASS" if ok else "")
    print(f"{name:58s} {n:4d} {cbr:5.3f} {rel:6.2f} {best:6.3f} {triv:6.3f} {ds_r:+6.2f} {gate}")

print(f"\n{sum(r[-1] for r in rows)}/{len(rows)} classes pass all gates")
if passing:
    comp = np.nanmean([(sk - np.nanmean(sk)) / np.nanstd(sk) for _, sk in passing], axis=0)
    pr, pp = partial(comp, fb, eci)
    biv = stats.pearsonr(comp, fb)
    print(f"\nsurviving composite ({len(passing)} classes): biv r={biv.statistic:+.2f} (p={biv.pvalue:.3f})  "
          f"partial|ECI={pr:+.2f} (p={pp:.3f})  r_ECI={stats.pearsonr(comp,eci).statistic:+.2f}")
    # also: full (ungated) lowprob aggregate for contrast
    sk_all, _ = skill_dec(qs)
    pr2, pp2 = partial(sk_all, fb, eci)
    print(f"ungated aggregate (all {len(qs)} q): partial|ECI={pr2:+.2f} (p={pp2:.3f})  "
          f"biv={stats.pearsonr(sk_all,fb).statistic:+.2f}")
