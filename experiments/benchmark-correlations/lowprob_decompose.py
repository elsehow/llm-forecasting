#!/usr/bin/env python3
"""Decompose the lowprob G2 failure: wrong prior vs no resolution.

Per class x model:
  - discrimination: AUC of predictions vs outcomes (prior-free)
  - prior gap: mean predicted P minus class base rate
  - recalibrated Brier: apply the single best monotone shift you could get by
    knowing the base rate (multiply odds by a constant so mean pred = cbr),
    i.e. what a model with this resolution WOULD score if handed the prior.
"""
import json
import numpy as np
from scipy import stats

RES = "/Users/elsehow/Projects/forecastbench-sim/data/results/lowprob_gates_eval.json"
TREE = "/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/lowprob-tree/questions_all.json"
MODELS = [
    "openai/gpt-4.1-2025-04-14", "openai/gpt-5-2025-08-07", "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5-nano-2025-08-07", "openai/gpt-5.1-2025-11-13", "openai/o3-2025-04-16",
    "openai/o3-mini-2025-01-31", "openai/o4-mini-2025-04-16",
    "anthropic/claude-sonnet-4-5-20250929", "anthropic/claude-haiku-4-5-20251001",
    "google/gemini-2.5-pro", "deepinfra/deepseek-ai/DeepSeek-V3", "xai/grok-4-0709",
]
d = json.load(open(RES))
tree = {q["question_id"]: q for q in json.load(open(TREE))["questions"]}

def pred(q, m):
    v = q["predictions"].get(m)
    return v.get("probability") if isinstance(v, dict) else v

def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]; neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return np.nan
    wins = sum((p > n_) + 0.5 * (p == n_) for p in pos for n_ in neg)
    return wins / (len(pos) * len(neg))

def shift_to_prior(ps, target):
    """Multiply odds by constant c so mean(shifted) == target."""
    ps = np.clip(np.array(ps), 1e-4, 1 - 1e-4)
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        z = 1 / (1 + np.exp(-(np.log(ps / (1 - ps)) + mid)))
        if z.mean() > target:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(np.log(ps / (1 - ps)) + (lo + hi) / 2)))

classes = {}
for q in d["questions"]:
    t = tree[q["question_id"]]
    k = (q["template_id"], t["parameters"]["target"], t["horizon"])
    classes.setdefault(k, []).append(q)

# pooled across all lowprob questions, per model
print(f"{'model':38s} {'AUC':>5s} {'meanP':>6s} {'cbr':>5s} {'brier':>6s} {'recal':>6s} {'triv':>6s}")
pooled_auc = {}
for m in MODELS:
    ps, ys, cbs = [], [], []
    for k, sub in classes.items():
        for q in sub:
            p = pred(q, m)
            if p is not None:
                ps.append(p); ys.append(float(q["ground_truth"]))
                cbs.append(tree[q["question_id"]]["parameters"]["class_base_rate"])
    ps, ys = np.array(ps), np.array(ys)
    a = auc(ps, ys)
    pooled_auc[m] = a
    brier = np.mean((ps - ys) ** 2)
    # per-class shift (uses each class's own base rate — the oracle prior)
    recal_sq = []
    for k, sub in classes.items():
        cps = [pred(q, m) for q in sub if pred(q, m) is not None]
        cys = [float(q["ground_truth"]) for q in sub if pred(q, m) is not None]
        cbr = tree[sub[0]["question_id"]]["parameters"]["class_base_rate"]
        if len(cps) >= 10:
            z = shift_to_prior(cps, cbr)
            recal_sq.extend((z - np.array(cys)) ** 2)
    recal = np.mean(recal_sq)
    triv = np.mean([(c - y) ** 2 for c, y in zip(cbs, ys)])
    print(f"{m.split('/')[-1]:38s} {a:5.2f} {ps.mean():6.2f} {np.mean(cbs):5.3f} {brier:6.3f} {recal:6.3f} {triv:6.3f}")

# does discrimination (AUC), a prior-free skill measure, transfer to FB | ECI?
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
av = np.array([pooled_auc[m] for m in MODELS])
fb = np.array([FB_ECI[m][0] for m in MODELS])
eci = np.array([FB_ECI[m][1] for m in MODELS])
rx = av - np.poly1d(np.polyfit(eci, av, 1))(eci)
ry = fb - np.poly1d(np.polyfit(eci, fb, 1))(eci)
pr = stats.pearsonr(rx, ry)
print(f"\nAUC->FB: biv r={stats.pearsonr(av,fb).statistic:+.2f} (p={stats.pearsonr(av,fb).pvalue:.3f})  "
      f"partial|ECI={pr.statistic:+.2f} (p={pr.pvalue:.3f})  r_ECI={stats.pearsonr(av,eci).statistic:+.2f}")
# split-half reliability of the AUC ranking
def auc_vec(subqs):
    out = []
    for m in MODELS:
        ps = [pred(q, m) for q in subqs if pred(q, m) is not None]
        ys = [float(q["ground_truth"]) for q in subqs if pred(q, m) is not None]
        out.append(auc(ps, ys))
    return np.array(out)
allq = d["questions"]
a1, a2 = auc_vec(allq[0::2]), auc_vec(allq[1::2])
rh = stats.spearmanr(a1, a2).statistic
print(f"AUC ranking split-half reliability (SB-corrected): {2*rh/(1+rh):.2f}")
