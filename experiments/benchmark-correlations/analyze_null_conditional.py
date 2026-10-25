#!/usr/bin/env python3
"""Demand-effect analysis: per model, compare forecasts on the SAME question
under three framings: unconditional, null-intervention (vaccine announced for
day 70 — after resolution; mechanically irrelevant), real intervention.

delta_null = p_null - p_uncond  (rational forecaster: ~0)
delta_real = p_cond - p_uncond  (rational: should track the true effect)
"""
import json
import numpy as np
from scipy import stats

NULL = json.load(open("/Users/elsehow/Projects/forecastbench-sim/worlds/pandemic/null_conditional_results.json"))
CACHE = json.load(open("/Users/elsehow/Projects/forecastbench-sim/worlds/pandemic/.response_cache.json"))

MODELS = list(NULL.keys())
qids = [f"s{s}_T{t}" for s in range(30) for t in (40, 60)]

print(f"{'model':38s} {'meanU':>6s} {'meanN':>6s} {'meanC':>6s} {'dNull':>7s} {'p':>6s} {'dReal':>7s} {'|dN|/|dR|':>9s}")
rows = {}
for m in MODELS:
    np_map = {p["question_id"].rsplit("_", 1)[0]: p["p"] for p in NULL[m]["preds"]}
    du, dn, dr = [], [], []
    pu_l, pn_l, pc_l = [], [], []
    for q in qids:
        pu = CACHE.get(f"{m}|{q}_unconditional")
        pc = CACHE.get(f"{m}|{q}_conditional")
        pn = np_map.get(q)
        if None in (pu, pc, pn):
            continue
        pu_l.append(pu); pn_l.append(pn); pc_l.append(pc)
        dn.append(pn - pu); dr.append(pc - pu)
    dn, dr = np.array(dn), np.array(dr)
    t, p = stats.ttest_1samp(dn, 0)
    ratio = np.abs(dn).mean() / max(np.abs(dr).mean(), 1e-9)
    rows[m] = dict(dnull=dn.mean(), dreal=dr.mean(), ratio=ratio)
    print(f"{m.split('/')[-1]:38s} {np.mean(pu_l):6.2f} {np.mean(pn_l):6.2f} {np.mean(pc_l):6.2f} "
          f"{dn.mean():+7.3f} {p:6.3f} {dr.mean():+7.3f} {ratio:9.2f}")

# does null suggestibility explain the conditional over-adjustment?
# over-adjustment proxy: how far meanC undershoots the true conditional base rate (0.217)
over = np.array([abs(np.mean([p['p'] for p in NULL[m]['preds']]) )for m in MODELS])  # placeholder unused
dnulls = np.array([rows[m]["dnull"] for m in MODELS])
# real over-adjustment measured earlier: meanP_cond - true rate; recompute
overshoot = []
for m in MODELS:
    pcs = [CACHE.get(f"{m}|{q}_conditional") for q in qids]
    pcs = [x for x in pcs if x is not None]
    overshoot.append(0.217 - np.mean(pcs))  # positive = undershoots true rate (over-adjusts down)
r = stats.pearsonr(dnulls, overshoot)
print(f"\ncorr(null delta, conditional overshoot): r={r.statistic:+.2f} (p={r.pvalue:.3f})")
print("(negative dNull = drops P when told about an irrelevant intervention = demand effect)")
