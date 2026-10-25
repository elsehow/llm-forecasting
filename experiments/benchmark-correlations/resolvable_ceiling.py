#!/usr/bin/env python3
"""Gate 0 / 'resolvable' ceiling test: for each mined low-prob class, can a
simple cross-game statistical baseline predict outcomes from turn-40 snapshot
features better than chance (out-of-sample AUC > 0.5)?

If not, climatology is Bayes-optimal for that class and NO forecaster could
beat it — the class is a calibration probe, not a skill probe.

Features per (game, civ) at snapshot turn 40: the 13 numeric series values at
turn 40 + their turn-20->40 deltas + current government one-hot (from events)
+ per-target flags (has the target tech/wonder already been discovered by
anyone; is the civ already in the target government).
"""
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Users/elsehow/Projects/forecastbench-sim")
SNAP = 40
SERIES = ["scores", "treasury", "population", "techs_known", "wonders_count",
          "cities_count", "territory_size", "science", "trade_production",
          "food_production", "shield_production", "units_count", "military_units_count"]

qd = json.loads((ROOT / "data/lowprob/lowprob_questions.json").read_text())
qs = qd["questions"]
classes = defaultdict(list)
for q in qs:
    classes[(q["template_id"], q.get("target", ""), q["horizon"])].append(q)

# who is the question about? parse civ name from question text, map via game civs
def civ_id_for(q, civs):
    for cid, c in civs.items():
        name = c.get("name", "")
        if name and name in q["question_text"]:
            return cid
    return None

game_cache = {}
def game(gid):
    if gid not in game_cache:
        game_cache[gid] = json.loads((ROOT / f"data/games/{gid}_data.json").read_text())
    return game_cache[gid]

def features(q):
    gd = game(q["game_id"])
    civs = gd["civilizations"]
    cid = civ_id_for(q, civs)
    ts = gd["time_series"]
    f = []
    def val(key, t, c):
        try:
            return float(ts[key][str(t)][str(c)])
        except (KeyError, TypeError, ValueError):
            return np.nan
    if cid is not None:
        for k in SERIES:
            v40, v20 = val(k, 40, cid), val(k, 20, cid)
            f += [v40, (v40 - v20) if not (np.isnan(v40) or np.isnan(v20)) else np.nan]
    else:
        f += [np.nan] * (2 * len(SERIES))
    # world aggregates
    for k in ("techs_known", "wonders_count", "scores"):
        vals = [val(k, 40, c) for c in civs]
        vals = [v for v in vals if not np.isnan(v)]
        f += [max(vals) if vals else np.nan, float(np.mean(vals)) if vals else np.nan]
    # events up to snapshot: target-specific flags
    target = q.get("target", "")
    tname = target.split("=")[-1] if target else ""
    n_target_events = 0
    cur_gov = None
    for e in gd.get("events", []):
        if e.get("turn", 1e9) > SNAP:
            continue
        if tname and tname.lower() in str(e.get("description", "")).lower():
            n_target_events += 1
        if e.get("type") == "government_change" and cid is not None and str(e.get("player_id")) == str(cid):
            cur_gov = e.get("metadata", {}).get("government", e.get("description", ""))
    f.append(float(n_target_events))
    f.append(1.0 if (tname and cur_gov and tname.lower() in str(cur_gov).lower()) else 0.0)
    return np.array(f, dtype=float)

def cv_auc(X, y, folds=5, seed=7):
    """L2 logistic via simple gradient descent, k-fold CV AUC."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    # impute + standardize
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    X = np.where(np.isnan(X), col_med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    scores = np.zeros(len(y))
    fold = np.arange(len(y)) % folds
    for f in range(folds):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 2 or y[tr].sum() > y[tr].size - 2:
            scores[te] = y[tr].mean()
            continue
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(300):
            z = X[tr] @ w + b
            p = 1 / (1 + np.exp(-z))
            g = X[tr].T @ (p - y[tr]) / tr.sum() + 0.05 * w
            gb = (p - y[tr]).mean()
            w -= 0.5 * g; b -= 0.5 * gb
        scores[te] = 1 / (1 + np.exp(-(X[te] @ w + b)))
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return np.nan, len(y)
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)), len(y)

print(f"{'class':55s} {'n':>6s} {'pos':>5s} {'cvAUC':>6s}  verdict")
results = {}
for k, sub in sorted(classes.items(), key=lambda kv: -len(kv[1])):
    y = np.array([float(q["ground_truth"]) for q in sub])
    if y.sum() < 8:
        continue
    X = np.vstack([features(q) for q in sub])
    a, n = cv_auc(X, y)
    # permutation p-value (rank vs 200 label shuffles)
    rngp = np.random.default_rng(3)
    null = []
    for _ in range(60):
        yp = rngp.permutation(y)
        an, _ = cv_auc(X, yp)
        null.append(an)
    p = float(np.mean([x >= a for x in null]))
    verdict = "RESOLVABLE" if (a > 0.58 and p < 0.1) else "calibration-only"
    results["|".join(map(str, k))] = {"cv_auc": float(a), "n": int(n), "p_perm": p, "verdict": verdict}
    name = f"{k[0]}|{k[1].split('=')[-1][:22]}|{k[2]}"
    print(f"{name:55s} {n:6d} {int(y.sum()):5d} {a:6.2f}  {verdict} (p={p:.2f})")

out = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/resolvable_ceiling_results.json")
out.write_text(json.dumps(results, indent=1))
print(f"-> {out}")
n_res = sum(1 for r in results.values() if r["verdict"] == "RESOLVABLE")
print(f"{n_res}/{len(results)} classes resolvable from snapshot features")
