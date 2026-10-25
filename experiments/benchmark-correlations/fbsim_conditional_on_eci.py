#!/usr/bin/env python3
"""Jason Abaluck's question (2026-07-08 email): does simulated-world forecasting
(FB-Sim) predict ForecastBench performance *conditional on ECI*?

Data:
  - Sim: non-H0 binary Brier, 30 models x 4655 questions
    (forecastbench-sim/data/results/uncond_nonh0_bin_cont_all30_20260307.json)
  - FB + ECI: curated table in llm-forecasting/experiments/benchmark-correlations/
    multi_benchmark_correlation.py (FB = baseline leaderboard, scratchpad; no news)
  - FB 95% CIs: leaderboard_baseline.csv -> noise ceiling
"""
import json, csv, itertools
import numpy as np
from scipy import stats

SIM = "/Users/elsehow/Projects/forecastbench-sim/data/results/uncond_nonh0_bin_cont_all30_20260307.json"
BC = "/Users/elsehow/Projects/llm-forecasting/experiments/benchmark-correlations"

# name -> (fb_overall, fb_dataset, eci)  [from multi_benchmark_correlation.py DATA]
FB_ECI = {
    "GPT-4o":            (59.7, 56.0, None),
    "GPT-4.1":           (63.3, 60.1, 137.43),
    "GPT-5":             (62.5, 61.8, 150.00),
    "GPT-5 Mini":        (62.9, 60.8, 144.32),
    "GPT-5 Nano":        (60.4, 57.6, 139.88),
    "GPT-5.1":           (59.0, 58.7, 149.74),
    "o3":                (64.8, 60.4, 146.62),
    "o3-mini":           (62.3, 58.9, 141.11),
    "o4-mini":           (63.8, 60.7, 145.06),
    "Claude Son. 4":     (61.7, 59.0, 142.33),
    "Claude Son. 4.5":   (64.3, 60.7, 146.61),
    "Claude Haiku 4.5":  (62.8, 59.4, 140.66),
    "Gemini 3 Pro":      (63.3, 61.3, 154.22),
    "Gemini 2.5 Pro":    (63.8, 61.0, 146.06),
    "Gemini 2.5 Flash":  (63.3, 60.5, None),
    "DeepSeek V3":       (61.9, 59.1, 132.96),
    "Grok 4":            (63.3, 60.4, 147.07),
}
SIM_ID = {
    "GPT-4o": "openai/gpt-4o",
    "GPT-4.1": "openai/gpt-4.1-2025-04-14",
    "GPT-5": "openai/gpt-5-2025-08-07",
    "GPT-5 Mini": "openai/gpt-5-mini-2025-08-07",
    "GPT-5 Nano": "openai/gpt-5-nano-2025-08-07",
    "GPT-5.1": "openai/gpt-5.1-2025-11-13",
    "o3": "openai/o3-2025-04-16",
    "o3-mini": "openai/o3-mini-2025-01-31",
    "o4-mini": "openai/o4-mini-2025-04-16",
    "Claude Son. 4": "anthropic/claude-sonnet-4-20250514",
    "Claude Son. 4.5": "anthropic/claude-sonnet-4-5-20250929",
    "Claude Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
    "Gemini 3 Pro": "google/gemini-3-pro-preview",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "DeepSeek V3": "together_ai/deepseek-ai/DeepSeek-V3",
    "Grok 4": "xai/grok-4-0709",
}
# leaderboard row name fragments for CI lookup (scratchpad rows, matching DATA)
LB_FRAG = {
    "GPT-4o": "GPT-4o (scratchpad",
    "GPT-4.1": "GPT-4.1-2025-04-14 (scratchpad",
    "GPT-5": "GPT-5-2025-08-07 (scratchpad",
    "GPT-5 Mini": "GPT-5-Mini-2025-08-07",
    "GPT-5 Nano": "GPT-5-Nano-2025-08-07",
    "GPT-5.1": "GPT-5.1-2025-11-13",
    "o3": "O3-2025-04-16 (scratchpad",
    "o3-mini": "O3-Mini-2025-01-31 (scratchpad",
    "o4-mini": "O4-Mini-2025-04-16 (scratchpad",
    "Claude Son. 4": "Claude-Sonnet-4-20250514",
    "Claude Son. 4.5": "Claude-Sonnet-4-5",
    "Claude Haiku 4.5": "Claude-Haiku-4-5",
    "Gemini 3 Pro": "Gemini-3-Pro",
    "Gemini 2.5 Pro": "Gemini-2.5-Pro",
    "Gemini 2.5 Flash": "Gemini-2.5-Flash",
    "DeepSeek V3": "DeepSeek-V3 (scratchpad",
    "Grok 4": "Grok-4-0709",
}

sim = json.load(open(SIM))["model_results"]
sim_brier = {n: sim[SIM_ID[n]]["binary"]["brier_score"] for n in SIM_ID if SIM_ID[n] in sim}

# CI lookup from baseline leaderboard
lb = list(csv.DictReader(open(f"{BC}/data/leaderboard_baseline.csv")))
def ci_se(name, col_ci, col_val):
    frag = LB_FRAG[name]
    cands = [r for r in lb if frag.lower() in r["Model"].lower()]
    # prefer scratchpad row (DATA used scratchpad where both exist)
    sc = [r for r in cands if "scratchpad" in r["Model"]]
    row = (sc or cands)[0] if cands else None
    if row is None:
        return None, None
    lo, hi = json.loads(row[col_ci])
    return float(row[col_val]), (hi - lo) / (2 * 1.96)

rows = []
for n, (fb_ov, fb_ds, eci) in FB_ECI.items():
    if n not in sim_brier:
        continue
    v_ov, se_ov = ci_se(n, "Overall 95% CI", "Overall")
    v_ds, se_ds = ci_se(n, "Dataset 95% CI", "Dataset")
    rows.append(dict(name=n, sim=-sim_brier[n], fb_ov=fb_ov, fb_ds=fb_ds, eci=eci,
                     se_ov=se_ov, se_ds=se_ds, lb_ov=v_ov, lb_ds=v_ds))

print(f"{'model':18s} {'simBrier':>8s} {'FBov':>5s} {'FBds':>5s} {'ECI':>7s} {'SEov':>5s} {'SEds':>5s} lb_check")
for r in sorted(rows, key=lambda r: -r["sim"]):
    print(f"{r['name']:18s} {-r['sim']:8.4f} {r['fb_ov']:5.1f} {r['fb_ds']:5.1f} "
          f"{r['eci'] if r['eci'] else float('nan'):7.2f} "
          f"{r['se_ov'] if r['se_ov'] else float('nan'):5.2f} "
          f"{r['se_ds'] if r['se_ds'] else float('nan'):5.2f} "
          f"{r['lb_ov']}/{r['lb_ds']}")

def ols_r2(X, y):
    X1 = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())), beta

def loo_r2(X, y):
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        X1 = np.column_stack([np.ones(m.sum())] + [x[m] for x in X])
        beta, *_ = np.linalg.lstsq(X1, y[m], rcond=None)
        preds[i] = np.concatenate([[1], [x[i] for x in X]]) @ beta
    return 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def partial_corr(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)

for target, se_key in [("fb_ov", "se_ov"), ("fb_ds", "se_ds")]:
    sub = [r for r in rows if r["eci"] is not None]
    y = np.array([r[target] for r in sub])
    simv = np.array([r["sim"] for r in sub])
    eci = np.array([r["eci"] for r in sub])
    ses = np.array([r[se_key] for r in sub if r[se_key] is not None])
    n = len(sub)
    print(f"\n===== target {target}  (n={n}) =====")
    print(f"bivariate: sim-FB rho_s={stats.spearmanr(simv,y).statistic:.3f} r={stats.pearsonr(simv,y).statistic:.3f} (p={stats.pearsonr(simv,y).pvalue:.3f})")
    print(f"           ECI-FB rho_s={stats.spearmanr(eci,y).statistic:.3f} r={stats.pearsonr(eci,y).statistic:.3f} (p={stats.pearsonr(eci,y).pvalue:.3f})")
    print(f"           sim-ECI r={stats.pearsonr(simv,eci).statistic:.3f}")
    pr, pp = partial_corr(simv, y, eci)
    print(f"partial corr sim-FB | ECI: r={pr:.3f} (p={pp:.3f})")
    r2_e, _ = ols_r2([eci], y)
    r2_es, _ = ols_r2([eci, simv], y)
    # F-test for adding sim
    F = (r2_es - r2_e) / ((1 - r2_es) / (n - 3))
    pF = 1 - stats.f.cdf(F, 1, n - 3)
    print(f"in-sample R2: ECI only {r2_e:.3f} -> ECI+sim {r2_es:.3f}  (dR2={r2_es-r2_e:+.3f}, F={F:.2f}, p={pF:.3f})")
    print(f"LOO CV-R2:   ECI only {loo_r2([eci],y):.3f} -> ECI+sim {loo_r2([eci,simv],y):.3f}  sim only {loo_r2([simv],y):.3f}")
    # noise ceiling: measurement variance in FB scores vs observed variance
    noise_var = np.mean(ses ** 2)
    obs_var = y.var(ddof=1)
    ceil = 1 - noise_var / obs_var
    print(f"noise ceiling: mean SE={np.sqrt(noise_var):.2f}, obs SD={np.sqrt(obs_var):.2f} -> max attainable R2 ~= {ceil:.3f}")
    print(f"  (reliability-adjusted: ECI+sim explains {r2_es/ceil:.0%} of explainable variance; ECI alone {r2_e/ceil:.0%})")
