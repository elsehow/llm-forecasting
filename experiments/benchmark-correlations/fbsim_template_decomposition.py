#!/usr/bin/env python3
"""Template-level decomposition of the sim->FB|ECI signal.

Per binary template:
  - split-half reliability of the model ranking it induces (even/odd questions)
  - corr with ECI, bivariate corr with FB, partial corr with FB | ECI (n=15)
  - wide-range bivariate (n=22, incl. weak models)
Mechanism probes:
  - decisiveness (mean |p-0.5|), ECE: partial corr with FB | ECI
  - honest LOO composite with within-fold template selection
"""
import json, csv
import numpy as np
from scipy import stats

SIMF = "/Users/elsehow/Projects/forecastbench-sim/data/results/uncond_nonh0_bin_cont_all30_20260307.json"
LB = "/Users/elsehow/Projects/llm-forecasting/experiments/benchmark-correlations/data/leaderboard_baseline.csv"

FB_ECI = {
    "GPT-4.1": (63.3, 60.1, 137.43), "GPT-5": (62.5, 61.8, 150.00),
    "GPT-5 Mini": (62.9, 60.8, 144.32), "GPT-5 Nano": (60.4, 57.6, 139.88),
    "GPT-5.1": (59.0, 58.7, 149.74), "o3": (64.8, 60.4, 146.62),
    "o3-mini": (62.3, 58.9, 141.11), "o4-mini": (63.8, 60.7, 145.06),
    "Claude Son. 4": (61.7, 59.0, 142.33), "Claude Son. 4.5": (64.3, 60.7, 146.61),
    "Claude Haiku 4.5": (62.8, 59.4, 140.66), "Gemini 3 Pro": (63.3, 61.3, 154.22),
    "Gemini 2.5 Pro": (63.8, 61.0, 146.06), "DeepSeek V3": (61.9, 59.1, 132.96),
    "Grok 4": (63.3, 60.4, 147.07),
}
SIM_ID = {
    "GPT-4.1": "openai/gpt-4.1-2025-04-14", "GPT-5": "openai/gpt-5-2025-08-07",
    "GPT-5 Mini": "openai/gpt-5-mini-2025-08-07", "GPT-5 Nano": "openai/gpt-5-nano-2025-08-07",
    "GPT-5.1": "openai/gpt-5.1-2025-11-13", "o3": "openai/o3-2025-04-16",
    "o3-mini": "openai/o3-mini-2025-01-31", "o4-mini": "openai/o4-mini-2025-04-16",
    "Claude Son. 4": "anthropic/claude-sonnet-4-20250514",
    "Claude Son. 4.5": "anthropic/claude-sonnet-4-5-20250929",
    "Claude Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
    "Gemini 3 Pro": "google/gemini-3-pro-preview", "Gemini 2.5 Pro": "google/gemini-2.5-pro",
    "DeepSeek V3": "together_ai/deepseek-ai/DeepSeek-V3", "Grok 4": "xai/grok-4-0709",
}
WIDE_LB = {  # extra sim models -> leaderboard fragment (FB dataset)
    "openai/gpt-3.5-turbo-0125": "GPT-3.5-Turbo-0125",
    "anthropic/claude-3-haiku-20240307": "Claude-3-Haiku-20240307",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B",
    "mistral/mistral-large-2407": "Mistral-Large-2407",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash (",
    "openai/gpt-4o": "GPT-4o (scratchpad",
}

data = json.load(open(SIMF))
mr = data["model_results"]
questions = [q for q in data["questions"] if q["question_type"] == "binary"]
names = list(FB_ECI)
fb_ds = np.array([FB_ECI[n][1] for n in names])
fb_ov = np.array([FB_ECI[n][0] for n in names])
eci = np.array([FB_ECI[n][2] for n in names])
ids = [SIM_ID[n] for n in names]

lb = list(csv.DictReader(open(LB)))
def lb_ds(frag):
    cands = [r for r in lb if frag.lower() in r["Model"].lower()]
    sc = [r for r in cands if "scratchpad" in r["Model"]] or cands
    return float(sc[0]["Dataset"]) if sc else None

templates = sorted({q["template_id"] for q in questions})

# per-model-per-template question-level briers, plus decisiveness
def qbrier(q, mid):
    p = q["predictions"].get(mid)
    if not p or p.get("probability") is None:
        return None
    return (p["probability"] - float(q["ground_truth"])) ** 2

def pdec(q, mid):
    p = q["predictions"].get(mid)
    return None if not p or p.get("probability") is None else abs(p["probability"] - 0.5)

def partial(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)

wide_ids = ids + [w for w in WIDE_LB if w in mr]
wide_fb = list(fb_ds) + [lb_ds(WIDE_LB[w]) for w in WIDE_LB if w in mr]

print(f"{'template':26s} {'nq':>5s} {'relia':>6s} {'r_ECI':>6s} {'biv_FBds':>8s} {'partial|ECI':>12s} {'p':>6s} {'wide_biv':>8s}")
tskill = {}
for t in templates:
    tq = [q for q in questions if q["template_id"] == t]
    skill = np.array([-np.mean([b for q in tq if (b := qbrier(q, m)) is not None]) for m in ids])
    tskill[t] = skill
    # split-half reliability of model ranking (even/odd questions, Spearman-Brown corrected)
    a = np.array([-np.mean([b for q in tq[0::2] if (b := qbrier(q, m)) is not None]) for m in ids])
    b = np.array([-np.mean([b_ for q in tq[1::2] if (b_ := qbrier(q, m)) is not None]) for m in ids])
    r_half = stats.spearmanr(a, b).statistic
    rel = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
    r_eci = stats.pearsonr(skill, eci).statistic
    biv = stats.pearsonr(skill, fb_ds).statistic
    pr, pp = partial(skill, fb_ds, eci)
    ws = np.array([-np.mean([b for q in tq if (b := qbrier(q, m)) is not None]) for m in wide_ids])
    wbiv = stats.pearsonr(ws, np.array(wide_fb)).statistic
    print(f"{t:26s} {len(tq):5d} {rel:6.2f} {r_eci:+6.2f} {biv:+8.2f} {pr:+12.2f} {pp:6.3f} {wbiv:+8.2f}")

# mechanism probes
dec = np.array([np.mean([d for q in questions if (d := pdec(q, m)) is not None]) for m in ids])
ece = np.array([mr[m]["binary"]["ece"] for m in ids])
overall = np.array([-mr[m]["binary"]["brier_score"] for m in ids])
print("\nmechanism probes (n=15):")
for lab, v in [("decisiveness |p-.5|", dec), ("ECE", ece), ("overall sim skill", overall)]:
    pr, pp = partial(v, fb_ds, eci)
    pro, ppo = partial(v, fb_ov, eci)
    print(f"  {lab:22s} r_ECI={stats.pearsonr(v,eci).statistic:+.2f}  partial|ECI: FBds {pr:+.2f} (p={pp:.3f})  FBov {pro:+.2f} (p={ppo:.3f})")
pr, pp = partial(dec, overall, eci)
print(f"  decisiveness vs sim skill | ECI: r={pr:+.2f} (p={pp:.3f})")

# honest LOO composite: select templates with positive training-fold partial, average z-scores
def loo_composite(target):
    n = len(names)
    preds = np.empty(n)
    chosen_all = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        chosen = []
        for t in templates:
            r, p = partial(tskill[t][m], target[m], eci[m])
            if r > 0.2:
                chosen.append(t)
        chosen_all.append(chosen)
        if chosen:
            comp = np.mean([(tskill[t] - tskill[t][m].mean()) / tskill[t][m].std() for t in chosen], axis=0)
            X1 = np.column_stack([np.ones(m.sum()), eci[m], comp[m]])
            beta, *_ = np.linalg.lstsq(X1, target[m], rcond=None)
            preds[i] = np.array([1, eci[i], comp[i]]) @ beta
        else:
            X1 = np.column_stack([np.ones(m.sum()), eci[m]])
            beta, *_ = np.linalg.lstsq(X1, target[m], rcond=None)
            preds[i] = np.array([1, eci[i]]) @ beta
    cv = 1 - ((target - preds) ** 2).sum() / ((target - target.mean()) ** 2).sum()
    from collections import Counter
    c = Counter(t for ch in chosen_all for t in ch)
    return cv, c

cv, c = loo_composite(fb_ds)
print(f"\nhonest LOO composite (templates w/ positive partial in-fold) -> FB dataset: CV-R2={cv:.3f}")
print("  selection frequency:", dict(c))
