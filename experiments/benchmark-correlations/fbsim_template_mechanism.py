#!/usr/bin/env python3
"""Mechanism per template: does hedging pay (decisiveness->skill corr across
models), how predictable is the template at the consensus level, and does
wonder_completed alone beat ECI-only in LOO?"""
import json
import numpy as np
from scipy import stats

SIMF = "/Users/elsehow/Projects/forecastbench-sim/data/results/uncond_nonh0_bin_cont_all30_20260307.json"
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
data = json.load(open(SIMF))
questions = [q for q in data["questions"] if q["question_type"] == "binary"]
names = list(FB_ECI)
ids = [SIM_ID[n] for n in names]
fb_ds = np.array([FB_ECI[n][1] for n in names])
eci = np.array([FB_ECI[n][2] for n in names])
ALL30 = list(data["model_results"].keys())

templates = sorted({q["template_id"] for q in questions})
print(f"{'template':26s} {'dec->skill':>10s} {'median-model':>12s} {'best-model':>10s} {'mean|p-.5|':>10s}")
for t in templates:
    tq = [q for q in questions if q["template_id"] == t]
    skills, decs = [], []
    for m in ids:
        bs, ds = [], []
        for q in tq:
            p = q["predictions"].get(m)
            if p and p.get("probability") is not None:
                bs.append((p["probability"] - float(q["ground_truth"])) ** 2)
                ds.append(abs(p["probability"] - 0.5))
        skills.append(-np.mean(bs)); decs.append(np.mean(ds))
    skills, decs = np.array(skills), np.array(decs)
    r_ds = stats.pearsonr(decs, skills).statistic
    # consensus: median across ALL 30 models per question
    med_briers, all_briers = [], {m: [] for m in ALL30}
    for q in tq:
        ps = [q["predictions"][m]["probability"] for m in ALL30
              if q["predictions"].get(m) and q["predictions"][m].get("probability") is not None]
        gt = float(q["ground_truth"])
        med_briers.append((np.median(ps) - gt) ** 2)
        for m in ALL30:
            p = q["predictions"].get(m)
            if p and p.get("probability") is not None:
                all_briers[m].append((p["probability"] - gt) ** 2)
    best = min(np.mean(v) for v in all_briers.values() if v)
    print(f"{t:26s} {r_ds:+10.2f} {np.mean(med_briers):12.3f} {best:10.3f} {decs.mean():10.2f}")

# wonder_completed alone in LOO vs ECI-only
def loo(Xcols, y):
    n = len(y); preds = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        X1 = np.column_stack([np.ones(m.sum())] + [x[m] for x in Xcols])
        beta, *_ = np.linalg.lstsq(X1, y[m], rcond=None)
        preds[i] = np.concatenate([[1], [x[i] for x in Xcols]]) @ beta
    return 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def tmpl_skill(t):
    tq = [q for q in questions if q["template_id"] == t]
    out = []
    for m in ids:
        bs = [(q["predictions"][m]["probability"] - float(q["ground_truth"])) ** 2
              for q in tq if q["predictions"].get(m) and q["predictions"][m].get("probability") is not None]
        out.append(-np.mean(bs))
    return np.array(out)

wonder = tmpl_skill("wonder_completed")
gov = tmpl_skill("government_at")
print(f"\nLOO CV-R2 -> FB dataset (n=15):")
print(f"  ECI only:            {loo([eci], fb_ds):+.3f}")
print(f"  ECI + wonder:        {loo([eci, wonder], fb_ds):+.3f}")
print(f"  wonder only:         {loo([wonder], fb_ds):+.3f}")
print(f"  ECI + wonder + gov:  {loo([eci, wonder, gov], fb_ds):+.3f}")
pr = stats.pearsonr(wonder, fb_ds)
print(f"  wonder biv r={pr.statistic:.3f} (p={pr.pvalue:.3f}); r with ECI={stats.pearsonr(wonder,eci).statistic:.3f}")
