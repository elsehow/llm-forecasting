#!/usr/bin/env python3
"""Robustness for the negative partial corr: jackknife influence, H0 vs non-H0
sim scores, rank-based partials, and paper-style bivariate with weak models."""
import json
import numpy as np
from scipy import stats

SIMDIR = "/Users/elsehow/Projects/forecastbench-sim/data/results"
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

nonh0 = json.load(open(f"{SIMDIR}/uncond_nonh0_bin_cont_all30_20260307.json"))["model_results"]
h0 = json.load(open(f"{SIMDIR}/uncond_h0_all30_20260307.json"))["model_results"]

names = list(FB_ECI)
fb_ov = np.array([FB_ECI[n][0] for n in names])
fb_ds = np.array([FB_ECI[n][1] for n in names])
eci = np.array([FB_ECI[n][2] for n in names])
sim_non = np.array([-nonh0[SIM_ID[n]]["binary"]["brier_score"] for n in names])
sim_h0 = np.array([-h0[SIM_ID[n]]["binary"]["brier_score"] for n in names])

def partial(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)

for simv, simlab in [(sim_non, "sim non-H0"), (sim_h0, "sim H0")]:
    for y, ylab in [(fb_ov, "FB overall"), (fb_ds, "FB dataset")]:
        r, p = partial(simv, y, eci)
        # rank-based partial
        rr, rp = partial(stats.rankdata(simv), stats.rankdata(y), stats.rankdata(eci))
        print(f"{simlab:11s} -> {ylab:10s}: partial r={r:+.3f} (p={p:.3f})  rank-partial={rr:+.3f} (p={rp:.3f})  biv r={stats.pearsonr(simv,y).statistic:+.3f}")

print("\njackknife partial corr (sim non-H0 -> FB dataset | ECI):")
base, _ = partial(sim_non, fb_ds, eci)
for i, n in enumerate(names):
    m = np.ones(len(names), bool); m[i] = False
    r, p = partial(sim_non[m], fb_ds[m], eci[m])
    flag = " <<<" if abs(r - base) > 0.15 else ""
    print(f"  drop {n:18s} r={r:+.3f} (p={p:.3f}){flag}")

print("\njackknife partial corr (sim non-H0 -> FB overall | ECI):")
base, _ = partial(sim_non, fb_ov, eci)
for i, n in enumerate(names):
    m = np.ones(len(names), bool); m[i] = False
    r, p = partial(sim_non[m], fb_ov[m], eci[m])
    flag = " <<<" if abs(r - base) > 0.15 else ""
    print(f"  drop {n:18s} r={r:+.3f} (p={p:.3f}){flag}")

# paper-style bivariate across ALL 30 sim models vs FB where leaderboard has them
# (weak models expand range: gpt-3.5, claude-3-haiku, mixtral, llama-3.3, mistral)
import csv
lb = list(csv.DictReader(open("/Users/elsehow/Projects/llm-forecasting/experiments/benchmark-correlations/data/leaderboard_baseline.csv")))
LB_MAP = {  # sim id -> leaderboard fragment
    "openai/gpt-3.5-turbo-0125": "GPT-3.5-Turbo-0125",
    "anthropic/claude-3-haiku-20240307": "Claude-3-Haiku-20240307",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B",
    "mistral/mistral-large-2407": "Mistral-Large-2407",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash (",
    "openai/gpt-4o": "GPT-4o (scratchpad",
}
pairs = []
for n in names:
    pairs.append((sim_non[list(names).index(n)], FB_ECI[n][1], n))
for sid, frag in LB_MAP.items():
    cands = [r for r in lb if frag.lower() in r["Model"].lower()]
    sc = [r for r in cands if "scratchpad" in r["Model"]] or cands
    if sc and sid in nonh0:
        pairs.append((-nonh0[sid]["binary"]["brier_score"], float(sc[0]["Dataset"]), sid.split("/")[-1]))
s = np.array([p[0] for p in pairs]); f = np.array([p[1] for p in pairs])
print(f"\nbivariate incl. weak models (n={len(pairs)}): sim -> FB dataset  "
      f"r={stats.pearsonr(s,f).statistic:.3f}  rho_s={stats.spearmanr(s,f).statistic:.3f}")
for p in sorted(pairs, key=lambda t: t[0]):
    print(f"  {p[2]:40s} simBrier={-p[0]:.4f} FBds={p[1]:.1f}")
