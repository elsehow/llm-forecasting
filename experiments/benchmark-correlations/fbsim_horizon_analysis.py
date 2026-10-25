#!/usr/bin/env python3
"""By-horizon and horizon x template decomposition, all conditional on ECI
(Jason's framing): partial corr of sim skill with FB-dataset | ECI."""
import json, re
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
TURN2H = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6", 270: "H7"}

data = json.load(open(SIMF))
names = list(FB_ECI)
ids = [SIM_ID[n] for n in names]
fb_ds = np.array([FB_ECI[n][1] for n in names])
eci = np.array([FB_ECI[n][2] for n in names])

qs = []
for q in data["questions"]:
    if q["question_type"] != "binary":
        continue
    turn = int(re.findall(r"turn (\d+)", q["question_text"])[-1])
    qs.append((TURN2H[turn], q["template_id"], q))

def partial(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)

def skill_vec(subset):
    out = []
    for m in ids:
        bs = [(q["predictions"][m]["probability"] - float(q["ground_truth"])) ** 2
              for q in subset if q["predictions"].get(m) and q["predictions"][m].get("probability") is not None]
        out.append(-np.mean(bs))
    return np.array(out)

def relia(subset):
    a = skill_vec(subset[0::2]); b = skill_vec(subset[1::2])
    r = stats.spearmanr(a, b).statistic
    return 2 * r / (1 + r) if r > -1 else np.nan

HORIZONS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
templates = sorted({t for _, t, _ in qs})

print("=== by horizon (all templates pooled), partial r with FB-ds | ECI ===")
print(f"{'hz':3s} {'nq':>5s} {'relia':>6s} {'partial|ECI':>11s} {'p':>6s} {'best_brier':>10s} {'base_rate':>9s}")
hz_partials = {}
for h in HORIZONS:
    sub = [q for hh, _, q in qs if hh == h]
    sk = skill_vec(sub)
    pr, pp = partial(sk, fb_ds, eci)
    hz_partials[h] = (pr, pp)
    best = max(skill_vec(sub))
    br = np.mean([float(q["ground_truth"]) for q in sub])
    print(f"{h:3s} {len(sub):5d} {relia(sub):6.2f} {pr:+11.2f} {pp:6.3f} {-best:10.3f} {br:9.3f}")

print("\n=== wonder_completed by horizon ===")
print(f"{'hz':3s} {'nq':>4s} {'partial|ECI':>11s} {'p':>6s} {'best_brier':>10s} {'base_rate':>9s}")
for h in HORIZONS:
    sub = [q for hh, t, q in qs if hh == h and t == "wonder_completed"]
    if len(sub) < 10: continue
    sk = skill_vec(sub)
    pr, pp = partial(sk, fb_ds, eci)
    best = max(sk)
    br = np.mean([float(q["ground_truth"]) for q in sub])
    print(f"{h:3s} {len(sub):4d} {pr:+11.2f} {pp:6.3f} {-best:10.3f} {br:9.3f}")

print("\n=== template x horizon: partial r | ECI (n_q per cell in parens) ===")
grid = {}
hdr = "template".ljust(26) + "".join(f"{h:>12s}" for h in HORIZONS)
print(hdr)
for t in templates:
    row = t.ljust(26)
    for h in HORIZONS:
        sub = [q for hh, tt, q in qs if hh == h and tt == t]
        if len(sub) < 20:
            grid[(t, h)] = (np.nan, np.nan, len(sub))
            row += f"{'—':>12s}"
            continue
        sk = skill_vec(sub)
        pr, pp = partial(sk, fb_ds, eci)
        grid[(t, h)] = (pr, pp, len(sub))
        star = "*" if pp < 0.05 else " "
        row += f"{pr:+8.2f}{star}({len(sub):3d})"[-12:].rjust(12)
    print(row)

json.dump({f"{t}|{h}": grid[(t, h)] for t, h in grid},
          open("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/horizon_grid.json", "w"))
json.dump(hz_partials,
          open("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/horizon_pooled.json", "w"))
