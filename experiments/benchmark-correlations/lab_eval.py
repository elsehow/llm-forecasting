#!/usr/bin/env python3
"""Models-vs-certified-instruments run.

Score frontier models on the lab-grown SKILL and CALIB classes, against the
per-instance ground-truth pi (branch-resampled). Scoring against pi (not the
binary outcome) deletes outcome noise, so small-n reliability is high.

Per model, per class:
  MSE_pi = mean_q (p_model - pi_q)^2        <- lower better
  climatology MSE = Var(pi)                 <- predict E[pi] constant
  perfect MSE = 0                           <- predict pi exactly
  skill = 1 - MSE_pi / Var(pi)              <- R^2-like; >0 beats climatology

Then: correlate lab skill vs ECI, and lab skill vs ForecastBench conditional
on ECI (Jason's extrapolation test, now on a CERTIFIED instrument).
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np

SCRATCH = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad")
CORPUS = json.load(open(SCRATCH / "lab_classes.json"))
CACHE_PATH = SCRATCH / "lab_eval_cache.json"      # model|kind|idx -> parsed float
RAW_PATH = SCRATCH / "lab_eval_raw.json"          # model|kind|idx -> raw text
RESULTS_PATH = SCRATCH / "lab_eval_results.json"

# ECI sample: spread across the available range (132.96 .. 150.00).
# (litellm id, display, eci)
MODELS = [
    ("deepinfra/deepseek-ai/DeepSeek-V3",       "DeepSeek-V3",       132.96),
    ("openai/gpt-4.1-2025-04-14",               "GPT-4.1",           137.43),
    ("openai/gpt-5-nano-2025-08-07",            "GPT-5 Nano",        139.88),
    ("anthropic/claude-haiku-4-5-20251001",     "Claude Haiku 4.5",  140.66),
    ("openai/o3-mini-2025-01-31",               "o3-mini",           141.11),
    ("openai/gpt-5-mini-2025-08-07",            "GPT-5 Mini",        144.32),
    ("openai/o4-mini-2025-04-16",               "o4-mini",           145.06),
    ("google/gemini-2.5-pro",                   "Gemini 2.5 Pro",    146.06),
    ("anthropic/claude-sonnet-4-5-20250929",    "Claude Son. 4.5",   146.61),
    ("openai/o3-2025-04-16",                    "o3",                146.62),
    ("xai/grok-4-0709",                         "Grok 4",            147.07),
    ("openai/gpt-5.1-2025-11-13",               "GPT-5.1",           149.74),
    ("openai/gpt-5-2025-08-07",                 "GPT-5",             150.00),
    ("anthropic/claude-opus-4-5-20251101",      "Claude Opus 4.5",   150.05),
]

# ForecastBench conditional-question Brier (from the Starsim run) per model,
# keyed by display name -> we pull it from scale_results.json if present.
SCALE = json.load(open("/Users/elsehow/Projects/forecastbench-sim/worlds/pandemic/scale_results.json"))


def parse_prob(text):
    if text is None:
        return None
    ms = re.findall(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])", text.strip())
    if not ms:
        ms = re.findall(r"\d+(?:\.\d+)?", text)
    if not ms:
        return None
    try:
        return min(max(float(ms[-1]), 0.0), 1.0)
    except ValueError:
        return None


def build_prompt(q):
    act = q["prefix_active"]
    n_agents = 5000
    thresh = int(q["thresh"])  # fixed absolute take-off threshold (225)
    seed0 = int(act[0])
    lines = [f"  day {d:2d}: {int(act[d]):4d} currently infectious" for d in range(len(act))]
    series = "\n".join(lines)
    return (
        f"You are an epidemiologist tracking an outbreak of a novel infectious "
        f"disease in a closed population of {n_agents} individuals. Each infected "
        f"person stays infectious for about 10 days; roughly 2% of infections are "
        f"fatal. People mix through a contact network (about 6 close contacts each). "
        f"The outbreak began with roughly {seed0} infectious individuals. Below is "
        f"the number currently infectious on each of the first 20 days.\n\n"
        f"OUTBREAK LOG (days 0-19):\n{series}\n\n"
        f"The key question is whether this outbreak takes off into a substantial "
        f"epidemic or fizzles out.\n\n"
        f"QUESTION: What is the probability that at least {thresh} people in total "
        f"become infected through onward person-to-person transmission by day 60 "
        f"(i.e. the outbreak grows into a substantial epidemic rather than dying "
        f"out)?\n\n"
        f"Give your probability as a single number between 0 and 1. "
        f"Respond with ONLY the number (e.g. 0.73)."
    )


def main():
    from fbsim_core.evaluation.models import get_models

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    raw = json.loads(RAW_PATH.read_text()) if RAW_PATH.exists() else {}

    questions = ([dict(q, kind="skill") for q in CORPUS["skill"]]
                 + [dict(q, kind="calib") for q in CORPUS["calib"]])
    print(f"Corpus: {len(CORPUS['skill'])} skill + {len(CORPUS['calib'])} calib "
          f"= {len(questions)} questions", flush=True)

    per_model = {}
    for mid, disp, eci in MODELS:
        model = get_models([mid])[0]
        print(f"\n=== {disp} ({mid}) ECI={eci} ===", flush=True)
        n_new = 0
        for q in questions:
            key = f"{mid}|{q['kind']}|{q['idx']}"
            if key in cache:
                continue
            txt = model.get_response(build_prompt(q), max_tokens=16000)
            raw[key] = txt
            cache[key] = parse_prob(txt)
            n_new += 1
            if n_new % 20 == 0:
                CACHE_PATH.write_text(json.dumps(cache))
                RAW_PATH.write_text(json.dumps(raw))
                print(f"  ...{n_new} new", flush=True)
        CACHE_PATH.write_text(json.dumps(cache))
        RAW_PATH.write_text(json.dumps(raw))

        # score against pi per class
        row = {"eci": eci}
        for kind in ("skill", "calib"):
            qs = [q for q in questions if q["kind"] == kind]
            pi = np.array([q["pi"] for q in qs])
            preds, tgt = [], []
            for q in qs:
                p = cache.get(f"{mid}|{kind}|{q['idx']}")
                if p is not None:
                    preds.append(p); tgt.append(q["pi"])
            preds, tgt = np.array(preds), np.array(tgt)
            varpi = pi.var()
            if len(preds):
                mse = float(((preds - tgt) ** 2).mean())
                skill = 1 - mse / varpi if varpi else float("nan")
                row[kind] = dict(n=len(preds), mse_pi=mse, skill=skill,
                                 mean_pred=float(preds.mean()),
                                 corr=float(np.corrcoef(preds, tgt)[0, 1]) if len(preds) > 2 else float("nan"))
            else:
                row[kind] = dict(n=0)
        per_model[disp] = row
        s = row.get("skill", {})
        c = row.get("calib", {})
        print(f"  SKILL: n={s.get('n')} MSE_pi={s.get('mse_pi',float('nan')):.4f} "
              f"skill={s.get('skill',float('nan')):+.3f} meanP={s.get('mean_pred',float('nan')):.3f} "
              f"corr(p,pi)={s.get('corr',float('nan')):+.2f}", flush=True)
        print(f"  CALIB: n={c.get('n')} MSE_pi={c.get('mse_pi',float('nan')):.4f} "
              f"meanP={c.get('mean_pred',float('nan')):.3f}", flush=True)

    RESULTS_PATH.write_text(json.dumps({"models": per_model,
                                        "corpus_stats": CORPUS["meta"]["stats"]}, indent=2))
    print(f"\n-> {RESULTS_PATH}")

    # ---- analysis: lab skill vs ECI, and does skill(p,pi) rank models? ----
    from scipy import stats as st
    disp = [m[1] for m in MODELS]
    eci = np.array([per_model[d]["eci"] for d in disp])
    skill = np.array([per_model[d]["skill"].get("skill", np.nan) for d in disp])
    corr = np.array([per_model[d]["skill"].get("corr", np.nan) for d in disp])
    ok = ~np.isnan(skill) & ~np.isnan(eci)
    print("\n" + "=" * 60)
    print("LAB SKILL CLASS — model ranking (scored vs ground-truth pi)")
    print("=" * 60)
    order = np.argsort(-skill)
    for i in order:
        print(f"  {disp[i]:20s} ECI={eci[i]:6.2f}  skill={skill[i]:+.3f}  corr(p,pi)={corr[i]:+.2f}")
    if ok.sum() > 3:
        r_es = st.pearsonr(eci[ok], skill[ok])
        r_ec = st.pearsonr(eci[ok], corr[ok])
        print(f"\ncorr(ECI, lab skill)      r={r_es.statistic:+.2f} (p={r_es.pvalue:.3f}, n={ok.sum()})")
        print(f"corr(ECI, corr(p,pi))     r={r_ec.statistic:+.2f} (p={r_ec.pvalue:.3f})")
        print("\n(High corr(ECI, skill) => the certified instrument tracks ECI;")
        print(" low/near-zero => it measures a capability ECI misses.)")


if __name__ == "__main__":
    main()
