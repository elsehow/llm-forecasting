#!/usr/bin/env python3
"""Null-intervention control for the Starsim over-adjustment finding.

Same 30 scenarios (seed 11). For each scenario x horizon, announce the SAME
vaccination campaign but scheduled on day 70 — after both resolution horizons
(T=40, 60) — and resolve on the CONTROL world. A rational forecaster ignores
it; lowering P(yes) relative to one's own unconditional forecast on the same
question = demand effect / over-adjustment to the mere mention of an
intervention.

Caches RAW responses (null_conditional_cache.json) and parses with
last-number-wins (immune to chain-of-thought preambles).
"""
import json, os, re, sys, time
from pathlib import Path

for line in Path("/Users/elsehow/Projects/forecastbench-sim/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
from fbsim_core.evaluation.models import load_api_keys_from_gcp, get_models
load_api_keys_from_gcp()
os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

from pandemic_world.scenarios import (
    sample_scenarios, NAMES, R_ID, S_ID, R_NAME, S_NAME, SNAPSHOT_DAY, HORIZONS)
from pandemic_world.scenarios import build_corpus
from pandemic_world.runner import run_region, to_world
from pandemic_world.report import context_blurb, build_prompt
from pandemic_world.templates import CASES_COMPARATIVE

MODELS = [
    "openai/gpt-4.1-2025-04-14", "openai/gpt-5-2025-08-07",
    "openai/gpt-5-mini-2025-08-07", "openai/gpt-5-nano-2025-08-07",
    "openai/gpt-5.1-2025-11-13", "openai/o3-2025-04-16",
    "openai/o3-mini-2025-01-31", "openai/o4-mini-2025-04-16",
    "anthropic/claude-sonnet-4-5-20250929", "anthropic/claude-haiku-4-5-20251001",
    "google/gemini-2.5-pro", "deepinfra/deepseek-ai/DeepSeek-V3", "xai/grok-4-0709",
]
NULL_DAY = 70  # after both horizons (40, 60)
CACHE_PATH = Path("/Users/elsehow/Projects/forecastbench-sim/worlds/pandemic/.null_conditional_cache.json")
OUT_PATH = Path("/Users/elsehow/Projects/forecastbench-sim/worlds/pandemic/null_conditional_results.json")

def parse_prob_last(text):
    if not text:
        return None
    ms = re.findall(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])", text)
    if not ms:
        return None
    try:
        return min(max(float(ms[-1]), 0.0), 1.0)
    except ValueError:
        return None

def build_null_corpus():
    corpus = []
    for sc in sample_scenarios(30, 11):
        kw = dict(vax_efficacy=sc["vax_efficacy"], vax_day=sc["vax_day"],
                  vax_coverage=sc["vax_coverage"], seed=sc["sim_seed"])
        s_series = run_region(sc["beta_S"], False, **kw)
        r_control = run_region(sc["beta_R"], False, **kw)
        control_world = to_world({R_ID: r_control, S_ID: s_series}, NAMES)
        null_text = (
            f"{R_NAME} will run a vaccination campaign on day {NULL_DAY} "
            f"({int(sc['vax_efficacy']*100)}% efficacy, {int(sc['vax_coverage']*100)}% coverage). "
            f"{S_NAME} has no planned intervention."
        )
        from fbsim_core.questions.schema import QuestionInstance, classify_horizon
        from fbsim_core.questions.resolver import QuestionResolver
        from pandemic_world.templates import REGISTRY
        resolver = QuestionResolver(REGISTRY)
        for T in HORIZONS:
            q_text = CASES_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)
            q = QuestionInstance(
                question_id=f"s{sc['scenario_id']}_T{T}_null",
                template_id="cases_comparative", resolution_turn=T,
                horizon=classify_horizon(SNAPSHOT_DAY, T),
                parameters={"region_a": R_NAME, "region_b": S_NAME,
                            "player_id_a": R_ID, "player_id_b": S_ID},
                question_text=q_text)
            res = resolver.resolve(q, control_world, SNAPSHOT_DAY)
            corpus.append({
                "question_id": q.question_id, "kind": "null_conditional",
                "horizon": T, "scenario_id": sc["scenario_id"],
                "question_text": q_text, "ground_truth": bool(res.answer),
                "context": context_blurb(control_world, [R_ID, S_ID], NAMES,
                                         SNAPSHOT_DAY, null_text),
            })
    return corpus

def main():
    corpus = build_null_corpus()
    print(f"null corpus: {len(corpus)} questions", flush=True)
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    results = {}
    for mid in MODELS:
        model = get_models([mid])[0]
        print(f"evaluating {mid} ({len(corpus)} q)...", flush=True)
        n_new = 0
        for c in corpus:
            ck = f"{mid}|{c['question_id']}"
            if ck in cache:
                continue
            raw = None
            for attempt in range(6):
                try:
                    raw = model.get_response(build_prompt(c["context"], c["question_text"]),
                                             max_tokens=16000)
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    w = 2 ** attempt * 5
                    print(f"  retry {attempt+1}/5 after {type(e).__name__}, {w}s", flush=True)
                    time.sleep(w)
            cache[ck] = raw
            n_new += 1
            if n_new % 10 == 0:
                CACHE_PATH.write_text(json.dumps(cache))
        CACHE_PATH.write_text(json.dumps(cache))
        briers, preds = [], []
        for c in corpus:
            p = parse_prob_last(cache.get(f"{mid}|{c['question_id']}"))
            if p is None:
                continue
            preds.append({"question_id": c["question_id"], "p": p,
                          "gt": c["ground_truth"]})
            briers.append((p - float(c["ground_truth"])) ** 2)
        import numpy as np
        results[mid] = {"brier": float(np.mean(briers)), "n": len(briers),
                        "mean_pred": float(np.mean([x["p"] for x in preds])),
                        "preds": preds}
        print(f"  brier={results[mid]['brier']:.4f} n={results[mid]['n']} "
              f"meanP={results[mid]['mean_pred']:.2f}", flush=True)
    OUT_PATH.write_text(json.dumps(results, indent=1))
    print(f"-> {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
