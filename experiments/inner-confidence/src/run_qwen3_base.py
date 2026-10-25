"""Run Qwen3-235B base on Fireworks via completions API."""
import json, math, os, random, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np, openai

CIVBENCH = Path("/Users/elsehow/Projects/civbench")
DATA_ROOT = CIVBENCH / "data/conditional/republic/baseline"
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")
MODEL_ID = "accounts/nick-p6n6002bmkx4/deployments/ed3m99wc"
MODEL_SHORT = "Qwen3-235B-base"
TARGET_SEEDS = {"seed0", "seed1", "seed2", "seed3", "seed4"}
DISRUPTABLE = {"population_continuous","territory_continuous","treasury_continuous","cities_count_continuous"}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
ALL_HORIZONS = {"H1","H2","H3","H4","H5","H6"}

API_KEY = [l.split("=",1)[1] for l in Path("/Users/elsehow/Projects/civbench/.env").read_text().splitlines() if l.startswith("FIREWORKS_API_KEY=")][0]
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")

def entropy_from_top5(top_lp_dict):
    if not top_lp_dict: return None
    probs = [math.exp(lp) for lp in top_lp_dict.values()]
    total = sum(probs)
    if total <= 0: return None
    probs = [p/total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)

def build_prompt(question, world_report):
    return f"""Below is a forecasting task on a FreeCiv game simulation.

=== EXAMPLE ===
World Report: Game seed99, Turn 60. Civ "Roman" has population 30, treasury 400, 12 techs, 20 cities, 280 tiles.

Question: What will Roman's population be at turn 180?

<<<PERCENTILES>>>
p10=20, p25=30, p50=45, p75=65, p90=85
<<<END>>>

=== TASK ===
## World Report
{world_report}

## Question
{question['question_text']}

<<<PERCENTILES>>>
"""

all_questions = []
for seed_dir in sorted(DATA_ROOT.iterdir()):
    if not seed_dir.is_dir() or seed_dir.name not in TARGET_SEEDS: continue
    qpath = seed_dir / "questions.json"
    if not qpath.exists(): continue
    with open(qpath) as f: qdata = json.load(f)
    wr_path = seed_dir / "world_report" / "turn_060_report.txt"
    if not wr_path.exists(): continue
    world_report = wr_path.read_text()
    for q in qdata["questions"]:
        if q["question_type"]=="continuous" and q["template_id"] in ALL_TEMPLATES and q.get("horizon") in ALL_HORIZONS:
            all_questions.append({"question": q, "world_report": world_report, "seed": seed_dir.name})

random.seed(42)
random.shuffle(all_questions)
print(f"[{MODEL_SHORT}] {len(all_questions)} questions")

results, errors = [], 0
for i, item in enumerate(all_questions):
    q = item["question"]
    prompt = build_prompt(q, item["world_report"])
    try:
        r = client.completions.create(model=MODEL_ID, prompt=prompt, max_tokens=200, temperature=0.0, logprobs=5)
        content = r.choices[0].text or ""
        lp = r.choices[0].logprobs
        # Extract max entropy from percentile tokens
        max_ent = None
        if lp and lp.tokens and lp.top_logprobs:
            text = "".join(lp.tokens)
            end_pos = text.find("<<<END>>>")
            if end_pos < 0: end_pos = len(text)
            pct_ents = []
            cp = 0
            for j, tok in enumerate(lp.tokens):
                ts, te = cp, cp + len(tok)
                if ts < end_pos and j < len(lp.top_logprobs) and lp.top_logprobs[j]:
                    e = entropy_from_top5(lp.top_logprobs[j])
                    if e is not None: pct_ents.append(e)
                cp = te
            if pct_ents: max_ent = max(pct_ents)
        results.append({
            "template_id": q["template_id"], "horizon": q["horizon"],
            "ground_truth": q["resolution"]["value_at_resolution"],
            "group": "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE",
            "model": MODEL_SHORT, "seed": item["seed"],
            "max_entropy_pct": max_ent,
        })
        if (i+1) % 25 == 0:
            parsed = sum(1 for r in results if r["max_entropy_pct"] is not None)
            print(f"  [{MODEL_SHORT}] {i+1}/{len(all_questions)}: {parsed} parsed")
    except Exception as e:
        errors += 1
        if errors <= 5: print(f"  [{MODEL_SHORT}] ERROR: {e.__class__.__name__}: {str(e)[:80]}")
    time.sleep(0.1)

out_path = RESULTS_DIR / "maxent_qwen3_235b_base.json"
with open(out_path, "w") as f: json.dump(results, f, indent=2)
parsed = sum(1 for r in results if r["max_entropy_pct"] is not None)
print(f"\n[{MODEL_SHORT}] Saved {len(results)} records ({parsed} parsed, {errors} errors)")
d = [r["max_entropy_pct"] for r in results if r["group"]=="DISRUPTABLE" and r["max_entropy_pct"] is not None]
s = [r["max_entropy_pct"] for r in results if r["group"]=="STABLE" and r["max_entropy_pct"] is not None]
if d and s: print(f"  DISRUPTABLE: {np.mean(d):.4f} (N={len(d)})  STABLE: {np.mean(s):.4f} (N={len(s)})  Delta: {np.mean(d)-np.mean(s):+.4f}")
