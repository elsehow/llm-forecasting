"""Run Qwen3-235B instruct and thinking through max entropy pipeline."""
import json, math, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path
import litellm, numpy as np

CIVBENCH = Path("/Users/elsehow/Projects/civbench")
DATA_ROOT = CIVBENCH / "data/conditional/republic/baseline"
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")

MODELS = [
    ("together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "Qwen3-235B-Instruct"),
    ("together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507", "Qwen3-235B-Thinking"),
]
TARGET_SEEDS = {"seed0", "seed1", "seed2", "seed3", "seed4"}
DISRUPTABLE = {"population_continuous", "territory_continuous", "treasury_continuous", "cities_count_continuous"}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
ALL_HORIZONS = {"H1", "H2", "H3", "H4", "H5", "H6"}

for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
    k, _, v = line.partition("=")
    if k in ("TOGETHERAI_API_KEY",): os.environ[k] = v

def entropy_from_top5(top_logprobs):
    if not top_logprobs: return None
    if isinstance(top_logprobs, dict):
        probs = [math.exp(lp) for lp in top_logprobs.values()]
    elif isinstance(top_logprobs, list) and hasattr(top_logprobs[0], 'logprob'):
        probs = [math.exp(t.logprob) for t in top_logprobs]
    else: return None
    total = sum(probs)
    if total <= 0: return None
    probs = [p/total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)

def get_all_token_entropies(logprobs_obj, response_text):
    all_ents, pct_ents = [], []
    if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
        tokens = [t.token for t in logprobs_obj.content]
        for t in logprobs_obj.content:
            all_ents.append(entropy_from_top5(t.top_logprobs))
    else:
        d = logprobs_obj.model_dump()
        tokens = d.get("tokens", [])
        for i, tok in enumerate(tokens):
            tlps = d.get("top_logprobs", [])
            if i < len(tlps) and tlps[i]:
                all_ents.append(entropy_from_top5(tlps[i]))
            else: all_ents.append(None)
    if not tokens: return [], []
    text = "".join(tokens)
    block_match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', text, re.DOTALL|re.IGNORECASE)
    if block_match:
        bs, be = block_match.start(), block_match.end()
        cp = 0
        for i, tok in enumerate(tokens):
            ts, te = cp, cp + len(tok)
            if ts >= bs and te <= be and i < len(all_ents) and all_ents[i] is not None:
                pct_ents.append(all_ents[i])
            cp = te
    return [e for e in all_ents if e is not None], pct_ents

def build_prompt(question, world_report):
    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

## World Report
{world_report}

## Question
{question['question_text']}

Provide your percentile estimates in this exact format:
<<<PERCENTILES>>>
p10=5, p25=10, p50=15, p75=20, p90=25
<<<END>>>

- p10 means you estimate there's a 10% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is above this number"""

def main():
    model_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_id, model_short = MODELS[model_idx]
    is_thinking = "Thinking" in model_short
    max_tokens = 8000 if is_thinking else 1000

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
            if q["question_type"] == "continuous" and q["template_id"] in ALL_TEMPLATES and q.get("horizon") in ALL_HORIZONS:
                all_questions.append({"question": q, "world_report": world_report, "seed": seed_dir.name})

    random.seed(42)
    random.shuffle(all_questions)
    print(f"[{model_short}] {len(all_questions)} questions (max_tokens={max_tokens})")

    results, errors = [], 0
    for i, item in enumerate(all_questions):
        q = item["question"]
        prompt = build_prompt(q, item["world_report"])
        try:
            r = litellm.completion(model=model_id, messages=[{"role":"user","content":prompt}],
                                   max_tokens=max_tokens, temperature=0.0, logprobs=True, top_logprobs=5)
            content = r.choices[0].message.content or ""
            lp = r.choices[0].logprobs
            all_ents, pct_ents = get_all_token_entropies(lp, content)
            results.append({
                "template_id": q["template_id"], "horizon": q["horizon"],
                "ground_truth": q["resolution"]["value_at_resolution"],
                "group": "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE",
                "model": model_short, "seed": item["seed"],
                "max_entropy_all": max(all_ents) if all_ents else None,
                "max_entropy_pct": max(pct_ents) if pct_ents else None,
            })
            if (i+1) % 50 == 0: print(f"  [{model_short}] {i+1}/{len(all_questions)}")
        except Exception as e:
            errors += 1
            if errors <= 3: print(f"  [{model_short}] ERROR: {e.__class__.__name__}: {str(e)[:80]}")
        time.sleep(0.05)

    slug = model_short.lower().replace(" ","_").replace("-","_").replace(".","_")
    out_path = RESULTS_DIR / f"maxent_{slug}.json"
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[{model_short}] Saved {len(results)} records ({errors} errors)")
    d = [r["max_entropy_pct"] for r in results if r["group"]=="DISRUPTABLE" and r.get("max_entropy_pct") is not None]
    s = [r["max_entropy_pct"] for r in results if r["group"]=="STABLE" and r.get("max_entropy_pct") is not None]
    if d and s: print(f"  DISRUPTABLE: {np.mean(d):.4f}  STABLE: {np.mean(s):.4f}  Delta: {np.mean(d)-np.mean(s):+.4f}")

if __name__ == "__main__": main()
