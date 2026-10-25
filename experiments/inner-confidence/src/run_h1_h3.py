"""Run inner confidence extraction for H1-H3 (close horizons) to complement existing H4-H7 data."""
import json, math, os, re, time
from collections import defaultdict
from pathlib import Path
import litellm
import numpy as np

CIVBENCH = Path("/Users/elsehow/Projects/civbench")
DATA_ROOT = CIVBENCH / "data/conditional/republic/baseline"
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")
MODELS = [
    ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-70B-Instruct-Turbo"),
    ("together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo", "Qwen2.5-7B-Instruct-Turbo"),
]
DISRUPTABLE = {"population_continuous","territory_continuous","treasury_continuous","cities_count_continuous"}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
TARGET_HORIZONS = {"H1","H2","H3"}

for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
    if line.startswith("TOGETHERAI_API_KEY="):
        os.environ["TOGETHERAI_API_KEY"] = line.split("=", 1)[1]
        break

def entropy_from_top_logprobs(top_lp_dict):
    if not top_lp_dict: return None
    probs = [math.exp(lp) for lp in top_lp_dict.values()]
    total = sum(probs)
    if total <= 0: return None
    probs = [p/total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)

def extract_quantile_entropies(tokens, token_logprobs, top_logprobs, response_text):
    results = []
    block_match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', response_text, re.DOTALL|re.IGNORECASE)
    if not block_match: return results
    block = block_match.group(1).strip()
    char_pos = 0
    token_char_ranges = []
    for tok in tokens:
        token_char_ranges.append((char_pos, char_pos + len(tok)))
        char_pos += len(tok)
    block_start = response_text.find(block)
    if block_start < 0: return results
    for line_idx, line in enumerate(block.strip().split('\n')):
        line = line.strip()
        if not line: continue
        for pmatch in re.finditer(r'p(\d+)\s*=\s*(-?\d+\.?\d*)', line, re.IGNORECASE):
            quantile = f'p{pmatch.group(1)}'
            declared_value = float(pmatch.group(2))
            value_str = pmatch.group(2)
            line_start = response_text.find(line, block_start)
            if line_start < 0: continue
            abs_start = line_start + pmatch.start(2)
            abs_end = abs_start + len(value_str)
            indices = [i for i,(ts,te) in enumerate(token_char_ranges) if te > abs_start and ts < abs_end]
            if not indices: continue
            ents = [e for i in indices if i < len(top_logprobs) and top_logprobs[i] for e in [entropy_from_top_logprobs(top_logprobs[i])] if e is not None]
            lps = [token_logprobs[i] for i in indices if i < len(token_logprobs)]
            if ents:
                results.append({'question_idx':line_idx,'quantile':quantile,'declared_value':declared_value,
                    'mean_token_entropy':float(np.mean(ents)),'n_tokens':len(indices),
                    'mean_token_logprob':float(np.mean(lps)) if lps else None})
    return results

def build_prompt(questions, world_report):
    q_text = '\n'.join(f'{i+1}. {q["question_text"]}' for i, q in enumerate(questions))
    n = len(questions)
    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES.

## World Report
{world_report}

## Questions
{q_text}

You may analyze the data, but you MUST end your response with percentile estimates in this exact format:
<<<PERCENTILES>>>
Q1: p10=5, p25=10, p50=15, p75=20, p90=25
Q2: p10=100, p25=200, p50=300, p75=400, p90=500
<<<END>>>

Provide one line of percentile estimates per question for all {n} questions, in order.
- p10 means you estimate there is a 10% chance the true value is below this number
- p50 (median) means you estimate there is a 50% chance the true value is below this number
- p90 means you estimate there is a 90% chance the true value is above this number"""

all_results = []
seeds = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("seed")])

for model_id, model_short in MODELS:
    print(f"══ {model_short} ══")
    for seed in seeds:
        qpath = DATA_ROOT / seed / "questions.json"
        if not qpath.exists(): continue
        with open(qpath) as f: qdata = json.load(f)
        questions = [q for q in qdata["questions"]
                     if q["question_type"]=="continuous" and q.get("horizon") in TARGET_HORIZONS
                     and q["template_id"] in ALL_TEMPLATES]
        if not questions: continue
        world_report = (DATA_ROOT / seed / "world_report" / "turn_060_report.txt").read_text()
        prompt = build_prompt(questions, world_report)
        try:
            r = litellm.completion(model=model_id, messages=[{"role":"user","content":prompt}],
                                   max_tokens=4000, temperature=0.0, logprobs=True, top_logprobs=5)
            content = r.choices[0].message.content or ""
            lp = r.choices[0].logprobs.model_dump()
            qd = extract_quantile_entropies(lp.get("tokens",[]), lp.get("token_logprobs",[]), lp.get("top_logprobs",[]), content)
            for d in qd:
                qi = d["question_idx"]
                if qi < len(questions):
                    q = questions[qi]
                    d["template_id"] = q["template_id"]
                    d["horizon"] = q["horizon"]
                    d["ground_truth"] = q["resolution"]["value_at_resolution"]
                    d["group"] = "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE"
                d["model"] = model_short
                d["seed"] = seed
            all_results.extend(qd)
            print(f"  {seed}: {len(qd)} values")
        except Exception as e:
            print(f"  {seed}: ERROR {e}")
        time.sleep(0.1)
    print()

# Save
out_path = RESULTS_DIR / "h1_h3_logprobs.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved {len(all_results)} records to {out_path}")

# Quick analysis: entropy by horizon
print("\n── Entropy by horizon (H1-H3, pooled across models) ──")
by_h = defaultdict(list)
for r in all_results:
    by_h[r.get("horizon","?")].append(r["mean_token_entropy"])
for h in ["H1","H2","H3"]:
    if h in by_h:
        print(f"  {h}: entropy = {np.mean(by_h[h]):.4f} (N={len(by_h[h])})")

# Load H4-H7 data for comparison
print("\n── Entropy by horizon (H4-H7, from existing data) ──")
with open(RESULTS_DIR / "full_logprobs.json") as f:
    h47_llama = json.load(f)
with open(RESULTS_DIR / "qwen25_7b_logprobs.json") as f:
    h47_qwen = json.load(f)
h47_all = h47_llama + h47_qwen
by_h47 = defaultdict(list)
for r in h47_all:
    by_h47[r.get("horizon","?")].append(r["mean_token_entropy"])
for h in ["H4","H5","H6","H7"]:
    if h in by_h47:
        print(f"  {h}: entropy = {np.mean(by_h47[h]):.4f} (N={len(by_h47[h])})")

# Full H1-H7 comparison
print("\n── Full H1-H7 comparison ──")
combined = defaultdict(list)
for r in all_results:
    combined[r.get("horizon","?")].append(r["mean_token_entropy"])
for r in h47_all:
    combined[r.get("horizon","?")].append(r["mean_token_entropy"])
for h in ["H1","H2","H3","H4","H5","H6","H7"]:
    if h in combined:
        print(f"  {h}: entropy = {np.mean(combined[h]):.4f} (N={len(combined[h])})")

# By horizon × group
print("\n── Entropy by horizon × group (H1-H7) ──")
by_hg = defaultdict(list)
for r in all_results + h47_all:
    by_hg[(r.get("horizon","?"), r.get("group","?"))].append(r["mean_token_entropy"])
print(f"  {'Horizon':8s} {'DISRUPTABLE':>12s} {'STABLE':>12s}")
for h in ["H1","H2","H3","H4","H5","H6","H7"]:
    d = by_hg.get((h,"DISRUPTABLE"), [])
    s = by_hg.get((h,"STABLE"), [])
    dm = f"{np.mean(d):.4f}" if d else "—"
    sm = f"{np.mean(s):.4f}" if s else "—"
    print(f"  {h:8s} {dm:>12s} {sm:>12s}")
