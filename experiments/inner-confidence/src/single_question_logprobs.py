"""
Inner confidence: one question per prompt, randomized order.
Isolates horizon effect from cascade/position confound.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/inner-confidence/src/single_question_logprobs.py [model_index]
    # model_index: 0 = Llama, 1 = Qwen (for parallel runs)
"""

import json
import math
import os
import random
import re
import sys
import time
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

DISRUPTABLE = {"population_continuous", "territory_continuous",
               "treasury_continuous", "cities_count_continuous"}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
ALL_HORIZONS = {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}

for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
    if line.startswith("TOGETHERAI_API_KEY="):
        os.environ["TOGETHERAI_API_KEY"] = line.split("=", 1)[1]
        break


def entropy_from_top_logprobs(top_lp_dict):
    if not top_lp_dict:
        return None
    probs = [math.exp(lp) for lp in top_lp_dict.values()]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def build_single_question_prompt(question, world_report):
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


def extract_single_entropies(tokens, token_logprobs, top_logprobs, response_text):
    results = []
    block_match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', response_text, re.DOTALL | re.IGNORECASE)
    if not block_match:
        return results

    block = block_match.group(1).strip()
    char_pos = 0
    token_char_ranges = []
    for tok in tokens:
        token_char_ranges.append((char_pos, char_pos + len(tok)))
        char_pos += len(tok)

    block_start = response_text.find(block)
    if block_start < 0:
        return results

    for pmatch in re.finditer(r'p(\d+)\s*=\s*(-?\d+\.?\d*)', block, re.IGNORECASE):
        quantile = f"p{pmatch.group(1)}"
        declared_value = float(pmatch.group(2))
        value_str = pmatch.group(2)

        abs_start = block_start + pmatch.start(2)
        abs_end = abs_start + len(value_str)

        indices = [i for i, (ts, te) in enumerate(token_char_ranges)
                   if te > abs_start and ts < abs_end]
        if not indices:
            continue

        ents = []
        for i in indices:
            if i < len(top_logprobs) and top_logprobs[i]:
                e = entropy_from_top_logprobs(top_logprobs[i])
                if e is not None:
                    ents.append(e)

        if ents:
            results.append({
                "quantile": quantile,
                "declared_value": declared_value,
                "mean_token_entropy": float(np.mean(ents)),
                "n_tokens": len(indices),
            })

    return results


def main():
    model_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_id, model_short = MODELS[model_idx]

    # Collect all questions
    all_questions = []
    for seed_dir in sorted(DATA_ROOT.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        qpath = seed_dir / "questions.json"
        if not qpath.exists():
            continue
        with open(qpath) as f:
            qdata = json.load(f)
        world_report = (seed_dir / "world_report" / "turn_060_report.txt").read_text()
        for q in qdata["questions"]:
            if (q["question_type"] == "continuous"
                    and q["template_id"] in ALL_TEMPLATES
                    and q.get("horizon") in ALL_HORIZONS):
                all_questions.append({
                    "question": q,
                    "world_report": world_report,
                    "seed": seed_dir.name,
                })

    # Randomize order
    random.seed(42)
    random.shuffle(all_questions)

    print(f"Model: {model_short}")
    print(f"Questions: {len(all_questions)} (randomized)")
    print()

    all_results = []
    errors = 0

    for i, item in enumerate(all_questions):
        q = item["question"]
        prompt = build_single_question_prompt(q, item["world_report"])

        try:
            r = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )
            content = r.choices[0].message.content or ""
            lp = r.choices[0].logprobs.model_dump()
            extracted = extract_single_entropies(
                lp.get("tokens", []),
                lp.get("token_logprobs", []),
                lp.get("top_logprobs", []),
                content,
            )

            for d in extracted:
                d["template_id"] = q["template_id"]
                d["horizon"] = q["horizon"]
                d["ground_truth"] = q["resolution"]["value_at_resolution"]
                d["group"] = "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE"
                d["model"] = model_short
                d["seed"] = item["seed"]

            all_results.extend(extracted)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(all_questions)}: {len(all_results)} values extracted")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR at {i}: {e.__class__.__name__}: {e}")

        time.sleep(0.05)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "llama" if model_idx == 0 else "qwen"
    out_path = RESULTS_DIR / f"single_q_{suffix}_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} records to {out_path} ({errors} errors)")

    # Analysis
    print(f"\n── {model_short}: Entropy by horizon ──")
    by_h = defaultdict(list)
    for r in all_results:
        by_h[r["horizon"]].append(r["mean_token_entropy"])
    for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        if h in by_h:
            print(f"  {h}: entropy = {np.mean(by_h[h]):.4f} (N={len(by_h[h])})")

    print(f"\n── Entropy by horizon × group ──")
    by_hg = defaultdict(list)
    for r in all_results:
        by_hg[(r["horizon"], r["group"])].append(r["mean_token_entropy"])
    print(f"  {'Horizon':8s} {'DISRUPTABLE':>12s} {'STABLE':>12s}")
    for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        d = by_hg.get((h, "DISRUPTABLE"), [])
        s = by_hg.get((h, "STABLE"), [])
        dm = f"{np.mean(d):.4f}" if d else "—"
        sm = f"{np.mean(s):.4f}" if s else "—"
        print(f"  {h:8s} {dm:>12s} {sm:>12s}")

    print(f"\n── Entropy by quantile (no cascade expected) ──")
    by_q = defaultdict(list)
    for r in all_results:
        by_q[r["quantile"]].append(r["mean_token_entropy"])
    for ql in ["p10", "p25", "p50", "p75", "p90"]:
        if ql in by_q:
            print(f"  {ql}: entropy = {np.mean(by_q[ql]):.4f} (N={len(by_q[ql])})")


if __name__ == "__main__":
    main()
