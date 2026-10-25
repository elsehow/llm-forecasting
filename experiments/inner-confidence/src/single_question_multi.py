"""
Inner confidence: one question per prompt, 4 new models.
Parallel-safe: pass model_index as argv[1].

Usage:
    uv run python experiments/inner-confidence/src/single_question_multi.py 0  # GPT-4.1
    uv run python experiments/inner-confidence/src/single_question_multi.py 1  # GPT-4o
    uv run python experiments/inner-confidence/src/single_question_multi.py 2  # Grok-4 non-reasoning
    uv run python experiments/inner-confidence/src/single_question_multi.py 3  # Grok-4 reasoning
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
    ("gpt-4.1-2025-04-14", "GPT-4.1"),
    ("gpt-4o", "GPT-4o"),
    ("xai/grok-4-fast-non-reasoning", "Grok-4-non-reasoning"),
    ("xai/grok-4-fast-reasoning", "Grok-4-reasoning"),
]

# 10 seeds for statistical power without full cost
TARGET_SEEDS = {"seed0", "seed1", "seed2", "seed3", "seed4",
                "seed5", "seed6", "seed7", "seed8", "seed9"}

DISRUPTABLE = {"population_continuous", "territory_continuous",
               "treasury_continuous", "cities_count_continuous"}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
ALL_HORIZONS = {"H1", "H2", "H3", "H4", "H5", "H6"}

# Load API keys
for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
    k, _, v = line.partition("=")
    if k in ("OPENAI_API_KEY", "XAI_API_KEY", "TOGETHERAI_API_KEY"):
        os.environ[k] = v


def entropy_from_top_logprobs_openai(top_logprobs_list):
    """OpenAI format: list of TopLogprob objects with .token and .logprob."""
    if not top_logprobs_list:
        return None
    probs = [math.exp(t.logprob) for t in top_logprobs_list]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def entropy_from_top_logprobs_dict(top_lp_dict):
    """Together AI format: dict of {token: logprob}."""
    if not top_lp_dict:
        return None
    probs = [math.exp(lp) for lp in top_lp_dict.values()]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def extract_entropies(logprobs_obj, response_text):
    """Extract entropy for each quantile value, handling both API formats."""
    results = []
    block_match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', response_text, re.DOTALL | re.IGNORECASE)
    if not block_match:
        return results

    block = block_match.group(1).strip()

    # Determine format
    if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
        # OpenAI format: logprobs.content is a list of TokenLogprob objects
        tokens = [t.token for t in logprobs_obj.content]
        top_lps = logprobs_obj.content  # keep as objects
        fmt = "openai"
    else:
        d = logprobs_obj.model_dump()
        tokens = d.get("tokens", [])
        top_lps = d.get("top_logprobs", [])
        fmt = "together"

    if not tokens:
        return results

    # Build char-to-token mapping
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

        # Find line containing this match
        line_start_in_block = block.rfind('\n', 0, pmatch.start())
        line_start_in_block = line_start_in_block + 1 if line_start_in_block >= 0 else 0
        line = block[line_start_in_block:].split('\n')[0]
        line_pos_in_response = response_text.find(line, block_start)
        if line_pos_in_response < 0:
            continue

        abs_start = line_pos_in_response + (pmatch.start() - line_start_in_block) + len(pmatch.group(0)) - len(value_str)
        abs_end = abs_start + len(value_str)

        indices = [i for i, (ts, te) in enumerate(token_char_ranges)
                   if te > abs_start and ts < abs_end]
        if not indices:
            continue

        ents = []
        for i in indices:
            if i < len(top_lps):
                if fmt == "openai":
                    e = entropy_from_top_logprobs_openai(top_lps[i].top_logprobs)
                else:
                    e = entropy_from_top_logprobs_dict(top_lps[i])
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

    # Collect questions
    all_questions = []
    for seed_dir in sorted(DATA_ROOT.iterdir()):
        if not seed_dir.is_dir() or seed_dir.name not in TARGET_SEEDS:
            continue
        qpath = seed_dir / "questions.json"
        if not qpath.exists():
            continue
        with open(qpath) as f:
            qdata = json.load(f)
        wr_path = seed_dir / "world_report" / "turn_060_report.txt"
        if not wr_path.exists():
            continue
        world_report = wr_path.read_text()
        for q in qdata["questions"]:
            if (q["question_type"] == "continuous"
                    and q["template_id"] in ALL_TEMPLATES
                    and q.get("horizon") in ALL_HORIZONS):
                all_questions.append({
                    "question": q,
                    "world_report": world_report,
                    "seed": seed_dir.name,
                })

    random.seed(42)
    random.shuffle(all_questions)

    print(f"[{model_short}] {len(all_questions)} questions (10 seeds, randomized)")

    all_results = []
    errors = 0

    for i, item in enumerate(all_questions):
        q = item["question"]
        prompt = build_prompt(q, item["world_report"])

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
            lp = r.choices[0].logprobs

            extracted = extract_entropies(lp, content)

            for d in extracted:
                d["template_id"] = q["template_id"]
                d["horizon"] = q["horizon"]
                d["ground_truth"] = q["resolution"]["value_at_resolution"]
                d["group"] = "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE"
                d["model"] = model_short
                d["seed"] = item["seed"]

            all_results.extend(extracted)

            if (i + 1) % 50 == 0:
                print(f"  [{model_short}] {i+1}/{len(all_questions)}: {len(all_results)} values")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{model_short}] ERROR at {i}: {e.__class__.__name__}: {str(e)[:100]}")

        time.sleep(0.05)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_short.lower().replace(" ", "_").replace("-", "_")
    out_path = RESULTS_DIR / f"single_q_{slug}_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{model_short}] Saved {len(all_results)} records to {out_path} ({errors} errors)")

    # Quick analysis
    print(f"\n[{model_short}] ── Template effect ──")
    for group in ["DISRUPTABLE", "STABLE"]:
        vals = [r["mean_token_entropy"] for r in all_results if r.get("group") == group]
        if vals:
            print(f"  {group:12s}: entropy = {np.mean(vals):.4f} (N={len(vals)})")

    print(f"\n[{model_short}] ── Horizon effect (disruptable only) ──")
    by_h = defaultdict(list)
    for r in all_results:
        if r.get("group") == "DISRUPTABLE":
            by_h[r["horizon"]].append(r["mean_token_entropy"])
    for h in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        if h in by_h:
            print(f"  {h}: entropy = {np.mean(by_h[h]):.4f} (N={len(by_h[h])})")


if __name__ == "__main__":
    main()
