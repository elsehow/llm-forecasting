"""
Compare two entropy measures:
1. max(entropy) across tokens in PERCENTILES block only
2. max(entropy) across ALL tokens in the response

Runs 5 models × 5 seeds × 30 questions = 750 API calls.
Generates mechanism_paired.png and mechanism_paired_alltokens.png.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/inner-confidence/src/max_entropy_comparison.py [model_index]
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
    ("gpt-4.1-2025-04-14", "GPT-4.1"),
    ("gpt-4o", "GPT-4o"),
    ("xai/grok-4-fast-non-reasoning", "Grok-4-non-reasoning"),
]

TARGET_SEEDS = {"seed0", "seed1", "seed2", "seed3", "seed4"}
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


def entropy_from_top5(top_logprobs):
    """Compute entropy from top-5 logprobs. Handles both API formats."""
    if not top_logprobs:
        return None
    # Together format: dict {token_str: logprob_float}
    if isinstance(top_logprobs, dict):
        probs = [math.exp(lp) for lp in top_logprobs.values()]
    # OpenAI format: list of objects with .logprob
    elif isinstance(top_logprobs, list) and hasattr(top_logprobs[0], 'logprob'):
        probs = [math.exp(t.logprob) for t in top_logprobs]
    else:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def get_all_token_entropies(logprobs_obj, response_text):
    """Extract entropy for every token in the response.
    Returns (all_entropies, percentile_block_entropies)."""

    all_ents = []
    pct_ents = []

    # Determine format
    if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
        # OpenAI format
        tokens = [t.token for t in logprobs_obj.content]
        for t in logprobs_obj.content:
            e = entropy_from_top5(t.top_logprobs)
            all_ents.append(e)
    else:
        d = logprobs_obj.model_dump()
        tokens = d.get("tokens", [])
        top_lps = d.get("top_logprobs", [])
        for i, tok in enumerate(tokens):
            if i < len(top_lps) and top_lps[i]:
                e = entropy_from_top5(top_lps[i]) if isinstance(top_lps[i], dict) else None
                all_ents.append(e)
            else:
                all_ents.append(None)

    if not tokens:
        return [], []

    # Find PERCENTILES block
    text = "".join(tokens)
    block_match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', text, re.DOTALL | re.IGNORECASE)
    if block_match:
        block_start = block_match.start()
        block_end = block_match.end()
        # Map character positions to token indices
        char_pos = 0
        for i, tok in enumerate(tokens):
            tok_start = char_pos
            tok_end = char_pos + len(tok)
            if tok_start >= block_start and tok_end <= block_end:
                if i < len(all_ents) and all_ents[i] is not None:
                    pct_ents.append(all_ents[i])
            char_pos = tok_end

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

    print(f"[{model_short}] {len(all_questions)} questions")

    results = []
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

            all_ents, pct_ents = get_all_token_entropies(lp, content)

            results.append({
                "template_id": q["template_id"],
                "horizon": q["horizon"],
                "ground_truth": q["resolution"]["value_at_resolution"],
                "group": "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE",
                "model": model_short,
                "seed": item["seed"],
                "max_entropy_all": max(all_ents) if all_ents else None,
                "max_entropy_pct": max(pct_ents) if pct_ents else None,
                "n_tokens_all": len(all_ents),
                "n_tokens_pct": len(pct_ents),
            })

            if (i + 1) % 50 == 0:
                print(f"  [{model_short}] {i+1}/{len(all_questions)}")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{model_short}] ERROR: {e.__class__.__name__}: {str(e)[:80]}")

        time.sleep(0.05)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_short.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    out_path = RESULTS_DIR / f"maxent_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{model_short}] Saved {len(results)} records to {out_path} ({errors} errors)")

    # Quick analysis
    for method in ["max_entropy_pct", "max_entropy_all"]:
        label = "PERCENTILES only" if "pct" in method else "ALL tokens"
        print(f"\n[{model_short}] ── {label} ──")
        for group in ["DISRUPTABLE", "STABLE"]:
            vals = [r[method] for r in results if r["group"] == group and r[method] is not None]
            if vals:
                print(f"  {group:12s}: {np.mean(vals):.4f} (N={len(vals)})")
        d = [r[method] for r in results if r["group"] == "DISRUPTABLE" and r[method] is not None]
        s = [r[method] for r in results if r["group"] == "STABLE" and r[method] is not None]
        if d and s:
            delta = np.mean(d) - np.mean(s)
            print(f"  Delta: {delta:+.4f}")


if __name__ == "__main__":
    main()
