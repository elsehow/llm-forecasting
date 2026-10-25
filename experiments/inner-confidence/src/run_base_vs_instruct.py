"""
RLHF gradient experiment: Llama-3-8B base vs instruct on Fireworks.
Base model uses completions API; instruct uses chat API.
Both return logprobs.

Usage:
    uv run python experiments/inner-confidence/src/run_base_vs_instruct.py 0  # base
    uv run python experiments/inner-confidence/src/run_base_vs_instruct.py 1  # instruct
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

import numpy as np
import openai

CIVBENCH = Path("/Users/elsehow/Projects/civbench")
DATA_ROOT = CIVBENCH / "data/conditional/republic/baseline"
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")

MODELS = [
    {
        "id": "accounts/nick-p6n6002bmkx4/deployments/qwx45n9m",
        "short": "Llama-3-8B-base",
        "is_base": True,
    },
    {
        "id": "accounts/nick-p6n6002bmkx4/deployments/xagzl5tj",
        "short": "Llama-3-8B-Instruct",
        "is_base": False,
    },
]

TARGET_SEEDS = {"seed0", "seed1", "seed2", "seed3", "seed4"}
DISRUPTABLE = {"population_continuous", "territory_continuous",
               "treasury_continuous", "cities_count_continuous"}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
ALL_HORIZONS = {"H1", "H2", "H3", "H4", "H5", "H6"}

API_KEY = ""
for env_path in [Path("/Users/elsehow/Projects/civbench/.env")]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="):
                API_KEY = line.split("=", 1)[1]


def entropy_from_top5_chat(top_logprobs):
    """OpenAI chat format: list of TopLogprob objects."""
    if not top_logprobs:
        return None
    probs = [math.exp(t.logprob) for t in top_logprobs]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def entropy_from_top5_completion(top_lp_dict):
    """Completions format: dict {token: logprob}."""
    if not top_lp_dict:
        return None
    probs = [math.exp(lp) for lp in top_lp_dict.values()]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def build_chat_prompt(question, world_report):
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


def build_completion_prompt(question, world_report):
    """Few-shot prompt for base model via completions API."""
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


def extract_pct_entropies_chat(logprobs_obj, response_text):
    """Extract max entropy from PERCENTILES block tokens (chat format)."""
    if not logprobs_obj or not logprobs_obj.content:
        return None
    tokens = [t.token for t in logprobs_obj.content]
    text = "".join(tokens)
    block_match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', text, re.DOTALL | re.IGNORECASE)
    if not block_match:
        return None
    bs, be = block_match.start(), block_match.end()
    pct_ents = []
    cp = 0
    for i, tok in enumerate(tokens):
        ts, te = cp, cp + len(tok)
        if ts >= bs and te <= be:
            e = entropy_from_top5_chat(logprobs_obj.content[i].top_logprobs)
            if e is not None:
                pct_ents.append(e)
        cp = te
    return max(pct_ents) if pct_ents else None


def extract_pct_entropies_completion(logprobs_obj, response_text):
    """Extract max entropy from PERCENTILES block tokens (completions format)."""
    tokens = logprobs_obj.tokens or []
    top_lps = logprobs_obj.top_logprobs or []
    if not tokens:
        return None
    # The completion starts INSIDE the PERCENTILES block (prompt ends with <<<PERCENTILES>>>\n)
    # So all tokens are percentile tokens until <<<END>>>
    pct_ents = []
    text = "".join(tokens)
    end_pos = text.find("<<<END>>>")
    cp = 0
    for i, tok in enumerate(tokens):
        ts, te = cp, cp + len(tok)
        if end_pos >= 0 and ts >= end_pos:
            break
        if i < len(top_lps) and top_lps[i]:
            e = entropy_from_top5_completion(top_lps[i])
            if e is not None:
                pct_ents.append(e)
        cp = te
    return max(pct_ents) if pct_ents else None


def main():
    model_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_cfg = MODELS[model_idx]
    model_id = model_cfg["id"]
    model_short = model_cfg["short"]
    is_base = model_cfg["is_base"]

    client = openai.OpenAI(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")

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
                    "question": q, "world_report": world_report, "seed": seed_dir.name,
                })

    random.seed(42)
    random.shuffle(all_questions)
    print(f"[{model_short}] {len(all_questions)} questions, is_base={is_base}")

    results = []
    errors = 0

    for i, item in enumerate(all_questions):
        q = item["question"]
        try:
            if is_base:
                prompt = build_completion_prompt(q, item["world_report"])
                r = client.completions.create(
                    model=model_id, prompt=prompt,
                    max_tokens=200, temperature=0.0, logprobs=5,
                )
                content = r.choices[0].text or ""
                max_ent = extract_pct_entropies_completion(r.choices[0].logprobs, content)
            else:
                prompt = build_chat_prompt(q, item["world_report"])
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000, temperature=0.0,
                    logprobs=True, top_logprobs=5,
                )
                content = r.choices[0].message.content or ""
                max_ent = extract_pct_entropies_chat(r.choices[0].logprobs, content)

            results.append({
                "template_id": q["template_id"],
                "horizon": q["horizon"],
                "ground_truth": q["resolution"]["value_at_resolution"],
                "group": "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE",
                "model": model_short,
                "seed": item["seed"],
                "max_entropy_pct": max_ent,
                "has_percentiles": "<<<PERCENTILES" in content.upper() or (is_base and max_ent is not None),
            })

            if (i + 1) % 25 == 0:
                parsed = sum(1 for r in results if r["max_entropy_pct"] is not None)
                print(f"  [{model_short}] {i+1}/{len(all_questions)}: {parsed} parsed")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{model_short}] ERROR: {e.__class__.__name__}: {str(e)[:80]}")

        time.sleep(0.05)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_short.lower().replace(" ", "_").replace("-", "_")
    out_path = RESULTS_DIR / f"maxent_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    parsed = sum(1 for r in results if r["max_entropy_pct"] is not None)
    print(f"\n[{model_short}] Saved {len(results)} records ({parsed} parsed, {errors} errors)")

    d = [r["max_entropy_pct"] for r in results if r["group"] == "DISRUPTABLE" and r["max_entropy_pct"] is not None]
    s = [r["max_entropy_pct"] for r in results if r["group"] == "STABLE" and r["max_entropy_pct"] is not None]
    if d and s:
        delta = np.mean(d) - np.mean(s)
        print(f"  DISRUPTABLE: {np.mean(d):.4f} (N={len(d)})")
        print(f"  STABLE:      {np.mean(s):.4f} (N={len(s)})")
        print(f"  Delta:       {delta:+.4f}")
    else:
        print(f"  Parsed: D={len(d)}, S={len(s)} — insufficient data")


if __name__ == "__main__":
    main()
