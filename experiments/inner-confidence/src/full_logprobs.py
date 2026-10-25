"""
Full inner confidence analysis: 3 models × all seeds × H4-H7.
Extracts token-level entropy on quantile output tokens from CivBench.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/inner-confidence/src/full_logprobs.py
"""

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import litellm
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

CIVBENCH = Path("/Users/elsehow/Projects/civbench")
DATA_ROOT = CIVBENCH / "data/conditional/republic/baseline"
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")

MODELS = [
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/deepseek-ai/DeepSeek-V3.1",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
]

DISRUPTABLE = {
    "population_continuous", "territory_continuous",
    "treasury_continuous", "cities_count_continuous",
}
STABLE = {"techs_continuous"}
ALL_TEMPLATES = DISRUPTABLE | STABLE
TARGET_HORIZONS = {"H4", "H5", "H6", "H7"}


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_seeds():
    seeds = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed"):
            continue
        qpath = d / "questions.json"
        if not qpath.exists():
            continue
        with open(qpath) as f:
            data = json.load(f)
        cont = [
            q for q in data["questions"]
            if q["question_type"] == "continuous"
            and q.get("horizon") in TARGET_HORIZONS
            and q["template_id"] in ALL_TEMPLATES
        ]
        if cont:
            seeds.append(d.name)
    return seeds


def load_seed_data(seed):
    data_dir = DATA_ROOT / seed
    with open(data_dir / "questions.json") as f:
        qdata = json.load(f)
    questions = [
        q for q in qdata["questions"]
        if q["question_type"] == "continuous"
        and q.get("horizon") in TARGET_HORIZONS
        and q["template_id"] in ALL_TEMPLATES
    ]
    world_report = (data_dir / "world_report" / "turn_060_report.txt").read_text()
    return questions, world_report


def build_prompt(questions, world_report):
    q_text = "\n".join(
        f"{i+1}. {q['question_text']}" for i, q in enumerate(questions)
    )
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
- p10 means you estimate there's a 10% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is above this number"""


def entropy_from_top_logprobs(top_lp_dict):
    if not top_lp_dict:
        return None
    logprobs = list(top_lp_dict.values())
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def extract_quantile_entropies(tokens, token_logprobs, top_logprobs, response_text):
    results = []
    block_match = re.search(
        r'<<<PERCENTILES?>>>(.*?)<<<END>>>',
        response_text, re.DOTALL | re.IGNORECASE
    )
    if not block_match:
        return results

    block = block_match.group(1).strip()

    # Build char-to-token mapping
    char_pos = 0
    token_char_ranges = []
    for i, tok in enumerate(tokens):
        tok_start = char_pos
        tok_end = char_pos + len(tok)
        token_char_ranges.append((tok_start, tok_end))
        char_pos = tok_end

    block_start_in_response = response_text.find(block)
    if block_start_in_response < 0:
        return results

    for line_idx, line in enumerate(block.strip().split('\n')):
        line = line.strip()
        if not line:
            continue

        pmatches = list(re.finditer(
            r'p(\d+)\s*=\s*(-?\d+\.?\d*)', line, re.IGNORECASE
        ))
        for pmatch in pmatches:
            quantile = f"p{pmatch.group(1)}"
            declared_value = float(pmatch.group(2))
            value_str = pmatch.group(2)

            line_start = response_text.find(line, block_start_in_response)
            if line_start < 0:
                continue

            abs_value_start = line_start + pmatch.start(2)
            abs_value_end = abs_value_start + len(value_str)

            value_token_indices = []
            for i, (ts, te) in enumerate(token_char_ranges):
                if te > abs_value_start and ts < abs_value_end:
                    value_token_indices.append(i)

            if not value_token_indices:
                continue

            entropies = []
            logprobs_list = []
            for ti in value_token_indices:
                if ti < len(top_logprobs) and top_logprobs[ti]:
                    ent = entropy_from_top_logprobs(top_logprobs[ti])
                    if ent is not None:
                        entropies.append(ent)
                if ti < len(token_logprobs):
                    logprobs_list.append(token_logprobs[ti])

            if entropies:
                results.append({
                    "question_idx": line_idx,
                    "quantile": quantile,
                    "declared_value": declared_value,
                    "mean_token_entropy": float(np.mean(entropies)),
                    "max_token_entropy": float(np.max(entropies)),
                    "mean_token_logprob": float(np.mean(logprobs_list)) if logprobs_list else None,
                    "n_tokens": len(value_token_indices),
                })

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
        if line.startswith("TOGETHERAI_API_KEY="):
            os.environ["TOGETHERAI_API_KEY"] = line.split("=", 1)[1]
            break

    seeds = get_seeds()
    print(f"Seeds with data: {len(seeds)}")
    print(f"Models: {len(MODELS)}")
    print(f"Expected calls: {len(seeds)} seeds × {len(MODELS)} models = {len(seeds) * len(MODELS)}")
    print()

    all_results = []
    errors = []

    for model_id in MODELS:
        model_short = model_id.split("/")[-1]
        print(f"══ {model_short} ══")

        for seed in seeds:
            questions, world_report = load_seed_data(seed)
            prompt = build_prompt(questions, world_report)

            try:
                response = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=5,
                )

                content = response.choices[0].message.content or ""
                lp_data = response.choices[0].logprobs.model_dump()
                tokens = lp_data.get("tokens", [])
                token_lps = lp_data.get("token_logprobs", [])
                top_lps = lp_data.get("top_logprobs", [])

                quantile_data = extract_quantile_entropies(tokens, token_lps, top_lps, content)

                # Annotate with metadata
                for qd in quantile_data:
                    qi = qd["question_idx"]
                    if qi < len(questions):
                        q = questions[qi]
                        qd["template_id"] = q["template_id"]
                        qd["horizon"] = q["horizon"]
                        qd["ground_truth"] = q["resolution"]["value_at_resolution"]
                        qd["group"] = "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE"
                    qd["model"] = model_short
                    qd["seed"] = seed

                all_results.extend(quantile_data)
                n_parsed = len(quantile_data)
                print(f"  {seed}: {n_parsed} quantile values extracted")

            except Exception as e:
                print(f"  {seed}: ERROR — {e.__class__.__name__}: {e}")
                errors.append({"model": model_short, "seed": seed, "error": str(e)})

            time.sleep(0.1)  # Light rate limiting

        print()

    # ── Save ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "full_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} records to {out_path}")
    if errors:
        err_path = RESULTS_DIR / "full_logprobs_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {err_path}")

    # ── Analysis ────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Per-model, group × quantile
    for model_short in [m.split("/")[-1] for m in MODELS]:
        model_data = [r for r in all_results if r["model"] == model_short]
        if not model_data:
            print(f"\n{model_short}: NO DATA")
            continue

        print(f"\n── {model_short} ({len(model_data)} values) ──")

        # Group comparison
        for group in ["DISRUPTABLE", "STABLE"]:
            ents = [r["mean_token_entropy"] for r in model_data if r.get("group") == group]
            if ents:
                print(f"  {group:12s}: mean entropy = {np.mean(ents):.4f} (N={len(ents)})")

        # Cascade
        print("  Quantile cascade:")
        for ql in ["p10", "p25", "p50", "p75", "p90"]:
            ents = [r["mean_token_entropy"] for r in model_data if r["quantile"] == ql]
            if ents:
                print(f"    {ql}: entropy = {np.mean(ents):.4f} (N={len(ents)})")

    # Pooled across models
    print(f"\n── POOLED ({len(all_results)} values) ──")
    for group in ["DISRUPTABLE", "STABLE"]:
        ents = [r["mean_token_entropy"] for r in all_results if r.get("group") == group]
        if ents:
            print(f"  {group:12s}: mean entropy = {np.mean(ents):.4f} (N={len(ents)})")

    print("  Quantile cascade (pooled):")
    for ql in ["p10", "p25", "p50", "p75", "p90"]:
        ents = [r["mean_token_entropy"] for r in all_results if r["quantile"] == ql]
        if ents:
            print(f"    {ql}: entropy = {np.mean(ents):.4f} (N={len(ents)})")


if __name__ == "__main__":
    main()
