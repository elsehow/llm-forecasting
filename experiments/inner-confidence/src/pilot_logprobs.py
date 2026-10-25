"""
Pilot: Extract token-level entropy on CivBench quantile outputs.
Tests whether overconfidence is in the model's beliefs or the decoding layer.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/inner-confidence/src/pilot_logprobs.py
"""

import json
import math
import os
import re
from pathlib import Path

import litellm
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

CIVBENCH = Path("/Users/elsehow/Projects/civbench")
SEED = "seed0"
DATA_DIR = CIVBENCH / "data/conditional/republic/baseline" / SEED
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")

MODEL = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"

DISRUPTABLE = {
    "population_continuous", "territory_continuous",
    "treasury_continuous", "cities_count_continuous",
}
STABLE = {"techs_continuous"}

# Run H4-H7 (far horizons where overconfidence peaks)
TARGET_HORIZONS = {"H4", "H5", "H6", "H7"}


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_data():
    with open(DATA_DIR / "questions.json") as f:
        qdata = json.load(f)
    questions = [
        q for q in qdata["questions"]
        if q["question_type"] == "continuous"
        and q["horizon"] in TARGET_HORIZONS
        and q["template_id"] in (DISRUPTABLE | STABLE)
    ]
    world_report = (DATA_DIR / "world_report" / "turn_060_report.txt").read_text()
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
    """Compute entropy from a dict of {token: logprob}.

    Uses renormalized probabilities over the top-k tokens.
    This is a lower bound on true entropy.
    """
    if not top_lp_dict:
        return None
    logprobs = list(top_lp_dict.values())
    # Convert to probabilities
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return None
    # Renormalize
    probs = [p / total for p in probs]
    # Entropy in nats
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return ent


def extract_quantile_entropies(tokens, token_logprobs, top_logprobs, response_text):
    """Extract entropy for each quantile value in the response.

    Returns list of dicts: {question_idx, quantile, declared_value, token_entropy, top_token_logprob}
    """
    results = []

    # Find the PERCENTILES block
    block_match = re.search(
        r'<<<PERCENTILES?>>>(.*?)<<<END>>>',
        response_text, re.DOTALL | re.IGNORECASE
    )
    if not block_match:
        return results

    block = block_match.group(1).strip()

    # Parse each line
    for line_idx, line in enumerate(block.strip().split('\n')):
        line = line.strip()
        if not line:
            continue

        # Extract quantile values from this line
        pmatches = list(re.finditer(
            r'p(\d+)\s*=\s*(-?\d+\.?\d*)', line, re.IGNORECASE
        ))
        for pmatch in pmatches:
            quantile = f"p{pmatch.group(1)}"
            declared_value = float(pmatch.group(2))
            # Find the numeric value's position in the full response
            value_str = pmatch.group(2)
            value_start_in_block = pmatch.start(2)

            # Now find corresponding tokens
            # Build character-to-token mapping from the token list
            # The response_text is the concatenation of all tokens
            char_pos = 0
            token_char_ranges = []
            for i, tok in enumerate(tokens):
                tok_start = char_pos
                tok_end = char_pos + len(tok)
                token_char_ranges.append((tok_start, tok_end))
                char_pos = tok_end

            # Find the position of the value string in the full response
            # Search for the PERCENTILES block start in response
            block_start_in_response = response_text.find(block)
            if block_start_in_response < 0:
                continue

            # Position of this line in the response
            line_start = response_text.find(line, block_start_in_response)
            if line_start < 0:
                continue

            # Absolute position of the numeric value in the response
            abs_value_start = line_start + pmatch.start(2)
            abs_value_end = abs_value_start + len(value_str)

            # Find tokens that overlap with this position
            value_token_indices = []
            for i, (ts, te) in enumerate(token_char_ranges):
                if te > abs_value_start and ts < abs_value_end:
                    value_token_indices.append(i)

            if not value_token_indices:
                continue

            # Compute entropy across the value tokens
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
                    "mean_token_entropy": np.mean(entropies),
                    "max_token_entropy": np.max(entropies),
                    "mean_token_logprob": np.mean(logprobs_list) if logprobs_list else None,
                    "n_tokens": len(value_token_indices),
                    "token_strings": [tokens[ti] for ti in value_token_indices],
                })

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load API key
    for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
        if line.startswith("TOGETHERAI_API_KEY="):
            os.environ["TOGETHERAI_API_KEY"] = line.split("=", 1)[1]
            break

    questions, world_report = load_data()
    prompt = build_prompt(questions, world_report)

    print(f"Loaded {len(questions)} questions (H4-H7, disruptable + techs)")
    print(f"Model: {MODEL}")
    print(f"Prompt: {len(prompt)} chars")
    print()

    # Call with logprobs
    print("Calling model with logprobs...")
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
    )

    content = response.choices[0].message.content or ""
    logprobs_data = response.choices[0].logprobs
    usage = response.usage
    print(f"Tokens: {usage.total_tokens} (prompt={usage.prompt_tokens}, completion={usage.completion_tokens})")
    print(f"Content: {len(content)} chars")
    print()

    # Extract token data
    lp_dict = logprobs_data.model_dump()
    tokens = lp_dict.get("tokens", [])
    token_logprobs = lp_dict.get("token_logprobs", [])
    top_lps = lp_dict.get("top_logprobs", [])

    print(f"Tokens in response: {len(tokens)}")

    # Extract quantile entropies
    quantile_data = extract_quantile_entropies(tokens, token_logprobs, top_lps, content)
    print(f"Extracted {len(quantile_data)} quantile values with entropy")
    print()

    if not quantile_data:
        print("No quantile data extracted. Showing response tail:")
        print(content[-500:])
        return

    # Map question indices to templates
    for qd in quantile_data:
        qi = qd["question_idx"]
        if qi < len(questions):
            q = questions[qi]
            qd["template_id"] = q["template_id"]
            qd["horizon"] = q["horizon"]
            qd["ground_truth"] = q["resolution"]["value_at_resolution"]
            qd["group"] = "STABLE" if q["template_id"] in STABLE else "DISRUPTABLE"
        else:
            qd["template_id"] = "unknown"
            qd["group"] = "unknown"

    # ── Results ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("ENTROPY BY TEMPLATE GROUP AND QUANTILE")
    print("=" * 80)
    print()

    # Group by (group, quantile)
    from collections import defaultdict
    grouped = defaultdict(list)
    for qd in quantile_data:
        grouped[(qd["group"], qd["quantile"])].append(qd["mean_token_entropy"])

    print(f"{'Group':12s} {'Quantile':8s} {'Mean Entropy':>12s} {'N':>4s}")
    print("-" * 40)
    for (group, quantile), entropies in sorted(grouped.items()):
        print(f"{group:12s} {quantile:8s} {np.mean(entropies):12.4f} {len(entropies):4d}")

    print()

    # Summary: disruptable vs stable
    disrupt_ent = [qd["mean_token_entropy"] for qd in quantile_data if qd["group"] == "DISRUPTABLE"]
    stable_ent = [qd["mean_token_entropy"] for qd in quantile_data if qd["group"] == "STABLE"]

    if disrupt_ent and stable_ent:
        print(f"DISRUPTABLE mean entropy: {np.mean(disrupt_ent):.4f} (N={len(disrupt_ent)})")
        print(f"STABLE mean entropy:      {np.mean(stable_ent):.4f} (N={len(stable_ent)})")
        print(f"Difference:               {np.mean(disrupt_ent) - np.mean(stable_ent):+.4f}")
        print()

    # Per-template detail
    print("Per-template mean entropy:")
    tmpl_ent = defaultdict(list)
    for qd in quantile_data:
        tmpl_ent[qd.get("template_id", "?")].append(qd["mean_token_entropy"])
    for tmpl, ents in sorted(tmpl_ent.items()):
        tag = "S" if tmpl in STABLE else "D"
        print(f"  {tmpl:30s} [{tag}] entropy={np.mean(ents):.4f} (N={len(ents)})")

    print()

    # Cascade test: entropy by quantile position within response
    print("Entropy by quantile level (cascade test):")
    ql_ent = defaultdict(list)
    for qd in quantile_data:
        ql_ent[qd["quantile"]].append(qd["mean_token_entropy"])
    for ql in ["p10", "p25", "p50", "p75", "p90"]:
        if ql in ql_ent:
            print(f"  {ql}: entropy={np.mean(ql_ent[ql]):.4f} (N={len(ql_ent[ql])})")

    # Save raw data
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pilot_logprobs.json"
    # Convert numpy types for JSON
    for qd in quantile_data:
        for k, v in qd.items():
            if isinstance(v, (np.floating, np.integer)):
                qd[k] = float(v)
    with open(out_path, "w") as f:
        json.dump(quantile_data, f, indent=2)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
