"""
ForecastBench inner confidence experiment (H2 test).

Tests whether token entropy differs between yfinance questions (volatile/unpredictable)
and Wikipedia questions (stable/knowledge-based).

H1 (capability-as-liability): models more *uncertain* on yfinance (higher entropy)
  => model "knows" it's guessing on volatile questions
H2 (overconfidence): models equally or more *confident* on yfinance (lower entropy)
  => model doesn't know what it doesn't know

No cascade possible here -- just a single probability token per question/horizon.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/inner-confidence/src/forecastbench_logprobs.py
"""

import json
import math
import os
import re
import time
from pathlib import Path

import litellm
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

QUESTION_SETS_DIR = Path("/Users/elsehow/Projects/forecastbench-datasets/datasets/question_sets")
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")

MODELS = [
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
]

# Use two question sets to get ~200 per source
QUESTION_SET_FILES = [
    "2025-03-02-llm.json",
    "2025-03-16-llm.json",
]

MAX_PER_SOURCE = 200
TARGET_SOURCES = {"yfinance", "wikipedia"}

# Use 1-week horizon (first resolution date) for clean single-probability output
TARGET_HORIZON_IDX = 0  # first resolution date = ~1 week

# ForecastBench ZERO_SHOT_NON_MARKET_PROMPT (for data-source questions)
NON_MARKET_PROMPT = """You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

You're going to predict the probability of the following potential outcome "at each of the resolution dates".

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Current value on {freeze_datetime}:
{freeze_datetime_value}

Value Explanation:
{freeze_datetime_value_explanation}

Today's Date: {today_date}

Question resolution dates: {list_of_resolution_dates}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. (For example, if there are n resolution dates, you would output different *p* for each resolution date) Do not output anything else.
Answer: {{ Insert answer here }}
"""


# ── Load questions ──────────────────────────────────────────────────────────

def load_questions():
    """Load yfinance and wikipedia questions from ForecastBench question sets."""
    questions_by_source = {"yfinance": [], "wikipedia": []}
    seen_ids = {"yfinance": set(), "wikipedia": set()}

    for fname in QUESTION_SET_FILES:
        fp = QUESTION_SETS_DIR / fname
        if not fp.exists():
            print(f"  WARNING: {fp} not found, skipping")
            continue

        with open(fp) as f:
            data = json.load(f)

        forecast_due_date = data["forecast_due_date"]

        for q in data["questions"]:
            source = q["source"]
            if source not in TARGET_SOURCES:
                continue
            if len(questions_by_source[source]) >= MAX_PER_SOURCE:
                continue
            # Deduplicate by ID (some IDs may be lists for combination questions)
            qid = str(q["id"])
            if qid in seen_ids[source]:
                continue
            seen_ids[source].add(qid)

            q["_forecast_due_date"] = forecast_due_date
            questions_by_source[source].append(q)

    return questions_by_source


def build_prompt(question):
    """Build the ForecastBench zero-shot non-market prompt."""
    forecast_due_date = question["_forecast_due_date"]

    # Format question text (replace template vars)
    q_text = question["question"]
    q_text = q_text.replace("{forecast_due_date}", forecast_due_date)
    q_text = q_text.replace("{resolution_date}", "each of the resolution dates provided below")

    background = question["background"]
    if question.get("market_info_resolution_criteria", "N/A") != "N/A":
        background += "\n" + question["market_info_resolution_criteria"]

    return NON_MARKET_PROMPT.format(
        question=q_text,
        background=background,
        resolution_criteria=question["resolution_criteria"],
        freeze_datetime=question["freeze_datetime"],
        freeze_datetime_value=question["freeze_datetime_value"],
        freeze_datetime_value_explanation=question["freeze_datetime_value_explanation"],
        today_date=forecast_due_date,
        list_of_resolution_dates=question["resolution_dates"],
    )


# ── Entropy computation ────────────────────────────────────────────────────

def entropy_from_top_logprobs(top_lp_dict):
    """Compute entropy from top-k logprobs dict {token_str: logprob}."""
    if not top_lp_dict:
        return None
    logprobs = list(top_lp_dict.values())
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def extract_probability_entropies(tokens, token_logprobs, top_logprobs, response_text):
    """
    Extract entropy for each *probability* output in the response.

    ForecastBench format: model outputs *0.75* for each resolution date.
    We find all *X.XX* patterns and compute entropy on the numeric tokens.

    Returns list of dicts with entropy info for each probability found.
    """
    results = []

    # Find all *probability* patterns in the response
    prob_matches = list(re.finditer(r'\*\s*(0?\.\d+|1\.0|0|1)\s*\*', response_text))
    if not prob_matches:
        # Also try without asterisks -- sometimes model outputs differently
        prob_matches = list(re.finditer(r'(0\.\d+|1\.0)', response_text))

    if not prob_matches:
        return results

    # Build character-to-token mapping
    full_text = "".join(tokens)
    char_pos = 0
    token_char_ranges = []
    for i, tok in enumerate(tokens):
        tok_start = char_pos
        tok_end = char_pos + len(tok)
        token_char_ranges.append((tok_start, tok_end))
        char_pos = tok_end

    # Map response text positions to token positions
    # The response_text should be the concatenation of tokens
    for match_idx, match in enumerate(prob_matches):
        # Get the numeric part (group 1)
        prob_str = match.group(1).strip()
        try:
            prob_value = float(prob_str)
        except ValueError:
            continue

        # Find the numeric tokens within this match
        # We want the position of the numeric part in the full token stream
        match_start = match.start(1)
        match_end = match.end(1)

        # Find overlapping tokens for the numeric part
        numeric_token_indices = []
        for i, (ts, te) in enumerate(token_char_ranges):
            if te > match_start and ts < match_end:
                tok_str = tokens[i].strip()
                # Only include tokens that are part of the number (digits, dot)
                if tok_str and (tok_str.replace('.', '').replace('-', '').isdigit()
                                or tok_str == '.' or tok_str in ['0', '1']):
                    numeric_token_indices.append(i)

        if not numeric_token_indices:
            continue

        # Compute entropy for numeric tokens
        entropies = []
        lps = []
        tok_strs = []
        for ti in numeric_token_indices:
            if ti < len(top_logprobs) and top_logprobs[ti]:
                ent = entropy_from_top_logprobs(top_logprobs[ti])
                if ent is not None:
                    entropies.append(ent)
            if ti < len(token_logprobs) and token_logprobs[ti] is not None:
                lps.append(token_logprobs[ti])
            if ti < len(tokens):
                tok_strs.append(tokens[ti])

        if entropies:
            results.append({
                "prob_index": match_idx,
                "declared_prob": prob_value,
                "mean_token_entropy": float(np.mean(entropies)),
                "max_token_entropy": float(np.max(entropies)),
                "min_token_entropy": float(np.min(entropies)),
                "mean_token_logprob": float(np.mean(lps)) if lps else None,
                "n_numeric_tokens": len(numeric_token_indices),
                "token_strings": tok_strs,
                "individual_entropies": [float(e) for e in entropies],
            })

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load API key
    env_path = Path("/Users/elsehow/Projects/forecastbench/v2/.env")
    for line in env_path.read_text().splitlines():
        if line.startswith("TOGETHERAI_API_KEY="):
            os.environ["TOGETHERAI_API_KEY"] = line.split("=", 1)[1]
            break

    # Load questions
    print("Loading ForecastBench questions...")
    questions_by_source = load_questions()
    for src, qs in questions_by_source.items():
        print(f"  {src}: {len(qs)} questions")

    all_questions = []
    for source, qs in questions_by_source.items():
        for q in qs:
            all_questions.append((source, q))

    print(f"\nTotal questions: {len(all_questions)}")
    print(f"Models: {len(MODELS)}")
    print(f"Expected API calls: {len(all_questions) * len(MODELS)}")
    print()

    all_results = []
    errors = []

    for model_id in MODELS:
        model_short = model_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {model_short}")
        print(f"{'='*60}")

        for q_idx, (source, question) in enumerate(all_questions):
            prompt = build_prompt(question)

            try:
                response = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=5,
                )

                content = response.choices[0].message.content or ""
                lp_data = response.choices[0].logprobs.model_dump()
                tokens = lp_data.get("tokens", [])
                token_lps = lp_data.get("token_logprobs", [])
                top_lps = lp_data.get("top_logprobs", [])

                prob_data = extract_probability_entropies(
                    tokens, token_lps, top_lps, content
                )

                # Annotate each probability with metadata
                n_horizons = len(question.get("resolution_dates", []))
                for pd_entry in prob_data:
                    horizon_idx = pd_entry["prob_index"]
                    res_date = None
                    if horizon_idx < n_horizons:
                        res_date = question["resolution_dates"][horizon_idx]

                    pd_entry.update({
                        "model": model_short,
                        "source": source,
                        "question_id": question["id"],
                        "horizon_index": horizon_idx,
                        "resolution_date": res_date,
                        "raw_response": content,
                    })
                    all_results.append(pd_entry)

                n_probs = len(prob_data)
                if (q_idx + 1) % 25 == 0 or q_idx == 0:
                    print(f"  [{q_idx+1}/{len(all_questions)}] {source}/{question['id'][:20]}: "
                          f"{n_probs} probs extracted, response: {content[:80]}")

            except Exception as e:
                print(f"  [{q_idx+1}] {source}/{question['id'][:20]}: ERROR - {e.__class__.__name__}: {e}")
                errors.append({
                    "model": model_short,
                    "source": source,
                    "question_id": question["id"],
                    "error": str(e),
                })

            time.sleep(0.05)  # Light rate limiting

    # ── Save ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "forecastbench_logprobs.json"

    # Strip raw_response for the saved file (too large)
    save_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "raw_response"}
        save_results.append(r_copy)

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved {len(save_results)} records to {out_path}")

    if errors:
        err_path = RESULTS_DIR / "forecastbench_logprobs_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {err_path}")

    # ── Analysis ────────────────────────────────────────────────────────
    analyze_results(all_results)


def analyze_results(all_results):
    """Print summary analysis."""
    print()
    print("=" * 80)
    print("RESULTS: INNER CONFIDENCE ON FORECASTBENCH")
    print("=" * 80)

    if not all_results:
        print("No results to analyze.")
        return

    # Use only the first horizon (1-week) for clean comparison
    # (all horizons give multiple probs per question)
    for horizon_label, horizon_filter in [
        ("All horizons (pooled)", lambda r: True),
        ("1-week horizon only", lambda r: r.get("horizon_index") == 0),
        ("1-month horizon only", lambda r: r.get("horizon_index") == 1),
        ("3-month horizon only", lambda r: r.get("horizon_index") == 2),
    ]:
        filtered = [r for r in all_results if horizon_filter(r)]
        if not filtered:
            continue

        print(f"\n{'─'*60}")
        print(f"  {horizon_label} (N={len(filtered)})")
        print(f"{'─'*60}")

        models = sorted(set(r["model"] for r in filtered))

        for model in models:
            model_data = [r for r in filtered if r["model"] == model]
            yf_data = [r for r in model_data if r["source"] == "yfinance"]
            wiki_data = [r for r in model_data if r["source"] == "wikipedia"]

            yf_ent = [r["mean_token_entropy"] for r in yf_data]
            wiki_ent = [r["mean_token_entropy"] for r in wiki_data]

            if not yf_ent or not wiki_ent:
                continue

            yf_mean = np.mean(yf_ent)
            wiki_mean = np.mean(wiki_ent)
            diff = yf_mean - wiki_mean

            # Welch's t-test
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(yf_ent, wiki_ent, equal_var=False)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(yf_ent) + np.var(wiki_ent)) / 2)
            cohen_d = diff / pooled_std if pooled_std > 0 else 0

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            direction = "HIGHER on yfinance (supports H1: knows uncertainty)" if diff > 0 else \
                        "LOWER on yfinance (supports H2: overconfident)"

            print(f"\n  {model}:")
            print(f"    yfinance  entropy: {yf_mean:.4f} (N={len(yf_ent)}, "
                  f"probs: mean={np.mean([r['declared_prob'] for r in yf_data]):.3f})")
            print(f"    wikipedia entropy: {wiki_mean:.4f} (N={len(wiki_ent)}, "
                  f"probs: mean={np.mean([r['declared_prob'] for r in wiki_data]):.3f})")
            print(f"    diff (yf - wiki): {diff:+.4f}  t={t_stat:.2f}  p={p_val:.4f} {sig}")
            print(f"    Cohen's d: {cohen_d:+.3f}")
            print(f"    => {direction}")

        # Pooled across models
        yf_all = [r["mean_token_entropy"] for r in filtered if r["source"] == "yfinance"]
        wiki_all = [r["mean_token_entropy"] for r in filtered if r["source"] == "wikipedia"]

        if yf_all and wiki_all:
            from scipy import stats
            yf_m = np.mean(yf_all)
            wiki_m = np.mean(wiki_all)
            diff = yf_m - wiki_m
            t_stat, p_val = stats.ttest_ind(yf_all, wiki_all, equal_var=False)
            pooled_std = np.sqrt((np.var(yf_all) + np.var(wiki_all)) / 2)
            cohen_d = diff / pooled_std if pooled_std > 0 else 0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            direction = "HIGHER on yfinance (supports H1)" if diff > 0 else \
                        "LOWER on yfinance (supports H2)"

            print(f"\n  POOLED (all models):")
            print(f"    yfinance  entropy: {yf_m:.4f} (N={len(yf_all)})")
            print(f"    wikipedia entropy: {wiki_m:.4f} (N={len(wiki_all)})")
            print(f"    diff (yf - wiki): {diff:+.4f}  t={t_stat:.2f}  p={p_val:.4f} {sig}")
            print(f"    Cohen's d: {cohen_d:+.3f}")
            print(f"    => {direction}")

    # Per-horizon entropy trend
    print(f"\n{'─'*60}")
    print(f"  ENTROPY BY HORIZON (longer = more uncertain?)")
    print(f"{'─'*60}")

    for model in sorted(set(r["model"] for r in all_results)):
        print(f"\n  {model}:")
        for src in ["yfinance", "wikipedia"]:
            ents_by_h = {}
            for r in all_results:
                if r["model"] == model and r["source"] == src:
                    h = r.get("horizon_index", -1)
                    if h not in ents_by_h:
                        ents_by_h[h] = []
                    ents_by_h[h].append(r["mean_token_entropy"])

            if ents_by_h:
                parts = []
                for h in sorted(ents_by_h.keys()):
                    parts.append(f"H{h}={np.mean(ents_by_h[h]):.4f}(N={len(ents_by_h[h])})")
                print(f"    {src:12s}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
