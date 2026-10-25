"""
Financial inner confidence analysis: token-level entropy on quantile forecasts.

Re-runs Llama-3.3-70B and Qwen2.5-7B on real-world financial price data with
logprobs=True to extract model confidence at the quantile output tokens.

Tests:
  H2 — volatile assets (bitcoin, natural_gas, silver) vs stable (wheat, wti_crude)
  H3 — cascade from p1 → p99 across quantile levels

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/inner-confidence/src/financial_logprobs.py
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
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/real-world-tail-risk/data")
RESULTS_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/inner-confidence/results")

MODELS = [
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
]

ASSETS = {
    "bitcoin": {"name": "Bitcoin", "unit": "USD"},
    "natural_gas": {"name": "Henry Hub Natural Gas", "unit": "$/MMBtu"},
    "silver": {"name": "Silver", "unit": "$/troy oz"},
    "wheat": {"name": "CBOT Wheat", "unit": "\u00a2/bushel"},
    "wti_crude": {"name": "WTI Crude Oil", "unit": "$/barrel"},
}

# Volatility classification based on annualized historical vol
VOLATILE = {"bitcoin", "natural_gas", "silver"}   # vol > 0.4
STABLE = {"wheat", "wti_crude"}                     # vol < 0.35

# 5 cutoff dates spread across the available range
# Need ~6mo history before, ~90 days after for horizons
CUTOFF_DATES = [
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
    "2025-09-01",
]

# 3 horizons (days from cutoff)
HORIZONS = [
    {"label": "H1", "days": 7},
    {"label": "H2", "days": 30},
    {"label": "H3", "days": 90},
]

QUANTILE_LABELS = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]


# ── Price loading ───────────────────────────────────────────────────────────

def load_prices(asset_id):
    """Load full price DataFrame for an asset."""
    df = pd.read_csv(DATA_DIR / "prices" / f"{asset_id}.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def price_history_csv(df, cutoff_date):
    """Format price history up to cutoff as CSV string."""
    subset = df[df.index <= pd.Timestamp(cutoff_date)]
    lines = ["Date,Close"]
    for date, row in subset.iterrows():
        lines.append(f"{date.strftime('%Y-%m-%d')},{float(row['Close']):.2f}")
    return "\n".join(lines)


def get_target_price(df, target_date):
    """Get the closing price on or nearest to target_date."""
    ts = pd.Timestamp(target_date)
    # Try exact date first, then nearest trading day
    if ts in df.index:
        return float(df.loc[ts, "Close"])
    nearby = df.index[df.index >= ts]
    if len(nearby) > 0:
        return float(df.loc[nearby[0], "Close"])
    nearby = df.index[df.index <= ts]
    if len(nearby) > 0:
        return float(df.loc[nearby[-1], "Close"])
    return None


# ── Prompt building (mirrors run_models.py) ─────────────────────────────────

def build_prompt(asset_name, unit, price_csv, questions):
    """Build prompt identical to real-world-tail-risk/run_models.py."""
    num_q = len(questions)
    questions_text = "\n".join(
        f"Q{i+1} ({q['target_date']}): p1=__, p5=__, p10=__, p25=__, p50=__, p75=__, p90=__, p95=__, p99=__"
        for i, q in enumerate(questions)
    )
    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing historical price data for a financial asset. Make predictions based on the price history below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable estimates, but always return numeric percentile values.

## Asset: {asset_name} ({unit})

## Price History (daily closing prices)
{price_csv}

## Questions

What will the closing price of {asset_name} be on the following dates?

{questions_text}

You may analyze the data, but you MUST end your response with percentile estimates in this exact format:
<<<PERCENTILES>>>
Q1: p1=2, p5=4, p10=5, p25=10, p50=15, p75=20, p90=25, p95=28, p99=35
Q2: p1=50, p5=80, p10=100, p25=200, p50=300, p75=400, p90=500, p95=600, p99=800
<<<END>>>

For each question, provide one line with percentile estimates for all {num_q} questions, in order.
- p1 means you estimate there's a 1% chance the true value is below this number
- p5 means you estimate there's a 5% chance the true value is below this number
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number
- p95 means you estimate there's a 95% chance the true value is below this number
- p99 means you estimate there's a 99% chance the true value is below this number"""


# ── Entropy extraction ──────────────────────────────────────────────────────

def entropy_from_top_logprobs(top_lp_dict):
    """Compute entropy from top-5 logprobs dict {token: logprob}."""
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
    """Extract entropy for each quantile value in the PERCENTILES block.

    Handles multi-token numbers by aggregating entropy across sub-tokens.
    """
    results = []
    block_match = re.search(
        r'<<<PERCENTILES?>>>(.*?)<<<END>>>',
        response_text, re.DOTALL | re.IGNORECASE
    )
    if not block_match:
        return results

    block = block_match.group(1).strip()

    # Build char-to-token mapping over full response
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
            r'p(\d+)\s*=\s*(-?[\d.,]+)', line, re.IGNORECASE
        ))
        for pmatch in pmatches:
            quantile = f"p{pmatch.group(1)}"
            value_str = pmatch.group(2).replace(",", "")
            try:
                declared_value = float(value_str)
            except ValueError:
                continue

            # Locate value tokens in response
            line_start = response_text.find(line, block_start_in_response)
            if line_start < 0:
                continue

            abs_value_start = line_start + pmatch.start(2)
            abs_value_end = abs_value_start + len(pmatch.group(2))

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
                if ti < len(token_logprobs) and token_logprobs[ti] is not None:
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


# ── Generate questions ──────────────────────────────────────────────────────

def generate_questions():
    """Generate all question specs: 5 assets × 5 cutoffs × 3 horizons = 75 questions."""
    questions = []
    price_dfs = {aid: load_prices(aid) for aid in ASSETS}

    for asset_id, info in ASSETS.items():
        df = price_dfs[asset_id]
        for cutoff in CUTOFF_DATES:
            cutoff_ts = pd.Timestamp(cutoff)
            # Verify we have history before cutoff
            hist = df[df.index <= cutoff_ts]
            if len(hist) < 30:
                print(f"  SKIP {asset_id} @ {cutoff}: only {len(hist)} history rows")
                continue

            cutoff_price = float(hist.iloc[-1]["Close"])

            for h in HORIZONS:
                target_ts = cutoff_ts + pd.Timedelta(days=h["days"])
                target_price = get_target_price(df, target_ts)
                if target_price is None:
                    print(f"  SKIP {asset_id} @ {cutoff} {h['label']}: no target price for {target_ts.date()}")
                    continue

                questions.append({
                    "asset_id": asset_id,
                    "asset_name": info["name"],
                    "unit": info["unit"],
                    "cutoff_date": cutoff,
                    "cutoff_price": cutoff_price,
                    "horizon": h["label"],
                    "horizon_days": h["days"],
                    "target_date": str(target_ts.date()),
                    "ground_truth": target_price,
                    "volatility_group": "VOLATILE" if asset_id in VOLATILE else "STABLE",
                })

    return questions


def group_questions_for_api(questions):
    """Group questions by (asset_id, cutoff_date) for batched API calls.

    Each API call sends all 3 horizons for one asset-cutoff pair.
    """
    groups = defaultdict(list)
    for q in questions:
        key = (q["asset_id"], q["cutoff_date"])
        groups[key].append(q)
    return dict(groups)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load API key
    for line in Path("/Users/elsehow/Projects/forecastbench/v2/.env").read_text().splitlines():
        if line.startswith("TOGETHERAI_API_KEY="):
            os.environ["TOGETHERAI_API_KEY"] = line.split("=", 1)[1]
            break

    # Generate questions
    questions = generate_questions()
    print(f"Generated {len(questions)} questions")
    print(f"  Volatile assets: {sum(1 for q in questions if q['volatility_group'] == 'VOLATILE')}")
    print(f"  Stable assets:   {sum(1 for q in questions if q['volatility_group'] == 'STABLE')}")

    groups = group_questions_for_api(questions)
    print(f"  API calls per model: {len(groups)}")
    print(f"  Models: {len(MODELS)}")
    print(f"  Total API calls: {len(groups) * len(MODELS)}")
    print()

    # Load price data
    price_dfs = {aid: load_prices(aid) for aid in ASSETS}

    all_results = []
    errors = []

    for model_id in MODELS:
        model_short = model_id.split("/")[-1]
        print(f"{'='*70}")
        print(f"MODEL: {model_short}")
        print(f"{'='*70}")

        call_num = 0
        for (asset_id, cutoff_date), q_group in sorted(groups.items()):
            call_num += 1
            info = ASSETS[asset_id]
            df = price_dfs[asset_id]

            # Build prompt
            csv_str = price_history_csv(df, cutoff_date)
            prompt = build_prompt(info["name"], info["unit"], csv_str, q_group)

            print(f"  [{call_num}/{len(groups)}] {asset_id} @ {cutoff_date}: "
                  f"{len(q_group)} horizons, ~{len(prompt)//4} tokens")

            try:
                response = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=5,
                )

                content = response.choices[0].message.content or ""
                lp_data = response.choices[0].logprobs
                if lp_data is None:
                    print(f"    WARNING: No logprobs returned")
                    continue

                lp_dict = lp_data.model_dump()
                tokens = lp_dict.get("tokens", [])
                token_lps = lp_dict.get("token_logprobs", [])
                top_lps = lp_dict.get("top_logprobs", [])

                if not tokens:
                    print(f"    WARNING: Empty tokens list")
                    continue

                quantile_data = extract_quantile_entropies(tokens, token_lps, top_lps, content)

                # Annotate with metadata
                for qd in quantile_data:
                    qi = qd["question_idx"]
                    if qi < len(q_group):
                        q = q_group[qi]
                        qd["asset_id"] = q["asset_id"]
                        qd["asset_name"] = q["asset_name"]
                        qd["cutoff_date"] = q["cutoff_date"]
                        qd["horizon"] = q["horizon"]
                        qd["horizon_days"] = q["horizon_days"]
                        qd["ground_truth"] = q["ground_truth"]
                        qd["cutoff_price"] = q["cutoff_price"]
                        qd["volatility_group"] = q["volatility_group"]
                    qd["model"] = model_short

                all_results.extend(quantile_data)
                print(f"    Extracted {len(quantile_data)} quantile entropy values")

            except Exception as e:
                print(f"    ERROR: {e.__class__.__name__}: {e}")
                errors.append({
                    "model": model_short,
                    "asset_id": asset_id,
                    "cutoff_date": cutoff_date,
                    "error": str(e),
                })

            time.sleep(0.2)  # Rate limiting

        print()

    # ── Save ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "financial_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} records to {out_path}")

    if errors:
        err_path = RESULTS_DIR / "financial_logprobs_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {err_path}")

    # ── Analysis ────────────────────────────────────────────────────────
    print_summary(all_results)


def print_summary(all_results):
    """Print comprehensive summary of results."""
    if not all_results:
        print("\nNO RESULTS TO ANALYZE")
        return

    print()
    print("=" * 80)
    print("FINANCIAL INNER CONFIDENCE — SUMMARY")
    print("=" * 80)
    print(f"Total quantile entropy records: {len(all_results)}")

    models = sorted(set(r["model"] for r in all_results))

    # ── H2 test: volatile vs stable ──────────────────────────────────
    print()
    print("─" * 70)
    print("H2 TEST: Volatile vs Stable Assets")
    print("  Volatile: bitcoin, natural_gas, silver (ann. vol > 0.40)")
    print("  Stable:   wheat, wti_crude (ann. vol < 0.35)")
    print("─" * 70)

    for model in models:
        md = [r for r in all_results if r["model"] == model]
        print(f"\n  {model} ({len(md)} values)")

        for group in ["VOLATILE", "STABLE"]:
            ents = [r["mean_token_entropy"] for r in md if r.get("volatility_group") == group]
            if ents:
                print(f"    {group:10s}: mean entropy = {np.mean(ents):.4f} "
                      f"(std={np.std(ents):.4f}, N={len(ents)})")

        vol_ents = [r["mean_token_entropy"] for r in md if r.get("volatility_group") == "VOLATILE"]
        stb_ents = [r["mean_token_entropy"] for r in md if r.get("volatility_group") == "STABLE"]
        if vol_ents and stb_ents:
            diff = np.mean(vol_ents) - np.mean(stb_ents)
            print(f"    DIFF (vol-stb): {diff:+.4f}")

    # Pooled
    print(f"\n  POOLED ({len(all_results)} values)")
    for group in ["VOLATILE", "STABLE"]:
        ents = [r["mean_token_entropy"] for r in all_results if r.get("volatility_group") == group]
        if ents:
            print(f"    {group:10s}: mean entropy = {np.mean(ents):.4f} "
                  f"(std={np.std(ents):.4f}, N={len(ents)})")
    vol_ents = [r["mean_token_entropy"] for r in all_results if r.get("volatility_group") == "VOLATILE"]
    stb_ents = [r["mean_token_entropy"] for r in all_results if r.get("volatility_group") == "STABLE"]
    if vol_ents and stb_ents:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(vol_ents, stb_ents)
        print(f"    t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # ── H3 test: quantile cascade p1 → p99 ──────────────────────────
    print()
    print("─" * 70)
    print("H3 TEST: Quantile Cascade (p1 → p99)")
    print("─" * 70)

    for model in models:
        md = [r for r in all_results if r["model"] == model]
        print(f"\n  {model} ({len(md)} values)")

        for ql in QUANTILE_LABELS:
            ents = [r["mean_token_entropy"] for r in md if r["quantile"] == ql]
            if ents:
                print(f"    {ql:4s}: entropy = {np.mean(ents):.4f} "
                      f"(std={np.std(ents):.4f}, N={len(ents)})")

    print(f"\n  POOLED")
    for ql in QUANTILE_LABELS:
        ents = [r["mean_token_entropy"] for r in all_results if r["quantile"] == ql]
        if ents:
            print(f"    {ql:4s}: entropy = {np.mean(ents):.4f} "
                  f"(std={np.std(ents):.4f}, N={len(ents)})")

    # ── Per-asset breakdown ──────────────────────────────────────────
    print()
    print("─" * 70)
    print("PER-ASSET BREAKDOWN")
    print("─" * 70)

    assets = sorted(set(r.get("asset_id", "?") for r in all_results))
    for model in models:
        md = [r for r in all_results if r["model"] == model]
        print(f"\n  {model}")
        for asset in assets:
            ad = [r for r in md if r.get("asset_id") == asset]
            if ad:
                vol_group = ad[0].get("volatility_group", "?")
                ents = [r["mean_token_entropy"] for r in ad]
                print(f"    {asset:15s} [{vol_group:8s}]: entropy = {np.mean(ents):.4f} "
                      f"(std={np.std(ents):.4f}, N={len(ents)})")

    # ── Entropy by horizon ───────────────────────────────────────────
    print()
    print("─" * 70)
    print("ENTROPY BY HORIZON")
    print("─" * 70)

    for model in models:
        md = [r for r in all_results if r["model"] == model]
        print(f"\n  {model}")
        for h in ["H1", "H2", "H3"]:
            hd = [r for r in md if r.get("horizon") == h]
            if hd:
                ents = [r["mean_token_entropy"] for r in hd]
                days = hd[0].get("horizon_days", "?")
                print(f"    {h} ({days}d): entropy = {np.mean(ents):.4f} "
                      f"(std={np.std(ents):.4f}, N={len(ents)})")


if __name__ == "__main__":
    main()
