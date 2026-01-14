#!/usr/bin/env python3
"""Resolve pending questions and run the conditional forecasting experiment.

Run this script on the resolution date to:
1. Fetch current prices for all tickers
2. Compare to baseline prices to determine YES/NO resolution
3. Run the conditional experiment (baseline and/or two-stage methods)

Usage:
    # Run both methods for comparison
    uv run python resolve_and_run.py --method both

    # Run only baseline (original method)
    uv run python resolve_and_run.py --method baseline

    # Run only two-stage (best performing method from scaffolding experiments)
    uv run python resolve_and_run.py --method two-stage
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import litellm

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "llm-forecasting" / "src"))

from llm_forecasting.sources import YahooFinanceSource


# ============================================================================
# TWO-STAGE PROMPTS (from scaffolding/two-stage experiment - best performing)
# ============================================================================

TWOSTAGE_CLASSIFY_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Most question pairs are UNRELATED. Before assuming a connection, consider:

1. Are these about the same entity, event, or domain?
2. Is there a DIRECT causal mechanism linking outcomes?
3. Would a subject matter expert see an obvious connection?

If you cannot identify a clear, specific mechanism—not just thematic similarity—classify as independent.

Classification:
- "correlated": Clear causal or logical link exists
- "independent": No meaningful connection, or only superficial similarity

Return only JSON: {{"classification": "correlated|independent", "reasoning": "one sentence"}}"""

TWOSTAGE_BASELINE_PROMPT = """Question: "{q_a}"

What is the probability this resolves YES? Give only your estimate, no explanation.

Return only JSON: {{"p_a": 0.XX}}"""

TWOSTAGE_BRACKET_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

You determined these are correlated. Now estimate:

Step 1: Is the correlation positive or negative?
- Positive: A=YES makes B=YES more likely (or vice versa)
- Negative: A=YES makes B=NO more likely (or vice versa)

Step 2: Give P(A), P(A|B=YES), P(A|B=NO).

Constraint: P(A) MUST fall between P(A|B=YES) and P(A|B=NO).

Return JSON: {{
  "direction": "positive|negative",
  "p_a": 0.XX,
  "p_a_given_b1": 0.XX,
  "p_a_given_b0": 0.XX
}}"""


async def resolve_questions(pairs_file: Path) -> dict:
    """Resolve all questions by comparing current prices to baseline."""
    with open(pairs_file) as f:
        data = json.load(f)

    print(f"Resolving questions from {pairs_file.name}")
    print(f"Expected resolution date: {data['resolution_date']}")
    print(f"Today: {date.today().isoformat()}")
    print("=" * 60)

    # Collect all unique tickers from yfinance questions
    tickers_to_resolve = {}
    for pair in data["pairs"]:
        for q in [pair["question_a"], pair["question_b"]]:
            if q["source"] == "yfinance" and q["id"] not in tickers_to_resolve:
                tickers_to_resolve[q["id"]] = {
                    "baseline_price": q["baseline_price"],
                    "baseline_date": q["baseline_date"],
                }

    print(f"\nResolving {len(tickers_to_resolve)} tickers...")

    # Fetch current prices
    source = YahooFinanceSource(
        tickers=[{"symbol": t, "name": t} for t in tickers_to_resolve.keys()]
    )

    resolutions = {}
    for ticker_id, baseline_info in tickers_to_resolve.items():
        resolution = await source.fetch_resolution(ticker_id)
        if resolution:
            current_price = resolution.value
            baseline_price = baseline_info["baseline_price"]

            # YES (1.0) if price increased, NO (0.0) if decreased or same
            resolved_value = 1.0 if current_price > baseline_price else 0.0
            pct_change = ((current_price - baseline_price) / baseline_price) * 100

            resolutions[ticker_id] = {
                "baseline_price": baseline_price,
                "current_price": current_price,
                "pct_change": pct_change,
                "resolution": resolved_value,
            }

            direction = "↑" if resolved_value == 1.0 else "↓"
            print(f"  {ticker_id}: ${baseline_price:.2f} → ${current_price:.2f} ({pct_change:+.1f}%) {direction}")
        else:
            print(f"  {ticker_id}: FAILED to fetch")

    await source.close()

    # Update pairs with resolutions
    resolved_pairs = []
    for pair in data["pairs"]:
        qa = pair["question_a"]
        qb = pair["question_b"]

        # Get resolutions
        res_a = resolutions.get(qa["id"])
        res_b = resolutions.get(qb["id"])

        if res_a and res_b:
            resolved_pairs.append({
                **pair,
                "resolution_a": res_a["resolution"],
                "resolution_b": res_b["resolution"],
                "details_a": res_a,
                "details_b": res_b,
            })
        else:
            print(f"  Skipping pair {pair['pair_id']}: missing resolution")

    print(f"\nResolved {len(resolved_pairs)} / {len(data['pairs'])} pairs")

    return {
        "resolution_date": date.today().isoformat(),
        "original_file": str(pairs_file),
        "pairs": resolved_pairs,
    }


# ============================================================================
# BASELINE PROMPT (original method)
# ============================================================================

PROMPT_TEMPLATE = """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Before giving your probability, briefly answer:
1. Why might these questions be INDEPENDENT? (1-2 sentences)
2. Is there a DIRECT causal mechanism that overrides this? (1-2 sentences)

If no direct causal link, keep your estimate similar to what you'd give without the context.

End your response with EXACTLY this format on its own line:
PROBABILITY: [your number between 0 and 1]

Your answer:"""


# ============================================================================
# LLM HELPERS
# ============================================================================

def extract_json(content: str) -> dict | None:
    """Extract JSON from response content."""
    if not content:
        return None
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
    return None


async def call_llm(prompt: str, model: str, thinking: bool) -> str | None:
    """Make an LLM call and return the content."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["temperature"] = 1  # Required when thinking is enabled
        kwargs["max_tokens"] = 4000
    else:
        kwargs["temperature"] = 0.3
        kwargs["max_tokens"] = 500

    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"    LLM Error: {e}")
        return None


# ============================================================================
# TWO-STAGE ELICITATION
# ============================================================================

async def twostage_classify(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Stage 1: Classify pair as correlated or independent."""
    prompt = TWOSTAGE_CLASSIFY_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "classification" in result:
        result["classification"] = result["classification"].lower().strip()
        return result
    return None


async def twostage_baseline(q_a: str, model: str, thinking: bool) -> dict | None:
    """Stage 2a: Baseline elicitation for independent pairs."""
    prompt = TWOSTAGE_BASELINE_PROMPT.format(q_a=q_a)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "p_a" in result:
        return {
            "method": "baseline",
            "p_a": result["p_a"],
            "p_a_given_b1": None,
            "p_a_given_b0": None,
        }
    return None


async def twostage_bracket(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Stage 2b: Bracket elicitation for correlated pairs."""
    prompt = TWOSTAGE_BRACKET_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["p_a", "p_a_given_b1", "p_a_given_b0"]):
        return {
            "method": "bracket",
            "p_a": result["p_a"],
            "p_a_given_b1": result["p_a_given_b1"],
            "p_a_given_b0": result["p_a_given_b0"],
            "direction": result.get("direction"),
        }
    return None


async def run_twostage(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Run full two-stage elicitation."""
    # Stage 1: Classification
    classify_result = await twostage_classify(q_a, q_b, model, thinking)
    if not classify_result:
        return None

    classification = classify_result["classification"]

    # Stage 2: Route based on classification
    if classification == "independent":
        stage2 = await twostage_baseline(q_a, model, thinking)
    else:
        stage2 = await twostage_bracket(q_a, q_b, model, thinking)

    if not stage2:
        return None

    return {
        "classification": classification,
        "classification_reasoning": classify_result.get("reasoning"),
        **stage2,
    }


# ============================================================================
# BASELINE ELICITATION (original method)
# ============================================================================

async def get_forecast(
    question: str,
    condition: str,
    model: str,
    thinking: bool = False,
) -> float | None:
    """Get a probability forecast from the model."""
    prompt = PROMPT_TEMPLATE.format(question=question, condition=condition)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["max_tokens"] = 4000

    response = await litellm.acompletion(**kwargs)

    if thinking and hasattr(response.choices[0].message, "thinking"):
        content = response.choices[0].message.content
    else:
        content = response.choices[0].message.content

    # Parse probability
    prob_match = re.search(r"PROBABILITY:\s*(0?\.\d+|1\.0|[01](?:\.\d+)?)", content, re.IGNORECASE)
    if prob_match:
        return float(prob_match.group(1))

    # Fallback patterns
    patterns = [
        r"(?:probability|estimate|forecast)[:\s]+(\d*\.?\d+)",
        r"(\d*\.?\d+)\s*(?:probability|chance)",
        r"^(0?\.\d+|1\.0)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 1:
                return val

    return None


async def run_baseline_experiment(
    resolved_data: dict,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict:
    """Run the baseline conditional forecasting experiment on resolved pairs."""
    pairs = resolved_data["pairs"]
    print(f"\nRunning BASELINE experiment on {len(pairs)} pairs")
    print(f"Model: {model}, Thinking: {thinking}")
    print("=" * 60)

    results = []
    for i, pair in enumerate(pairs):
        qa = pair["question_a"]
        qb = pair["question_b"]
        res_a = pair["resolution_a"]
        res_b = pair["resolution_b"]

        print(f"\n[{i+1}/{len(pairs)}] {pair['category']}: {qa['id']} ↔ {qb['id']}")

        # Get P(A), P(A|B=1), P(A|B=0)
        p_a = await get_forecast(qa["text"], "No additional context.", model, thinking)
        p_a_given_b1 = await get_forecast(
            qa["text"],
            f"Context: The following resolved YES: {qb['text']}",
            model,
            thinking,
        )
        p_a_given_b0 = await get_forecast(
            qa["text"],
            f"Context: The following resolved NO: {qb['text']}",
            model,
            thinking,
        )

        if None in (p_a, p_a_given_b1, p_a_given_b0):
            print(f"  SKIPPED: Failed to parse forecasts")
            continue

        # Calculate metrics
        p_a_given_actual_b = p_a_given_b1 if res_b == 1.0 else p_a_given_b0
        brier_independence = (p_a - res_a) ** 2
        brier_conditional = (p_a_given_actual_b - res_a) ** 2
        improvement = brier_independence - brier_conditional
        sensitivity = abs(p_a_given_b1 - p_a_given_b0)

        print(f"  P(A)={p_a:.2f}, P(A|B=1)={p_a_given_b1:.2f}, P(A|B=0)={p_a_given_b0:.2f}")
        print(f"  Improvement: {improvement:+.4f}, Sensitivity: {sensitivity:.2f}")

        results.append({
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "reason": pair["reason"],
            "p_a": p_a,
            "p_a_given_b1": p_a_given_b1,
            "p_a_given_b0": p_a_given_b0,
            "p_a_given_actual_b": p_a_given_actual_b,
            "resolution_a": res_a,
            "resolution_b": res_b,
            "brier_independence": brier_independence,
            "brier_conditional": brier_conditional,
            "improvement": improvement,
            "sensitivity": sensitivity,
            "text_a": qa["text"][:100],
            "text_b": qb["text"][:100],
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY BY CATEGORY")
    print("=" * 60)

    summary = {}
    for cat in ["strong", "weak", "none"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            n = len(cat_results)
            mean_imp = sum(r["improvement"] for r in cat_results) / n
            mean_sens = sum(r["sensitivity"] for r in cat_results) / n
            wins = sum(1 for r in cat_results if r["improvement"] > 0)
            summary[cat] = {
                "n": n,
                "mean_improvement": mean_imp,
                "mean_sensitivity": mean_sens,
                "conditional_win_rate": wins / n,
            }
            print(f"\n{cat.upper()} (n={n}):")
            print(f"  Mean improvement:  {mean_imp:+.4f}")
            print(f"  Mean sensitivity:  {mean_sens:.3f}")
            print(f"  Conditional wins:  {wins}/{n} ({wins/n:.0%})")

    return {
        "method": "baseline",
        "results": results,
        "metadata": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "thinking": thinking,
            "num_pairs": len(pairs),
            "num_successful": len(results),
        },
        "summary": summary,
    }


async def run_twostage_experiment(
    resolved_data: dict,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict:
    """Run the two-stage conditional forecasting experiment on resolved pairs."""
    pairs = resolved_data["pairs"]
    print(f"\nRunning TWO-STAGE experiment on {len(pairs)} pairs")
    print(f"Model: {model}, Thinking: {thinking}")
    print("=" * 60)

    results = []
    for i, pair in enumerate(pairs):
        qa = pair["question_a"]
        qb = pair["question_b"]
        res_a = pair["resolution_a"]
        res_b = pair["resolution_b"]

        print(f"\n[{i+1}/{len(pairs)}] {pair['category']}: {qa['id']} ↔ {qb['id']}")

        # Run two-stage elicitation
        ts_result = await run_twostage(qa["text"], qb["text"], model, thinking)

        if not ts_result:
            print(f"  SKIPPED: Failed to get two-stage forecast")
            continue

        classification = ts_result["classification"]
        p_a = ts_result["p_a"]
        p_a_given_b1 = ts_result.get("p_a_given_b1")
        p_a_given_b0 = ts_result.get("p_a_given_b0")

        # For independent pairs, conditionals are same as unconditional
        if classification == "independent":
            p_a_given_b1 = p_a
            p_a_given_b0 = p_a
            print(f"  Classified: INDEPENDENT → P(A)={p_a:.2f}")
        else:
            print(f"  Classified: CORRELATED → P(A)={p_a:.2f}, P(A|B=1)={p_a_given_b1:.2f}, P(A|B=0)={p_a_given_b0:.2f}")

        # Calculate metrics
        p_a_given_actual_b = p_a_given_b1 if res_b == 1.0 else p_a_given_b0
        brier_independence = (p_a - res_a) ** 2
        brier_conditional = (p_a_given_actual_b - res_a) ** 2
        improvement = brier_independence - brier_conditional
        sensitivity = abs(p_a_given_b1 - p_a_given_b0)

        print(f"  Improvement: {improvement:+.4f}, Sensitivity: {sensitivity:.2f}")

        results.append({
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "reason": pair["reason"],
            "classification": classification,
            "classification_reasoning": ts_result.get("classification_reasoning"),
            "routed_to": ts_result.get("method"),
            "p_a": p_a,
            "p_a_given_b1": p_a_given_b1,
            "p_a_given_b0": p_a_given_b0,
            "p_a_given_actual_b": p_a_given_actual_b,
            "resolution_a": res_a,
            "resolution_b": res_b,
            "brier_independence": brier_independence,
            "brier_conditional": brier_conditional,
            "improvement": improvement,
            "sensitivity": sensitivity,
            "text_a": qa["text"][:100],
            "text_b": qb["text"][:100],
        })

    # Summary
    print("\n" + "=" * 60)
    print("TWO-STAGE SUMMARY BY CATEGORY")
    print("=" * 60)

    summary = {}
    for cat in ["strong", "weak", "none"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            n = len(cat_results)
            mean_imp = sum(r["improvement"] for r in cat_results) / n
            mean_sens = sum(r["sensitivity"] for r in cat_results) / n
            wins = sum(1 for r in cat_results if r["improvement"] > 0)
            classified_correlated = sum(1 for r in cat_results if r["classification"] == "correlated")
            summary[cat] = {
                "n": n,
                "mean_improvement": mean_imp,
                "mean_sensitivity": mean_sens,
                "conditional_win_rate": wins / n,
                "classified_correlated": classified_correlated,
                "classified_independent": n - classified_correlated,
            }
            print(f"\n{cat.upper()} (n={n}):")
            print(f"  Mean improvement:      {mean_imp:+.4f}")
            print(f"  Mean sensitivity:      {mean_sens:.3f}")
            print(f"  Conditional wins:      {wins}/{n} ({wins/n:.0%})")
            print(f"  Classified correlated: {classified_correlated}/{n}")

    return {
        "method": "two-stage",
        "results": results,
        "metadata": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "thinking": thinking,
            "num_pairs": len(pairs),
            "num_successful": len(results),
        },
        "summary": summary,
    }


async def main():
    parser = argparse.ArgumentParser(description="Resolve questions and run experiment")
    parser.add_argument(
        "pairs_file",
        type=str,
        nargs="?",
        default="pending_pairs_2026-01-14.json",
        help="Path to pending pairs JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=True,
        help="Enable extended thinking (default: True)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_false",
        dest="thinking",
        help="Disable extended thinking",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve questions, don't run experiment",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "two-stage", "both"],
        default="both",
        help="Experiment method: baseline (original), two-stage (best from scaffolding), or both for comparison",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    pairs_file = output_dir / args.pairs_file

    if not pairs_file.exists():
        print(f"ERROR: {pairs_file} not found")
        return

    # Step 1: Resolve questions
    resolved_data = await resolve_questions(pairs_file)

    # Save resolved data
    resolved_file = output_dir / f"resolved_{date.today().isoformat()}.json"
    with open(resolved_file, "w") as f:
        json.dump(resolved_data, f, indent=2)
    print(f"\nSaved resolutions to {resolved_file}")

    if args.resolve_only:
        return

    # Step 2: Run experiment(s)
    model_name = args.model.split("/")[-1].replace("-", "_")
    thinking_str = "_thinking" if args.thinking else ""
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    if args.method in ["baseline", "both"]:
        baseline_results = await run_baseline_experiment(
            resolved_data,
            model=args.model,
            thinking=args.thinking,
        )
        all_results["baseline"] = baseline_results

        results_file = results_dir / f"baseline_{model_name}{thinking_str}_{date.today().isoformat()}.json"
        with open(results_file, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\nBaseline results saved to {results_file}")

    if args.method in ["two-stage", "both"]:
        twostage_results = await run_twostage_experiment(
            resolved_data,
            model=args.model,
            thinking=args.thinking,
        )
        all_results["two-stage"] = twostage_results

        results_file = results_dir / f"twostage_{model_name}{thinking_str}_{date.today().isoformat()}.json"
        with open(results_file, "w") as f:
            json.dump(twostage_results, f, indent=2)
        print(f"\nTwo-stage results saved to {results_file}")

    # If running both, print comparison summary
    if args.method == "both" and "baseline" in all_results and "two-stage" in all_results:
        print("\n" + "=" * 60)
        print("COMPARISON: BASELINE vs TWO-STAGE")
        print("=" * 60)

        for cat in ["strong", "weak", "none"]:
            bl_summary = all_results["baseline"]["summary"].get(cat, {})
            ts_summary = all_results["two-stage"]["summary"].get(cat, {})

            if bl_summary and ts_summary:
                print(f"\n{cat.upper()}:")
                print(f"  Brier improvement:")
                print(f"    Baseline:   {bl_summary.get('mean_improvement', 0):+.4f}")
                print(f"    Two-Stage:  {ts_summary.get('mean_improvement', 0):+.4f}")
                print(f"  Win rate:")
                print(f"    Baseline:   {bl_summary.get('conditional_win_rate', 0):.0%}")
                print(f"    Two-Stage:  {ts_summary.get('conditional_win_rate', 0):.0%}")

        # Save combined comparison
        comparison_file = results_dir / f"comparison_{model_name}{thinking_str}_{date.today().isoformat()}.json"
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison saved to {comparison_file}")


if __name__ == "__main__":
    asyncio.run(main())
