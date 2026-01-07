#!/usr/bin/env python3
"""Resolve pending questions and run the conditional forecasting experiment.

Run this script on the resolution date to:
1. Fetch current prices for all tickers
2. Compare to baseline prices to determine YES/NO resolution
3. Run the conditional experiment
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import litellm

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "llm-forecasting" / "src"))

from llm_forecasting.sources import YahooFinanceSource


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


async def run_experiment(
    resolved_data: dict,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict:
    """Run the conditional forecasting experiment on resolved pairs."""
    pairs = resolved_data["pairs"]
    print(f"\nRunning experiment on {len(pairs)} pairs")
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

    # Step 2: Run experiment
    experiment_results = await run_experiment(
        resolved_data,
        model=args.model,
        thinking=args.thinking,
    )

    # Save results
    model_name = args.model.split("/")[-1].replace("-", "_")
    thinking_str = "_thinking" if args.thinking else ""
    results_file = output_dir / f"results/longitudinal_{model_name}{thinking_str}_{date.today().isoformat()}.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
