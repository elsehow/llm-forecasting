#!/usr/bin/env python3
"""Full Bayesian consistency check for conditional forecasting.

Elicits BOTH directions:
- P(A), P(A|B=1), P(A|B=0)
- P(B), P(B|A=1), P(B|A=0)

Then checks if Bayes' rule holds:
  P(A|B) * P(B) = P(B|A) * P(A)

Usage:
    uv run python experiments/fb-conditional/bayesian_check.py
    uv run python experiments/fb-conditional/bayesian_check.py --model gpt-5.2
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm

# Reuse the elicitation from run_experiment
from run_experiment import elicit_probability


async def run_pair_symmetric(pair: dict, model: str, thinking: bool = False) -> dict:
    """Run symmetric elicitation on a question pair.

    Elicits P(A), P(A|B=1), P(A|B=0), P(B), P(B|A=1), P(B|A=0).
    """
    text_a = pair["text_a"]
    text_b = pair["text_b"]

    # Elicit all 6 probabilities in parallel
    (p_a, p_a_given_b1, p_a_given_b0,
     p_b, p_b_given_a1, p_b_given_a0) = await asyncio.gather(
        # A direction
        elicit_probability(text_a, condition=None, model=model, thinking=thinking, prompt_style="skeptical"),
        elicit_probability(text_a, condition=f'"{text_b}" resolved YES.', model=model, thinking=thinking, prompt_style="skeptical"),
        elicit_probability(text_a, condition=f'"{text_b}" resolved NO.', model=model, thinking=thinking, prompt_style="skeptical"),
        # B direction
        elicit_probability(text_b, condition=None, model=model, thinking=thinking, prompt_style="skeptical"),
        elicit_probability(text_b, condition=f'"{text_a}" resolved YES.', model=model, thinking=thinking, prompt_style="skeptical"),
        elicit_probability(text_b, condition=f'"{text_a}" resolved NO.', model=model, thinking=thinking, prompt_style="skeptical"),
    )

    if any(p is None for p in [p_a, p_a_given_b1, p_a_given_b0, p_b, p_b_given_a1, p_b_given_a0]):
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Failed to elicit some probabilities",
        }

    # Bayes check: P(A|B=1) * P(B) should equal P(B|A=1) * P(A) (approximately)
    # More precisely: P(A,B) = P(A|B)*P(B) = P(B|A)*P(A)

    # For B=1 case:
    lhs_b1 = p_a_given_b1 * p_b          # P(A|B=1) * P(B=1)
    rhs_b1 = p_b_given_a1 * p_a          # P(B=1|A=1) * P(A=1) -- but this is P(A=1,B=1), same as lhs

    # Actually, let's be more careful. Bayes says:
    # P(A=1|B=1) = P(B=1|A=1) * P(A=1) / P(B=1)
    # So: P(A=1|B=1) * P(B=1) = P(B=1|A=1) * P(A=1)

    # Joint probability P(A=1, B=1) two ways:
    joint_ab_11_via_a = p_a_given_b1 * p_b      # P(A=1|B=1) * P(B=1)
    joint_ab_11_via_b = p_b_given_a1 * p_a      # P(B=1|A=1) * P(A=1)

    # Joint probability P(A=1, B=0) two ways:
    joint_ab_10_via_a = p_a_given_b0 * (1 - p_b)  # P(A=1|B=0) * P(B=0)
    joint_ab_10_via_b = (1 - p_b_given_a1) * p_a  # P(B=0|A=1) * P(A=1)

    # Joint probability P(A=0, B=1) two ways:
    joint_ab_01_via_a = (1 - p_a_given_b1) * p_b      # P(A=0|B=1) * P(B=1)
    joint_ab_01_via_b = p_b_given_a0 * (1 - p_a)      # P(B=1|A=0) * P(A=0)

    # Consistency errors
    error_11 = abs(joint_ab_11_via_a - joint_ab_11_via_b)
    error_10 = abs(joint_ab_10_via_a - joint_ab_10_via_b)
    error_01 = abs(joint_ab_01_via_a - joint_ab_01_via_b)

    # Also check law of total probability
    # P(A) should equal P(A|B=1)*P(B) + P(A|B=0)*P(B=0)
    p_a_reconstructed = p_a_given_b1 * p_b + p_a_given_b0 * (1 - p_b)
    lotp_error_a = abs(p_a - p_a_reconstructed)

    p_b_reconstructed = p_b_given_a1 * p_a + p_b_given_a0 * (1 - p_a)
    lotp_error_b = abs(p_b - p_b_reconstructed)

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        # Raw probabilities
        "p_a": p_a,
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
        "p_b": p_b,
        "p_b_given_a1": p_b_given_a1,
        "p_b_given_a0": p_b_given_a0,
        # Joint probabilities (two ways each)
        "joint_11_via_a": joint_ab_11_via_a,
        "joint_11_via_b": joint_ab_11_via_b,
        "joint_10_via_a": joint_ab_10_via_a,
        "joint_10_via_b": joint_ab_10_via_b,
        "joint_01_via_a": joint_ab_01_via_a,
        "joint_01_via_b": joint_ab_01_via_b,
        # Bayes errors
        "bayes_error_11": error_11,
        "bayes_error_10": error_10,
        "bayes_error_01": error_01,
        "bayes_error_mean": (error_11 + error_10 + error_01) / 3,
        # Law of total probability errors
        "lotp_error_a": lotp_error_a,
        "lotp_error_b": lotp_error_b,
        # Question text
        "text_a": text_a[:80],
        "text_b": text_b[:80],
    }


def print_analysis(results: list[dict]):
    """Print analysis of Bayesian consistency."""
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 70)
    print("BAYESIAN CONSISTENCY ANALYSIS")
    print("=" * 70)
    print()
    print("We elicit P(A), P(A|B), P(B), P(B|A) and check if Bayes' rule holds:")
    print("  P(A|B=1) * P(B) = P(B|A=1) * P(A)  [should be equal]")
    print()

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n = len(cat_results)

        # Bayes errors
        mean_bayes_error = sum(r["bayes_error_mean"] for r in cat_results) / n
        max_bayes_error = max(r["bayes_error_mean"] for r in cat_results)

        # Count "consistent" (error < threshold)
        threshold = 0.05
        consistent = sum(1 for r in cat_results if r["bayes_error_mean"] < threshold)

        # LOTP errors
        mean_lotp_a = sum(r["lotp_error_a"] for r in cat_results) / n
        mean_lotp_b = sum(r["lotp_error_b"] for r in cat_results) / n

        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Bayes consistency (error < {threshold}): {consistent}/{n} ({100*consistent/n:.0f}%)")
        print(f"  Mean Bayes error: {mean_bayes_error:.3f}")
        print(f"  Max Bayes error:  {max_bayes_error:.3f}")
        print(f"  Mean LOTP error (A): {mean_lotp_a:.3f}")
        print(f"  Mean LOTP error (B): {mean_lotp_b:.3f}")

        # Show worst examples
        worst = sorted(cat_results, key=lambda r: r["bayes_error_mean"], reverse=True)[:2]
        print(f"\n  Worst Bayes violations:")
        for r in worst:
            print(f"    Error={r['bayes_error_mean']:.3f}: {r['text_a'][:40]}...")
            print(f"      P(A)={r['p_a']:.2f}, P(A|B=1)={r['p_a_given_b1']:.2f}, P(A|B=0)={r['p_a_given_b0']:.2f}")
            print(f"      P(B)={r['p_b']:.2f}, P(B|A=1)={r['p_b_given_a1']:.2f}, P(B|A=0)={r['p_b_given_a0']:.2f}")
            print(f"      Joint(1,1): {r['joint_11_via_a']:.3f} vs {r['joint_11_via_b']:.3f}")

    # Overall summary
    all_errors = [r["bayes_error_mean"] for r in valid]
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total pairs analyzed: {len(valid)}")
    print(f"Mean Bayes error: {sum(all_errors)/len(all_errors):.3f}")
    print(f"Pairs with error < 0.05: {sum(1 for e in all_errors if e < 0.05)}/{len(valid)}")
    print(f"Pairs with error < 0.10: {sum(1 for e in all_errors if e < 0.10)}/{len(valid)}")
    print(f"Pairs with error > 0.20: {sum(1 for e in all_errors if e > 0.20)}/{len(valid)}")


async def main():
    parser = argparse.ArgumentParser(description="Bayesian consistency check")
    parser.add_argument("--pairs", type=str, default="experiments/fb-conditional/pairs_filtered.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        return

    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])
    if args.limit:
        pairs = pairs[:args.limit]

    if args.output is None:
        model_short = args.model.split("/")[-1].replace("anthropic_", "")
        thinking_suffix = "_thinking" if args.thinking else ""
        args.output = f"experiments/fb-conditional/results/bayesian_{model_short}{thinking_suffix}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running Bayesian consistency check on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(3)  # Lower concurrency since we're doing 6 calls per pair

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:50]}...")
            result = await run_pair_symmetric(pair, args.model, thinking=args.thinking)
            if "error" not in result:
                print(f"    Bayes error: {result['bayes_error_mean']:.3f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print analysis
    print_analysis(results)

    # Save results
    output = {
        "results": results,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "model": args.model,
            "thinking": args.thinking,
            "num_pairs": len(pairs),
            "num_successful": len([r for r in results if "error" not in r]),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
