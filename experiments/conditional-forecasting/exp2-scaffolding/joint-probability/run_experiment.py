#!/usr/bin/env python3
"""Joint probability table elicitation experiment.

Tests whether eliciting joint probability tables improves coherence and
Brier scores over direct conditional elicitation.

Usage:
    uv run python experiments/fb-conditional/scaffolding/joint-probability/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/joint-probability/run_experiment.py --limit 5
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm


JOINT_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Estimate the joint probability distribution over outcomes. All four cells must sum to 1.0.

        B=YES    B=NO
A=YES   [  ]     [  ]
A=NO    [  ]     [  ]

Think carefully about how A and B relate before filling in values.

Return only valid JSON: {{"a1_b1": 0.XX, "a1_b0": 0.XX, "a0_b1": 0.XX, "a0_b0": 0.XX}}"""


async def elicit_joint_table(
    q_a: str,
    q_b: str,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict | None:
    """Elicit a joint probability table for a pair of questions."""
    prompt = JOINT_PROMPT.format(q_a=q_a, q_b=q_b)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["temperature"] = 1  # Required for thinking mode
    else:
        kwargs["temperature"] = 0.3

    try:
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content

        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            table = json.loads(json_match.group())
            # Validate keys
            required_keys = {"a1_b1", "a1_b0", "a0_b1", "a0_b0"}
            if required_keys.issubset(table.keys()):
                return table
    except Exception as e:
        print(f"    Error: {e}")

    return None


def validate_table(table: dict) -> bool:
    """Check if table sums to 1.0 (within tolerance)."""
    total = table["a1_b1"] + table["a1_b0"] + table["a0_b1"] + table["a0_b0"]
    return abs(total - 1.0) <= 0.01


def derive_probabilities(table: dict) -> dict:
    """Derive marginal and conditional probabilities from joint table."""
    a1_b1 = table["a1_b1"]
    a1_b0 = table["a1_b0"]
    a0_b1 = table["a0_b1"]
    a0_b0 = table["a0_b0"]

    # Marginals
    p_a = a1_b1 + a1_b0
    p_b = a1_b1 + a0_b1

    # Conditionals (with division safety)
    p_a_given_b1 = a1_b1 / (a1_b1 + a0_b1) if (a1_b1 + a0_b1) > 0.001 else 0.5
    p_a_given_b0 = a1_b0 / (a1_b0 + a0_b0) if (a1_b0 + a0_b0) > 0.001 else 0.5
    p_b_given_a1 = a1_b1 / (a1_b1 + a1_b0) if (a1_b1 + a1_b0) > 0.001 else 0.5
    p_b_given_a0 = a0_b1 / (a0_b1 + a0_b0) if (a0_b1 + a0_b0) > 0.001 else 0.5

    return {
        "p_a": p_a,
        "p_b": p_b,
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
        "p_b_given_a1": p_b_given_a1,
        "p_b_given_a0": p_b_given_a0,
    }


def compute_brier_metrics(derived: dict, resolution_a: float, resolution_b: float) -> dict:
    """Compute Brier scores and improvement."""
    p_a = derived["p_a"]
    p_a_given_actual_b = derived["p_a_given_b1"] if resolution_b == 1.0 else derived["p_a_given_b0"]

    brier_independence = (p_a - resolution_a) ** 2
    brier_conditional = (p_a_given_actual_b - resolution_a) ** 2
    improvement = brier_independence - brier_conditional

    sensitivity = abs(derived["p_a_given_b1"] - derived["p_a_given_b0"])

    return {
        "brier_independence": brier_independence,
        "brier_conditional": brier_conditional,
        "improvement": improvement,
        "sensitivity": sensitivity,
        "p_a_given_actual_b": p_a_given_actual_b,
    }


def analyze_direction(derived: dict) -> str:
    """Analyze direction of update."""
    p_a = derived["p_a"]
    p_b1 = derived["p_a_given_b1"]
    p_b0 = derived["p_a_given_b0"]

    eps = 0.02
    delta_b1 = p_b1 - p_a
    delta_b0 = p_b0 - p_a

    if abs(delta_b1) < eps and abs(delta_b0) < eps:
        return "no_update"
    elif delta_b1 > eps and delta_b0 < -eps:
        return "monotonic_up" if p_b1 > p_a > p_b0 else "partial"
    elif delta_b1 < -eps and delta_b0 > eps:
        return "monotonic_down" if p_b1 < p_a < p_b0 else "partial"
    elif (delta_b1 > eps and delta_b0 > eps) or (delta_b1 < -eps and delta_b0 < -eps):
        return "inconsistent"
    else:
        return "partial"


async def run_pair(pair: dict, model: str, thinking: bool) -> dict:
    """Run joint table elicitation on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]

    table = await elicit_joint_table(q_a, q_b, model=model, thinking=thinking)

    if table is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Failed to elicit joint table",
        }

    table_valid = validate_table(table)
    derived = derive_probabilities(table)
    brier = compute_brier_metrics(derived, pair["resolution_a"], pair["resolution_b"])
    direction = analyze_direction(derived)

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": pair["resolution_a"],
        "resolution_b": pair["resolution_b"],
        "table": table,
        "table_valid": table_valid,
        "derived": derived,
        "brier": brier,
        "direction": direction,
    }


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 70)
    print("JOINT PROBABILITY TABLE EXPERIMENT RESULTS")
    print("=" * 70)

    if not valid:
        print("\nNo valid results to analyze.")
        return

    # Format compliance
    valid_tables = sum(1 for r in valid if r["table_valid"])
    print(f"\nFormat compliance: {valid_tables}/{len(valid)} ({100*valid_tables/len(valid):.0f}%) tables sum to 1.0")

    # Group by category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    print("\n" + "-" * 70)
    print("RESULTS BY CATEGORY")
    print("-" * 70)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n = len(cat_results)

        # Brier improvement
        improvements = [r["brier"]["improvement"] for r in cat_results]
        mean_improvement = sum(improvements) / n
        wins = sum(1 for i in improvements if i > 0)

        # Direction analysis
        directions = [r["direction"] for r in cat_results]
        monotonic = sum(1 for d in directions if d in ("monotonic_up", "monotonic_down"))
        inconsistent = sum(1 for d in directions if d == "inconsistent")
        no_update = sum(1 for d in directions if d == "no_update")

        # Sensitivity (for false positive check on none)
        sensitivities = [r["brier"]["sensitivity"] for r in cat_results]
        false_positives = sum(1 for s in sensitivities if s > 0.05)

        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Brier improvement:  {mean_improvement:+.4f} (wins: {wins}/{n} = {100*wins/n:.0f}%)")
        print(f"  Direction:")
        print(f"    Monotonic:     {monotonic}/{n} ({100*monotonic/n:.0f}%)")
        print(f"    Inconsistent:  {inconsistent}/{n} ({100*inconsistent/n:.0f}%)")
        print(f"    No update:     {no_update}/{n} ({100*no_update/n:.0f}%)")
        if cat == "none":
            print(f"  False positives (sensitivity > 0.05): {false_positives}/{n} ({100*false_positives/n:.0f}%)")

    # Bayes consistency (by construction, should be 100%)
    print("\n" + "-" * 70)
    print("BAYESIAN CONSISTENCY")
    print("-" * 70)
    print("By construction, joint tables are 100% Bayes-consistent.")
    print("P(A|B) × P(B) = P(B|A) × P(A) is guaranteed by deriving from joint.")


def print_comparison(results: list[dict]):
    """Print comparison to baseline."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        print("\nNo valid results for comparison.")
        return

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Compute metrics
    valid_tables = sum(1 for r in valid if r["table_valid"])

    strong = by_category.get("strong", [])
    weak = by_category.get("weak", [])
    none = by_category.get("none", [])

    brier_strong = sum(r["brier"]["improvement"] for r in strong) / len(strong) if strong else 0
    brier_weak = sum(r["brier"]["improvement"] for r in weak) / len(weak) if weak else 0
    brier_none = sum(r["brier"]["improvement"] for r in none) / len(none) if none else 0

    mono_strong = sum(1 for r in strong if r["direction"] in ("monotonic_up", "monotonic_down"))
    mono_strong_pct = 100 * mono_strong / len(strong) if strong else 0

    fp_none = sum(1 for r in none if r["brier"]["sensitivity"] > 0.05)
    fp_none_pct = 100 * fp_none / len(none) if none else 0

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (Sonnet 4 + thinking, direct elicitation)")
    print("=" * 70)

    print("""
| Metric                     | Baseline | Joint Table | Delta   |
|----------------------------|----------|-------------|---------|""")
    print(f"| Format compliance          | —        | {100*valid_tables/len(valid):.0f}%         | —       |")
    print(f"| Brier (strong)             | +0.007   | {brier_strong:+.3f}       | {brier_strong - 0.007:+.3f}   |")
    print(f"| Brier (weak)               | -0.002   | {brier_weak:+.3f}       | {brier_weak - (-0.002):+.3f}   |")
    print(f"| Brier (none)               | -0.015   | {brier_none:+.3f}       | {brier_none - (-0.015):+.3f}   |")
    print(f"| Direction correct (strong) | 57%      | {mono_strong_pct:.0f}%         | {mono_strong_pct - 57:+.0f}%    |")
    print(f"| False positives (none)     | 7%       | {fp_none_pct:.0f}%          | {fp_none_pct - 7:+.0f}%     |")
    print(f"| Bayes consistent (strong)  | 50%      | 100%        | +50%    |")


async def main():
    parser = argparse.ArgumentParser(description="Joint probability table experiment")
    parser.add_argument("--pairs", type=str, default="experiments/fb-conditional/pairs_filtered.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_false", dest="thinking")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load pairs
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        return

    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])
    if args.limit:
        pairs = pairs[:args.limit]

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"experiments/fb-conditional/scaffolding/joint-probability/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running joint probability experiment on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(5)

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:50]}...")
            result = await run_pair(pair, args.model, args.thinking)
            if "error" not in result:
                print(f"    Table valid: {result['table_valid']}, Direction: {result['direction']}, Improvement: {result['brier']['improvement']:+.3f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print summary
    print_summary(results)
    print_comparison(results)

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
