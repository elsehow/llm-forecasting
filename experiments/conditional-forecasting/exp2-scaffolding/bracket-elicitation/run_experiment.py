#!/usr/bin/env python3
"""Bracket elicitation experiment.

Tests whether forcing explicit direction commitment before probability estimation
reduces impossible updates and improves direction accuracy.

Usage:
    uv run python experiments/fb-conditional/scaffolding/bracket-elicitation/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/bracket-elicitation/run_experiment.py --limit 5
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


BRACKET_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Step 1: Relationship
Are these questions positively correlated, negatively correlated, or independent?
- Positive: A=YES makes B=YES more likely (or vice versa)
- Negative: A=YES makes B=YES less likely (or vice versa)
- Independent: Knowing one tells you nothing about the other

Step 2: Probabilities
Give your estimates for:
- P(A): probability A resolves YES
- P(A|B=YES): probability A resolves YES, given B resolved YES
- P(A|B=NO): probability A resolves YES, given B resolved NO

CONSTRAINT: If you said "positive" or "negative", your P(A) MUST fall between P(A|B=YES) and P(A|B=NO). If "independent", all three should be approximately equal.

Think through the relationship carefully before giving numbers.

Return only valid JSON:
{{
  "direction": "positive|negative|independent",
  "mechanism": "one sentence explaining why",
  "p_a": 0.XX,
  "p_a_given_b1": 0.XX,
  "p_a_given_b0": 0.XX
}}"""


async def elicit_bracket(
    q_a: str,
    q_b: str,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict | None:
    """Elicit bracket probabilities for a pair of questions."""
    prompt = BRACKET_PROMPT.format(q_a=q_a, q_b=q_b)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["temperature"] = 1  # Required when thinking is enabled
    else:
        kwargs["temperature"] = 0.3

    try:
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content

        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate keys
            required_keys = {"direction", "mechanism", "p_a", "p_a_given_b1", "p_a_given_b0"}
            if required_keys.issubset(result.keys()):
                # Normalize direction
                result["direction"] = result["direction"].lower().strip()
                return result
    except Exception as e:
        print(f"    Error: {e}")

    return None


def check_constraint_satisfied(result: dict) -> bool:
    """Check if the stated constraint is satisfied."""
    direction = result["direction"]
    p_a = result["p_a"]
    p_a_given_b1 = result["p_a_given_b1"]
    p_a_given_b0 = result["p_a_given_b0"]

    if direction == "independent":
        # All three should be approximately equal (within 0.05)
        return abs(p_a_given_b1 - p_a_given_b0) < 0.05
    else:
        # P(A) must fall between the conditionals
        min_cond = min(p_a_given_b1, p_a_given_b0)
        max_cond = max(p_a_given_b1, p_a_given_b0)
        return min_cond <= p_a <= max_cond


def check_impossible_update(result: dict) -> bool:
    """Check if update is impossible (both conditionals same side of P(A))."""
    p_a = result["p_a"]
    p_a_given_b1 = result["p_a_given_b1"]
    p_a_given_b0 = result["p_a_given_b0"]

    # Both conditionals above P(A) or both below P(A)
    both_above = (p_a_given_b1 > p_a) and (p_a_given_b0 > p_a)
    both_below = (p_a_given_b1 < p_a) and (p_a_given_b0 < p_a)

    return both_above or both_below


def check_direction_correct(stated_direction: str, p_a_given_b1: float, p_a_given_b0: float) -> bool | None:
    """Check if stated direction matches implied direction from probabilities.

    For strong pairs, we infer 'ground truth' from the probability pattern:
    - If P(A|B=YES) > P(A|B=NO), the relationship is positive
    - If P(A|B=YES) < P(A|B=NO), the relationship is negative
    """
    diff = p_a_given_b1 - p_a_given_b0
    eps = 0.02  # Threshold for "no meaningful difference"

    if abs(diff) < eps:
        implied = "independent"
    elif diff > 0:
        implied = "positive"
    else:
        implied = "negative"

    # Normalize stated direction
    stated = stated_direction.lower().strip()
    if stated in ("positive", "negative", "independent"):
        return stated == implied
    return None


def compute_brier_metrics(result: dict, resolution_a: float, resolution_b: float) -> dict:
    """Compute Brier scores and improvement."""
    p_a = result["p_a"]
    p_a_given_actual_b = result["p_a_given_b1"] if resolution_b == 1.0 else result["p_a_given_b0"]

    brier_independence = (p_a - resolution_a) ** 2
    brier_conditional = (p_a_given_actual_b - resolution_a) ** 2
    improvement = brier_independence - brier_conditional

    sensitivity = abs(result["p_a_given_b1"] - result["p_a_given_b0"])

    return {
        "brier_independence": brier_independence,
        "brier_conditional": brier_conditional,
        "improvement": improvement,
        "sensitivity": sensitivity,
        "p_a_given_actual_b": p_a_given_actual_b,
    }


async def run_pair(pair: dict, model: str, thinking: bool) -> dict:
    """Run bracket elicitation on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]

    result = await elicit_bracket(q_a, q_b, model=model, thinking=thinking)

    if result is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Failed to elicit bracket",
        }

    constraint_satisfied = check_constraint_satisfied(result)
    impossible_update = check_impossible_update(result)
    direction_correct = check_direction_correct(
        result["direction"], result["p_a_given_b1"], result["p_a_given_b0"]
    )
    brier = compute_brier_metrics(result, pair["resolution_a"], pair["resolution_b"])

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": pair["resolution_a"],
        "resolution_b": pair["resolution_b"],
        "direction": result["direction"],
        "mechanism": result["mechanism"],
        "p_a": result["p_a"],
        "p_a_given_b1": result["p_a_given_b1"],
        "p_a_given_b0": result["p_a_given_b0"],
        "constraint_satisfied": constraint_satisfied,
        "impossible_update": impossible_update,
        "direction_correct": direction_correct,
        "brier": brier,
    }


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 70)
    print("BRACKET ELICITATION EXPERIMENT RESULTS")
    print("=" * 70)

    if not valid:
        print("\nNo valid results to summarize.")
        return

    # Group by category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Overall constraint satisfaction
    constraint_satisfied = sum(1 for r in valid if r["constraint_satisfied"])
    print(f"\nConstraint compliance: {constraint_satisfied}/{len(valid)} ({100*constraint_satisfied/len(valid):.0f}%)")

    print("\n" + "-" * 70)
    print("RESULTS BY CATEGORY")
    print("-" * 70)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n = len(cat_results)

        # Constraint satisfaction
        satisfied = sum(1 for r in cat_results if r["constraint_satisfied"])

        # Impossible updates
        impossible = sum(1 for r in cat_results if r["impossible_update"])

        # Direction accuracy (for strong pairs, check if stated direction matches reality)
        direction_checks = [r["direction_correct"] for r in cat_results if r["direction_correct"] is not None]
        direction_correct = sum(1 for d in direction_checks if d)
        direction_total = len(direction_checks)

        # Brier improvement
        improvements = [r["brier"]["improvement"] for r in cat_results]
        mean_improvement = sum(improvements) / n if n > 0 else 0
        wins = sum(1 for i in improvements if i > 0)

        # False positives (for none category: said positive/negative when should be independent)
        if cat == "none":
            false_positives = sum(1 for r in cat_results if r["direction"] != "independent")
            fp_pct = 100 * false_positives / n if n > 0 else 0
        else:
            false_positives = None
            fp_pct = None

        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Constraint satisfied: {satisfied}/{n} ({100*satisfied/n:.0f}%)")
        print(f"  Impossible updates:   {impossible}/{n} ({100*impossible/n:.0f}%)")
        if direction_total > 0:
            print(f"  Direction correct:    {direction_correct}/{direction_total} ({100*direction_correct/direction_total:.0f}%)")
        print(f"  Brier improvement:    {mean_improvement:+.4f} (wins: {wins}/{n} = {100*wins/n:.0f}%)")
        if fp_pct is not None:
            print(f"  False positives:      {false_positives}/{n} ({fp_pct:.0f}%)")

    # Show some mechanism examples
    print("\n" + "-" * 70)
    print("SAMPLE MECHANISMS")
    print("-" * 70)
    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue
        cat_results = by_category[cat][:3]  # First 3 of each category
        for r in cat_results:
            print(f"\n[{cat}] {r['direction']}: {r.get('mechanism', 'N/A')[:80]}")


def print_comparison(results: list[dict]):
    """Print comparison table to baseline."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    strong = by_category.get("strong", [])
    weak = by_category.get("weak", [])
    none = by_category.get("none", [])

    # Compute metrics
    constraint_all = sum(1 for r in valid if r["constraint_satisfied"])
    constraint_pct = 100 * constraint_all / len(valid) if valid else 0

    impossible_strong = sum(1 for r in strong if r["impossible_update"])
    impossible_strong_pct = 100 * impossible_strong / len(strong) if strong else 0

    impossible_none = sum(1 for r in none if r["impossible_update"])
    impossible_none_pct = 100 * impossible_none / len(none) if none else 0

    direction_checks_strong = [r["direction_correct"] for r in strong if r["direction_correct"] is not None]
    direction_correct_strong = sum(1 for d in direction_checks_strong if d)
    direction_pct = 100 * direction_correct_strong / len(direction_checks_strong) if direction_checks_strong else 0

    fp_none = sum(1 for r in none if r["direction"] != "independent")
    fp_pct = 100 * fp_none / len(none) if none else 0

    brier_strong = sum(r["brier"]["improvement"] for r in strong) / len(strong) if strong else 0
    brier_none = sum(r["brier"]["improvement"] for r in none) / len(none) if none else 0

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (Sonnet 4 + thinking, direct elicitation)")
    print("=" * 70)

    print("""
+---------------------------------+----------+---------+-------+
|             Metric              | Baseline | Bracket | Delta |
+---------------------------------+----------+---------+-------+""")
    print(f"| Impossible updates (strong)     | 14%      | {impossible_strong_pct:5.0f}%  | {impossible_strong_pct - 14:+5.0f}% |")
    print(f"| Impossible updates (none)       | ?        | {impossible_none_pct:5.0f}%  | ?     |")
    print(f"| Direction correct (strong)      | 57%      | {direction_pct:5.0f}%  | {direction_pct - 57:+5.0f}% |")
    print(f"| False positives (none)          | 7%       | {fp_pct:5.0f}%  | {fp_pct - 7:+5.0f}% |")
    print(f"| Constraint satisfied            | -        | {constraint_pct:5.0f}%  | -     |")
    print(f"| Brier (strong)                  | +0.007   | {brier_strong:+.3f} | {brier_strong - 0.007:+.3f} |")
    print(f"| Brier (none)                    | -0.015   | {brier_none:+.3f} | {brier_none - (-0.015):+.3f} |")
    print("+---------------------------------+----------+---------+-------+")


def print_analysis(results: list[dict]):
    """Print analysis answering the key questions."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    strong = by_category.get("strong", [])
    weak = by_category.get("weak", [])
    none = by_category.get("none", [])

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # 1. Constraint compliance
    constraint_all = sum(1 for r in valid if r["constraint_satisfied"])
    print(f"\n1. CONSTRAINT COMPLIANCE")
    print(f"   {constraint_all}/{len(valid)} ({100*constraint_all/len(valid):.0f}%) responses satisfy the stated constraint.")
    violated = [r for r in valid if not r["constraint_satisfied"]]
    if violated:
        print(f"   Violations by category:")
        for cat in ["strong", "weak", "none"]:
            cat_violated = [r for r in violated if r.get("category") == cat]
            if cat_violated:
                print(f"     - {cat}: {len(cat_violated)}")

    # 2. Impossible update rate
    impossible_strong = sum(1 for r in strong if r["impossible_update"])
    print(f"\n2. IMPOSSIBLE UPDATE RATE")
    print(f"   Strong pairs: {impossible_strong}/{len(strong)} ({100*impossible_strong/len(strong):.0f}%) vs 14% baseline")
    if impossible_strong / len(strong) < 0.14:
        print(f"   -> Explicit direction commitment REDUCED impossible updates")
    else:
        print(f"   -> No improvement over baseline")

    # 3. Direction accuracy
    direction_checks = [r["direction_correct"] for r in strong if r["direction_correct"] is not None]
    direction_correct = sum(1 for d in direction_checks if d)
    direction_pct = 100 * direction_correct / len(direction_checks) if direction_checks else 0
    print(f"\n3. DIRECTION ACCURACY (strong pairs)")
    print(f"   {direction_correct}/{len(direction_checks)} ({direction_pct:.0f}%) correct vs 57% baseline")
    if direction_pct > 57:
        print(f"   -> Explicit direction commitment IMPROVED accuracy")
    else:
        print(f"   -> No improvement over baseline")

    # 4. Independence detection
    independent_none = sum(1 for r in none if r["direction"] == "independent")
    hallucinated = sum(1 for r in none if r["direction"] != "independent")
    print(f"\n4. INDEPENDENCE DETECTION (none pairs)")
    print(f"   Correctly identified as independent: {independent_none}/{len(none)} ({100*independent_none/len(none):.0f}%)")
    print(f"   Hallucinated correlations: {hallucinated}/{len(none)} ({100*hallucinated/len(none):.0f}%)")
    # Show what directions were hallucinated
    if hallucinated:
        hallucinated_positive = sum(1 for r in none if r["direction"] == "positive")
        hallucinated_negative = sum(1 for r in none if r["direction"] == "negative")
        print(f"     - Said 'positive': {hallucinated_positive}")
        print(f"     - Said 'negative': {hallucinated_negative}")

    # 5. Mechanism quality (sample)
    print(f"\n5. MECHANISM QUALITY (sample)")
    print("   Strong pairs:")
    for r in strong[:2]:
        print(f"     [{r['direction']}] {r.get('mechanism', 'N/A')[:70]}...")
    print("   None pairs:")
    for r in none[:2]:
        print(f"     [{r['direction']}] {r.get('mechanism', 'N/A')[:70]}...")

    # 6. Key finding
    print(f"\n6. KEY FINDING")
    improvement_rate = (0.14 - impossible_strong / len(strong)) if strong else 0
    direction_improvement = direction_pct - 57
    if improvement_rate > 0.05 and direction_improvement > 5:
        print("   Forcing direction commitment first DOES reduce impossible updates")
        print("   and improve direction accuracy.")
    elif improvement_rate > 0.05:
        print("   Forcing direction commitment reduces impossible updates but")
        print("   does not improve direction accuracy.")
    elif direction_improvement > 5:
        print("   Forcing direction commitment improves direction accuracy but")
        print("   does not reduce impossible updates.")
    else:
        print("   Forcing direction commitment does NOT meaningfully improve")
        print("   over baseline on either metric.")


async def main():
    parser = argparse.ArgumentParser(description="Bracket elicitation experiment")
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
        args.output = f"experiments/fb-conditional/scaffolding/bracket-elicitation/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running bracket elicitation experiment on {len(pairs)} pairs")
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
                status = "OK" if result["constraint_satisfied"] else "VIOLATED"
                print(f"    {result['direction']} | constraint: {status} | impossible: {result['impossible_update']} | improvement: {result['brier']['improvement']:+.3f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print outputs
    print_summary(results)
    print_comparison(results)
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
