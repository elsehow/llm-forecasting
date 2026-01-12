#!/usr/bin/env python3
"""Mechanized Bayes experiment.

Tests whether forcing explicit Bayesian computation—eliciting marginals and joints
separately, then computing conditionals—improves accuracy.

Protocol (Multi-Turn, fresh context each):
  Turn 1: P(A) marginal
  Turn 2: P(B) marginal
  Turn 3: P(A and B) joint
  Turn 4: P(A and not B) joint reverse

Derive conditionals: P(A|B) = P(A,B) / P(B), P(A|~B) = P(A,~B) / P(~B)

Usage:
    uv run python experiments/fb-conditional/scaffolding/mechanized-bayes/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/mechanized-bayes/run_experiment.py --limit 5
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


# Prompts for each turn (fresh context, no conversation history)
PROMPT_MARGINAL_A = """Question: "{q_a}"

What is the probability this resolves YES?
Return only JSON: {{"p_a": 0.XX}}"""

PROMPT_MARGINAL_B = """Question: "{q_b}"

What is the probability this resolves YES?
Return only JSON: {{"p_b": 0.XX}}"""

PROMPT_JOINT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

What is the probability that BOTH resolve YES?

Note: This must be <= min(P(A), P(B)). If they're independent, P(A and B) = P(A) x P(B).
If positively correlated, P(A and B) > P(A) x P(B).
If negatively correlated, P(A and B) < P(A) x P(B).

Return only JSON: {{"p_joint": 0.XX}}"""

PROMPT_JOINT_REVERSE = """Questions:
- A: "{q_a}"
- B: "{q_b}"

What is the probability that A resolves YES and B resolves NO?

Return only JSON: {{"p_a1_b0": 0.XX}}"""


async def elicit_single(
    prompt: str,
    expected_key: str,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> float | None:
    """Elicit a single probability estimate."""
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
            data = json.loads(json_match.group())
            if expected_key in data:
                value = float(data[expected_key])
                # Clamp to [0, 1]
                return max(0.0, min(1.0, value))
    except Exception as e:
        print(f"    Error in elicit_single: {e}")

    return None


async def elicit_mechanized_bayes(
    q_a: str,
    q_b: str,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict | None:
    """Run the 4-turn mechanized Bayes protocol.

    Each turn uses fresh context (no conversation history) to test
    whether each estimate is independently reasonable.
    """
    # Turn 1: Marginal A
    p_a = await elicit_single(
        PROMPT_MARGINAL_A.format(q_a=q_a),
        "p_a",
        model=model,
        thinking=thinking,
    )
    if p_a is None:
        return None

    # Turn 2: Marginal B
    p_b = await elicit_single(
        PROMPT_MARGINAL_B.format(q_b=q_b),
        "p_b",
        model=model,
        thinking=thinking,
    )
    if p_b is None:
        return None

    # Turn 3: Joint P(A and B)
    p_a1_b1 = await elicit_single(
        PROMPT_JOINT.format(q_a=q_a, q_b=q_b),
        "p_joint",
        model=model,
        thinking=thinking,
    )
    if p_a1_b1 is None:
        return None

    # Turn 4: Joint reverse P(A and not B)
    p_a1_b0 = await elicit_single(
        PROMPT_JOINT_REVERSE.format(q_a=q_a, q_b=q_b),
        "p_a1_b0",
        model=model,
        thinking=thinking,
    )
    if p_a1_b0 is None:
        return None

    return {
        "p_a": p_a,
        "p_b": p_b,
        "p_a1_b1": p_a1_b1,
        "p_a1_b0": p_a1_b0,
    }


def derive_conditionals(elicited: dict) -> dict:
    """Derive conditional probabilities from elicited values."""
    p_a = elicited["p_a"]
    p_b = elicited["p_b"]
    p_a1_b1 = elicited["p_a1_b1"]
    p_a1_b0 = elicited["p_a1_b0"]

    # Derive conditionals (with division safety)
    p_a_given_b1 = p_a1_b1 / p_b if p_b > 0.001 else 0.5
    p_a_given_b0 = p_a1_b0 / (1 - p_b) if (1 - p_b) > 0.001 else 0.5

    # Clamp to [0, 1] in case of estimation errors
    p_a_given_b1 = max(0.0, min(1.0, p_a_given_b1))
    p_a_given_b0 = max(0.0, min(1.0, p_a_given_b0))

    return {
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
    }


def compute_validity(elicited: dict) -> dict:
    """Check validity constraints on elicited values."""
    p_a = elicited["p_a"]
    p_b = elicited["p_b"]
    p_a1_b1 = elicited["p_a1_b1"]
    p_a1_b0 = elicited["p_a1_b0"]

    # Joint validity: P(A,B) <= min(P(A), P(B))
    joint_valid = p_a1_b1 <= min(p_a, p_b) + 0.01  # Small tolerance

    # Marginal consistency: P(A,B) + P(A,~B) should equal P(A)
    implied_p_a = p_a1_b1 + p_a1_b0
    marginal_consistent = abs(implied_p_a - p_a) < 0.05

    return {
        "joint_valid": joint_valid,
        "marginal_consistent": marginal_consistent,
        "implied_p_a": implied_p_a,
        "marginal_error": abs(implied_p_a - p_a),
    }


def compute_correlation_ratio(elicited: dict) -> float:
    """Compute correlation ratio: P(A,B) / (P(A) * P(B)).

    > 1 means positive correlation
    < 1 means negative correlation
    = 1 means independence
    """
    p_a = elicited["p_a"]
    p_b = elicited["p_b"]
    p_a1_b1 = elicited["p_a1_b1"]

    independence_baseline = p_a * p_b
    if independence_baseline < 0.001:
        return 1.0  # Avoid division by zero

    return p_a1_b1 / independence_baseline


def compute_brier_metrics(
    derived: dict,
    elicited: dict,
    resolution_a: float,
    resolution_b: float,
) -> dict:
    """Compute Brier scores and improvement."""
    p_a = elicited["p_a"]
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


def analyze_direction(derived: dict, elicited: dict) -> str:
    """Analyze direction of update."""
    p_a = elicited["p_a"]
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
    """Run mechanized Bayes on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]

    elicited = await elicit_mechanized_bayes(q_a, q_b, model=model, thinking=thinking)

    if elicited is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Failed to elicit probabilities",
        }

    derived = derive_conditionals(elicited)
    validity = compute_validity(elicited)
    correlation_ratio = compute_correlation_ratio(elicited)
    brier = compute_brier_metrics(derived, elicited, pair["resolution_a"], pair["resolution_b"])
    direction = analyze_direction(derived, elicited)

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": pair["resolution_a"],
        "resolution_b": pair["resolution_b"],
        "elicited": elicited,
        "derived": derived,
        "validity": validity,
        "correlation_ratio": correlation_ratio,
        "brier": brier,
        "direction": direction,
    }


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 70)
    print("MECHANIZED BAYES EXPERIMENT RESULTS")
    print("=" * 70)

    if not valid:
        print("\nNo valid results to analyze.")
        return

    # Validity checks
    joint_valid = sum(1 for r in valid if r["validity"]["joint_valid"])
    marginal_consistent = sum(1 for r in valid if r["validity"]["marginal_consistent"])

    print(f"\nValidity (n={len(valid)}):")
    print(f"  Joint valid (P(A,B) <= min):    {joint_valid}/{len(valid)} ({100*joint_valid/len(valid):.0f}%)")
    print(f"  Marginal consistent:            {marginal_consistent}/{len(valid)} ({100*marginal_consistent/len(valid):.0f}%)")

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

        # Correlation ratio analysis
        ratios = [r["correlation_ratio"] for r in cat_results]
        ratio_above_1 = sum(1 for r in ratios if r > 1.05)
        ratio_near_1 = sum(1 for r in ratios if 0.95 <= r <= 1.05)
        ratio_below_1 = sum(1 for r in ratios if r < 0.95)
        mean_ratio = sum(ratios) / n

        # Sensitivity (for false positive check on none)
        sensitivities = [r["brier"]["sensitivity"] for r in cat_results]
        false_positives = sum(1 for s in sensitivities if s > 0.05)

        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Brier improvement:  {mean_improvement:+.4f} (wins: {wins}/{n} = {100*wins/n:.0f}%)")
        print(f"  Direction:")
        print(f"    Monotonic:     {monotonic}/{n} ({100*monotonic/n:.0f}%)")
        print(f"    Inconsistent:  {inconsistent}/{n} ({100*inconsistent/n:.0f}%)")
        print(f"    No update:     {no_update}/{n} ({100*no_update/n:.0f}%)")
        print(f"  Correlation ratio (mean): {mean_ratio:.2f}")
        print(f"    > 1 (positive): {ratio_above_1}/{n} ({100*ratio_above_1/n:.0f}%)")
        print(f"    ~ 1 (none):     {ratio_near_1}/{n} ({100*ratio_near_1/n:.0f}%)")
        print(f"    < 1 (negative): {ratio_below_1}/{n} ({100*ratio_below_1/n:.0f}%)")
        if cat == "none":
            print(f"  False positives (sensitivity > 0.05): {false_positives}/{n} ({100*false_positives/n:.0f}%)")


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
    joint_valid = sum(1 for r in valid if r["validity"]["joint_valid"])
    marginal_consistent = sum(1 for r in valid if r["validity"]["marginal_consistent"])

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

    # Correlation ratio detection
    corr_strong_above = sum(1 for r in strong if r["correlation_ratio"] > 1.05)
    corr_strong_pct = 100 * corr_strong_above / len(strong) if strong else 0

    corr_none_near = sum(1 for r in none if 0.95 <= r["correlation_ratio"] <= 1.05)
    corr_none_pct = 100 * corr_none_near / len(none) if none else 0

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (Sonnet 4 + thinking, direct elicitation)")
    print("=" * 70)

    print("""
| Metric                         | Baseline | Mechanized | Delta   |
|--------------------------------|----------|------------|---------|""")
    print(f"| Joint valid (<= min)           | -        | {100*joint_valid/len(valid):.0f}%        | -       |")
    print(f"| Marginal consistent            | -        | {100*marginal_consistent/len(valid):.0f}%        | -       |")
    print(f"| Correlation ratio > 1 (strong) | -        | {corr_strong_pct:.0f}%        | -       |")
    print(f"| Correlation ratio ~ 1 (none)   | -        | {corr_none_pct:.0f}%        | -       |")
    print(f"| Brier (strong)                 | +0.007   | {brier_strong:+.3f}      | {brier_strong - 0.007:+.3f}   |")
    print(f"| Brier (weak)                   | -0.002   | {brier_weak:+.3f}      | {brier_weak - (-0.002):+.3f}   |")
    print(f"| Brier (none)                   | -0.015   | {brier_none:+.3f}      | {brier_none - (-0.015):+.3f}   |")
    print(f"| Direction correct (strong)     | 57%      | {mono_strong_pct:.0f}%        | {mono_strong_pct - 57:+.0f}%    |")
    print(f"| False positives (none)         | 7%       | {fp_none_pct:.0f}%         | {fp_none_pct - 7:+.0f}%     |")


def print_analysis(results: list[dict]):
    """Print analysis answering key questions."""
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
    none = by_category.get("none", [])

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # 1. Joint validity
    joint_valid = sum(1 for r in valid if r["validity"]["joint_valid"])
    print(f"\n1. Joint validity: {100*joint_valid/len(valid):.0f}% of P(A,B) estimates satisfy P(A,B) <= min(P(A), P(B))")

    # 2. Marginal consistency
    marginal_consistent = sum(1 for r in valid if r["validity"]["marginal_consistent"])
    avg_error = sum(r["validity"]["marginal_error"] for r in valid) / len(valid)
    print(f"\n2. Marginal consistency: {100*marginal_consistent/len(valid):.0f}% have P(A,B) + P(A,~B) ~ P(A)")
    print(f"   Average marginal error: {avg_error:.3f}")

    # 3. Correlation detection via ratio
    if strong:
        strong_above = sum(1 for r in strong if r["correlation_ratio"] > 1.05)
        strong_below = sum(1 for r in strong if r["correlation_ratio"] < 0.95)
        print(f"\n3. Correlation detection (strong pairs, n={len(strong)}):")
        print(f"   Ratio > 1 (detected positive): {strong_above} ({100*strong_above/len(strong):.0f}%)")
        print(f"   Ratio < 1 (detected negative): {strong_below} ({100*strong_below/len(strong):.0f}%)")

    if none:
        none_near = sum(1 for r in none if 0.95 <= r["correlation_ratio"] <= 1.05)
        print(f"\n   On 'none' pairs (n={len(none)}):")
        print(f"   Ratio ~ 1 (correctly independent): {none_near} ({100*none_near/len(none):.0f}%)")

    # 4. Key finding
    print("\n4. Key finding:")
    if joint_valid / len(valid) > 0.9 and marginal_consistent / len(valid) > 0.8:
        print("   Models produce coherent probability estimates when asked for components separately.")
    else:
        print("   Models struggle with coherence even with mechanized decomposition.")

    # 5. Comparison insight
    print("\n5. Comparison insight:")
    print("   - Joint Table: Model fills out full distribution, we derive conditionals")
    print("   - Mechanized Bayes: Model estimates components separately, we compute")
    print("   If Mechanized Bayes works better -> models struggle with full table cognitive load")
    print("   If Joint Table works better -> models need to see whole structure")
    print("   If neither helps -> problem is correlation detection, not computation")


async def main():
    parser = argparse.ArgumentParser(description="Mechanized Bayes experiment")
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
        args.output = f"experiments/fb-conditional/scaffolding/mechanized-bayes/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running mechanized Bayes experiment on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()

    # Run with concurrency limit (lower than other experiments due to 4 calls per pair)
    semaphore = asyncio.Semaphore(3)

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:50]}...")
            result = await run_pair(pair, args.model, args.thinking)
            if "error" not in result:
                v = result["validity"]
                print(f"    Joint valid: {v['joint_valid']}, Marginal err: {v['marginal_error']:.3f}, "
                      f"Corr ratio: {result['correlation_ratio']:.2f}, Brier: {result['brier']['improvement']:+.3f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print summary
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
