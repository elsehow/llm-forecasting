#!/usr/bin/env python3
"""Strength Elicitation experiment.

Tests whether eliciting correlation strength (1-10 scale) separately from direction
produces better-calibrated probability updates than direct conditional elicitation.

Hypothesis: Models may be better at relative strength judgments than absolute
probability adjustments.

Usage:
    uv run python experiments/magnitude/strength-elicitation/run_experiment.py
    uv run python experiments/magnitude/strength-elicitation/run_experiment.py --limit 5
"""

import argparse
import asyncio
import json
import math
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm


# Stage 1: Independence classification (same as Two-Stage)
STAGE1_PROMPT = """Questions:
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


# Two-Stage Baseline prompt (for comparison)
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


# Strength elicitation prompt
STRENGTH_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

You've determined these questions are correlated.

Step 1: DIRECTION
Is the correlation positive or negative?
- Positive: A=YES makes B=YES more likely
- Negative: A=YES makes B=NO more likely

Step 2: STRENGTH
How strong is this correlation? Rate 1-10:
- 1-2: Very weak (barely detectable relationship)
- 3-4: Weak (noticeable but small effect)
- 5-6: Moderate (meaningful effect)
- 7-8: Strong (substantial effect)
- 9-10: Very strong (one nearly determines the other)

Step 3: BASE PROBABILITY
What is P(A) unconditionally?

Return JSON:
{{
  "direction": "positive|negative",
  "strength": <1-10>,
  "p_a": 0.XX,
  "reasoning": "one sentence on strength rating"
}}"""


# Baseline prompt (for independent pairs)
BASELINE_PROMPT = """Question: "{q_a}"

What is the probability this resolves YES? Give only your estimate, no explanation.

Return only JSON: {{"p_a": 0.XX}}"""


async def call_llm(prompt: str, model: str, thinking: bool) -> str | None:
    """Make an LLM call and return the content."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["temperature"] = 1
    else:
        kwargs["temperature"] = 0.3

    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"    LLM Error: {e}")
        return None


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


def compute_conditionals_linear(p_a: float, direction: str, strength: int, max_shift: float = 0.3) -> tuple[float, float]:
    """Linear mapping: strength 1 -> 0.03, strength 10 -> max_shift."""
    shift = (strength / 10) * max_shift

    if direction == "positive":
        p_a_given_b1 = min(0.99, p_a + shift)
        p_a_given_b0 = max(0.01, p_a - shift)
    else:
        p_a_given_b1 = max(0.01, p_a - shift)
        p_a_given_b0 = min(0.99, p_a + shift)

    return p_a_given_b1, p_a_given_b0


def compute_conditionals_sigmoid(p_a: float, direction: str, strength: int, max_shift: float = 0.3) -> tuple[float, float]:
    """Sigmoid mapping: compressed at extremes, more responsive in middle."""
    # Sigmoid-like scaling: more responsive in middle range
    # s(x) = max_shift * (1 / (1 + e^(-0.8*(x-5.5))))
    x = strength
    sigmoid_val = 1 / (1 + math.exp(-0.8 * (x - 5.5)))
    shift = max_shift * sigmoid_val

    if direction == "positive":
        p_a_given_b1 = min(0.99, p_a + shift)
        p_a_given_b0 = max(0.01, p_a - shift)
    else:
        p_a_given_b1 = max(0.01, p_a - shift)
        p_a_given_b0 = min(0.99, p_a + shift)

    return p_a_given_b1, p_a_given_b0


def compute_conditionals_quadratic(p_a: float, direction: str, strength: int, max_shift: float = 0.3) -> tuple[float, float]:
    """Quadratic mapping: larger differences at high strengths."""
    shift = max_shift * (strength / 10) ** 2

    if direction == "positive":
        p_a_given_b1 = min(0.99, p_a + shift)
        p_a_given_b0 = max(0.01, p_a - shift)
    else:
        p_a_given_b1 = max(0.01, p_a - shift)
        p_a_given_b0 = min(0.99, p_a + shift)

    return p_a_given_b1, p_a_given_b0


async def stage1_classify(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Stage 1: Classify pair as correlated or independent."""
    prompt = STAGE1_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "classification" in result:
        result["classification"] = result["classification"].lower().strip()
        return result
    return None


async def get_baseline(q_a: str, model: str, thinking: bool) -> dict | None:
    """Get baseline P(A) for independent pairs."""
    prompt = BASELINE_PROMPT.format(q_a=q_a)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "p_a" in result:
        return {"p_a": result["p_a"]}
    return None


async def get_twostage_bracket(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Get Two-Stage baseline (bracket elicitation) for comparison."""
    prompt = TWOSTAGE_BRACKET_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["direction", "p_a", "p_a_given_b1", "p_a_given_b0"]):
        return {
            "direction": result["direction"].lower().strip(),
            "p_a": result["p_a"],
            "p_a_given_b1": result["p_a_given_b1"],
            "p_a_given_b0": result["p_a_given_b0"],
        }
    return None


async def get_strength_elicitation(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Get strength-based elicitation."""
    prompt = STRENGTH_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["direction", "strength", "p_a"]):
        return {
            "direction": result["direction"].lower().strip(),
            "strength": int(result["strength"]),
            "p_a": result["p_a"],
            "reasoning": result.get("reasoning", ""),
        }
    return None


def compute_brier(p_a_given_b_actual: float, resolution_a: float) -> float:
    """Compute Brier score."""
    return (p_a_given_b_actual - resolution_a) ** 2


async def run_pair(pair: dict, model: str, thinking: bool) -> dict:
    """Run experiment on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]
    resolution_a = pair["resolution_a"]
    resolution_b = pair["resolution_b"]

    # Stage 1: Classification
    stage1 = await stage1_classify(q_a, q_b, model, thinking)

    if stage1 is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Stage 1 classification failed",
        }

    result = {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": resolution_a,
        "resolution_b": resolution_b,
        "stage1": stage1,
    }

    # If independent, just get baseline
    if stage1["classification"] == "independent":
        baseline = await get_baseline(q_a, model, thinking)
        if baseline is None:
            result["error"] = "Baseline elicitation failed"
            return result

        result["baseline"] = {
            "p_a": baseline["p_a"],
            "p_a_given_b1": baseline["p_a"],
            "p_a_given_b0": baseline["p_a"],
        }
        result["strength_elicitation"] = None
        result["computed"] = None

        # Brier for baseline
        brier_baseline = compute_brier(baseline["p_a"], resolution_a)
        result["brier"] = {
            "baseline": brier_baseline,
            "strength_linear": brier_baseline,
            "strength_sigmoid": brier_baseline,
            "strength_quadratic": brier_baseline,
            "improvement_linear": 0,
            "improvement_sigmoid": 0,
            "improvement_quadratic": 0,
        }
        return result

    # For correlated pairs, run both methods
    # 1. Two-Stage baseline (bracket)
    twostage = await get_twostage_bracket(q_a, q_b, model, thinking)
    if twostage is None:
        result["error"] = "Two-Stage bracket elicitation failed"
        return result

    result["baseline"] = twostage

    # 2. Strength elicitation
    strength = await get_strength_elicitation(q_a, q_b, model, thinking)
    if strength is None:
        result["error"] = "Strength elicitation failed"
        return result

    result["strength_elicitation"] = strength

    # 3. Compute conditionals via formulas
    p_a = strength["p_a"]
    direction = strength["direction"]
    s = strength["strength"]

    linear_b1, linear_b0 = compute_conditionals_linear(p_a, direction, s)
    sigmoid_b1, sigmoid_b0 = compute_conditionals_sigmoid(p_a, direction, s)
    quadratic_b1, quadratic_b0 = compute_conditionals_quadratic(p_a, direction, s)

    result["computed"] = {
        "linear": {"p_a_given_b1": linear_b1, "p_a_given_b0": linear_b0},
        "sigmoid": {"p_a_given_b1": sigmoid_b1, "p_a_given_b0": sigmoid_b0},
        "quadratic": {"p_a_given_b1": quadratic_b1, "p_a_given_b0": quadratic_b0},
    }

    # 4. Compute Brier scores
    # Baseline: use Two-Stage bracket result
    baseline_p_given_actual = twostage["p_a_given_b1"] if resolution_b == 1.0 else twostage["p_a_given_b0"]
    brier_baseline = compute_brier(baseline_p_given_actual, resolution_a)

    # Strength methods
    linear_p_given_actual = linear_b1 if resolution_b == 1.0 else linear_b0
    sigmoid_p_given_actual = sigmoid_b1 if resolution_b == 1.0 else sigmoid_b0
    quadratic_p_given_actual = quadratic_b1 if resolution_b == 1.0 else quadratic_b0

    brier_linear = compute_brier(linear_p_given_actual, resolution_a)
    brier_sigmoid = compute_brier(sigmoid_p_given_actual, resolution_a)
    brier_quadratic = compute_brier(quadratic_p_given_actual, resolution_a)

    result["brier"] = {
        "baseline": brier_baseline,
        "strength_linear": brier_linear,
        "strength_sigmoid": brier_sigmoid,
        "strength_quadratic": brier_quadratic,
        "improvement_linear": brier_baseline - brier_linear,
        "improvement_sigmoid": brier_baseline - brier_sigmoid,
        "improvement_quadratic": brier_baseline - brier_quadratic,
    }

    return result


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        print("\nNo valid results to summarize.")
        return

    print("\n" + "=" * 70)
    print("STRENGTH ELICITATION EXPERIMENT RESULTS")
    print("=" * 70)

    # Group by category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Overall summary table
    print("\n" + "-" * 70)
    print("BRIER SCORES BY CATEGORY AND METHOD")
    print("-" * 70)

    print("\n| Category | N | Baseline | Linear | Sigmoid | Quadratic |")
    print("|----------|---|----------|--------|---------|-----------|")

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue
        cat_results = by_category[cat]
        n = len(cat_results)

        mean_baseline = sum(r["brier"]["baseline"] for r in cat_results) / n
        mean_linear = sum(r["brier"]["strength_linear"] for r in cat_results) / n
        mean_sigmoid = sum(r["brier"]["strength_sigmoid"] for r in cat_results) / n
        mean_quadratic = sum(r["brier"]["strength_quadratic"] for r in cat_results) / n

        print(f"| {cat:8} | {n} | {mean_baseline:.4f}   | {mean_linear:.4f} | {mean_sigmoid:.4f}  | {mean_quadratic:.4f}    |")

    # Improvement table
    print("\n" + "-" * 70)
    print("BRIER IMPROVEMENT (Baseline - Strength)")
    print("-" * 70)

    print("\n| Category | Linear | Sigmoid | Quadratic |")
    print("|----------|--------|---------|-----------|")

    for cat in ["strong", "weak"]:
        if cat not in by_category:
            continue
        cat_results = [r for r in by_category[cat] if r["strength_elicitation"] is not None]
        if not cat_results:
            continue
        n = len(cat_results)

        mean_imp_linear = sum(r["brier"]["improvement_linear"] for r in cat_results) / n
        mean_imp_sigmoid = sum(r["brier"]["improvement_sigmoid"] for r in cat_results) / n
        mean_imp_quadratic = sum(r["brier"]["improvement_quadratic"] for r in cat_results) / n

        print(f"| {cat:8} | {mean_imp_linear:+.4f} | {mean_imp_sigmoid:+.4f}  | {mean_imp_quadratic:+.4f}    |")

    # Strength distribution
    print("\n" + "-" * 70)
    print("STRENGTH RATINGS BY CATEGORY")
    print("-" * 70)

    for cat in ["strong", "weak"]:
        if cat not in by_category:
            continue
        cat_results = [r for r in by_category[cat] if r["strength_elicitation"] is not None]
        if not cat_results:
            continue

        strengths = [r["strength_elicitation"]["strength"] for r in cat_results]
        mean_strength = sum(strengths) / len(strengths)
        min_strength = min(strengths)
        max_strength = max(strengths)

        print(f"\n{cat.upper()} (n={len(cat_results)}):")
        print(f"  Mean strength: {mean_strength:.1f}")
        print(f"  Range: {min_strength} - {max_strength}")
        print(f"  Distribution: {sorted(strengths)}")


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
    weak = by_category.get("weak", [])
    none = by_category.get("none", [])

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # 1. Does strength elicitation improve Brier?
    strong_correlated = [r for r in strong if r["strength_elicitation"] is not None]
    if strong_correlated:
        mean_imp_linear = sum(r["brier"]["improvement_linear"] for r in strong_correlated) / len(strong_correlated)
        mean_imp_sigmoid = sum(r["brier"]["improvement_sigmoid"] for r in strong_correlated) / len(strong_correlated)
        mean_imp_quadratic = sum(r["brier"]["improvement_quadratic"] for r in strong_correlated) / len(strong_correlated)

        print(f"\n1. DOES STRENGTH ELICITATION IMPROVE BRIER? (strong pairs)")
        print(f"   Linear:    {mean_imp_linear:+.4f}")
        print(f"   Sigmoid:   {mean_imp_sigmoid:+.4f}")
        print(f"   Quadratic: {mean_imp_quadratic:+.4f}")

        best_method = max([("linear", mean_imp_linear), ("sigmoid", mean_imp_sigmoid), ("quadratic", mean_imp_quadratic)], key=lambda x: x[1])
        if best_method[1] > 0:
            print(f"   -> YES, {best_method[0]} improves by {best_method[1]:+.4f}")
        else:
            print(f"   -> NO, all methods are equal or worse than baseline")

    # 2. Do strength ratings discriminate?
    strong_strengths = [r["strength_elicitation"]["strength"] for r in strong if r["strength_elicitation"]]
    weak_strengths = [r["strength_elicitation"]["strength"] for r in weak if r["strength_elicitation"]]

    if strong_strengths and weak_strengths:
        mean_strong = sum(strong_strengths) / len(strong_strengths)
        mean_weak = sum(weak_strengths) / len(weak_strengths)

        print(f"\n2. DO STRENGTH RATINGS DISCRIMINATE?")
        print(f"   Mean strength (strong pairs): {mean_strong:.1f}")
        print(f"   Mean strength (weak pairs):   {mean_weak:.1f}")
        print(f"   Difference: {mean_strong - mean_weak:.1f}")

        if mean_strong > mean_weak:
            print(f"   -> YES, strong pairs rated higher by {mean_strong - mean_weak:.1f} points")
        else:
            print(f"   -> NO, weak pairs actually rated equal or higher")

    # 3. Which formula works best?
    if strong_correlated:
        wins_linear = sum(1 for r in strong_correlated if r["brier"]["improvement_linear"] > 0)
        wins_sigmoid = sum(1 for r in strong_correlated if r["brier"]["improvement_sigmoid"] > 0)
        wins_quadratic = sum(1 for r in strong_correlated if r["brier"]["improvement_quadratic"] > 0)
        n = len(strong_correlated)

        print(f"\n3. WHICH FORMULA WORKS BEST? (win rate on strong pairs)")
        print(f"   Linear:    {wins_linear}/{n} ({100*wins_linear/n:.0f}%)")
        print(f"   Sigmoid:   {wins_sigmoid}/{n} ({100*wins_sigmoid/n:.0f}%)")
        print(f"   Quadratic: {wins_quadratic}/{n} ({100*wins_quadratic/n:.0f}%)")

    # 4. Where does it fail?
    print(f"\n4. WHERE DOES IT FAIL?")

    # Check for direction mismatches
    if strong_correlated:
        direction_mismatches = []
        for r in strong_correlated:
            baseline_dir = r["baseline"]["direction"]
            strength_dir = r["strength_elicitation"]["direction"]
            if baseline_dir != strength_dir:
                direction_mismatches.append(r)

        if direction_mismatches:
            print(f"   Direction mismatches: {len(direction_mismatches)}/{len(strong_correlated)}")
            for r in direction_mismatches[:2]:
                print(f"     - Baseline: {r['baseline']['direction']}, Strength: {r['strength_elicitation']['direction']}")
                print(f"       Reason: {r.get('reason', 'N/A')[:50]}")
        else:
            print(f"   Direction mismatches: 0/{len(strong_correlated)}")

    # 5. Key finding
    print(f"\n5. KEY FINDING")
    if strong_correlated:
        # Is bottleneck in strength estimation, formula mapping, or base probability?
        # Compare variance in improvements

        # Check if base P(A) differs
        pa_diffs = [abs(r["baseline"]["p_a"] - r["strength_elicitation"]["p_a"]) for r in strong_correlated]
        mean_pa_diff = sum(pa_diffs) / len(pa_diffs) if pa_diffs else 0

        print(f"   Mean |P(A) baseline - P(A) strength|: {mean_pa_diff:.3f}")

        if mean_pa_diff > 0.1:
            print(f"   -> BOTTLENECK: Base probability estimation differs significantly")
        elif mean_strong <= mean_weak + 0.5 if strong_strengths and weak_strengths else False:
            print(f"   -> BOTTLENECK: Strength ratings don't discriminate")
        else:
            print(f"   -> BOTTLENECK: Formula mapping (try different max_shift or formula)")


async def main():
    parser = argparse.ArgumentParser(description="Strength elicitation experiment")
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
        args.output = f"experiments/magnitude/strength-elicitation/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running strength elicitation experiment on {len(pairs)} pairs")
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
                if result["strength_elicitation"]:
                    strength = result["strength_elicitation"]["strength"]
                    imp = result["brier"]["improvement_linear"]
                    print(f"    Strength: {strength}, Improvement (linear): {imp:+.4f}")
                else:
                    print(f"    Independent (no strength elicitation)")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print outputs
    print_summary(results)
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
