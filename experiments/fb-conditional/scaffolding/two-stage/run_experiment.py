#!/usr/bin/env python3
"""Two-stage elicitation experiment.

Tests whether separating independence classification from probability elicitation
can combine Bracket's benefits (100% direction accuracy) with baseline's low
false positive rate (7%).

Usage:
    uv run python experiments/fb-conditional/scaffolding/two-stage/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/two-stage/run_experiment.py --limit 5
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


# Stage 1: Independence classification with skeptical prompt
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


# Stage 2a: Baseline direct elicitation (for independent pairs)
STAGE2_BASELINE_PROMPT = """Question: "{q_a}"

What is the probability this resolves YES? Give only your estimate, no explanation.

Return only JSON: {{"p_a": 0.XX}}"""


# Stage 2b: Bracket elicitation (for correlated pairs)
STAGE2_BRACKET_PROMPT = """Questions:
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


async def call_llm(prompt: str, model: str, thinking: bool) -> str | None:
    """Make an LLM call and return the content."""
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


async def stage1_classify(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Stage 1: Classify pair as correlated or independent."""
    prompt = STAGE1_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "classification" in result:
        result["classification"] = result["classification"].lower().strip()
        return result
    return None


async def stage2_baseline(q_a: str, model: str, thinking: bool) -> dict | None:
    """Stage 2a: Baseline direct elicitation for independent pairs."""
    prompt = STAGE2_BASELINE_PROMPT.format(q_a=q_a)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and "p_a" in result:
        return {
            "method": "baseline",
            "p_a": result["p_a"],
            "p_a_given_b1": None,
            "p_a_given_b0": None,
            "direction": None,
        }
    return None


async def stage2_bracket(q_a: str, q_b: str, model: str, thinking: bool) -> dict | None:
    """Stage 2b: Bracket elicitation for correlated pairs."""
    prompt = STAGE2_BRACKET_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model, thinking)
    result = extract_json(content)

    if result and all(k in result for k in ["direction", "p_a", "p_a_given_b1", "p_a_given_b0"]):
        return {
            "method": "bracket",
            "p_a": result["p_a"],
            "p_a_given_b1": result["p_a_given_b1"],
            "p_a_given_b0": result["p_a_given_b0"],
            "direction": result["direction"].lower().strip(),
        }
    return None


def compute_brier(stage2: dict, resolution_a: float, resolution_b: float) -> float:
    """Compute Brier score based on routing method."""
    if stage2["method"] == "baseline":
        # Use P(A) directly
        return (stage2["p_a"] - resolution_a) ** 2
    else:
        # Use P(A|B=actual)
        p_a_given_b_actual = stage2["p_a_given_b1"] if resolution_b == 1.0 else stage2["p_a_given_b0"]
        return (p_a_given_b_actual - resolution_a) ** 2


def compute_brier_improvement(stage2: dict, resolution_a: float, resolution_b: float) -> float:
    """Compute Brier improvement (independence baseline - conditional/direct)."""
    brier_actual = compute_brier(stage2, resolution_a, resolution_b)
    brier_independence = (stage2["p_a"] - resolution_a) ** 2
    return brier_independence - brier_actual


def check_direction_correct(stage2: dict) -> bool | None:
    """Check if stated direction matches probability pattern."""
    if stage2["method"] != "bracket":
        return None

    p_a_given_b1 = stage2["p_a_given_b1"]
    p_a_given_b0 = stage2["p_a_given_b0"]
    diff = p_a_given_b1 - p_a_given_b0

    if abs(diff) < 0.02:
        return stage2["direction"] == "independent"
    elif diff > 0:
        return stage2["direction"] == "positive"
    else:
        return stage2["direction"] == "negative"


def check_false_positive(stage1: dict, stage2: dict) -> bool:
    """Check if this is a false positive (independent pair treated as correlated with sensitivity)."""
    if stage1["classification"] == "independent":
        return False  # Routed to baseline, no false positive possible

    # Routed to bracket - check sensitivity
    if stage2["method"] == "bracket" and stage2["p_a_given_b1"] is not None:
        sensitivity = abs(stage2["p_a_given_b1"] - stage2["p_a_given_b0"])
        return sensitivity > 0.05

    return False


async def run_pair(pair: dict, model: str, thinking: bool) -> dict:
    """Run two-stage elicitation on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]

    # Stage 1: Classification
    stage1 = await stage1_classify(q_a, q_b, model, thinking)

    if stage1 is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Stage 1 classification failed",
        }

    # Stage 2: Route based on classification
    if stage1["classification"] == "independent":
        stage2 = await stage2_baseline(q_a, model, thinking)
    else:
        stage2 = await stage2_bracket(q_a, q_b, model, thinking)

    if stage2 is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "stage1": stage1,
            "error": "Stage 2 elicitation failed",
        }

    # Compute metrics
    brier = compute_brier(stage2, pair["resolution_a"], pair["resolution_b"])
    brier_improvement = compute_brier_improvement(stage2, pair["resolution_a"], pair["resolution_b"])
    direction_correct = check_direction_correct(stage2)
    false_positive = check_false_positive(stage1, stage2)

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": pair["resolution_a"],
        "resolution_b": pair["resolution_b"],
        "stage1": stage1,
        "stage2": stage2,
        "derived": {
            "brier": brier,
            "brier_improvement": brier_improvement,
            "direction_correct": direction_correct,
            "false_positive": false_positive,
        },
    }


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        print("\nNo valid results to summarize.")
        return

    print("\n" + "=" * 70)
    print("TWO-STAGE ELICITATION EXPERIMENT RESULTS")
    print("=" * 70)

    # Group by category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Stage 1 Classification accuracy
    print("\n" + "-" * 70)
    print("STAGE 1: INDEPENDENCE CLASSIFICATION")
    print("-" * 70)

    print("\n┌──────────┬───────────────────────┬────────────────────────┬──────────┐")
    print("│ Category │ Classified Correlated │ Classified Independent │ Accuracy │")
    print("├──────────┼───────────────────────┼────────────────────────┼──────────┤")

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue
        cat_results = by_category[cat]
        n = len(cat_results)
        correlated = sum(1 for r in cat_results if r["stage1"]["classification"] == "correlated")
        independent = n - correlated

        # Accuracy: strong should be correlated, none should be independent
        if cat == "strong":
            accuracy = 100 * correlated / n
        elif cat == "none":
            accuracy = 100 * independent / n
        else:
            accuracy = None

        acc_str = f"{accuracy:.0f}%" if accuracy is not None else "—"
        print(f"│ {cat:8} │ {correlated:>2}/{n:<2}                  │ {independent:>2}/{n:<2}                   │ {acc_str:>8} │")

    print("└──────────┴───────────────────────┴────────────────────────┴──────────┘")

    # Routing distribution
    total_correlated = sum(1 for r in valid if r["stage1"]["classification"] == "correlated")
    total_independent = len(valid) - total_correlated
    print(f"\nRouting: {total_correlated} to Bracket, {total_independent} to Baseline")

    # Stage 2 results by category
    print("\n" + "-" * 70)
    print("STAGE 2: ELICITATION RESULTS BY CATEGORY")
    print("-" * 70)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n = len(cat_results)

        # Brier improvement
        improvements = [r["derived"]["brier_improvement"] for r in cat_results]
        mean_improvement = sum(improvements) / n if n > 0 else 0
        wins = sum(1 for i in improvements if i > 0)

        # Direction accuracy (for bracket-routed pairs)
        bracket_results = [r for r in cat_results if r["stage2"]["method"] == "bracket"]
        direction_checks = [r["derived"]["direction_correct"] for r in bracket_results if r["derived"]["direction_correct"] is not None]
        direction_correct = sum(1 for d in direction_checks if d)
        direction_total = len(direction_checks)

        # False positives (for none category)
        if cat == "none":
            false_positives = sum(1 for r in cat_results if r["derived"]["false_positive"])
            fp_pct = 100 * false_positives / n if n > 0 else 0
        else:
            false_positives = None
            fp_pct = None

        # Classification breakdown
        to_bracket = sum(1 for r in cat_results if r["stage2"]["method"] == "bracket")
        to_baseline = n - to_bracket

        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Routing: {to_bracket} → Bracket, {to_baseline} → Baseline")
        print(f"  Brier improvement: {mean_improvement:+.4f} (wins: {wins}/{n} = {100*wins/n:.0f}%)")
        if direction_total > 0:
            print(f"  Direction correct (Bracket): {direction_correct}/{direction_total} ({100*direction_correct/direction_total:.0f}%)")
        if fp_pct is not None:
            print(f"  False positives: {false_positives}/{n} ({fp_pct:.0f}%)")


def print_comparison(results: list[dict]):
    """Print comparison table to baselines."""
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

    # Direction accuracy on strong (only bracket-routed)
    strong_bracket = [r for r in strong if r["stage2"]["method"] == "bracket"]
    direction_checks = [r["derived"]["direction_correct"] for r in strong_bracket if r["derived"]["direction_correct"] is not None]
    direction_correct = sum(1 for d in direction_checks if d)
    # Adjust for misclassified strong pairs (they don't get direction check)
    strong_misclassified = len(strong) - len(strong_bracket)
    # Overall direction: count misclassified as "wrong" since they didn't get direction benefit
    overall_direction_pct = 100 * direction_correct / len(strong) if strong else 0

    # False positives on none
    fp_none = sum(1 for r in none if r["derived"]["false_positive"])
    fp_pct = 100 * fp_none / len(none) if none else 0

    # Brier scores
    brier_strong = sum(r["derived"]["brier_improvement"] for r in strong) / len(strong) if strong else 0
    brier_none = sum(r["derived"]["brier_improvement"] for r in none) / len(none) if none else 0

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINES")
    print("=" * 70)

    print("""
┌────────────────────────────┬──────────┬─────────┬───────────┬─────────┐
│           Metric           │ Baseline │ Bracket │ Two-Stage │ Target  │
├────────────────────────────┼──────────┼─────────┼───────────┼─────────┤""")
    print(f"│ Direction correct (strong) │ 57%      │ 100%    │ {overall_direction_pct:>5.0f}%    │ >90%    │")
    print(f"│ False positives (none)     │ 7%       │ 13%     │ {fp_pct:>5.0f}%    │ <10%    │")
    print(f"│ Brier (strong)             │ +0.007   │ +0.013  │ {brier_strong:>+6.3f}   │ >+0.010 │")
    print(f"│ Brier (none)               │ -0.015   │ -0.005  │ {brier_none:>+6.3f}   │ <-0.010 │")
    print("└────────────────────────────┴──────────┴─────────┴───────────┴─────────┘")

    # Success check
    print("\n" + "-" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("-" * 70)
    fp_success = fp_pct < 10
    direction_success = overall_direction_pct > 90
    brier_success = brier_strong > 0.010

    print(f"  False positives < 10%: {'✓' if fp_success else '✗'} ({fp_pct:.0f}%)")
    print(f"  Direction accuracy > 90%: {'✓' if direction_success else '✗'} ({overall_direction_pct:.0f}%)")
    print(f"  Brier (strong) > +0.010: {'✓' if brier_success else '✗'} ({brier_strong:+.3f})")

    if fp_success and direction_success:
        print("\n  → Two-Stage is WORTH IT: combines low FP with high direction accuracy")
    elif not fp_success and not direction_success:
        print("\n  → Two-Stage is NOT worth it: worse than both alternatives")
    else:
        print("\n  → Two-Stage has mixed results: evaluate tradeoffs")


def print_error_analysis(results: list[dict]):
    """Print error analysis."""
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

    # Error types
    strong_fn = [r for r in strong if r["stage1"]["classification"] == "independent"]  # False negative
    none_fp = [r for r in none if r["stage1"]["classification"] == "correlated"]  # False positive
    weak_independent = [r for r in weak if r["stage1"]["classification"] == "independent"]
    weak_correlated = [r for r in weak if r["stage1"]["classification"] == "correlated"]

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print("""
┌───────────────────────────┬───────┬────────────────────────┐
│        Error Type         │ Count │         Impact         │
├───────────────────────────┼───────┼────────────────────────┤""")
    print(f"│ Strong → Independent (FN) │ {len(strong_fn):>5} │ Lost direction benefit │")
    print(f"│ None → Correlated (FP)    │ {len(none_fp):>5} │ Gained false positive  │")
    print(f"│ Weak → Independent        │ {len(weak_independent):>5} │ Neutral?               │")
    print(f"│ Weak → Correlated         │ {len(weak_correlated):>5} │ Neutral?               │")
    print("└───────────────────────────┴───────┴────────────────────────┘")

    # Show reasoning for errors
    if strong_fn:
        print("\nStrong pairs misclassified as independent:")
        for r in strong_fn[:3]:
            print(f"  - {r.get('reason', 'N/A')[:60]}...")
            print(f"    Reasoning: {r['stage1'].get('reasoning', 'N/A')[:60]}")

    if none_fp:
        print("\nNone pairs misclassified as correlated:")
        for r in none_fp[:3]:
            print(f"  - {r.get('reason', 'N/A')[:60]}...")
            print(f"    Reasoning: {r['stage1'].get('reasoning', 'N/A')[:60]}")


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

    # 1. Does the skeptical prompt work?
    none_independent = sum(1 for r in none if r["stage1"]["classification"] == "independent")
    none_independent_pct = 100 * none_independent / len(none) if none else 0
    print(f"\n1. DOES THE SKEPTICAL PROMPT WORK?")
    print(f"   Independence classification on none pairs: {none_independent}/{len(none)} ({none_independent_pct:.0f}%)")
    print(f"   vs Bracket implicit rate: 87%")
    if none_independent_pct > 87:
        print(f"   → YES, skeptical prompt improved independence detection")
    else:
        print(f"   → NO, skeptical prompt did not improve over Bracket")

    # 2. Do we lose strong pair benefits?
    strong_misclassified = sum(1 for r in strong if r["stage1"]["classification"] == "independent")
    strong_correct = len(strong) - strong_misclassified
    print(f"\n2. DO WE LOSE STRONG PAIR BENEFITS?")
    print(f"   Strong pairs correctly routed to Bracket: {strong_correct}/{len(strong)} ({100*strong_correct/len(strong):.0f}%)")
    if strong_misclassified > 0:
        # Brier impact of misclassified
        misclassified_brier = [r["derived"]["brier_improvement"] for r in strong if r["stage1"]["classification"] == "independent"]
        mean_misclassified_brier = sum(misclassified_brier) / len(misclassified_brier) if misclassified_brier else 0
        print(f"   Misclassified pairs ({strong_misclassified}): mean Brier improvement = {mean_misclassified_brier:+.3f}")

    # 3. Net false positive rate
    actual_fp = sum(1 for r in none if r["derived"]["false_positive"])
    actual_fp_pct = 100 * actual_fp / len(none) if none else 0
    print(f"\n3. NET FALSE POSITIVE RATE")
    print(f"   None pairs with sensitivity > 0.05: {actual_fp}/{len(none)} ({actual_fp_pct:.0f}%)")
    print(f"   vs Bracket: 13%, vs Baseline: 7%")
    if actual_fp_pct < 10:
        print(f"   → SUCCESS: Below 10% target")
    elif actual_fp_pct < 13:
        print(f"   → PARTIAL: Better than Bracket but above 10% target")
    else:
        print(f"   → FAILURE: Not better than Bracket")

    # 4. Is two-stage worth the complexity?
    brier_strong = sum(r["derived"]["brier_improvement"] for r in strong) / len(strong) if strong else 0
    brier_none = sum(r["derived"]["brier_improvement"] for r in none) / len(none) if none else 0
    print(f"\n4. IS TWO-STAGE WORTH THE COMPLEXITY?")
    print(f"   Overall Brier:")
    print(f"     Strong: {brier_strong:+.3f} (Bracket: +0.013, Baseline: +0.007)")
    print(f"     None:   {brier_none:+.3f} (Bracket: -0.005, Baseline: -0.015)")

    # 5. Classification reasoning quality
    print(f"\n5. CLASSIFICATION REASONING QUALITY")
    print("   Sample reasoning for each category:")
    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue
        sample = by_category[cat][0] if by_category[cat] else None
        if sample:
            reasoning = sample["stage1"].get("reasoning", "N/A")[:70]
            classification = sample["stage1"]["classification"]
            print(f"   [{cat}] {classification}: {reasoning}...")

    # 6. Recommendation
    strong_correct_pct = 100 * strong_correct / len(strong) if strong else 0
    print(f"\n6. RECOMMENDATION")
    if actual_fp_pct < 10 and strong_correct_pct > 90:
        print("   → USE TWO-STAGE: Achieves both low FP and high direction accuracy")
    elif actual_fp_pct >= 13:
        print("   → USE PURE BRACKET: Two-stage doesn't reduce FP enough to justify complexity")
    elif strong_correct_pct <= 90:
        print("   → USE PURE BRACKET: Too many strong pairs misclassified")
    else:
        print("   → EVALUATE TRADEOFFS: Mixed results, depends on use case")


async def main():
    parser = argparse.ArgumentParser(description="Two-stage elicitation experiment")
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
        args.output = f"experiments/fb-conditional/scaffolding/two-stage/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running two-stage elicitation experiment on {len(pairs)} pairs")
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
                classification = result["stage1"]["classification"]
                method = result["stage2"]["method"]
                improvement = result["derived"]["brier_improvement"]
                print(f"    Stage1: {classification} → Stage2: {method} | improvement: {improvement:+.3f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print outputs
    print_summary(results)
    print_comparison(results)
    print_error_analysis(results)
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
