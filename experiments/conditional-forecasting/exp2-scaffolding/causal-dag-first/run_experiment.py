#!/usr/bin/env python3
"""Causal DAG First experiment.

Tests whether forcing explicit causal structure commitment before forecasting
improves direction accuracy and reduces hallucinated correlations.

Usage:
    uv run python experiments/fb-conditional/scaffolding/causal-dag-first/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/causal-dag-first/run_experiment.py --limit 5
    uv run python experiments/fb-conditional/scaffolding/causal-dag-first/run_experiment.py --generate-ground-truth
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


CAUSAL_DAG_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

STEP 1: CAUSAL STRUCTURE

What is the causal relationship between these questions? Choose exactly one:

(a) A → B: A's outcome causally influences B's outcome
(b) B → A: B's outcome causally influences A's outcome
(c) A ← C → B: Common cause (some third factor influences both)
(d) Independent: No meaningful causal connection

STEP 2: MECHANISM (skip if independent)

If you chose (a), (b), or (c), describe the causal mechanism in one sentence.

STEP 3: DIRECTION OF CORRELATION (skip if independent)

Given your causal structure:
- If A=YES, does this make B=YES more likely or less likely?
- Answer: "positive" (both move together) or "negative" (they move opposite)

STEP 4: PROBABILITIES

Estimate:
- P(A): probability A resolves YES
- P(B): probability B resolves YES
- P(A|B=YES): probability A resolves YES, given B resolved YES
- P(A|B=NO): probability A resolves YES, given B resolved NO

Your conditionals MUST be consistent with your stated causal structure and correlation direction.

Return JSON:
{{
  "structure": "a_causes_b|b_causes_a|common_cause|independent",
  "mechanism": "..." or null,
  "correlation_direction": "positive|negative" or null,
  "p_a": 0.XX,
  "p_b": 0.XX,
  "p_a_given_b1": 0.XX,
  "p_a_given_b0": 0.XX
}}"""


GROUND_TRUTH_PROMPT = """You are labeling the causal structure between two forecasting questions for a research study.

Questions:
- A: "{q_a}"
- B: "{q_b}"

Relationship category provided by human: {category}
Reason provided: {reason}

What is the TRUE causal relationship? Choose exactly one:

(a) a_causes_b: A's outcome would causally influence B's outcome (A → B)
(b) b_causes_a: B's outcome would causally influence A's outcome (B → A)
(c) common_cause: Both are influenced by a common third factor (A ← C → B)
(d) independent: No meaningful causal connection exists

Important considerations:
- For "strong" pairs: there IS a causal relationship - determine which type
- For "weak" pairs: may have common cause or weak causal link
- For "none" pairs: should be "independent"
- Temporal order matters: earlier events can cause later ones, not vice versa
- "common_cause" means both are affected by the same underlying factor

Return only valid JSON: {{"structure": "a_causes_b|b_causes_a|common_cause|independent", "reasoning": "one sentence"}}"""


async def generate_ground_truth(pair: dict, model: str = "claude-sonnet-4-20250514") -> dict:
    """Generate ground truth causal structure label for a pair."""
    prompt = GROUND_TRUTH_PROMPT.format(
        q_a=pair["text_a"],
        q_b=pair["text_b"],
        category=pair.get("category", "unknown"),
        reason=pair.get("reason", "")
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content

        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "structure": result.get("structure", "unknown"),
                "reasoning": result.get("reasoning", "")
            }
    except Exception as e:
        print(f"    Ground truth error: {e}")

    # Default based on category
    if pair.get("category") == "none":
        return {"structure": "independent", "reasoning": "Unrelated questions"}
    return {"structure": "unknown", "reasoning": "Failed to generate"}


async def elicit_causal_dag(
    q_a: str,
    q_b: str,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> dict | None:
    """Elicit causal DAG and probabilities for a pair of questions."""
    prompt = CAUSAL_DAG_PROMPT.format(q_a=q_a, q_b=q_b)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 3000}
        kwargs["temperature"] = 1
    else:
        kwargs["temperature"] = 0.3

    try:
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content

        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            required_keys = {"structure", "p_a", "p_a_given_b1", "p_a_given_b0"}
            if required_keys.issubset(result.keys()):
                result["structure"] = normalize_structure(result["structure"])
                if result.get("correlation_direction"):
                    result["correlation_direction"] = result["correlation_direction"].lower().strip()
                return result
    except Exception as e:
        print(f"    Error: {e}")

    return None


def normalize_structure(structure: str) -> str:
    """Normalize structure string to canonical form."""
    s = structure.lower().strip()
    if s in ("a_causes_b", "a causes b", "a->b", "a → b"):
        return "a_causes_b"
    elif s in ("b_causes_a", "b causes a", "b->a", "b → a"):
        return "b_causes_a"
    elif s in ("common_cause", "common cause", "a<-c->b", "a ← c → b"):
        return "common_cause"
    elif s in ("independent", "none", "no connection"):
        return "independent"
    return s


def check_direction_consistent(result: dict) -> bool | None:
    """Check if stated correlation direction matches actual conditionals."""
    direction = result.get("correlation_direction")
    if direction is None or result.get("structure") == "independent":
        return None

    p_a_given_b1 = result["p_a_given_b1"]
    p_a_given_b0 = result["p_a_given_b0"]
    diff = p_a_given_b1 - p_a_given_b0

    eps = 0.02
    if direction == "positive":
        return diff > -eps  # P(A|B=YES) >= P(A|B=NO)
    elif direction == "negative":
        return diff < eps   # P(A|B=YES) <= P(A|B=NO)
    return None


def check_structure_correct(stated: str, ground_truth: str) -> bool | None:
    """Check if stated structure matches ground truth."""
    if ground_truth == "unknown":
        return None
    return stated == ground_truth


def check_arrow_correct(stated: str, ground_truth: str) -> bool | None:
    """Check if causal arrow direction is correct (for directional ground truths)."""
    if ground_truth not in ("a_causes_b", "b_causes_a"):
        return None
    if stated not in ("a_causes_b", "b_causes_a"):
        return None  # Can't assess if they said common_cause or independent
    return stated == ground_truth


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


async def run_pair(pair: dict, ground_truth: dict, model: str, thinking: bool) -> dict:
    """Run causal DAG elicitation on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]

    result = await elicit_causal_dag(q_a, q_b, model=model, thinking=thinking)

    if result is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "category": pair.get("category"),
            "error": "Failed to elicit causal DAG",
        }

    gt_structure = ground_truth.get("structure", "unknown")
    structure_correct = check_structure_correct(result["structure"], gt_structure)
    arrow_correct = check_arrow_correct(result["structure"], gt_structure)
    direction_consistent = check_direction_consistent(result)
    correctly_independent = (
        result["structure"] == "independent"
        if pair.get("category") == "none"
        else None
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
        "ground_truth_structure": gt_structure,
        "ground_truth_reasoning": ground_truth.get("reasoning", ""),
        "response": {
            "structure": result["structure"],
            "mechanism": result.get("mechanism"),
            "correlation_direction": result.get("correlation_direction"),
            "p_a": result["p_a"],
            "p_b": result.get("p_b"),
            "p_a_given_b1": result["p_a_given_b1"],
            "p_a_given_b0": result["p_a_given_b0"],
        },
        "analysis": {
            "structure_correct": structure_correct,
            "arrow_correct": arrow_correct,
            "direction_consistent": direction_consistent,
            "correctly_independent": correctly_independent,
        },
        "brier": brier,
    }


def print_structure_classification_table(results: list[dict]):
    """Print structure classification distribution by category."""
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 70)
    print("STRUCTURE CLASSIFICATION")
    print("=" * 70)

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    print("""
┌──────────┬─────┬─────┬────────┬─────────────┐
│ Category │ A→B │ B→A │ Common │ Independent │
├──────────┼─────┼─────┼────────┼─────────────┤""")

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue
        cat_results = by_category[cat]
        n = len(cat_results)

        structures = [r["response"]["structure"] for r in cat_results]
        a_causes_b = sum(1 for s in structures if s == "a_causes_b")
        b_causes_a = sum(1 for s in structures if s == "b_causes_a")
        common = sum(1 for s in structures if s == "common_cause")
        independent = sum(1 for s in structures if s == "independent")

        print(f"│ {cat.capitalize():<8} │ {100*a_causes_b/n:3.0f}% │ {100*b_causes_a/n:3.0f}% │ {100*common/n:5.0f}%  │ {100*independent/n:10.0f}%  │")

    print("└──────────┴─────┴─────┴────────┴─────────────┘")


def print_accuracy_table(results: list[dict]):
    """Print accuracy metrics by category."""
    valid = [r for r in results if "error" not in r]

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    print("\n" + "-" * 70)
    print("ACCURACY METRICS")
    print("-" * 70)

    print("""
┌─────────────────────────┬────────┬──────┬──────┐
│         Metric          │ Strong │ Weak │ None │
├─────────────────────────┼────────┼──────┼──────┤""")

    # Structure correct
    def pct_or_dash(results_list, key):
        vals = [r["analysis"][key] for r in results_list if r["analysis"][key] is not None]
        if not vals:
            return "—"
        return f"{100*sum(vals)/len(vals):.0f}%"

    strong = by_category.get("strong", [])
    weak = by_category.get("weak", [])
    none = by_category.get("none", [])

    print(f"│ Structure correct       │ {pct_or_dash(strong, 'structure_correct'):>6} │ {pct_or_dash(weak, 'structure_correct'):>4} │ {pct_or_dash(none, 'structure_correct'):>4} │")
    print(f"│ Arrow direction correct │ {pct_or_dash(strong, 'arrow_correct'):>6} │ {'—':>4} │ {'—':>4} │")
    print(f"│ Internal consistency    │ {pct_or_dash(strong, 'direction_consistent'):>6} │ {pct_or_dash(weak, 'direction_consistent'):>4} │ {pct_or_dash(none, 'direction_consistent'):>4} │")

    # Brier improvement
    def brier_str(results_list):
        if not results_list:
            return "—"
        improvements = [r["brier"]["improvement"] for r in results_list]
        return f"{sum(improvements)/len(improvements):+.3f}"

    print(f"│ Brier improvement       │ {brier_str(strong):>6} │ {brier_str(weak):>4} │ {brier_str(none):>4} │")
    print("└─────────────────────────┴────────┴──────┴──────┘")


def print_comparison_table(results: list[dict]):
    """Print comparison to baseline."""
    valid = [r for r in results if "error" not in r]

    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    strong = by_category.get("strong", [])
    none = by_category.get("none", [])

    # Compute metrics
    # Direction correct (using internal consistency as proxy)
    dir_consistent = [r["analysis"]["direction_consistent"] for r in strong if r["analysis"]["direction_consistent"] is not None]
    dir_correct_pct = 100 * sum(dir_consistent) / len(dir_consistent) if dir_consistent else 0

    # False positives (none pairs that were NOT labeled independent)
    false_positives = sum(1 for r in none if r["response"]["structure"] != "independent")
    fp_pct = 100 * false_positives / len(none) if none else 0

    # Brier
    brier_strong = sum(r["brier"]["improvement"] for r in strong) / len(strong) if strong else 0
    brier_none = sum(r["brier"]["improvement"] for r in none) / len(none) if none else 0

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (Sonnet 4 + thinking, direct elicitation)")
    print("=" * 70)

    print("""
┌────────────────────────────┬──────────┬───────────┬───────┐
│           Metric           │ Baseline │ DAG First │ Delta │
├────────────────────────────┼──────────┼───────────┼───────┤""")
    print(f"│ Direction correct (strong) │ 57%      │ {dir_correct_pct:6.0f}%   │ {dir_correct_pct - 57:+5.0f}% │")
    print(f"│ False positives (none)     │ 7%       │ {fp_pct:6.0f}%   │ {fp_pct - 7:+5.0f}% │")
    print(f"│ Brier (strong)             │ +0.007   │ {brier_strong:+7.3f}   │ {brier_strong - 0.007:+5.3f} │")
    print(f"│ Brier (none)               │ -0.015   │ {brier_none:+7.3f}   │ {brier_none - (-0.015):+5.3f} │")
    print("└────────────────────────────┴──────────┴───────────┴───────┘")


def print_analysis(results: list[dict]):
    """Print detailed analysis answering the key research questions."""
    valid = [r for r in results if "error" not in r]

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

    # 1. Independence detection
    independent_none = sum(1 for r in none if r["response"]["structure"] == "independent")
    false_positives = sum(1 for r in none if r["response"]["structure"] != "independent")
    print(f"\n1. INDEPENDENCE DETECTION")
    print(f"   On 'none' pairs:")
    print(f"     Correctly identified as independent: {independent_none}/{len(none)} ({100*independent_none/len(none):.0f}%)")
    print(f"     False positive (hallucinated correlation): {false_positives}/{len(none)} ({100*false_positives/len(none):.0f}%)")
    print(f"   Baseline false positive rate: 7%")
    if 100*false_positives/len(none) < 7:
        print(f"   -> IMPROVED: Explicit structure reduced hallucinated correlations")
    elif 100*false_positives/len(none) > 15:
        print(f"   -> WORSE: Explicit structure prompt may encourage finding correlations")
    else:
        print(f"   -> SIMILAR to baseline")

    # 2. Structure accuracy
    structure_checks = [r["analysis"]["structure_correct"] for r in strong if r["analysis"]["structure_correct"] is not None]
    structure_correct = sum(1 for s in structure_checks if s)
    print(f"\n2. STRUCTURE ACCURACY")
    print(f"   On 'strong' pairs:")
    if structure_checks:
        print(f"     Correct structure: {structure_correct}/{len(structure_checks)} ({100*structure_correct/len(structure_checks):.0f}%)")
    else:
        print(f"     (No ground truth available for comparison)")

    # 3. Arrow confusion
    print(f"\n3. ARROW CONFUSION (strong pairs)")
    structures = [r["response"]["structure"] for r in strong]
    a_causes_b = sum(1 for s in structures if s == "a_causes_b")
    b_causes_a = sum(1 for s in structures if s == "b_causes_a")
    common = sum(1 for s in structures if s == "common_cause")
    independent_strong = sum(1 for s in structures if s == "independent")

    print(f"   Distribution: A→B: {a_causes_b}, B→A: {b_causes_a}, Common: {common}, Independent: {independent_strong}")

    # Check for arrow_correct where applicable
    arrow_checks = [r for r in strong if r["analysis"]["arrow_correct"] is not None]
    if arrow_checks:
        arrow_correct = sum(1 for r in arrow_checks if r["analysis"]["arrow_correct"])
        print(f"   Arrow direction correct (where applicable): {arrow_correct}/{len(arrow_checks)} ({100*arrow_correct/len(arrow_checks):.0f}%)")

    if common > len(strong) * 0.5:
        print(f"   -> WARNING: Model often defaults to 'common cause' ({100*common/len(strong):.0f}%) - possible hedging")

    # 4. Internal consistency
    dir_checks = [r["analysis"]["direction_consistent"] for r in strong if r["analysis"]["direction_consistent"] is not None]
    dir_consistent = sum(1 for d in dir_checks if d)
    print(f"\n4. INTERNAL CONSISTENCY")
    print(f"   Does stated correlation direction match conditional probabilities?")
    if dir_checks:
        print(f"   Strong pairs: {dir_consistent}/{len(dir_checks)} ({100*dir_consistent/len(dir_checks):.0f}%) consistent")

    # 5. Mechanism quality
    print(f"\n5. MECHANISM QUALITY (samples)")
    print("   Strong pairs:")
    for r in strong[:3]:
        mechanism = r["response"].get("mechanism", "N/A")
        if mechanism:
            print(f"     [{r['response']['structure']}] {mechanism[:70]}...")
    print("   None pairs (should be null or minimal):")
    for r in none[:3]:
        mechanism = r["response"].get("mechanism", "N/A")
        if mechanism:
            print(f"     [{r['response']['structure']}] {mechanism[:70] if mechanism else 'null'}...")

    # 6. Does structure predict accuracy?
    print(f"\n6. DOES STRUCTURE PREDICT ACCURACY?")
    # Compare Brier improvement for structure_correct vs structure_incorrect
    correct_brier = [r["brier"]["improvement"] for r in strong if r["analysis"]["structure_correct"] == True]
    incorrect_brier = [r["brier"]["improvement"] for r in strong if r["analysis"]["structure_correct"] == False]

    if correct_brier and incorrect_brier:
        avg_correct = sum(correct_brier) / len(correct_brier)
        avg_incorrect = sum(incorrect_brier) / len(incorrect_brier)
        print(f"   When structure correct: Brier improvement = {avg_correct:+.4f}")
        print(f"   When structure wrong:   Brier improvement = {avg_incorrect:+.4f}")
        if avg_correct > avg_incorrect:
            print(f"   -> Correct structure DOES predict better Brier scores")
        else:
            print(f"   -> Correct structure does NOT reliably predict better Brier")
    else:
        print(f"   (Insufficient data for comparison)")

    # 7. Key finding
    print(f"\n7. KEY FINDING")
    fp_rate = 100*false_positives/len(none) if none else 0
    dir_rate = 100*dir_consistent/len(dir_checks) if dir_checks else 0

    if fp_rate < 10 and dir_rate > 70:
        print("   Explicit causal DAG commitment IMPROVES both independence detection")
        print("   and direction consistency. Causal reasoning is a productive scaffold.")
    elif fp_rate > 15:
        print("   Explicit causal structure prompt INCREASES hallucinated correlations.")
        print("   The prompt may prime models to find relationships that don't exist.")
    elif dir_rate < 50:
        print("   Models can state causal structures but struggle to translate them")
        print("   into consistent probability estimates. The bottleneck is numerical.")
    else:
        print("   Mixed results - explicit causal reasoning shows modest improvements")
        print("   but is not a silver bullet for conditional forecasting.")

    print(f"\n8. CAUSAL CONFUSION HYPOTHESIS")
    print("   This tests whether models detect correlation but confuse causal direction.")
    if arrow_checks:
        arrow_pct = 100*arrow_correct/len(arrow_checks) if arrow_checks else 0
        if arrow_pct < 60:
            print(f"   -> CONFIRMED: Arrow accuracy is {arrow_pct:.0f}% - direction confusion is real")
        elif arrow_pct > 80:
            print(f"   -> REFUTED: Arrow accuracy is {arrow_pct:.0f}% - models identify direction correctly")
        else:
            print(f"   -> INCONCLUSIVE: Arrow accuracy is {arrow_pct:.0f}%")
    else:
        print("   -> Cannot assess without ground truth causal directions")


async def generate_all_ground_truths(pairs: list[dict], model: str) -> dict:
    """Generate ground truth labels for all pairs."""
    print(f"\nGenerating ground truth labels for {len(pairs)} pairs...")

    semaphore = asyncio.Semaphore(5)
    ground_truths = {}

    async def gen_one(i: int, pair: dict):
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:40]}...")
            result = await generate_ground_truth(pair, model=model)
            pair_id = f"{pair['id_a']}_{pair['id_b']}"
            ground_truths[pair_id] = result
            print(f"    -> {result['structure']}")

    await asyncio.gather(*[gen_one(i, p) for i, p in enumerate(pairs)])
    return ground_truths


async def main():
    parser = argparse.ArgumentParser(description="Causal DAG First experiment")
    parser.add_argument("--pairs", type=str, default="experiments/fb-conditional/pairs_filtered.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_false", dest="thinking")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ground-truth-file", type=str, default=None,
                       help="Path to existing ground truth JSON file")
    parser.add_argument("--generate-ground-truth", action="store_true",
                       help="Generate ground truth labels only (don't run experiment)")
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

    # Ground truth handling
    gt_file = Path(args.ground_truth_file) if args.ground_truth_file else Path(
        "experiments/fb-conditional/scaffolding/causal-dag-first/ground_truth.json"
    )

    if args.generate_ground_truth:
        # Generate and save ground truth only
        ground_truths = await generate_all_ground_truths(pairs, model=args.model)
        gt_file.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_file, "w") as f:
            json.dump(ground_truths, f, indent=2)
        print(f"\nGround truth saved to {gt_file}")
        return

    # Load or generate ground truth
    if gt_file.exists():
        print(f"Loading ground truth from {gt_file}")
        with open(gt_file) as f:
            ground_truths = json.load(f)
    else:
        print("No ground truth file found. Generating...")
        ground_truths = await generate_all_ground_truths(pairs, model=args.model)
        gt_file.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_file, "w") as f:
            json.dump(ground_truths, f, indent=2)
        print(f"Ground truth saved to {gt_file}")

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"experiments/fb-conditional/scaffolding/causal-dag-first/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning Causal DAG First experiment on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(5)

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            pair_id = f"{pair['id_a']}_{pair['id_b']}"
            gt = ground_truths.get(pair_id, {"structure": "unknown", "reasoning": ""})
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:40]}...")
            result = await run_pair(pair, gt, args.model, args.thinking)
            if "error" not in result:
                print(f"    Structure: {result['response']['structure']} (GT: {gt['structure']})")
                print(f"    Direction: {result['response'].get('correlation_direction', 'N/A')} | Improvement: {result['brier']['improvement']:+.3f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print outputs
    print_structure_classification_table(results)
    print_accuracy_table(results)
    print_comparison_table(results)
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
