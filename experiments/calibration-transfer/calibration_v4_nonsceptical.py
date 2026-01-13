#!/usr/bin/env python3
"""Calibration Transfer Experiment v4: Non-Skeptical Stage 1.

Uses a non-skeptical Stage 1 prompt to find more "related" pairs, enabling
a proper test of conditional calibration hypothesis.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/calibration-transfer/calibration_v4_nonsceptical.py
    uv run python experiments/calibration-transfer/calibration_v4_nonsceptical.py --skip-elicitation
"""

import argparse
import asyncio
import json
import random
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

load_dotenv()

import litellm

# Configuration
CUTOFF_DATE = date(2025, 10, 1)
DB_PATH = Path("data/forecastbench.db")
DATA_DIR = Path("experiments/calibration-transfer/data")
RESULTS_DIR = Path("experiments/calibration-transfer/results")

# NON-SKEPTICAL Stage 1 prompt (high recall, accepts false positives)
STAGE1_NONSCEPTICAL_PROMPT = """You are evaluating whether two forecasting questions might be related.

Question A: {q_a}
Question B: {q_b}

Consider whether these questions might share any connection:
- Direct causal relationship (A causes B or vice versa)
- Common cause (both affected by the same underlying factor)
- Conceptual overlap (same domain, related outcomes)
- Any plausible mechanism linking them

Be INCLUSIVE in your assessment. We're collecting potential relationships for further analysis. It's better to flag a relationship that turns out to be weak than to miss a real one.

Respond with:
- "RELATED" if there's any plausible connection (even weak or indirect)
- "UNRELATED" if the questions are clearly independent (different domains, no conceivable mechanism)

Return only JSON: {{"classification": "related|unrelated", "reasoning": "one sentence"}}"""


STAGE2_CONDITIONAL_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

These questions may be related. Estimate:

1. P(B) - The unconditional probability that B resolves YES
2. P(B|A=YES) - The probability B resolves YES, given A resolves YES
3. P(B|A=NO) - The probability B resolves YES, given A resolves NO

Constraint: P(B) MUST fall between P(B|A=YES) and P(B|A=NO).

Return JSON: {{
  "p_b": 0.XX,
  "p_b_given_a1": 0.XX,
  "p_b_given_a0": 0.XX
}}"""


STAGE2_BASELINE_PROMPT = """Question: "{q_b}"

What is the probability this resolves YES? Give only your estimate, no explanation.

Return only JSON: {{"p_b": 0.XX}}"""


async def call_llm(prompt: str, model: str) -> str | None:
    """Make an LLM call."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"    LLM Error: {e}")
        return None


def extract_json(content: str) -> dict | None:
    """Extract JSON from response."""
    if not content:
        return None
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
    return None


async def elicit_p_b_given_a(q_a: str, q_b: str, model: str) -> dict | None:
    """Two-Stage elicitation with non-skeptical Stage 1."""
    # Stage 1: Classify with non-skeptical prompt
    prompt1 = STAGE1_NONSCEPTICAL_PROMPT.format(q_a=q_a, q_b=q_b)
    content1 = await call_llm(prompt1, model)
    stage1 = extract_json(content1)

    if stage1 is None or "classification" not in stage1:
        return None

    classification = stage1["classification"].lower().strip()

    # Normalize to related/unrelated
    if classification in ["related", "correlated"]:
        classification = "related"
    elif classification in ["unrelated", "independent"]:
        classification = "unrelated"

    stage1["classification"] = classification

    # Stage 2: Elicit
    if classification == "related":
        prompt2 = STAGE2_CONDITIONAL_PROMPT.format(q_a=q_a, q_b=q_b)
        content2 = await call_llm(prompt2, model)
        stage2 = extract_json(content2)

        if stage2 is None or "p_b_given_a1" not in stage2:
            return None

        return {
            "stage1": stage1,
            "p_b_given_a1": stage2["p_b_given_a1"],
            "p_b_given_a0": stage2.get("p_b_given_a0", stage2.get("p_b", 0.5)),
            "p_b": stage2.get("p_b", 0.5),
            "method": "conditional",
        }
    else:
        prompt2 = STAGE2_BASELINE_PROMPT.format(q_b=q_b)
        content2 = await call_llm(prompt2, model)
        stage2 = extract_json(content2)

        if stage2 is None or "p_b" not in stage2:
            return None

        return {
            "stage1": stage1,
            "p_b_given_a1": stage2["p_b"],
            "p_b_given_a0": stage2["p_b"],
            "p_b": stage2["p_b"],
            "method": "baseline",
        }


def extract_postcutoff_questions() -> list[dict]:
    """Extract post-cutoff binary questions with resolutions (clean data)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
    SELECT DISTINCT
        q.id,
        q.text,
        q.source,
        q.category,
        r.value as resolution_value,
        r.date as resolution_date
    FROM resolutions r
    JOIN questions q ON r.question_id = q.id
    WHERE r.date >= ?
      AND q.question_type = 'binary'
      AND r.value IN (0.0, 1.0)
    ORDER BY r.date DESC
    """

    cursor = conn.execute(query, (CUTOFF_DATE.isoformat(),))
    questions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return questions


def generate_stratified_pairs(questions: list[dict], max_pairs: int = 200) -> list[dict]:
    """Generate pairs with stratified sampling for resolution diversity."""
    # Split by resolution
    yes_qs = [q for q in questions if q["resolution_value"] == 1.0]
    no_qs = [q for q in questions if q["resolution_value"] == 0.0]

    print(f"Questions: {len(yes_qs)} YES, {len(no_qs)} NO")

    random.seed(42)

    pairs = []

    # A=YES, B=YES pairs
    n_yes_yes = max_pairs // 4
    for _ in range(n_yes_yes):
        if len(yes_qs) < 2:
            break
        q_a, q_b = random.sample(yes_qs, 2)
        pairs.append({
            "a_id": q_a["id"],
            "b_id": q_b["id"],
            "a_text": q_a["text"],
            "b_text": q_b["text"],
            "a_resolution": q_a["resolution_value"],
            "b_resolution": q_b["resolution_value"],
        })

    # A=YES, B=NO pairs
    n_yes_no = max_pairs // 4
    for _ in range(n_yes_no):
        if not yes_qs or not no_qs:
            break
        q_a = random.choice(yes_qs)
        q_b = random.choice(no_qs)
        pairs.append({
            "a_id": q_a["id"],
            "b_id": q_b["id"],
            "a_text": q_a["text"],
            "b_text": q_b["text"],
            "a_resolution": q_a["resolution_value"],
            "b_resolution": q_b["resolution_value"],
        })

    # A=NO, B=YES pairs
    n_no_yes = max_pairs // 4
    for _ in range(n_no_yes):
        if not no_qs or not yes_qs:
            break
        q_a = random.choice(no_qs)
        q_b = random.choice(yes_qs)
        pairs.append({
            "a_id": q_a["id"],
            "b_id": q_b["id"],
            "a_text": q_a["text"],
            "b_text": q_b["text"],
            "a_resolution": q_a["resolution_value"],
            "b_resolution": q_b["resolution_value"],
        })

    # A=NO, B=NO pairs
    n_no_no = max_pairs // 4
    for _ in range(n_no_no):
        if len(no_qs) < 2:
            break
        q_a, q_b = random.sample(no_qs, 2)
        pairs.append({
            "a_id": q_a["id"],
            "b_id": q_b["id"],
            "a_text": q_a["text"],
            "b_text": q_b["text"],
            "a_resolution": q_a["resolution_value"],
            "b_resolution": q_b["resolution_value"],
        })

    random.shuffle(pairs)

    # Count
    a1 = sum(1 for p in pairs if p["a_resolution"] == 1.0)
    b1 = sum(1 for p in pairs if p["b_resolution"] == 1.0)
    print(f"Generated {len(pairs)} pairs: A=1: {a1}, B=1: {b1}")

    return pairs


async def run_elicitation(pairs: list[dict], model: str) -> list[dict]:
    """Run Two-Stage elicitation with non-skeptical Stage 1 on all pairs."""
    results = []
    semaphore = asyncio.Semaphore(5)

    async def process_pair(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair['a_id'][:20]}... x {pair['b_id'][:20]}...")

            elicitation = await elicit_p_b_given_a(
                pair["a_text"], pair["b_text"], model
            )

            if elicitation is None:
                return {**pair, "error": "Elicitation failed"}

            # Ground truth depends on A's resolution
            if pair["a_resolution"] == 1.0:
                p_llm = elicitation["p_b_given_a1"]
            else:
                p_llm = elicitation["p_b_given_a0"]

            p_correct = pair["b_resolution"]
            error = p_llm - p_correct

            result = {
                **pair,
                "elicitation": elicitation,
                "p_llm": p_llm,
                "p_correct": p_correct,
                "error": error,
                "classification": elicitation["stage1"]["classification"],
                "method": elicitation["method"],
            }

            print(f"    [{elicitation['stage1']['classification'][:3]}] P(B|A={int(pair['a_resolution'])}) = {p_llm:.2f}, actual = {p_correct}, error = {error:+.2f}")
            return result

    results = await asyncio.gather(*[
        process_pair(i, p) for i, p in enumerate(pairs)
    ])

    return results


def analyze_by_classification(results: list[dict]) -> dict:
    """Analyze errors by LLM classification (related vs unrelated)."""
    valid = [r for r in results if "p_llm" in r and not isinstance(r.get("error"), str)]

    related = [r for r in valid if r["classification"] == "related"]
    unrelated = [r for r in valid if r["classification"] == "unrelated"]

    def compute_stats(group, name):
        if not group:
            return {"name": name, "n": 0, "error": "No data"}

        errors = [r["error"] for r in group]
        p_llm = [r["p_llm"] for r in group]
        p_correct = [r["p_correct"] for r in group]
        a1 = sum(1 for r in group if r["a_resolution"] == 1.0)
        b1 = sum(1 for r in group if r["b_resolution"] == 1.0)

        return {
            "name": name,
            "n": len(group),
            "n_a1": a1,
            "n_b1": b1,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mean_p_llm": float(np.mean(p_llm)),
            "mean_p_correct": float(np.mean(p_correct)),
        }

    return {
        "related": compute_stats(related, "related"),
        "unrelated": compute_stats(unrelated, "unrelated"),
        "all": compute_stats(valid, "all"),
    }


def train_calibration(results: list[dict], group: str = "all") -> dict:
    """Train calibration on all pairs."""
    valid = [r for r in results if "p_llm" in r and not isinstance(r.get("error"), str)]

    if group == "all":
        train_data = valid
    else:
        train_data = [r for r in valid if r["classification"] == group]

    if len(train_data) < 5:
        return {"error": f"Not enough data in {group} group ({len(train_data)} pairs)"}

    X = np.array([[r["p_llm"]] for r in train_data])
    y = np.array([r["p_correct"] for r in train_data])

    model = LinearRegression()
    model.fit(X, y)

    alpha = model.coef_[0]
    beta = model.intercept_

    # Interpretation
    if alpha < 1:
        interp = f"Scale down by {(1-alpha)*100:.0f}%"
    else:
        interp = f"Scale up by {(alpha-1)*100:.0f}%"
    if beta > 0.01:
        interp += f", add +{beta:.2f} baseline"
    elif beta < -0.01:
        interp += f", subtract {-beta:.2f} baseline"

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "interpretation": interp,
        "train_size": len(train_data),
        "train_group": group,
    }


def validate_by_classification(results: list[dict], calibration: dict) -> dict:
    """Validate calibration on each classification group."""
    if "error" in calibration:
        return {"error": calibration["error"]}

    alpha = calibration["alpha"]
    beta = calibration["beta"]

    valid = [r for r in results if "p_llm" in r and not isinstance(r.get("error"), str)]
    related = [r for r in valid if r["classification"] == "related"]
    unrelated = [r for r in valid if r["classification"] == "unrelated"]

    def compute_validation(group, name):
        if len(group) < 2:
            return {"name": name, "n": len(group), "error": "Not enough data"}

        p_llm = np.array([r["p_llm"] for r in group])
        p_correct = np.array([r["p_correct"] for r in group])

        brier_uncal = np.mean((p_llm - p_correct) ** 2)
        p_cal = np.clip(alpha * p_llm + beta, 0, 1)
        brier_cal = np.mean((p_cal - p_correct) ** 2)
        improvement = brier_uncal - brier_cal

        # Bootstrap
        n_bootstrap = 1000
        improvements = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(p_llm), len(p_llm), replace=True)
            bu = np.mean((p_llm[idx] - p_correct[idx]) ** 2)
            bc = np.mean((p_cal[idx] - p_correct[idx]) ** 2)
            improvements.append(bu - bc)

        p_value = float(np.mean(np.array(improvements) <= 0))

        return {
            "name": name,
            "n": len(group),
            "brier_uncalibrated": float(brier_uncal),
            "brier_calibrated": float(brier_cal),
            "improvement": float(improvement),
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
        }

    return {
        "related": compute_validation(related, "related"),
        "unrelated": compute_validation(unrelated, "unrelated"),
        "all": compute_validation(valid, "all"),
    }


def test_conditional_hypothesis(validation: dict) -> dict:
    """Test whether calibration helps more on related pairs than unrelated."""
    related = validation.get("related", {})
    unrelated = validation.get("unrelated", {})

    if "error" in related or "error" in unrelated:
        return {
            "error": "Insufficient data in one or both groups",
            "related_error": related.get("error"),
            "unrelated_error": unrelated.get("error"),
        }

    related_improvement = related.get("improvement", 0)
    unrelated_improvement = unrelated.get("improvement", 0)
    difference = related_improvement - unrelated_improvement

    # Is the hypothesis confirmed?
    related_sig = related.get("significant", False)
    related_better = difference > 0

    if related_sig and related_better:
        verdict = "CONFIRMED - Calibration helps more on related pairs"
    elif related_sig and not related_better:
        verdict = "REVERSED - Calibration helps more on unrelated pairs"
    elif related_better:
        verdict = "WEAK SIGNAL - Related improvement > unrelated, but not significant"
    else:
        verdict = "NO DIFFERENCE - Similar improvement across groups"

    return {
        "related_improvement": related_improvement,
        "unrelated_improvement": unrelated_improvement,
        "difference": difference,
        "related_significant": related_sig,
        "hypothesis_verdict": verdict,
    }


async def main():
    parser = argparse.ArgumentParser(description="Calibration Transfer v4: Non-Skeptical Stage 1")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--limit", type=int, default=200, help="Max pairs")
    parser.add_argument("--skip-elicitation", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("CALIBRATION TRANSFER v4: NON-SKEPTICAL STAGE 1")
    print("=" * 70)
    print("\nGoal: Find more 'related' pairs to test conditional calibration hypothesis")
    print("Using post-cutoff data (clean, no contamination)\n")

    # Step 1: Extract questions
    print("-" * 70)
    print("STEP 1: Extract post-cutoff questions")
    print("-" * 70)
    questions = extract_postcutoff_questions()
    print(f"Found {len(questions)} post-cutoff questions")

    # Step 2: Generate pairs
    print("\n" + "-" * 70)
    print("STEP 2: Generate stratified pairs")
    print("-" * 70)
    pairs = generate_stratified_pairs(questions, max_pairs=args.limit)

    # Step 3: Elicitation
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "v4_pairs_nonsceptical.json"

    if args.skip_elicitation and cache_path.exists():
        print("\n" + "-" * 70)
        print("STEP 3: Loading cached elicitation results")
        print("-" * 70)
        with open(cache_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results")
    else:
        print("\n" + "-" * 70)
        print("STEP 3: Run Two-Stage elicitation (non-skeptical)")
        print("-" * 70)
        results = await run_elicitation(pairs, args.model)

        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {len(results)} results to {cache_path}")

    # Step 4: Analyze by classification
    print("\n" + "-" * 70)
    print("STEP 4: Analyze by classification")
    print("-" * 70)
    analysis = analyze_by_classification(results)

    n_related = analysis["related"]["n"] if "n" in analysis["related"] else 0
    n_unrelated = analysis["unrelated"]["n"] if "n" in analysis["unrelated"] else 0
    n_total = n_related + n_unrelated
    pct_related = n_related / n_total * 100 if n_total > 0 else 0

    print(f"\n>>> Non-skeptical prompt found {n_related} 'related' pairs ({pct_related:.1f}%) <<<")

    for group_name in ["related", "unrelated", "all"]:
        stats = analysis[group_name]
        if "error" in stats:
            print(f"\n{group_name.upper()}: {stats['error']}")
            continue
        print(f"\n{group_name.upper()} (n={stats['n']}):")
        print(f"  A=1: {stats['n_a1']}, B=1: {stats['n_b1']}")
        print(f"  Mean error: {stats['mean_error']:.3f}")
        print(f"  Mean P(LLM): {stats['mean_p_llm']:.3f}")
        print(f"  Mean P(correct): {stats['mean_p_correct']:.3f}")

    # Step 5: Train calibration on ALL pairs
    print("\n" + "-" * 70)
    print("STEP 5: Train calibration on ALL pairs")
    print("-" * 70)
    calibration = train_calibration(results, group="all")

    if "error" in calibration:
        print(f"Error: {calibration['error']}")
    else:
        print(f"Learned: calibrated = {calibration['alpha']:.3f} * raw + {calibration['beta']:.3f}")
        print(f"Interpretation: {calibration['interpretation']}")
        print(f"Trained on {calibration['train_size']} pairs")

        # Compare to v1/v3
        print("\n  Comparison to previous experiments:")
        print(f"    v1 (post-cutoff): α=0.790, β=0.288")
        print(f"    v3 (pre-cutoff):  α=0.996, β=0.158")
        print(f"    v4 (this run):    α={calibration['alpha']:.3f}, β={calibration['beta']:.3f}")

    # Step 6: Validate by classification
    print("\n" + "-" * 70)
    print("STEP 6: Validate by classification")
    print("-" * 70)
    validation = validate_by_classification(results, calibration)

    if "error" in validation:
        print(f"Error: {validation['error']}")
    else:
        print("\n┌─────────────┬─────┬──────────────┬────────────┬────────────┬─────────┐")
        print("│ Group       │  N  │ Uncal Brier  │ Cal Brier  │ Improvement│ p-value │")
        print("├─────────────┼─────┼──────────────┼────────────┼────────────┼─────────┤")
        for group_name in ["related", "unrelated", "all"]:
            v = validation[group_name]
            if "error" in v:
                print(f"│ {group_name:11} │ {v['n']:3} │ {'N/A':12} │ {'N/A':10} │ {'N/A':10} │ {'N/A':7} │")
            else:
                sig_marker = "*" if v['significant'] else " "
                print(f"│ {group_name:11} │ {v['n']:3} │ {v['brier_uncalibrated']:.4f}       │ {v['brier_calibrated']:.4f}     │ {v['improvement']:+.4f}    │ {v['p_value']:.3f}{sig_marker}  │")
        print("└─────────────┴─────┴──────────────┴────────────┴────────────┴─────────┘")

    # Step 7: Test conditional calibration hypothesis
    print("\n" + "-" * 70)
    print("STEP 7: Test conditional calibration hypothesis")
    print("-" * 70)
    hypothesis = test_conditional_hypothesis(validation)

    if "error" in hypothesis:
        print(f"Error: {hypothesis['error']}")
    else:
        print(f"\n  Related improvement:   {hypothesis['related_improvement']:+.4f}")
        print(f"  Unrelated improvement: {hypothesis['unrelated_improvement']:+.4f}")
        print(f"  Difference:            {hypothesis['difference']:+.4f}")
        print(f"\n>>> {hypothesis['hypothesis_verdict']} <<<")

    # Step 8: Verdict
    print("\n" + "-" * 70)
    print("STEP 8: Overall Verdict")
    print("-" * 70)

    # Success criteria
    pct_related_target = 30
    pct_related_ok = pct_related >= pct_related_target

    related_v = validation.get("related", {})
    related_sig = related_v.get("significant", False)
    related_better = hypothesis.get("difference", 0) > 0 if "difference" in hypothesis else False

    if pct_related_ok and related_sig and related_better:
        verdict = "STRONG SUCCESS - Conditional calibration hypothesis confirmed"
    elif pct_related_ok and related_sig:
        verdict = "PARTIAL SUCCESS - Calibration helps on related pairs (not better than unrelated)"
    elif pct_related_ok:
        verdict = "WEAK SIGNAL - Found related pairs but calibration not significant"
    elif pct_related < 10:
        verdict = "FAILURE - Non-skeptical prompt still not finding related pairs"
    else:
        verdict = "PARTIAL SUCCESS - Some related pairs found, mixed calibration results"

    print(f"\n>>> {verdict} <<<")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "analysis": {k: v for k, v in analysis.items()},
        "calibration": calibration,
        "validation": validation,
        "hypothesis_test": hypothesis,
        "verdict": verdict,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "model": args.model,
            "n_pairs": len(results),
            "pct_related": pct_related,
            "post_cutoff": True,
            "prompt_type": "non-skeptical",
        }
    }

    with open(RESULTS_DIR / "calibration_v4_nonsceptical.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR / 'calibration_v4_nonsceptical.json'}")

    return output


if __name__ == "__main__":
    asyncio.run(main())
