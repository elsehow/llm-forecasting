#!/usr/bin/env python3
"""Calibration Transfer Experiment v3: Pre-Cutoff Data.

Uses pre-cutoff resolutions for resolution diversity, at cost of potential
data contamination. Tests whether magnitude bias patterns are robust to
contamination.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/calibration-transfer/calibration_v3_precutoff.py
    uv run python experiments/calibration-transfer/calibration_v3_precutoff.py --skip-elicitation
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

# Two-Stage prompts
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


STAGE2_CONDITIONAL_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

These questions are correlated. Estimate:

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
    """Two-Stage elicitation for P(B|A)."""
    # Stage 1: Classify
    prompt1 = STAGE1_PROMPT.format(q_a=q_a, q_b=q_b)
    content1 = await call_llm(prompt1, model)
    stage1 = extract_json(content1)

    if stage1 is None or "classification" not in stage1:
        return None

    stage1["classification"] = stage1["classification"].lower().strip()

    # Stage 2: Elicit
    if stage1["classification"] == "correlated":
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


def extract_precutoff_questions() -> list[dict]:
    """Extract pre-cutoff binary questions with resolutions."""
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
    WHERE r.date < ?
      AND q.question_type = 'binary'
      AND r.value IN (0.0, 1.0)
    ORDER BY r.date DESC
    """

    cursor = conn.execute(query, (CUTOFF_DATE.isoformat(),))
    questions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return questions


def generate_stratified_pairs(questions: list[dict], max_pairs: int = 100) -> list[dict]:
    """Generate pairs with stratified sampling for resolution diversity."""
    # Split by resolution
    yes_qs = [q for q in questions if q["resolution_value"] == 1.0]
    no_qs = [q for q in questions if q["resolution_value"] == 0.0]

    print(f"Questions: {len(yes_qs)} YES, {len(no_qs)} NO")

    random.seed(42)

    # Generate pairs where A=YES (so we can compute P(B|A=YES) ground truth)
    # Stratify B to get ~50% YES, ~50% NO
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

    # A=NO, B=YES pairs (for comparison)
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
    """Run Two-Stage elicitation on all pairs."""
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
    """Analyze errors by LLM classification (correlated vs independent)."""
    valid = [r for r in results if "p_llm" in r and not isinstance(r.get("error"), str)]

    correlated = [r for r in valid if r["classification"] == "correlated"]
    independent = [r for r in valid if r["classification"] == "independent"]

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
        "correlated": compute_stats(correlated, "correlated"),
        "independent": compute_stats(independent, "independent"),
        "all": compute_stats(valid, "all"),
    }


def train_calibration(results: list[dict], group: str = "correlated") -> dict:
    """Train calibration on a specific group."""
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
    correlated = [r for r in valid if r["classification"] == "correlated"]
    independent = [r for r in valid if r["classification"] == "independent"]

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
            "significant": p_value < 0.05,
        }

    return {
        "correlated": compute_validation(correlated, "correlated"),
        "independent": compute_validation(independent, "independent"),
        "all": compute_validation(valid, "all"),
    }


async def main():
    parser = argparse.ArgumentParser(description="Calibration Transfer v3: Pre-Cutoff")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--limit", type=int, default=100, help="Max pairs")
    parser.add_argument("--skip-elicitation", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("CALIBRATION TRANSFER v3: PRE-CUTOFF DATA")
    print("=" * 70)
    print("\n⚠️  CONTAMINATION WARNING: Using pre-cutoff resolutions.")
    print("    Models may have seen these outcomes during training.")
    print("    Results are PRELIMINARY only.\n")

    # Step 1: Extract questions
    print("-" * 70)
    print("STEP 1: Extract pre-cutoff questions")
    print("-" * 70)
    questions = extract_precutoff_questions()
    print(f"Found {len(questions)} pre-cutoff questions")

    # Step 2: Generate pairs
    print("\n" + "-" * 70)
    print("STEP 2: Generate stratified pairs")
    print("-" * 70)
    pairs = generate_stratified_pairs(questions, max_pairs=args.limit)

    # Step 3: Elicitation
    cache_path = DATA_DIR / "v3_pairs_precutoff.json"

    if args.skip_elicitation and cache_path.exists():
        print("\n" + "-" * 70)
        print("STEP 3: Loading cached elicitation results")
        print("-" * 70)
        with open(cache_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results")
    else:
        print("\n" + "-" * 70)
        print("STEP 3: Run Two-Stage elicitation")
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

    for group_name in ["correlated", "independent", "all"]:
        stats = analysis[group_name]
        if "error" in stats:
            print(f"\n{group_name.upper()}: {stats['error']}")
            continue
        print(f"\n{group_name.upper()} (n={stats['n']}):")
        print(f"  A=1: {stats['n_a1']}, B=1: {stats['n_b1']}")
        print(f"  Mean error: {stats['mean_error']:.3f}")
        print(f"  Mean P(LLM): {stats['mean_p_llm']:.3f}")
        print(f"  Mean P(correct): {stats['mean_p_correct']:.3f}")

    # Step 5: Train calibration on correlated pairs
    print("\n" + "-" * 70)
    print("STEP 5: Train calibration on correlated pairs")
    print("-" * 70)
    calibration = train_calibration(results, group="correlated")

    if "error" in calibration:
        print(f"Error: {calibration['error']}")
        # Fallback to all pairs
        print("Falling back to all pairs...")
        calibration = train_calibration(results, group="all")

    if "error" not in calibration:
        print(f"Learned: calibrated = {calibration['alpha']:.3f} * raw + {calibration['beta']:.3f}")
        print(f"Interpretation: {calibration['interpretation']}")
        print(f"Trained on {calibration['train_size']} pairs ({calibration['train_group']})")

        # Compare to v1
        print("\n  Comparison to v1 (post-cutoff, full corpus):")
        print(f"    v1: α=0.790, β=0.288")
        print(f"    v3: α={calibration['alpha']:.3f}, β={calibration['beta']:.3f}")

        # Check similarity
        alpha_diff = abs(calibration['alpha'] - 0.790)
        beta_diff = abs(calibration['beta'] - 0.288)
        if alpha_diff < 0.3 and beta_diff < 0.3:
            print("    → SIMILAR: Bias pattern robust to contamination")
        else:
            print("    → DIFFERENT: Contamination may affect bias pattern")

    # Step 6: Validate by classification
    print("\n" + "-" * 70)
    print("STEP 6: Validate by classification")
    print("-" * 70)
    validation = validate_by_classification(results, calibration)

    if "error" in validation:
        print(f"Error: {validation['error']}")
    else:
        print("\n┌─────────────┬────┬──────────────┬────────────┬────────────┬─────────┐")
        print("│ Group       │  N │ Uncal Brier  │ Cal Brier  │ Improvement│ p-value │")
        print("├─────────────┼────┼──────────────┼────────────┼────────────┼─────────┤")
        for group_name in ["correlated", "independent", "all"]:
            v = validation[group_name]
            if "error" in v:
                print(f"│ {group_name:11} │ {v['n']:2} │ {'N/A':12} │ {'N/A':10} │ {'N/A':10} │ {'N/A':7} │")
            else:
                print(f"│ {group_name:11} │ {v['n']:2} │ {v['brier_uncalibrated']:.4f}       │ {v['brier_calibrated']:.4f}     │ {v['improvement']:+.4f}    │ {v['p_value']:.3f}   │")
        print("└─────────────┴────┴──────────────┴────────────┴────────────┴─────────┘")

    # Step 7: Verdict
    print("\n" + "-" * 70)
    print("STEP 7: Verdict")
    print("-" * 70)

    corr_v = validation.get("correlated", {})
    indep_v = validation.get("independent", {})

    corr_improves = corr_v.get("improvement", 0) > 0
    corr_sig = corr_v.get("significant", False)
    corr_better = corr_v.get("improvement", 0) > indep_v.get("improvement", 0)

    # Contamination check
    alpha_similar = abs(calibration.get("alpha", 0) - 0.790) < 0.3 if "alpha" in calibration else False
    beta_similar = abs(calibration.get("beta", 0) - 0.288) < 0.3 if "beta" in calibration else False
    bias_similar = alpha_similar and beta_similar

    if corr_sig and corr_better and bias_similar:
        verdict = "STRONG PRELIMINARY SIGNAL - Calibration helps on correlated pairs, bias pattern matches v1"
    elif corr_improves and bias_similar:
        verdict = "WEAK PRELIMINARY SIGNAL - Calibration helps, bias pattern matches v1 (not significant)"
    elif corr_improves and not bias_similar:
        verdict = "CONTAMINATION RED FLAG - Calibration helps but bias pattern differs from v1"
    elif not corr_improves and not bias_similar:
        verdict = "NO SIGNAL + CONTAMINATION - Results likely corrupt"
    else:
        verdict = "AMBIGUOUS - Mixed results"

    print(f"\n>>> {verdict} <<<")

    # Save results
    output = {
        "analysis": {k: v for k, v in analysis.items()},
        "calibration": calibration,
        "validation": validation,
        "verdict": verdict,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "model": args.model,
            "n_pairs": len(results),
            "pre_cutoff": True,
            "contamination_warning": True,
        }
    }

    with open(RESULTS_DIR / "calibration_v3_precutoff.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR / 'calibration_v3_precutoff.json'}")

    return output


if __name__ == "__main__":
    asyncio.run(main())
