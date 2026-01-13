#!/usr/bin/env python3
"""Calibration Transfer Experiment.

Tests whether we can learn magnitude calibration from resolution data.

Hypothesis: LLMs know direction but not magnitude. We can learn a correction
function from historical resolutions.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/calibration-transfer/calibration_experiment.py

    # Resume from cached predictions
    uv run python experiments/calibration-transfer/calibration_experiment.py --skip-elicitation

    # Limit pairs for testing
    uv run python experiments/calibration-transfer/calibration_experiment.py --limit 20
"""

import argparse
import asyncio
import json
import random
import re
import sqlite3
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

load_dotenv()

import litellm

# Configuration
CUTOFF_DATE = date(2025, 10, 1)
DB_PATH = Path("data/forecastbench.db")
DATA_DIR = Path("experiments/calibration-transfer/data")
RESULTS_DIR = Path("experiments/calibration-transfer/results")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Prompts for Two-Stage Elicitation (adapted for P(B|A))
# ============================================================================

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


# ============================================================================
# LLM Utilities
# ============================================================================

async def call_llm(prompt: str, model: str, thinking: bool = False) -> str | None:
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


# ============================================================================
# Two-Stage Elicitation for P(B|A)
# ============================================================================

async def stage1_classify(q_a: str, q_b: str, model: str) -> dict | None:
    """Stage 1: Classify pair as correlated or independent."""
    prompt = STAGE1_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model)
    result = extract_json(content)

    if result and "classification" in result:
        result["classification"] = result["classification"].lower().strip()
        return result
    return None


async def stage2_conditional(q_a: str, q_b: str, model: str) -> dict | None:
    """Stage 2: Elicit P(B|A) for correlated pairs."""
    prompt = STAGE2_CONDITIONAL_PROMPT.format(q_a=q_a, q_b=q_b)
    content = await call_llm(prompt, model)
    result = extract_json(content)

    if result and all(k in result for k in ["p_b", "p_b_given_a1", "p_b_given_a0"]):
        return {
            "method": "conditional",
            "p_b": result["p_b"],
            "p_b_given_a1": result["p_b_given_a1"],
            "p_b_given_a0": result["p_b_given_a0"],
        }
    return None


async def stage2_baseline(q_b: str, model: str) -> dict | None:
    """Stage 2: Baseline elicitation for independent pairs."""
    prompt = STAGE2_BASELINE_PROMPT.format(q_b=q_b)
    content = await call_llm(prompt, model)
    result = extract_json(content)

    if result and "p_b" in result:
        return {
            "method": "baseline",
            "p_b": result["p_b"],
            "p_b_given_a1": result["p_b"],  # Use P(B) for independent
            "p_b_given_a0": result["p_b"],
        }
    return None


async def elicit_p_b_given_a(q_a: str, q_b: str, model: str) -> dict | None:
    """Full Two-Stage elicitation for P(B|A)."""
    # Stage 1: Classify
    stage1 = await stage1_classify(q_a, q_b, model)
    if stage1 is None:
        return None

    # Stage 2: Elicit based on classification
    if stage1["classification"] == "correlated":
        stage2 = await stage2_conditional(q_a, q_b, model)
    else:
        stage2 = await stage2_baseline(q_b, model)

    if stage2 is None:
        return None

    return {
        "stage1": stage1,
        "stage2": stage2,
        "p_b_given_a1": stage2["p_b_given_a1"],
        "p_b_given_a0": stage2["p_b_given_a0"],
        "p_b": stage2["p_b"],
    }


# ============================================================================
# Data Extraction
# ============================================================================

def extract_resolved_questions() -> list[dict]:
    """Extract all binary questions resolved after cutoff date."""
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
    WHERE r.date > ?
      AND q.question_type = 'binary'
      AND r.value IN (0.0, 1.0)
    ORDER BY r.date DESC
    """

    cursor = conn.execute(query, (CUTOFF_DATE.isoformat(),))
    questions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    print(f"Found {len(questions)} binary questions resolved after {CUTOFF_DATE}")
    return questions


def generate_pairs(questions: list[dict], max_pairs: int | None = None) -> list[dict]:
    """Generate question pairs where A resolved to 1.

    Uses stratified sampling to ensure balanced representation of B outcomes.
    """
    # Filter to questions where resolution = 1 (these become A)
    resolved_yes = [q for q in questions if q["resolution_value"] == 1.0]
    resolved_no = [q for q in questions if q["resolution_value"] == 0.0]

    print(f"Questions resolved YES: {len(resolved_yes)}")
    print(f"Questions resolved NO: {len(resolved_no)}")

    # Generate pairs with B resolved YES (A=YES paired with B=YES)
    pairs_b_yes = []
    for q_a in resolved_yes:
        for q_b in resolved_yes:
            if q_a["id"] == q_b["id"]:
                continue
            pairs_b_yes.append({
                "pair_key": f"{q_a['id']}_{q_b['id']}",
                "a_id": q_a["id"],
                "b_id": q_b["id"],
                "a_text": q_a["text"],
                "b_text": q_b["text"],
                "a_resolution": q_a["resolution_value"],
                "b_resolution": q_b["resolution_value"],
                "a_category": q_a["category"],
                "b_category": q_b["category"],
            })

    # Generate pairs with B resolved NO (A=YES paired with B=NO)
    pairs_b_no = []
    for q_a in resolved_yes:
        for q_b in resolved_no:
            pairs_b_no.append({
                "pair_key": f"{q_a['id']}_{q_b['id']}",
                "a_id": q_a["id"],
                "b_id": q_b["id"],
                "a_text": q_a["text"],
                "b_text": q_b["text"],
                "a_resolution": q_a["resolution_value"],
                "b_resolution": q_b["resolution_value"],
                "a_category": q_a["category"],
                "b_category": q_b["category"],
            })

    print(f"Generated {len(pairs_b_yes)} pairs where B=YES")
    print(f"Generated {len(pairs_b_no)} pairs where B=NO")

    # Stratified sampling: 50% from each group
    if max_pairs:
        random.seed(42)
        half = max_pairs // 2

        # Sample from each group
        sample_b_yes = random.sample(pairs_b_yes, min(half, len(pairs_b_yes)))
        sample_b_no = random.sample(pairs_b_no, min(half, len(pairs_b_no)))

        unique_pairs = sample_b_yes + sample_b_no
        random.shuffle(unique_pairs)
        print(f"Stratified sample: {len(sample_b_yes)} B=YES, {len(sample_b_no)} B=NO")
    else:
        unique_pairs = pairs_b_yes + pairs_b_no
        random.seed(42)
        random.shuffle(unique_pairs)

    print(f"Total pairs: {len(unique_pairs)}")

    return unique_pairs


# ============================================================================
# Main Experiment
# ============================================================================

async def run_elicitation(pairs: list[dict], model: str) -> list[dict]:
    """Run Two-Stage elicitation on all pairs."""
    results = []
    semaphore = asyncio.Semaphore(5)  # Limit concurrency

    async def process_pair(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair['a_id'][:20]}... x {pair['b_id'][:20]}...")

            elicitation = await elicit_p_b_given_a(
                pair["a_text"], pair["b_text"], model
            )

            if elicitation is None:
                return {**pair, "error": "Elicitation failed"}

            # Ground truth: did B resolve YES when A resolved YES?
            # Since we only include pairs where A resolved YES, this is just B's resolution
            p_correct = pair["b_resolution"]  # 0 or 1

            # LLM prediction: P(B|A=YES)
            p_llm = elicitation["p_b_given_a1"]

            # Magnitude error
            error = p_llm - p_correct

            result = {
                **pair,
                "elicitation": elicitation,
                "p_llm": p_llm,
                "p_correct": p_correct,
                "error": error,
                "classification": elicitation["stage1"]["classification"],
            }

            print(f"    P(B|A) = {p_llm:.2f}, actual = {p_correct}, error = {error:+.2f}")
            return result

    results = await asyncio.gather(*[
        process_pair(i, p) for i, p in enumerate(pairs)
    ])

    return results


def analyze_results(results: list[dict]) -> dict:
    """Analyze magnitude errors and compute summary statistics."""
    # Filter successful elicitations
    valid = [r for r in results if "p_llm" in r]
    errors = [r for r in valid if not isinstance(r.get("error"), str)]

    print(f"\nValid elicitations: {len(errors)}/{len(results)}")

    if not errors:
        return {"error": "No valid elicitations"}

    # Extract data
    p_llm = np.array([r["p_llm"] for r in errors])
    p_correct = np.array([r["p_correct"] for r in errors])
    magnitude_errors = np.array([r["error"] for r in errors])

    # Summary statistics
    stats_dict = {
        "n_pairs": len(errors),
        "mean_error": float(np.mean(magnitude_errors)),
        "std_error": float(np.std(magnitude_errors)),
        "median_error": float(np.median(magnitude_errors)),
        "min_error": float(np.min(magnitude_errors)),
        "max_error": float(np.max(magnitude_errors)),
        "mean_p_llm": float(np.mean(p_llm)),
        "mean_p_correct": float(np.mean(p_correct)),
    }

    # Classification breakdown
    correlated = [r for r in errors if r["classification"] == "correlated"]
    independent = [r for r in errors if r["classification"] == "independent"]

    stats_dict["n_correlated"] = len(correlated)
    stats_dict["n_independent"] = len(independent)

    if correlated:
        stats_dict["correlated_mean_error"] = float(np.mean([r["error"] for r in correlated]))
    if independent:
        stats_dict["independent_mean_error"] = float(np.mean([r["error"] for r in independent]))

    return stats_dict


def train_calibration(results: list[dict], test_size: float = 0.33) -> dict:
    """Train and validate linear calibration model."""
    # Filter valid results
    errors = [r for r in results if "p_llm" in r and not isinstance(r.get("error"), str)]

    if len(errors) < 10:
        return {"error": "Not enough data for calibration"}

    # Prepare data
    X = np.array([[r["p_llm"]] for r in errors])
    y = np.array([r["p_correct"] for r in errors])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"\nTraining calibration model...")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    alpha = model.coef_[0]
    beta = model.intercept_

    print(f"Learned: calibrated = {alpha:.3f} * raw + {beta:.3f}")

    # Predictions
    y_pred_uncalibrated = X_test.flatten()
    y_pred_calibrated = model.predict(X_test).flatten()

    # Clip calibrated predictions to [0, 1]
    y_pred_calibrated = np.clip(y_pred_calibrated, 0, 1)

    # Brier scores
    brier_uncalibrated = np.mean((y_pred_uncalibrated - y_test) ** 2)
    brier_calibrated = np.mean((y_pred_calibrated - y_test) ** 2)
    improvement = brier_uncalibrated - brier_calibrated

    print(f"\nTest set Brier scores:")
    print(f"  Uncalibrated: {brier_uncalibrated:.4f}")
    print(f"  Calibrated:   {brier_calibrated:.4f}")
    print(f"  Improvement:  {improvement:+.4f}")

    # Bootstrap significance test
    n_bootstrap = 1000
    improvements = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        brier_uncal = np.mean((y_pred_uncalibrated[idx] - y_test[idx]) ** 2)
        brier_cal = np.mean((y_pred_calibrated[idx] - y_test[idx]) ** 2)
        improvements.append(brier_uncal - brier_cal)

    improvements = np.array(improvements)
    p_value = np.mean(improvements <= 0)
    ci_lower = np.percentile(improvements, 2.5)
    ci_upper = np.percentile(improvements, 97.5)

    print(f"\nBootstrap analysis (n={n_bootstrap}):")
    print(f"  p-value: {p_value:.4f}")
    print(f"  95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")

    # Interpretation
    if alpha < 1:
        interpretation = f"LLM overshoots by {(1-alpha)*100:.0f}% on average"
    else:
        interpretation = f"LLM undershoots by {(alpha-1)*100:.0f}% on average"

    if beta > 0:
        interpretation += f", with +{beta:.2f} baseline shift"
    elif beta < 0:
        interpretation += f", with {beta:.2f} baseline shift"

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "interpretation": interpretation,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "brier_uncalibrated": float(brier_uncalibrated),
        "brier_calibrated": float(brier_calibrated),
        "brier_improvement": float(improvement),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "significant": bool(p_value < 0.05),
    }


def determine_verdict(calibration: dict) -> str:
    """Determine experiment verdict."""
    if "error" in calibration:
        return "FAILURE - Not enough data"

    if calibration["brier_improvement"] > 0 and calibration["p_value"] < 0.05:
        return "SUCCESS - Calibration improves Brier (p < 0.05)"
    elif calibration["brier_improvement"] > 0 and calibration["p_value"] < 0.1:
        return "PARTIAL SUCCESS - Calibration improves Brier (marginally significant)"
    elif calibration["brier_improvement"] > 0:
        return "PARTIAL SUCCESS - Calibration improves Brier (not significant)"
    else:
        return "FAILURE - Calibration does not improve Brier"


def fmt(val, fmt_spec=".3f"):
    """Format a value, returning 'N/A' if None."""
    if val is None or val == 'N/A':
        return 'N/A'
    try:
        return f"{val:{fmt_spec}}"
    except (ValueError, TypeError):
        return str(val)


def write_findings(
    questions: list[dict],
    pairs: list[dict],
    results: list[dict],
    stats: dict,
    calibration: dict,
    verdict: str,
) -> None:
    """Write FINDINGS.md with experiment results."""
    findings_path = Path("experiments/calibration-transfer/FINDINGS.md")

    # Format values with fallbacks
    n_yes = len([q for q in questions if q['resolution_value'] == 1.0])
    n_no = len([q for q in questions if q['resolution_value'] == 0.0])

    corr_err = stats.get('correlated_mean_error')
    indep_err = stats.get('independent_mean_error')

    content = f"""# Calibration Transfer Experiment Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

**Verdict:** {verdict}

## Data

| Metric | Value |
|--------|-------|
| Total questions in corpus (post-cutoff) | {len(questions)} |
| Questions resolved YES | {n_yes} |
| Questions resolved NO | {n_no} |
| Pairs generated | {len(pairs)} |
| Usable pairs (A=YES) | {stats.get('n_pairs', 'N/A')} |

## Magnitude Error Analysis

| Metric | Value |
|--------|-------|
| Mean error | {fmt(stats.get('mean_error'))} |
| Std error | {fmt(stats.get('std_error'))} |
| Median error | {fmt(stats.get('median_error'))} |
| Range | [{fmt(stats.get('min_error'))}, {fmt(stats.get('max_error'))}] |

### By Classification

| Classification | Count | Mean Error |
|----------------|-------|------------|
| Correlated | {stats.get('n_correlated', 'N/A')} | {fmt(corr_err) if corr_err else 'N/A'} |
| Independent | {stats.get('n_independent', 'N/A')} | {fmt(indep_err) if indep_err else 'N/A'} |

## Calibration Model

**Learned function:** `calibrated = {fmt(calibration.get('alpha'))} * raw + {fmt(calibration.get('beta'))}`

**Interpretation:** {calibration.get('interpretation', 'N/A')}

## Validation Results

| Metric | Value |
|--------|-------|
| Train size | {calibration.get('train_size', 'N/A')} |
| Test size | {calibration.get('test_size', 'N/A')} |
| Uncalibrated Brier | {fmt(calibration.get('brier_uncalibrated'), '.4f')} |
| Calibrated Brier | {fmt(calibration.get('brier_calibrated'), '.4f')} |
| Improvement | {fmt(calibration.get('brier_improvement'), '+.4f')} |
| p-value | {fmt(calibration.get('p_value'), '.4f')} |
| 95% CI | [{fmt(calibration.get('ci_95_lower'), '+.4f')}, {fmt(calibration.get('ci_95_upper'), '+.4f')}] |
| Statistically significant | {'Yes' if calibration.get('significant') else 'No'} |

## Interpretation

"""

    if "SUCCESS" in verdict:
        content += """
The experiment validates the core hypothesis: LLMs know direction but miscalibrate magnitude.
The learned calibration function can correct for this systematic bias.

**Implications for Causal Discovery Engine:**
- Phase 1 PASSED: Calibration transfer is viable
- Proceed to Phase 2: Relationship discovery
- Integration path: Wire calibration into `propagation.py`
"""
    elif "PARTIAL" in verdict:
        content += """
The experiment shows partial support for the hypothesis. Calibration improves Brier scores
but the effect is not statistically robust. More data or domain-specific calibration may be needed.

**Implications:**
- Phase 1 shows promise but needs more validation
- Consider: larger dataset, domain stratification, or better elicitation
"""
    else:
        content += """
The experiment does not support the hypothesis. Magnitude errors may be noise rather than
systematic bias, or the linear calibration model is inadequate.

**Implications:**
- Phase 1 FAILED: Calibration transfer may not be viable
- Consider: nonlinear calibration, feature engineering, or abandoning this approach
"""

    content += f"""

## Raw Results

See:
- `data/pairs_with_ground_truth.json` - All pairs with predictions
- `data/magnitude_errors.json` - Error distribution
- `results/calibration_model.json` - Model coefficients
- `results/validation_results.json` - Full validation metrics
"""

    with open(findings_path, "w") as f:
        f.write(content)

    print(f"\nWrote findings to {findings_path}")


async def main():
    parser = argparse.ArgumentParser(description="Calibration Transfer Experiment")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max pairs to process (default: 200)")
    parser.add_argument("--skip-elicitation", action="store_true",
                        help="Skip elicitation, use cached results")
    args = parser.parse_args()

    print("=" * 70)
    print("CALIBRATION TRANSFER EXPERIMENT")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Cutoff date: {CUTOFF_DATE}")
    print(f"Max pairs: {args.limit}")

    # Step 1: Extract questions
    print("\n" + "-" * 70)
    print("STEP 1: Extract resolved questions")
    print("-" * 70)
    questions = extract_resolved_questions()

    # Save questions
    with open(DATA_DIR / "questions.json", "w") as f:
        json.dump(questions, f, indent=2, default=str)

    # Step 2: Generate pairs
    print("\n" + "-" * 70)
    print("STEP 2: Generate question pairs")
    print("-" * 70)
    pairs = generate_pairs(questions, max_pairs=args.limit)

    # Check for cached results
    cache_path = DATA_DIR / "pairs_with_ground_truth.json"

    if args.skip_elicitation and cache_path.exists():
        print("\n" + "-" * 70)
        print("STEP 3: Loading cached elicitation results")
        print("-" * 70)
        with open(cache_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results")
    else:
        # Step 3: Run elicitation
        print("\n" + "-" * 70)
        print("STEP 3: Run Two-Stage elicitation")
        print("-" * 70)
        results = await run_elicitation(pairs, args.model)

        # Save results
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {len(results)} results to {cache_path}")

    # Step 4: Analyze magnitude errors
    print("\n" + "-" * 70)
    print("STEP 4: Analyze magnitude errors")
    print("-" * 70)
    stats = analyze_results(results)

    print(f"\nSummary statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Save error analysis
    with open(DATA_DIR / "magnitude_errors.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Step 5: Train calibration model
    print("\n" + "-" * 70)
    print("STEP 5: Train calibration model")
    print("-" * 70)
    calibration = train_calibration(results)

    # Save calibration model
    with open(RESULTS_DIR / "calibration_model.json", "w") as f:
        json.dump(calibration, f, indent=2)

    # Save validation results
    with open(RESULTS_DIR / "validation_results.json", "w") as f:
        json.dump({
            "calibration": calibration,
            "stats": stats,
            "metadata": {
                "model": args.model,
                "cutoff_date": str(CUTOFF_DATE),
                "n_questions": len(questions),
                "n_pairs": len(pairs),
                "run_at": datetime.now().isoformat(),
            }
        }, f, indent=2)

    # Step 6: Determine verdict
    print("\n" + "-" * 70)
    print("STEP 6: Verdict")
    print("-" * 70)
    verdict = determine_verdict(calibration)
    print(f"\n>>> {verdict} <<<")

    # Write findings
    write_findings(questions, pairs, results, stats, calibration, verdict)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return verdict


if __name__ == "__main__":
    asyncio.run(main())
