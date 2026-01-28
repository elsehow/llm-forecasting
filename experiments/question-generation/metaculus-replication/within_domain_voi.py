#!/usr/bin/env python3
"""
Metaculus Within-Domain Multi-Prompt Correlation Validation.

Compares two ρ estimation approaches on Metaculus:
1. **E2E 1-call**: Holistic prompt asking for ρ + posteriors (P(A), P(A|B=YES), P(A|B=NO))
2. **ρ-only**: Simple single-step prompt asking only for ρ

Key research question:
Does the E2E holistic prompting context improve ρ direction accuracy?
- Polymarket found E2E direction (27.3%) > ρ-only direction (20.0%)
- This experiment tests whether that advantage generalizes

Key differences from Polymarket:
- Ground truth: resolution agreement (+1 if same, -1 if different) vs Δp
- No ceiling: can't compute market ρ VOI without price data
- No market P(B): removed from prompt, tests "pure" LLM correlation estimation

Aligned design decisions (matching polymarket experiments):
- E2E 1-call prompt structure (same format, asks for ρ + posteriors)
- ρ-only prompt (from llm_forecasting.voi.RHO_ESTIMATION_PROMPT)
- LLM estimation process (two calls per pair - E2E and ρ-only)
- Category-based analysis (within-category splits)
- Direction accuracy metric (does sign(ρ) match ground truth?)
- Output format (JSON with per-pair details and summary metrics)

Why this matters:
- If E2E > ρ-only: Holistic context helps (validates Polymarket finding)
- If E2E ≈ ρ-only: No benefit from posteriors-alongside-ρ
- If both fail: Fundamental limitation in LLM correlation estimation

Success criteria:
- Sign accuracy > 55%: Direction capability validated
- Sign accuracy ~50%: Random, direction doesn't generalize
- Sign accuracy < 50%: LLMs systematically wrong
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
import litellm
from dotenv import load_dotenv

# Force line-buffered output for progress visibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load .env from monorepo root
_monorepo_root = Path(__file__).resolve().parents[3]
load_dotenv(_monorepo_root / ".env")

from llm_forecasting.market_data import MarketDataStorage, MarketStatus

# Configuration
DB_PATH = Path(__file__).parents[3] / "data" / "forecastbench.db"
OUTPUT_DIR = Path(__file__).parent / "data"
MIN_QUESTIONS_PER_CATEGORY = 50
MAX_PAIRS_PER_CATEGORY = 100  # Same ballpark as Polymarket's ~150 total pairs
RANDOM_SEED = 42

# Model to use for LLM estimation (same as Polymarket experiment)
MODEL = "anthropic/claude-sonnet-4-20250514"


# =============================================================================
# E2E 1-CALL PROMPT (adapted from Polymarket - removed market probability)
# =============================================================================

E2E_1CALL_PROMPT = """You are a forecaster estimating conditional probabilities.

Question A: "{question_a}"
Question B: "{question_b}"

Step 1: First, estimate the correlation coefficient (ρ) between these questions.
- ρ > 0: They tend to resolve together (if B resolves YES, A more likely YES)
- ρ = 0: Independent
- ρ < 0: They resolve oppositely (if B resolves YES, A more likely NO)

Step 2: Then, estimate P(A), P(A|B=YES), and P(A|B=NO), using your ρ estimate
to calibrate the magnitude of update.

Respond with JSON only:
{{"rho_estimate": <float -1 to +1>, "p_a": <float 0-1>, "p_a_given_b_yes": <float 0-1>, "p_a_given_b_no": <float 0-1>, "reasoning": "<brief explanation>"}}"""


# =============================================================================
# ρ-ONLY PROMPT (from llm_forecasting.voi - simple single-step)
# =============================================================================

RHO_ONLY_PROMPT = """You are estimating the correlation between two prediction market questions.

Question A: "{question_a}"
Question B: "{question_b}"

Estimate the correlation coefficient (ρ) between these two questions. This measures how much knowing the outcome of one question tells you about the other:
- ρ = +1: Perfect positive correlation (if A is YES, B is definitely YES)
- ρ = 0: Independent (knowing A tells you nothing about B)
- ρ = -1: Perfect negative correlation (if A is YES, B is definitely NO)

Think about:
- Are these questions about related events?
- Would one outcome make the other more or less likely?
- Are they measuring the same underlying phenomenon?

Respond with JSON only:
{{"rho": <float from -1 to +1>, "reasoning": "<brief explanation>"}}"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ResolvedQuestion:
    """A resolved Metaculus question."""
    id: str
    title: str
    category: str
    resolved_value: float  # 0.0 or 1.0


@dataclass
class QuestionPair:
    """A pair of questions for correlation validation."""
    question_a: ResolvedQuestion
    question_b: ResolvedQuestion
    # E2E estimates
    rho_e2e: float | None = None
    p_a: float | None = None
    p_a_given_b_yes: float | None = None
    p_a_given_b_no: float | None = None
    reasoning_e2e: str = ""
    error_e2e: str | None = None
    # ρ-only estimates
    rho_only: float | None = None
    reasoning_only: str = ""
    error_only: str | None = None
    # Ground truth
    same_resolution: bool = False  # Did both resolve the same way?
    ground_truth: int = 0  # +1 if same, -1 if different
    # Derived
    llm_direction: int = 0  # sign of (p_a_given_b_yes - p_a)


# =============================================================================
# CATEGORY NORMALIZATION
# =============================================================================

def normalize_category(raw_cat: str | None) -> str:
    """Normalize category name for consistency."""
    if not raw_cat:
        return "uncategorized"

    cat_lower = raw_cat.lower()
    if "politic" in cat_lower or "election" in cat_lower:
        return "politics"
    elif "economy" in cat_lower or "business" in cat_lower or "crypto" in cat_lower:
        return "economy"
    elif "geopolitic" in cat_lower:
        return "geopolitics"
    elif "health" in cat_lower or "pandemic" in cat_lower:
        return "health"
    elif "technology" in cat_lower or "computing" in cat_lower:
        return "technology"
    elif "ai" in cat_lower or "artificial intelligence" in cat_lower:
        return "ai"
    elif "sport" in cat_lower or "entertainment" in cat_lower:
        return "sports"
    elif "environment" in cat_lower or "climate" in cat_lower:
        return "environment"
    elif "space" in cat_lower or "science" in cat_lower:
        return "science"
    elif "nuclear" in cat_lower:
        return "nuclear"
    elif "law" in cat_lower:
        return "law"
    else:
        return cat_lower


# =============================================================================
# DATA LOADING
# =============================================================================

async def load_resolved_questions() -> dict[str, list[ResolvedQuestion]]:
    """Load resolved Metaculus questions grouped by category."""
    storage = MarketDataStorage(DB_PATH)

    # Get all resolved Metaculus questions
    markets = await storage.get_markets(
        platform="metaculus",
        status=MarketStatus.RESOLVED,
    )
    await storage.close()

    print(f"Loaded {len(markets)} resolved Metaculus questions")

    # Group by category
    questions_by_category: dict[str, list[ResolvedQuestion]] = defaultdict(list)
    yes_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)

    for market in markets:
        # Skip if missing resolution
        if market.resolved_value is None:
            continue

        # Get normalized category
        raw_cat = market.topic_categories[0] if market.topic_categories else None
        category = normalize_category(raw_cat)

        questions_by_category[category].append(ResolvedQuestion(
            id=market.id,
            title=market.title,
            category=category,
            resolved_value=market.resolved_value,
        ))
        total_counts[category] += 1
        if market.resolved_value == 1.0:
            yes_counts[category] += 1

    # Filter to categories with enough questions
    filtered = {
        cat: qs for cat, qs in questions_by_category.items()
        if len(qs) >= MIN_QUESTIONS_PER_CATEGORY
    }

    print(f"\nCategories with n>={MIN_QUESTIONS_PER_CATEGORY}:")
    for cat, qs in sorted(filtered.items(), key=lambda x: -len(x[1])):
        yes_rate = yes_counts[cat] / total_counts[cat] if total_counts[cat] > 0 else 0
        print(f"  {cat}: {len(qs)} (YES rate: {yes_rate:.1%})")

    return filtered


def generate_pairs(
    questions: list[ResolvedQuestion],
    max_pairs: int = MAX_PAIRS_PER_CATEGORY,
) -> list[QuestionPair]:
    """Generate random question pairs within a category.

    Ground truth: did both questions resolve the same way?
    - same_resolution=True, ground_truth=+1: both YES or both NO
    - same_resolution=False, ground_truth=-1: one YES, one NO
    """
    random.seed(RANDOM_SEED)

    # Generate all possible pairs
    all_pairs = []
    for i, q_a in enumerate(questions):
        for q_b in questions[i+1:]:
            same_resolution = q_a.resolved_value == q_b.resolved_value
            ground_truth = 1 if same_resolution else -1

            all_pairs.append(QuestionPair(
                question_a=q_a,
                question_b=q_b,
                same_resolution=same_resolution,
                ground_truth=ground_truth,
            ))

    # Sample if too many pairs
    if len(all_pairs) > max_pairs:
        all_pairs = random.sample(all_pairs, max_pairs)

    return all_pairs


# =============================================================================
# E2E 1-CALL LLM ESTIMATION
# =============================================================================

async def estimate_e2e_1call(
    question_a: str,
    question_b: str,
    model: str = MODEL,
) -> dict:
    """Estimate posteriors using E2E 1-call prompt.

    Returns dict with:
    - rho_estimate: LLM's internal ρ estimate
    - p_a: LLM's unconditional P(A)
    - p_a_given_b_yes: LLM's P(A|B=YES)
    - p_a_given_b_no: LLM's P(A|B=NO)
    - reasoning: LLM's explanation
    """
    prompt = E2E_1CALL_PROMPT.format(
        question_a=question_a,
        question_b=question_b,
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        # Normalize and clamp values
        return {
            "rho_estimate": max(-1.0, min(1.0, float(result.get("rho_estimate", 0.0)))),
            "p_a": max(0.01, min(0.99, float(result.get("p_a", 0.5)))),
            "p_a_given_b_yes": max(0.01, min(0.99, float(result.get("p_a_given_b_yes", 0.5)))),
            "p_a_given_b_no": max(0.01, min(0.99, float(result.get("p_a_given_b_no", 0.5)))),
            "reasoning": result.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "rho_estimate": 0.0,
            "p_a": 0.5,
            "p_a_given_b_yes": 0.5,
            "p_a_given_b_no": 0.5,
            "reasoning": "",
            "error": str(e),
        }


async def estimate_e2e_for_pairs(
    pairs: list[QuestionPair],
    category: str,
) -> list[QuestionPair]:
    """Estimate posteriors using E2E 1-call for all pairs in a category."""
    print(f"\n[{category}] Estimating E2E 1-call for {len(pairs)} pairs ({MODEL})...")

    for i, pair in enumerate(pairs):
        if i % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] Processing...")

        result = await estimate_e2e_1call(
            pair.question_a.title,
            pair.question_b.title,
        )

        pair.rho_e2e = result["rho_estimate"]
        pair.p_a = result["p_a"]
        pair.p_a_given_b_yes = result["p_a_given_b_yes"]
        pair.p_a_given_b_no = result["p_a_given_b_no"]
        pair.reasoning_e2e = result["reasoning"]
        pair.error_e2e = result["error"]

        # Direction from LLM posteriors
        if result["p_a_given_b_yes"] > result["p_a"]:
            pair.llm_direction = 1
        elif result["p_a_given_b_yes"] < result["p_a"]:
            pair.llm_direction = -1
        else:
            pair.llm_direction = 0

        if result["error"]:
            print(f"    Error: {result['error'][:50]}...")

    return pairs


# =============================================================================
# ρ-ONLY LLM ESTIMATION
# =============================================================================

async def estimate_rho_only(
    question_a: str,
    question_b: str,
    model: str = MODEL,
) -> dict:
    """Estimate ρ using simple single-step prompt (no posteriors).

    Returns dict with:
    - rho: LLM's ρ estimate
    - reasoning: LLM's explanation
    - error: None or error message
    """
    prompt = RHO_ONLY_PROMPT.format(
        question_a=question_a,
        question_b=question_b,
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        # Accept both "rho" and "rho_estimate" for compatibility
        rho = result.get("rho", result.get("rho_estimate", 0.0))

        return {
            "rho": max(-1.0, min(1.0, float(rho))),
            "reasoning": result.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "rho": 0.0,
            "reasoning": "",
            "error": str(e),
        }


async def estimate_rho_only_for_pairs(
    pairs: list[QuestionPair],
    category: str,
) -> list[QuestionPair]:
    """Estimate ρ using simple single-step prompt for all pairs in a category."""
    print(f"\n[{category}] Estimating ρ-only for {len(pairs)} pairs ({MODEL})...")

    for i, pair in enumerate(pairs):
        if i % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] Processing...")

        result = await estimate_rho_only(
            pair.question_a.title,
            pair.question_b.title,
        )

        pair.rho_only = result["rho"]
        pair.reasoning_only = result["reasoning"]
        pair.error_only = result["error"]

        if result["error"]:
            print(f"    Error: {result['error'][:50]}...")

    return pairs


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(
    pairs: list[QuestionPair],
    rho_field: str = "rho_e2e",
    error_field: str = "error_e2e",
) -> dict:
    """Compute correlation and direction accuracy metrics.

    Args:
        pairs: List of question pairs with ρ estimates
        rho_field: Which ρ field to use ("rho_e2e" or "rho_only")
        error_field: Which error field to use ("error_e2e" or "error_only")

    Matches Polymarket E2E 1-call metrics where applicable:
    - r(ρ_llm, ground_truth): does ρ correlate with resolution agreement?
    - Direction accuracy: does sign(ρ) match ground_truth?

    No ceiling comparison (can't compute market ρ VOI without price data).
    """
    # Filter to pairs with valid estimates
    valid_pairs = [
        p for p in pairs
        if getattr(p, rho_field) is not None and getattr(p, error_field) is None
    ]

    if len(valid_pairs) < 10:
        return {"n": len(valid_pairs), "error": "Too few valid pairs"}

    # Extract arrays
    rho_estimates = np.array([getattr(p, rho_field) for p in valid_pairs])
    ground_truths = np.array([p.ground_truth for p in valid_pairs])

    # Filter NaN
    mask = ~(np.isnan(rho_estimates) | np.isnan(ground_truths))
    rho_estimates = rho_estimates[mask]
    ground_truths = ground_truths[mask]

    if len(rho_estimates) < 10:
        return {"n": len(rho_estimates), "error": "Too few valid pairs after filtering"}

    # Check for zero variance
    if np.std(rho_estimates) == 0:
        return {"n": len(rho_estimates), "error": "Zero variance in ρ estimates"}

    # Spearman correlation: r(ρ_llm, resolution_agreement)
    r, p_value = stats.spearmanr(rho_estimates, ground_truths)

    # Direction accuracy: does sign(ρ) predict same_resolution?
    # Positive ρ should predict ground_truth=+1 (same resolution)
    # Negative ρ should predict ground_truth=-1 (different resolution)
    rho_signs = np.sign(rho_estimates)
    # Only count pairs where LLM made a prediction (non-zero ρ)
    prediction_mask = rho_signs != 0
    if np.sum(prediction_mask) > 0:
        correct = np.sum(rho_signs[prediction_mask] == ground_truths[prediction_mask])
        direction_accuracy = float(correct / np.sum(prediction_mask))
        n_predictions = int(np.sum(prediction_mask))
    else:
        direction_accuracy = None
        n_predictions = 0

    # Additional stats
    pct_same_resolution = float(np.mean(ground_truths == 1))
    mean_abs_rho = float(np.mean(np.abs(rho_estimates)))
    pct_positive_rho = float(np.mean(rho_estimates > 0))
    pct_near_zero = float(np.mean(np.abs(rho_estimates) < 0.05))

    return {
        "n": int(len(rho_estimates)),
        "n_predictions": n_predictions,
        "r": float(r),
        "p": float(p_value),
        "direction_accuracy": direction_accuracy,
        "pct_same_resolution": pct_same_resolution,
        "mean_abs_rho": mean_abs_rho,
        "pct_positive_rho": pct_positive_rho,
        "pct_near_zero": pct_near_zero,
    }


def compute_comparison_metrics(pairs: list[QuestionPair]) -> dict:
    """Compute paired comparison metrics between E2E and ρ-only approaches.

    Uses McNemar's test for paired binary comparisons on direction accuracy.
    """
    # Get pairs where both methods made non-zero predictions
    valid_pairs = [
        p for p in pairs
        if (p.rho_e2e is not None and p.error_e2e is None and p.rho_e2e != 0 and
            p.rho_only is not None and p.error_only is None and p.rho_only != 0)
    ]

    if len(valid_pairs) < 10:
        return {"n": len(valid_pairs), "error": "Too few pairs with both estimates"}

    # Compute direction correctness for each pair
    e2e_correct = np.array([
        1 if np.sign(p.rho_e2e) == p.ground_truth else 0
        for p in valid_pairs
    ])
    only_correct = np.array([
        1 if np.sign(p.rho_only) == p.ground_truth else 0
        for p in valid_pairs
    ])

    # McNemar's test contingency table
    # b: E2E correct, ρ-only wrong
    # c: E2E wrong, ρ-only correct
    b = np.sum((e2e_correct == 1) & (only_correct == 0))
    c = np.sum((e2e_correct == 0) & (only_correct == 1))

    # McNemar's test (with continuity correction)
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 1 else 0
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat = 0.0
        p_value = 1.0

    # Direction accuracies
    e2e_acc = float(np.mean(e2e_correct))
    only_acc = float(np.mean(only_correct))

    # Agreement: how often do they make the same prediction?
    same_prediction = np.sum(np.sign([p.rho_e2e for p in valid_pairs]) ==
                             np.sign([p.rho_only for p in valid_pairs]))
    agreement_rate = float(same_prediction / len(valid_pairs))

    return {
        "n": len(valid_pairs),
        "e2e_direction_accuracy": e2e_acc,
        "rho_only_direction_accuracy": only_acc,
        "e2e_advantage": e2e_acc - only_acc,
        "mcnemar_b": int(b),  # E2E correct, ρ-only wrong
        "mcnemar_c": int(c),  # E2E wrong, ρ-only correct
        "mcnemar_stat": float(mcnemar_stat),
        "p_value": float(p_value),
        "agreement_rate": agreement_rate,
    }


def print_metrics(name: str, m: dict):
    """Print metrics in a formatted way (matches Polymarket style)."""
    if m.get("error"):
        print(f"\n{name}: {m['error']} (n={m.get('n', 0)})")
        return

    print(f"\n{name} (n={m['n']}):")
    print(f"  r(ρ_llm, resolution):     {m['r']:.3f} (p={m['p']:.4f})")
    if m.get('direction_accuracy') is not None:
        print(f"  Direction accuracy:       {m['direction_accuracy']:.1%} ({m['n_predictions']} predictions)")
    else:
        print(f"  Direction accuracy:       N/A (no non-zero ρ predictions)")
    print(f"  % same resolution:        {m['pct_same_resolution']:.1%}")
    print(f"  mean |ρ|:                 {m['mean_abs_rho']:.3f}")
    print(f"  % positive ρ:             {m['pct_positive_rho']:.1%}")
    print(f"  % near-zero ρ (<0.05):    {m['pct_near_zero']:.1%}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the Metaculus multi-prompt correlation comparison experiment."""
    print("=" * 70)
    print("METACULUS MULTI-PROMPT CORRELATION COMPARISON")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Start time: {datetime.now().isoformat()}")
    print("\nComparing two ρ estimation approaches:")
    print("  1. E2E 1-call: Holistic prompt with ρ + posteriors")
    print("  2. ρ-only: Simple single-step prompt")
    print("\nTest: Which approach better predicts resolution correlation?")

    # Load questions
    print("\n[1/5] Loading data...")
    questions_by_category = await load_resolved_questions()

    if not questions_by_category:
        print("No categories with sufficient questions found!")
        return

    # Process each category
    print("\n[2/5] Generating pairs...")
    all_pairs: list[QuestionPair] = []
    pairs_by_category: dict[str, list[QuestionPair]] = {}

    for category, questions in sorted(questions_by_category.items(), key=lambda x: -len(x[1])):
        pairs = generate_pairs(questions)
        pairs_by_category[category] = pairs
        all_pairs.extend(pairs)
        print(f"  {category}: {len(pairs)} pairs")

    print(f"\nTotal pairs: {len(all_pairs)}")

    # E2E 1-call estimation
    print("\n[3/5] E2E 1-call posterior estimation...")
    for category, pairs in pairs_by_category.items():
        await estimate_e2e_for_pairs(pairs, category)

    # ρ-only estimation
    print("\n[4/5] ρ-only estimation...")
    for category, pairs in pairs_by_category.items():
        await estimate_rho_only_for_pairs(pairs, category)

    # Compute metrics
    print("\n[5/5] Computing metrics...")

    # E2E metrics
    e2e_by_category = {}
    for category, pairs in pairs_by_category.items():
        e2e_by_category[category] = compute_metrics(pairs, "rho_e2e", "error_e2e")
    e2e_overall = compute_metrics(all_pairs, "rho_e2e", "error_e2e")

    # ρ-only metrics
    only_by_category = {}
    for category, pairs in pairs_by_category.items():
        only_by_category[category] = compute_metrics(pairs, "rho_only", "error_only")
    only_overall = compute_metrics(all_pairs, "rho_only", "error_only")

    # Comparison metrics
    comparison = compute_comparison_metrics(all_pairs)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: E2E 1-CALL")
    print("=" * 70)
    print_metrics("OVERALL (E2E)", e2e_overall)

    print("\n" + "=" * 70)
    print("RESULTS: ρ-ONLY")
    print("=" * 70)
    print_metrics("OVERALL (ρ-only)", only_overall)

    print("\n" + "=" * 70)
    print("COMPARISON: E2E vs ρ-ONLY")
    print("=" * 70)
    if comparison.get("error"):
        print(f"\n{comparison['error']}")
    else:
        print(f"\nPaired comparison (n={comparison['n']}):")
        print(f"  E2E direction accuracy:     {comparison['e2e_direction_accuracy']:.1%}")
        print(f"  ρ-only direction accuracy:  {comparison['rho_only_direction_accuracy']:.1%}")
        print(f"  E2E advantage:              {comparison['e2e_advantage']:+.1%}")
        print(f"\nMcNemar's test:")
        print(f"  E2E correct, ρ-only wrong:  {comparison['mcnemar_b']}")
        print(f"  E2E wrong, ρ-only correct:  {comparison['mcnemar_c']}")
        print(f"  χ² statistic:               {comparison['mcnemar_stat']:.3f}")
        print(f"  p-value:                    {comparison['p_value']:.4f}")
        print(f"\nAgreement rate:               {comparison['agreement_rate']:.1%}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    e2e_acc = e2e_overall.get("direction_accuracy")
    only_acc = only_overall.get("direction_accuracy")

    if e2e_acc is not None and only_acc is not None:
        print(f"\nE2E direction accuracy:    {e2e_acc:.1%}")
        print(f"ρ-only direction accuracy: {only_acc:.1%}")

        if e2e_acc > only_acc + 0.05:
            print("\n✓ E2E ADVANTAGE: Holistic prompting improves direction accuracy")
            print("  This validates the Polymarket finding (E2E 27.3% vs ρ-only 20.0%)")
        elif only_acc > e2e_acc + 0.05:
            print("\n✗ ρ-ONLY ADVANTAGE: Simple prompt outperforms holistic prompt")
            print("  Posteriors-alongside-ρ may confuse the model")
        else:
            print("\n~ NO DIFFERENCE: Both approaches perform similarly")
            print("  E2E's advantage on Polymarket may not generalize")

        # Compare to random baseline
        if e2e_acc > 0.55 or only_acc > 0.55:
            print("\n✓ ABOVE RANDOM: At least one approach predicts direction")
        elif e2e_acc > 0.50 or only_acc > 0.50:
            print("\n~ MARGINAL: Slightly above random")
        else:
            print("\n✗ AT/BELOW RANDOM: Neither approach works")

    # Comparison to Polymarket
    print("\n" + "-" * 70)
    print("COMPARISON TO POLYMARKET EXPERIMENTS")
    print("-" * 70)
    print("Polymarket results:")
    print("  E2E direction:   27.3%")
    print("  ρ-only direction: 20.0%")
    print("  E2E advantage:   +7.3pp")
    print("\nMetaculus results:")
    if e2e_acc is not None and only_acc is not None:
        print(f"  E2E direction:   {e2e_acc:.1%}")
        print(f"  ρ-only direction: {only_acc:.1%}")
        print(f"  E2E advantage:   {(e2e_acc - only_acc):+.1%}")

        if comparison.get("p_value") is not None:
            sig = "significant" if comparison["p_value"] < 0.05 else "not significant"
            print(f"\n  Statistical significance: {sig} (p={comparison['p_value']:.4f})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "experiment": "metaculus_multi_prompt_comparison",
            "description": "Compares E2E 1-call vs ρ-only prompts on Metaculus",
            "model": MODEL,
            "prompts": ["e2e_1call", "rho_only"],
            "n_total_pairs": len(all_pairs),
            "db_path": str(DB_PATH),
            "random_seed": RANDOM_SEED,
            "max_pairs_per_category": MAX_PAIRS_PER_CATEGORY,
            "run_at": datetime.now().isoformat(),
        },
        "summary": {
            "e2e": e2e_overall,
            "rho_only": only_overall,
            "comparison": comparison,
        },
        "by_category": {
            "e2e": e2e_by_category,
            "rho_only": only_by_category,
        },
        "interpretation": {
            "e2e_advantage": "E2E holistic prompting helps if positive",
            "p_value": "McNemar's test for paired direction accuracy",
            "agreement_rate": "How often both approaches make the same prediction",
        },
        "pairs": [
            {
                "question_a": p.question_a.title,
                "question_b": p.question_b.title,
                "category": p.question_a.category,
                "resolved_a": p.question_a.resolved_value,
                "resolved_b": p.question_b.resolved_value,
                "same_resolution": p.same_resolution,
                "ground_truth": p.ground_truth,
                # E2E estimates
                "rho_e2e": p.rho_e2e,
                "p_a": p.p_a,
                "p_a_given_b_yes": p.p_a_given_b_yes,
                "p_a_given_b_no": p.p_a_given_b_no,
                "llm_direction": p.llm_direction,
                "reasoning_e2e": p.reasoning_e2e,
                "error_e2e": p.error_e2e,
                # ρ-only estimates
                "rho_only": p.rho_only,
                "reasoning_only": p.reasoning_only,
                "error_only": p.error_only,
            }
            for p in all_pairs
        ],
    }

    output_path = OUTPUT_DIR / "metaculus_multi_prompt.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved to: {output_path}")
    print(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
