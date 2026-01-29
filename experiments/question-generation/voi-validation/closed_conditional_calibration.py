"""Closed-Question Conditional Calibration Experiment.

Test LLM conditional forecasting ability using resolved questions where we have
ground truth, rather than open questions where we can only compare to market
correlations.

Design:
- Use curated_pairs_nontrivial.json (34 pairs)
- Query Polymarket API for current resolution status of ALL questions
- For pairs where BOTH questions have resolved:
  - Ask LLM: P(X|Q=yes), P(X|Q=no)
  - Select P(X|Q=actual) based on Q's actual resolution
  - Compute Brier score: (P(X|Q=actual) - X_outcome)²
- Compare to marginal baseline (P(X) without conditioning)

Key insight: This tests whether LLMs can do conditional forecasting directly,
isolated from market dynamics that confounded Q2.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/closed_conditional_calibration.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from monorepo root before importing llm_forecasting
_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import httpx
import numpy as np
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).parent
CURATED_PAIRS_FILE = SCRIPT_DIR / "curated_pairs_nontrivial.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "closed_conditional_calibration_results.json"

# Model - same as Q2 for comparison
MODEL = "claude-opus-4-5-20251101"

# Polymarket API
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Prompt for conditional probability elicitation
CONDITIONAL_PROB_PROMPT = """You are estimating conditional probabilities for a forecasting scenario.

QUESTION X (target): "{question_x}"
QUESTION Q (condition): "{question_q}"

Your task: Estimate the conditional probabilities:

1. P(X=YES | Q=YES): If Q resolved YES, what is the probability X resolves YES?

2. P(X=YES | Q=NO): If Q resolved NO, what is the probability X resolves YES?

Think carefully about:
- What is the logical/causal relationship between these questions?
- How does knowing Q's outcome change your belief about X?
- Consider all the ways Q being true/false could affect X

Respond with JSON only:
{{"p_x_given_q_yes": <float 0.0-1.0>, "p_x_given_q_no": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


def load_curated_pairs() -> list[dict]:
    """Load curated pairs from JSON file."""
    with open(CURATED_PAIRS_FILE) as f:
        data = json.load(f)
    return data["curated_pairs"]


async def fetch_market_by_condition_id(
    client: httpx.AsyncClient,
    condition_id: str,
) -> dict | None:
    """Fetch market data from Polymarket by condition ID."""
    try:
        params = {"condition_ids": condition_id}
        response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        markets = response.json()
        if markets:
            return markets[0]
        return None
    except Exception as e:
        print(f"    Error fetching {condition_id}: {e}")
        return None


def parse_resolution(market: dict) -> tuple[str | None, bool | None]:
    """Parse resolution status and outcome from market data.

    Returns:
        Tuple of (status, outcome) where:
        - status: 'resolved', 'open', or 'closed'
        - outcome: True (YES), False (NO), or None
    """
    if not market:
        return None, None

    uma_status = market.get("umaResolutionStatus")

    if uma_status == "resolved":
        # Get the outcome
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            prices = json.loads(market.get("outcomePrices", "[]"))

            # Find which outcome won (price = 1.0)
            for i, price in enumerate(prices):
                if float(price) > 0.99:
                    outcome_str = outcomes[i].lower()
                    return "resolved", bool(outcome_str == "yes")
                elif float(price) < 0.01:
                    continue

            # If no clear 1.0, check if resolved to YES (index 0 typically)
            if len(prices) >= 2:
                if float(prices[0]) > 0.5:
                    return "resolved", True
                else:
                    return "resolved", False

        except (json.JSONDecodeError, TypeError, IndexError, ValueError):
            return "resolved", None

    elif market.get("closed") or not market.get("active"):
        return "closed", None

    return "open", None


def make_json_serializable(obj):
    """Convert numpy types and bools to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


async def refresh_resolution_status(pairs: list[dict]) -> list[dict]:
    """Refresh resolution status for all pairs from Polymarket API.

    Returns enriched pairs with current resolution status.
    """
    print("\nRefreshing resolution status from Polymarket API...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        enriched_pairs = []

        for i, pair in enumerate(pairs):
            print(f"  [{i+1}/{len(pairs)}] Checking pair...")

            # Fetch both markets
            market_a = await fetch_market_by_condition_id(
                client, pair["condition_id_a"]
            )
            market_b = await fetch_market_by_condition_id(
                client, pair["condition_id_b"]
            )

            # Parse resolution status
            status_a, outcome_a = parse_resolution(market_a)
            status_b, outcome_b = parse_resolution(market_b)

            enriched = {
                **pair,
                "market_a_status": status_a,
                "market_a_outcome": outcome_a,
                "market_a_title": market_a.get("question", pair["question_a"]) if market_a else pair["question_a"],
                "market_b_status": status_b,
                "market_b_outcome": outcome_b,
                "market_b_title": market_b.get("question", pair["question_b"]) if market_b else pair["question_b"],
            }
            enriched_pairs.append(enriched)

            # Rate limiting
            await asyncio.sleep(0.1)

    return enriched_pairs


def filter_fully_resolved(pairs: list[dict]) -> list[dict]:
    """Filter to pairs where BOTH questions have resolved."""
    fully_resolved = []

    for pair in pairs:
        status_a = pair.get("market_a_status")
        status_b = pair.get("market_b_status")
        outcome_a = pair.get("market_a_outcome")
        outcome_b = pair.get("market_b_outcome")

        if (
            status_a == "resolved"
            and status_b == "resolved"
            and outcome_a is not None
            and outcome_b is not None
        ):
            fully_resolved.append(pair)

    return fully_resolved


async def estimate_conditionals(
    question_x: str,
    question_q: str,
    model: str = MODEL,
) -> tuple[float, float, str]:
    """Estimate P(X|Q=yes) and P(X|Q=no) using LLM.

    Returns:
        Tuple of (p_x_given_q_yes, p_x_given_q_no, reasoning)
    """
    import litellm

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": CONDITIONAL_PROB_PROMPT.format(
                    question_x=question_x,
                    question_q=question_q,
                )
            }],
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
        p_x_yes = float(result.get("p_x_given_q_yes", 0.5))
        p_x_no = float(result.get("p_x_given_q_no", 0.5))
        reasoning = result.get("reasoning", "")

        # Clamp to valid range
        p_x_yes = max(0.01, min(0.99, p_x_yes))
        p_x_no = max(0.01, min(0.99, p_x_no))

        return p_x_yes, p_x_no, reasoning

    except Exception as e:
        return 0.5, 0.5, f"Error: {e}"


def brier_score(predicted: float, actual: bool) -> float:
    """Compute Brier score: (predicted - actual)^2."""
    actual_value = 1.0 if actual else 0.0
    return (predicted - actual_value) ** 2


async def run_experiment(pairs: list[dict]) -> list[dict]:
    """Run the closed-question conditional calibration experiment."""
    print(f"\nRunning experiment on {len(pairs)} fully-resolved pairs...")
    print(f"Model: {MODEL}")

    results = []

    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Processing {pair['classification']['category']}...")

        # Determine which question is Q (resolved first per original data)
        # and which is X (the other one)
        original_resolved = pair["resolved"]  # "A" or "B"

        if original_resolved == "A":
            question_q = pair["question_a"]
            question_x = pair["question_b"]
            q_outcome = pair["market_a_outcome"]
            x_outcome = pair["market_b_outcome"]
        else:
            question_q = pair["question_b"]
            question_x = pair["question_a"]
            q_outcome = pair["market_b_outcome"]
            x_outcome = pair["market_a_outcome"]

        # Get LLM estimates of conditionals
        p_x_yes, p_x_no, reasoning = await estimate_conditionals(
            question_x=question_x,
            question_q=question_q,
            model=MODEL,
        )

        # Select the conditional matching Q's actual outcome
        if q_outcome:  # Q resolved YES
            p_x_conditional = p_x_yes
        else:  # Q resolved NO
            p_x_conditional = p_x_no

        # Compute Brier score
        brier = brier_score(p_x_conditional, x_outcome)

        # Marginal baseline: P(X) = 0.5 (no information)
        # Or we could use the original market price, but we don't have it for all
        baseline_brier = brier_score(0.5, x_outcome)

        # Store result
        results.append({
            "question_x": question_x,
            "question_q": question_q,
            "category": pair["classification"]["category"],
            "q_outcome": q_outcome,
            "x_outcome": x_outcome,
            # LLM estimates
            "p_x_given_q_yes": p_x_yes,
            "p_x_given_q_no": p_x_no,
            "p_x_conditional": p_x_conditional,
            "reasoning": reasoning,
            # Brier scores
            "brier_conditional": brier,
            "brier_baseline": baseline_brier,
            "brier_improvement": baseline_brier - brier,
        })

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    if not results:
        return {"error": "No results"}

    brier_conditional = [r["brier_conditional"] for r in results]
    brier_baseline = [r["brier_baseline"] for r in results]
    improvements = [r["brier_improvement"] for r in results]

    summary = {
        "n": len(results),
        "mean_brier_conditional": float(np.mean(brier_conditional)),
        "mean_brier_baseline": float(np.mean(brier_baseline)),
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "median_improvement": float(np.median(improvements)),
        # Did conditioning help?
        "n_improved": sum(1 for i in improvements if i > 0),
        "n_hurt": sum(1 for i in improvements if i < 0),
        "pct_improved": sum(1 for i in improvements if i > 0) / len(results),
    }

    # Statistical test: is improvement > 0?
    if len(improvements) >= 3:
        t_stat, p_value = stats.ttest_1samp(improvements, 0)
        summary["t_stat"] = float(t_stat)
        summary["p_value"] = float(p_value)
        summary["significant_at_05"] = p_value < 0.05

    return summary


def compute_by_category(results: list[dict]) -> dict:
    """Compute summary by category."""
    by_cat = {}
    categories = set(r["category"] for r in results)

    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)

        if n < 2:
            by_cat[cat] = {"n": n, "insufficient_data": True}
            continue

        improvements = [r["brier_improvement"] for r in cat_results]
        brier_cond = [r["brier_conditional"] for r in cat_results]

        by_cat[cat] = {
            "n": n,
            "mean_brier_conditional": float(np.mean(brier_cond)),
            "mean_improvement": float(np.mean(improvements)),
            "n_improved": sum(1 for i in improvements if i > 0),
        }

    return by_cat


async def main():
    """Run the closed-question conditional calibration experiment."""
    print("=" * 70)
    print("Closed-Question Conditional Calibration Experiment")
    print("=" * 70)

    # Load curated pairs
    print("\nLoading curated pairs...")
    pairs = load_curated_pairs()
    print(f"  Loaded {len(pairs)} pairs")

    # Refresh resolution status from Polymarket
    enriched_pairs = await refresh_resolution_status(pairs)

    # Count resolution status
    status_counts = {"both_resolved": 0, "one_resolved": 0, "none_resolved": 0}
    for p in enriched_pairs:
        status_a = p.get("market_a_status")
        status_b = p.get("market_b_status")
        if status_a == "resolved" and status_b == "resolved":
            status_counts["both_resolved"] += 1
        elif status_a == "resolved" or status_b == "resolved":
            status_counts["one_resolved"] += 1
        else:
            status_counts["none_resolved"] += 1

    print("\nResolution status:")
    print(f"  Both resolved: {status_counts['both_resolved']}")
    print(f"  One resolved: {status_counts['one_resolved']}")
    print(f"  None resolved: {status_counts['none_resolved']}")

    # Filter to fully resolved pairs
    fully_resolved = filter_fully_resolved(enriched_pairs)
    print(f"\nFully resolved pairs for analysis: {len(fully_resolved)}")

    if not fully_resolved:
        print("\nNo fully resolved pairs found. Cannot run experiment.")
        # Save partial results
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": MODEL,
                "n_total_pairs": len(pairs),
            },
            "status_counts": status_counts,
            "error": "No fully resolved pairs found",
        }
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nPartial results saved to: {OUTPUT_FILE}")
        return

    # Show which pairs we have
    print("\nFully resolved pairs:")
    for p in fully_resolved:
        cat = p["classification"]["category"]
        print(f"  - [{cat}] {p['question_a'][:50]}... vs {p['question_b'][:50]}...")

    # Run experiment
    results = await run_experiment(fully_resolved)

    # Compute summary
    summary = compute_summary(results)
    by_category = compute_by_category(results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n--- Overall (n={summary['n']}) ---")
    print(f"  Mean Brier (LLM conditional): {summary['mean_brier_conditional']:.4f}")
    print(f"  Mean Brier (baseline 0.5):    {summary['mean_brier_baseline']:.4f}")
    print(f"  Mean improvement:             {summary['mean_improvement']:+.4f}")
    print(f"  Improved: {summary['n_improved']}/{summary['n']} ({summary['pct_improved']:.1%})")

    if "p_value" in summary:
        sig = "*" if summary["significant_at_05"] else ""
        print(f"  t-test (improvement > 0):     t={summary['t_stat']:.2f}, p={summary['p_value']:.4f}{sig}")

    print("\n--- By Category ---")
    for cat, metrics in sorted(by_category.items()):
        if metrics.get("insufficient_data"):
            print(f"  {cat}: n={metrics['n']} (insufficient)")
        else:
            print(f"  {cat}: n={metrics['n']}, brier={metrics['mean_brier_conditional']:.4f}, improvement={metrics['mean_improvement']:+.4f}")

    # Interpretation
    print("\n--- Interpretation ---")
    if summary["mean_improvement"] > 0:
        if summary.get("significant_at_05"):
            print("  POSITIVE: Conditioning significantly improves predictions")
            print("  → LLMs CAN do conditional forecasting when tested against ground truth")
        else:
            print("  WEAK POSITIVE: Conditioning helps but not significantly")
            print("  → Need more data to confirm")
    else:
        print("  NEGATIVE: Conditioning does not improve predictions")
        print("  → Consistent with Q2 failure - LLMs cannot estimate conditionals")

    # Comparison to Q2
    print("\n--- Comparison to Q2 ---")
    print("  Q2 tested: Does LLM VOI correlate with market VOI?")
    print("  This tests: Does LLM P(X|Q) predict actual X outcomes?")
    if summary["mean_improvement"] > 0 and summary.get("p_value", 1) < 0.1:
        print("  → If Q2 failed but this succeeds: market dynamics issue, not LLM capability")
    elif summary["mean_improvement"] <= 0:
        print("  → If both fail: LLMs genuinely cannot estimate conditionals")

    # Save results
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "n_total_pairs": len(pairs),
            "n_fully_resolved": len(fully_resolved),
        },
        "status_counts": status_counts,
        "summary": summary,
        "by_category": by_category,
        "results": results,
        "fully_resolved_pairs": fully_resolved,
    }

    # Make JSON serializable (handle numpy/bool types)
    output = make_json_serializable(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
