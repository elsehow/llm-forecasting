"""Closed-Question Conditional Calibration - Metaculus Extension.

Extends the closed-question experiment to Metaculus pairs where both Q and X
have resolved. The metaculus_comovement_pairs.json has 1011 pairs with Q resolved
and 82 unique X questions - many of which may have also resolved.

Design:
- Load metaculus_comovement_pairs.json (has Q resolved, X probabilities)
- Query Metaculus API to find which X questions have now resolved
- For pairs where BOTH Q and X resolved:
  - Ask LLM: P(X|Q=yes), P(X|Q=no)
  - Select P(X|Q=actual) based on Q's actual resolution
  - Compute Brier score: (P(X|Q=actual) - X_outcome)²
- Compare to marginal baseline

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/closed_conditional_metaculus.py
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from monorepo root
_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import httpx
import numpy as np
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).parent
METACULUS_PAIRS_FILE = SCRIPT_DIR.parent / "metaculus-replication" / "data" / "metaculus_comovement_pairs.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "closed_conditional_metaculus_results.json"

# Model - same as Q2 for comparison
MODEL = "claude-opus-4-5-20251101"

# Metaculus API
METACULUS_API_URL = "https://www.metaculus.com/api"
METACULUS_API_KEY = os.environ.get("METACULUS_API_KEY", "")

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


def load_metaculus_pairs() -> list[dict]:
    """Load Metaculus comovement pairs."""
    with open(METACULUS_PAIRS_FILE) as f:
        data = json.load(f)
    return data["pairs"]


async def fetch_x_resolutions(x_ids: set[int]) -> dict[int, float | None]:
    """Fetch resolution status for X questions from Metaculus API.

    Returns dict mapping x_id -> resolution (1.0 for YES, 0.0 for NO, None for ambiguous/unresolved)
    """
    print(f"\nFetching resolution status for {len(x_ids)} unique X questions...")

    headers = {}
    if METACULUS_API_KEY:
        headers["Authorization"] = f"Token {METACULUS_API_KEY}"

    resolved = {}

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for i, x_id in enumerate(sorted(x_ids)):
            if i > 0 and i % 10 == 0:
                print(f"  Checked {i}/{len(x_ids)}...")

            try:
                resp = await client.get(f"{METACULUS_API_URL}/posts/{x_id}/")
                resp.raise_for_status()
                d = resp.json()

                is_resolved = d.get("resolved", False)
                if is_resolved:
                    question = d.get("question", {})
                    res = question.get("resolution", "").lower()
                    title = d.get("title", "")[:50]

                    if res == "yes":
                        resolved[x_id] = 1.0
                    elif res == "no":
                        resolved[x_id] = 0.0
                    else:
                        resolved[x_id] = None  # ambiguous/annulled

                # Rate limiting - 2 seconds between requests to avoid 429
                await asyncio.sleep(2.0)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    print(f"  Rate limited at {x_id}, waiting 30s...")
                    await asyncio.sleep(30)
                    # Retry once
                    try:
                        resp = await client.get(f"{METACULUS_API_URL}/posts/{x_id}/")
                        resp.raise_for_status()
                        d = resp.json()
                        is_resolved = d.get("resolved", False)
                        if is_resolved:
                            question = d.get("question", {})
                            res = question.get("resolution", "").lower()
                            if res == "yes":
                                resolved[x_id] = 1.0
                            elif res == "no":
                                resolved[x_id] = 0.0
                            else:
                                resolved[x_id] = None
                    except Exception:
                        pass
                else:
                    print(f"  Error fetching {x_id}: {e}")
            except Exception as e:
                print(f"  Error fetching {x_id}: {e}")

    print(f"  Found {len(resolved)} resolved X questions")
    return resolved


def filter_fully_resolved(pairs: list[dict], x_resolutions: dict[int, float | None]) -> list[dict]:
    """Filter to pairs where both Q and X have clear resolutions."""
    fully_resolved = []

    for pair in pairs:
        q_res = pair.get("q_resolution")
        x_id = pair.get("x_id")
        x_res = x_resolutions.get(x_id)

        # Need clear YES/NO resolutions for both (not None/ambiguous)
        if q_res is not None and x_res is not None:
            # Convert q_resolution to bool
            q_outcome = q_res > 0.5
            x_outcome = x_res > 0.5

            fully_resolved.append({
                **pair,
                "q_outcome": q_outcome,
                "x_outcome": x_outcome,
                "x_resolution": x_res,
            })

    return fully_resolved


async def estimate_conditionals(
    question_x: str,
    question_q: str,
    model: str = MODEL,
) -> tuple[float, float, str]:
    """Estimate P(X|Q=yes) and P(X|Q=no) using LLM."""
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


async def run_experiment(pairs: list[dict], max_pairs: int | None = None) -> list[dict]:
    """Run the conditional calibration experiment."""
    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"\nRunning experiment on {len(pairs)} fully-resolved pairs...")
    print(f"Model: {MODEL}")

    results = []

    for i, pair in enumerate(pairs):
        if i > 0 and i % 10 == 0:
            print(f"  Processed {i}/{len(pairs)}...")

        question_q = pair["q_title"]
        question_x = pair["x_title"]
        q_outcome = pair["q_outcome"]
        x_outcome = pair["x_outcome"]

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

        # Marginal baseline: use x_prob_before if available, else 0.5
        x_prob_before = pair.get("x_prob_before", 0.5)
        baseline_brier = brier_score(x_prob_before, x_outcome)

        # Also compute Brier for naive 0.5 baseline
        naive_brier = brier_score(0.5, x_outcome)

        # Store result
        results.append({
            "q_id": pair["q_id"],
            "x_id": pair["x_id"],
            "question_q": question_q,
            "question_x": question_x,
            "q_outcome": q_outcome,
            "x_outcome": x_outcome,
            "rho": pair.get("rho"),  # Historical correlation
            # Market data
            "x_prob_before": x_prob_before,
            "x_prob_after": pair.get("x_prob_after"),
            "x_delta_p": pair.get("x_delta_p"),
            # LLM estimates
            "p_x_given_q_yes": p_x_yes,
            "p_x_given_q_no": p_x_no,
            "p_x_conditional": p_x_conditional,
            "reasoning": reasoning,
            # Brier scores
            "brier_conditional": brier,
            "brier_baseline": baseline_brier,  # vs x_prob_before
            "brier_naive": naive_brier,  # vs 0.5
            "improvement_vs_baseline": baseline_brier - brier,
            "improvement_vs_naive": naive_brier - brier,
        })

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    if not results:
        return {"error": "No results"}

    brier_conditional = [r["brier_conditional"] for r in results]
    brier_baseline = [r["brier_baseline"] for r in results]
    brier_naive = [r["brier_naive"] for r in results]
    improvement_baseline = [r["improvement_vs_baseline"] for r in results]
    improvement_naive = [r["improvement_vs_naive"] for r in results]

    summary = {
        "n": len(results),
        "mean_brier_conditional": float(np.mean(brier_conditional)),
        "mean_brier_baseline": float(np.mean(brier_baseline)),
        "mean_brier_naive": float(np.mean(brier_naive)),
        "mean_improvement_vs_baseline": float(np.mean(improvement_baseline)),
        "mean_improvement_vs_naive": float(np.mean(improvement_naive)),
        "std_improvement_baseline": float(np.std(improvement_baseline)),
        "median_improvement_baseline": float(np.median(improvement_baseline)),
        # Counts
        "n_improved_vs_baseline": sum(1 for i in improvement_baseline if i > 0),
        "n_improved_vs_naive": sum(1 for i in improvement_naive if i > 0),
        "pct_improved_vs_baseline": sum(1 for i in improvement_baseline if i > 0) / len(results),
        "pct_improved_vs_naive": sum(1 for i in improvement_naive if i > 0) / len(results),
    }

    # Statistical test: is improvement > 0?
    if len(improvement_baseline) >= 3:
        t_stat, p_value = stats.ttest_1samp(improvement_baseline, 0)
        summary["t_stat_vs_baseline"] = float(t_stat)
        summary["p_value_vs_baseline"] = float(p_value)
        summary["significant_at_05"] = p_value < 0.05

    return summary


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


async def main():
    """Run the Metaculus closed-question conditional calibration experiment."""
    print("=" * 70)
    print("Closed-Question Conditional Calibration - Metaculus")
    print("=" * 70)

    # Load Metaculus pairs
    print("\nLoading Metaculus comovement pairs...")
    pairs = load_metaculus_pairs()
    print(f"  Loaded {len(pairs)} pairs")

    # Get unique X question IDs
    x_ids = set(p["x_id"] for p in pairs)
    print(f"  Unique X questions: {len(x_ids)}")

    # Fetch X resolutions from API
    x_resolutions = await fetch_x_resolutions(x_ids)

    # Filter to fully resolved pairs
    fully_resolved = filter_fully_resolved(pairs, x_resolutions)
    print(f"\nFully resolved pairs: {len(fully_resolved)}")

    if not fully_resolved:
        print("\nNo fully resolved pairs found. Cannot run experiment.")
        return

    # Show sample
    print("\nSample of fully resolved pairs:")
    for p in fully_resolved[:5]:
        print(f"  Q: {p['q_title'][:50]}...")
        print(f"  X: {p['x_title'][:50]}...")
        print(f"  Q={p['q_outcome']}, X={p['x_outcome']}")
        print()

    # Run experiment (limit to avoid too many API calls during testing)
    results = await run_experiment(fully_resolved, max_pairs=50)

    # Compute summary
    summary = compute_summary(results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n--- Overall (n={summary['n']}) ---")
    print(f"  Mean Brier (LLM conditional): {summary['mean_brier_conditional']:.4f}")
    print(f"  Mean Brier (market baseline): {summary['mean_brier_baseline']:.4f}")
    print(f"  Mean Brier (naive 0.5):       {summary['mean_brier_naive']:.4f}")
    print(f"  Improvement vs baseline:      {summary['mean_improvement_vs_baseline']:+.4f}")
    print(f"  Improvement vs naive:         {summary['mean_improvement_vs_naive']:+.4f}")
    print(f"  Improved vs baseline: {summary['n_improved_vs_baseline']}/{summary['n']} ({summary['pct_improved_vs_baseline']:.1%})")

    if "p_value_vs_baseline" in summary:
        sig = "*" if summary["significant_at_05"] else ""
        print(f"  t-test (improvement > 0):     t={summary['t_stat_vs_baseline']:.2f}, p={summary['p_value_vs_baseline']:.4f}{sig}")

    # Interpretation
    print("\n--- Interpretation ---")
    if summary["mean_improvement_vs_baseline"] > 0:
        if summary.get("significant_at_05"):
            print("  POSITIVE: LLM conditionals beat market baseline significantly")
        else:
            print("  WEAK POSITIVE: LLM conditionals help but not significantly")
    else:
        print("  NEGATIVE: LLM conditionals don't beat market baseline")

    # Save results
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "n_total_pairs": len(pairs),
            "n_x_resolved": len(x_resolutions),
            "n_fully_resolved": len(fully_resolved),
            "n_tested": len(results),
        },
        "summary": summary,
        "results": results,
        "x_resolutions": {str(k): v for k, v in x_resolutions.items()},
    }

    output = make_json_serializable(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
