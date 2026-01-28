"""Q2 Experiment: Can LLMs Estimate VOI Inputs Well Enough?

Test whether Opus 4.5 can estimate P(X|Q=yes), P(X|Q=no) well enough that
LLM-computed VOI correlates with market-derived VOI.

Key design choices:
- P(X) from market (observable)
- P(Q) from market (observable) — NOT LLM-estimated
- P(X|Q=yes), P(X|Q=no) from LLM

This isolates the conditional estimation question. Prior experiments showed
LLM P(Q) estimation hurt entropy VOI (r=-0.023 vs r=0.112 with P(Q)=0.5).
Using market P(Q) tests whether the problem is the conditionals themselves.

Ground truth: market-derived VOI using 60-day rho correlation.
LLM estimate: VOI computed from LLM conditionals + market P(X), P(Q).

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/q2_llm_voi_estimation.py
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

import numpy as np
from scipy import stats

from llm_forecasting.voi import (
    entropy_voi,
    entropy_voi_from_rho,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
CURATED_PAIRS_FILE = SCRIPT_DIR / "curated_pairs_nontrivial.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "q2_llm_voi_estimation_results.json"

# Model to use - Opus 4.5 for best conditional estimation
MODEL = "claude-opus-4-5-20251101"

# Prompt for conditional probability elicitation (market-informed)
# Key change from prior experiments: We PROVIDE P(X) and P(Q) from market
# and ONLY ask for the conditionals
CONDITIONAL_PROB_PROMPT = """You are estimating conditional probabilities for a forecasting scenario.

QUESTION X (target - we want to forecast this): "{question_x}"
QUESTION Q (signal - resolves first): "{question_q}"

Market data (these are given - use them as context):
- P(X) = {p_x:.3f} (current market probability for X)
- P(Q) = {p_q:.3f} (current market probability for Q)

Your task: Estimate the two conditional probabilities:

1. P(X | Q=yes): If Q resolves YES, what is the probability X resolves YES?

2. P(X | Q=no): If Q resolves NO, what is the probability X resolves YES?

Think carefully about:
- How are these questions related? (causal, competitive, shared drivers?)
- What does learning Q's outcome tell us about X?
- The conditionals must be consistent: P(X) = P(Q) * P(X|Q=yes) + (1-P(Q)) * P(X|Q=no)

Respond with JSON only:
{{"p_x_given_q_yes": <float 0.0-1.0>, "p_x_given_q_no": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


def load_data() -> list[dict]:
    """Load curated pairs with classifications."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)
    return curated["curated_pairs"]


async def estimate_conditionals(
    question_x: str,
    question_q: str,
    p_x: float,
    p_q: float,
    model: str = MODEL,
) -> tuple[float, float, str]:
    """Estimate P(X|Q=yes) and P(X|Q=no) using LLM.

    Unlike prior experiments, we provide market P(X) and P(Q) as context.
    The LLM only needs to estimate the conditionals.

    Args:
        question_x: The target question (X)
        question_q: The signal question (Q, resolves first)
        p_x: Market probability P(X)
        p_q: Market probability P(Q)
        model: LLM model to use

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
                    p_x=p_x,
                    p_q=p_q,
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
        p_x_yes = float(result.get("p_x_given_q_yes", p_x))
        p_x_no = float(result.get("p_x_given_q_no", p_x))
        reasoning = result.get("reasoning", "")

        # Clamp to valid range
        p_x_yes = max(0.01, min(0.99, p_x_yes))
        p_x_no = max(0.01, min(0.99, p_x_no))

        return p_x_yes, p_x_no, reasoning

    except Exception as e:
        # Fallback: no information (conditionals equal prior)
        return p_x, p_x, f"Error: {e}"


async def run_experiment(pairs: list[dict]) -> list[dict]:
    """Run Q2 experiment: LLM conditional estimation vs market VOI."""
    print(f"\nRunning Q2 experiment on {len(pairs)} pairs...")
    print(f"Model: {MODEL}")

    results = []

    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Processing {pair['classification']['category']}...")

        # Determine which market is X (target) and which is Q (signal)
        # The resolved market is Q (we learn its outcome first)
        resolved = pair["resolved"]  # "A" or "B"

        if resolved == "A":
            question_q = pair["question_a"]
            question_x = pair["question_b"]
            # P(X) is the "other" market price at resolution
            # P(Q) is harder - we use 0.5 as baseline since we don't have it
            # Actually, for this experiment we need market prices BEFORE resolution
            # other_price_at_resolution is the price of the unresolved market
            p_x = pair["other_price_at_resolution"]
        else:
            question_q = pair["question_b"]
            question_x = pair["question_a"]
            p_x = pair["other_price_at_resolution"]

        # For P(Q), we don't have the exact market price at resolution time
        # Use 0.5 as the neutral assumption (maximizes entropy)
        # This is conservative - actual P(Q) might be more extreme
        p_q = 0.5

        # Get ground-truth VOI from rho
        rho = pair["rho"]
        ground_truth_voi = entropy_voi_from_rho(p_x, p_q, rho)

        # Get LLM estimates of conditionals
        p_x_yes, p_x_no, reasoning = await estimate_conditionals(
            question_x=question_x,
            question_q=question_q,
            p_x=p_x,
            p_q=p_q,
            model=MODEL,
        )

        # Compute LLM VOI from estimated conditionals
        llm_voi = entropy_voi(p_x, p_q, p_x_yes, p_x_no)

        # Compute ground-truth conditionals from rho for comparison
        from llm_forecasting.voi import rho_to_posteriors
        gt_p_x_yes, gt_p_x_no = rho_to_posteriors(rho, p_x, p_q)

        # Compute diagnostics
        llm_spread = abs(p_x_yes - p_x_no)
        gt_spread = abs(gt_p_x_yes - gt_p_x_no)
        llm_direction = 1 if p_x_yes > p_x_no else (-1 if p_x_yes < p_x_no else 0)
        gt_direction = 1 if rho > 0 else (-1 if rho < 0 else 0)
        direction_match = llm_direction == gt_direction

        results.append({
            "question_x": question_x,
            "question_q": question_q,
            "category": pair["classification"]["category"],
            "resolved": resolved,
            # Market data
            "p_x_market": p_x,
            "p_q_market": p_q,
            "rho": rho,
            # Ground truth
            "ground_truth_voi": ground_truth_voi,
            "gt_p_x_yes": gt_p_x_yes,
            "gt_p_x_no": gt_p_x_no,
            "gt_spread": gt_spread,
            # LLM estimates
            "llm_p_x_yes": p_x_yes,
            "llm_p_x_no": p_x_no,
            "llm_voi": llm_voi,
            "llm_spread": llm_spread,
            "llm_reasoning": reasoning,
            # Diagnostics
            "direction_match": direction_match,
            "spread_ratio": llm_spread / gt_spread if gt_spread > 0.001 else None,
        })

    return results


def compute_correlations(results: list[dict]) -> dict:
    """Compute correlation metrics between ground-truth and LLM VOI."""
    gt_voi = np.array([r["ground_truth_voi"] for r in results])
    llm_voi = np.array([r["llm_voi"] for r in results])

    # Primary metric: Spearman correlation (robust to outliers)
    spearman_r, spearman_p = stats.spearmanr(gt_voi, llm_voi)

    # Secondary: Pearson for comparison
    pearson_r, pearson_p = stats.pearsonr(gt_voi, llm_voi)

    # Direction accuracy
    direction_matches = sum(1 for r in results if r["direction_match"])
    direction_accuracy = direction_matches / len(results)

    # Spread correlation (how well does LLM estimate |P(X|yes) - P(X|no)|)
    gt_spreads = np.array([r["gt_spread"] for r in results])
    llm_spreads = np.array([r["llm_spread"] for r in results])
    spread_r, spread_p = stats.spearmanr(gt_spreads, llm_spreads)

    # Top-k agreement (do high-VOI pairs match?)
    gt_top5_idx = set(np.argsort(gt_voi)[-5:])
    llm_top5_idx = set(np.argsort(llm_voi)[-5:])
    top5_overlap = len(gt_top5_idx & llm_top5_idx) / 5

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "direction_accuracy": direction_accuracy,
        "spread_spearman_r": spread_r,
        "spread_spearman_p": spread_p,
        "top5_agreement": top5_overlap,
        "n": len(results),
    }


def compute_by_category(results: list[dict]) -> dict:
    """Compute correlations by classification category."""
    by_category = {}
    categories = set(r["category"] for r in results)

    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)

        if n < 3:
            by_category[cat] = {"n": n, "insufficient_data": True}
            continue

        gt_voi = np.array([r["ground_truth_voi"] for r in cat_results])
        llm_voi = np.array([r["llm_voi"] for r in cat_results])

        r, p = stats.spearmanr(gt_voi, llm_voi)
        direction_acc = sum(1 for x in cat_results if x["direction_match"]) / n

        by_category[cat] = {
            "n": n,
            "spearman_r": r,
            "spearman_p": p,
            "direction_accuracy": direction_acc,
            "mean_gt_voi": float(np.mean(gt_voi)),
            "mean_llm_voi": float(np.mean(llm_voi)),
        }

    return by_category


async def main():
    """Run the Q2 LLM VOI estimation experiment."""
    print("=" * 70)
    print("Q2 Experiment: Can LLMs Estimate VOI Inputs Well Enough?")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    pairs = load_data()
    print(f"  Loaded {len(pairs)} curated pairs")

    # Count by category
    categories = {}
    for p in pairs:
        cat = p["classification"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\n  By category:")
    for cat, n in sorted(categories.items()):
        print(f"    {cat}: {n}")

    # Run experiment
    results = await run_experiment(pairs)

    # Compute correlations
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    correlations = compute_correlations(results)

    print("\n--- Overall Metrics (n={}) ---".format(correlations["n"]))
    print(f"  Spearman r (GT VOI vs LLM VOI): {correlations['spearman_r']:.3f} (p={correlations['spearman_p']:.4f})")
    print(f"  Pearson r: {correlations['pearson_r']:.3f}")
    print(f"  Direction accuracy: {correlations['direction_accuracy']:.1%}")
    print(f"  Spread correlation: {correlations['spread_spearman_r']:.3f}")
    print(f"  Top-5 agreement: {correlations['top5_agreement']:.1%}")

    # By category
    print("\n--- By Category ---")
    by_category = compute_by_category(results)

    # Structural pairs (key test)
    structural_cats = ["mutually_exclusive", "sequential_prerequisite"]
    structural_results = [r for r in results if r["category"] in structural_cats]
    if len(structural_results) >= 3:
        gt_voi = np.array([r["ground_truth_voi"] for r in structural_results])
        llm_voi = np.array([r["llm_voi"] for r in structural_results])
        r, p = stats.spearmanr(gt_voi, llm_voi)
        print(f"\n  STRUCTURAL (mutually_exclusive + sequential_prerequisite):")
        print(f"    n = {len(structural_results)}")
        print(f"    Spearman r = {r:.3f} (p = {p:.4f})")
        dir_acc = sum(1 for x in structural_results if x["direction_match"]) / len(structural_results)
        print(f"    Direction accuracy = {dir_acc:.1%}")

    print("\n  Individual categories:")
    for cat, metrics in sorted(by_category.items()):
        if metrics.get("insufficient_data"):
            print(f"    {cat}: n={metrics['n']} (insufficient)")
        else:
            print(f"    {cat}: n={metrics['n']}, r={metrics['spearman_r']:.3f}, dir_acc={metrics['direction_accuracy']:.1%}")

    # Interpretation
    print("\n--- Interpretation ---")
    overall_r = correlations["spearman_r"]
    if overall_r >= 0.5:
        verdict = "YES - LLMs can estimate VOI inputs"
    elif overall_r >= 0.3:
        verdict = "PARTIAL - Works for some pair types"
    else:
        verdict = "NO - LLMs cannot reliably estimate conditionals"
    print(f"  Overall verdict: {verdict}")

    # Compare to baselines
    print("\n--- Baseline Comparison ---")
    print("  | Metric | Value |")
    print("  |--------|-------|")
    print(f"  | Q2 Spearman r (this experiment) | {overall_r:.3f} |")
    print("  | Q1 structural r (market-derived) | 0.77-0.83 |")
    print("  | Phase 2 entropy VOI r (P(Q)=0.5) | 0.112 |")
    print("  | Phase 2 entropy VOI r (LLM P(Q)) | -0.023 |")

    # Save results
    output = {
        "metadata": {
            "model": MODEL,
            "n_pairs": len(results),
            "generated_at": datetime.now().isoformat(),
            "experiment": "Q2 LLM VOI Estimation",
        },
        "summary": {
            "spearman_r_all": correlations["spearman_r"],
            "spearman_p_all": correlations["spearman_p"],
            "direction_accuracy": correlations["direction_accuracy"],
            "top_5_agreement": correlations["top5_agreement"],
            "verdict": verdict,
        },
        "correlations": correlations,
        "by_category": by_category,
        "results": results,
    }

    # Add structural subset if available
    if len(structural_results) >= 3:
        gt_voi = np.array([r["ground_truth_voi"] for r in structural_results])
        llm_voi = np.array([r["llm_voi"] for r in structural_results])
        r, p = stats.spearmanr(gt_voi, llm_voi)
        output["summary"]["spearman_r_structural"] = r
        output["summary"]["n_structural"] = len(structural_results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
