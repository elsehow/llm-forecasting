"""Q2 Experiment v2: E2E 1-call prompt for conditional estimation.

Same as Q2 v1, but uses the E2E 1-call prompt from Conditional Forecasting
which achieved 85% direction accuracy on the original validation set.

Key difference from v1:
- v1: Direct conditional estimation (ask for P(X|Q=yes), P(X|Q=no))
- v2: E2E 1-call (estimate ρ first, then use it to calibrate conditionals)

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/q2_llm_voi_estimation_v2.py
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
    rho_to_posteriors,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
CURATED_PAIRS_FILE = SCRIPT_DIR / "curated_pairs_nontrivial.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "q2_llm_voi_estimation_v2_results.json"

# Model to use - Opus 4.5 for best conditional estimation
MODEL = "claude-opus-4-5-20251101"

# E2E 1-call prompt (from Conditional Forecasting project)
# Key insight: estimating ρ first helps calibrate the magnitude of conditionals
E2E_1CALL_PROMPT = """You are a forecaster estimating conditional probabilities.

Question X (target - we want to forecast this): "{question_x}"
Question Q (signal - resolves first): "{question_q}"

Market data (given):
- P(X) = {p_x:.1%} (current market probability for X)
- P(Q) = {p_q:.1%} (current market probability for Q)

Step 1: First, estimate the correlation coefficient (ρ) between these questions.
- ρ > 0: They tend to move together (if Q resolves YES, X becomes more likely)
- ρ = 0: Independent
- ρ < 0: They move oppositely (if Q resolves YES, X becomes less likely)

Think carefully about:
- Are they competing for the same outcome? (negative ρ)
- Does one enable or require the other? (positive ρ)
- Do they share a common driver? (positive ρ)
- Are they truly independent? (ρ ≈ 0)

Step 2: Then, estimate P(X|Q=YES) and P(X|Q=NO), using your ρ estimate
to calibrate the magnitude of update.

Important: The conditionals should be consistent with your ρ estimate.
- If ρ > 0: P(X|Q=YES) > P(X) > P(X|Q=NO)
- If ρ < 0: P(X|Q=YES) < P(X) < P(X|Q=NO)
- If ρ = 0: P(X|Q=YES) ≈ P(X|Q=NO) ≈ P(X)

Respond with JSON only:
{{"rho_estimate": <float -1 to +1>, "p_x_given_q_yes": <float 0-1>, "p_x_given_q_no": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def load_data() -> list[dict]:
    """Load curated pairs with classifications."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)
    return curated["curated_pairs"]


async def estimate_conditionals_e2e(
    question_x: str,
    question_q: str,
    p_x: float,
    p_q: float,
    model: str = MODEL,
) -> tuple[float, float, float, str]:
    """Estimate conditionals using E2E 1-call approach.

    Returns:
        Tuple of (rho_estimate, p_x_given_q_yes, p_x_given_q_no, reasoning)
    """
    import litellm

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": E2E_1CALL_PROMPT.format(
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
        rho_est = float(result.get("rho_estimate", 0.0))
        p_x_yes = float(result.get("p_x_given_q_yes", p_x))
        p_x_no = float(result.get("p_x_given_q_no", p_x))
        reasoning = result.get("reasoning", "")

        # Clamp to valid range
        rho_est = max(-1.0, min(1.0, rho_est))
        p_x_yes = max(0.01, min(0.99, p_x_yes))
        p_x_no = max(0.01, min(0.99, p_x_no))

        return rho_est, p_x_yes, p_x_no, reasoning

    except Exception as e:
        return 0.0, p_x, p_x, f"Error: {e}"


async def run_experiment(pairs: list[dict]) -> list[dict]:
    """Run Q2 v2 experiment: E2E conditional estimation vs market VOI."""
    print(f"\nRunning Q2 v2 experiment (E2E 1-call) on {len(pairs)} pairs...")
    print(f"Model: {MODEL}")

    results = []

    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Processing {pair['classification']['category']}...")

        # Determine which market is X (target) and which is Q (signal)
        resolved = pair["resolved"]  # "A" or "B"

        if resolved == "A":
            question_q = pair["question_a"]
            question_x = pair["question_b"]
            p_x = pair["other_price_at_resolution"]
        else:
            question_q = pair["question_b"]
            question_x = pair["question_a"]
            p_x = pair["other_price_at_resolution"]

        p_q = 0.5  # Neutral assumption

        # Get ground-truth VOI from rho
        rho = pair["rho"]
        ground_truth_voi = entropy_voi_from_rho(p_x, p_q, rho)

        # Get LLM estimates using E2E 1-call
        rho_est, p_x_yes, p_x_no, reasoning = await estimate_conditionals_e2e(
            question_x=question_x,
            question_q=question_q,
            p_x=p_x,
            p_q=p_q,
            model=MODEL,
        )

        # Compute LLM VOI from estimated conditionals
        llm_voi = entropy_voi(p_x, p_q, p_x_yes, p_x_no)

        # Compute ground-truth conditionals from rho for comparison
        gt_p_x_yes, gt_p_x_no = rho_to_posteriors(rho, p_x, p_q)

        # Compute diagnostics
        llm_spread = abs(p_x_yes - p_x_no)
        gt_spread = abs(gt_p_x_yes - gt_p_x_no)

        # Direction based on LLM's explicit rho estimate
        llm_direction = 1 if rho_est > 0.05 else (-1 if rho_est < -0.05 else 0)
        gt_direction = 1 if rho > 0.05 else (-1 if rho < -0.05 else 0)
        direction_match = llm_direction == gt_direction

        # Also check if conditionals match direction
        cond_direction = 1 if p_x_yes > p_x_no else (-1 if p_x_yes < p_x_no else 0)
        cond_direction_match = cond_direction == gt_direction

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
            "llm_rho_estimate": rho_est,
            "llm_p_x_yes": p_x_yes,
            "llm_p_x_no": p_x_no,
            "llm_voi": llm_voi,
            "llm_spread": llm_spread,
            "llm_reasoning": reasoning,
            # Diagnostics
            "rho_direction_match": direction_match,
            "cond_direction_match": cond_direction_match,
            "rho_error": rho_est - rho,
            "spread_ratio": llm_spread / gt_spread if gt_spread > 0.001 else None,
        })

    return results


def compute_correlations(results: list[dict]) -> dict:
    """Compute correlation metrics between ground-truth and LLM VOI."""
    gt_voi = np.array([r["ground_truth_voi"] for r in results])
    llm_voi = np.array([r["llm_voi"] for r in results])

    # Primary metric: Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(gt_voi, llm_voi)
    pearson_r, pearson_p = stats.pearsonr(gt_voi, llm_voi)

    # Direction accuracy (based on explicit rho estimate)
    rho_direction_matches = sum(1 for r in results if r["rho_direction_match"])
    rho_direction_accuracy = rho_direction_matches / len(results)

    # Direction accuracy (based on conditionals)
    cond_direction_matches = sum(1 for r in results if r["cond_direction_match"])
    cond_direction_accuracy = cond_direction_matches / len(results)

    # Rho estimation accuracy
    gt_rhos = np.array([r["rho"] for r in results])
    llm_rhos = np.array([r["llm_rho_estimate"] for r in results])
    rho_corr, rho_p = stats.spearmanr(gt_rhos, llm_rhos)
    rho_mae = np.mean(np.abs(llm_rhos - gt_rhos))

    # Top-k agreement
    gt_top5_idx = set(np.argsort(gt_voi)[-5:])
    llm_top5_idx = set(np.argsort(llm_voi)[-5:])
    top5_overlap = len(gt_top5_idx & llm_top5_idx) / 5

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "rho_direction_accuracy": rho_direction_accuracy,
        "cond_direction_accuracy": cond_direction_accuracy,
        "rho_estimation_r": rho_corr,
        "rho_estimation_mae": rho_mae,
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
        rho_dir_acc = sum(1 for x in cat_results if x["rho_direction_match"]) / n

        # Rho estimation for this category
        gt_rhos = np.array([x["rho"] for x in cat_results])
        llm_rhos = np.array([x["llm_rho_estimate"] for x in cat_results])
        rho_corr, _ = stats.spearmanr(gt_rhos, llm_rhos)

        by_category[cat] = {
            "n": n,
            "spearman_r": r,
            "spearman_p": p,
            "rho_direction_accuracy": rho_dir_acc,
            "rho_estimation_r": rho_corr,
            "mean_gt_voi": float(np.mean(gt_voi)),
            "mean_llm_voi": float(np.mean(llm_voi)),
        }

    return by_category


async def main():
    """Run the Q2 v2 LLM VOI estimation experiment."""
    print("=" * 70)
    print("Q2 v2 Experiment: E2E 1-call for Conditional Estimation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    pairs = load_data()
    print(f"  Loaded {len(pairs)} curated pairs")

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
    print(f"  Rho direction accuracy: {correlations['rho_direction_accuracy']:.1%}")
    print(f"  Conditional direction accuracy: {correlations['cond_direction_accuracy']:.1%}")
    print(f"  Rho estimation r: {correlations['rho_estimation_r']:.3f}")
    print(f"  Rho estimation MAE: {correlations['rho_estimation_mae']:.3f}")
    print(f"  Top-5 agreement: {correlations['top5_agreement']:.1%}")

    # By category
    print("\n--- By Category ---")
    by_category = compute_by_category(results)

    # Structural pairs
    structural_cats = ["mutually_exclusive", "sequential_prerequisite"]
    structural_results = [r for r in results if r["category"] in structural_cats]
    if len(structural_results) >= 3:
        gt_voi = np.array([r["ground_truth_voi"] for r in structural_results])
        llm_voi = np.array([r["llm_voi"] for r in structural_results])
        r, p = stats.spearmanr(gt_voi, llm_voi)
        print(f"\n  STRUCTURAL (mutually_exclusive + sequential_prerequisite):")
        print(f"    n = {len(structural_results)}")
        print(f"    Spearman r = {r:.3f} (p = {p:.4f})")
        dir_acc = sum(1 for x in structural_results if x["rho_direction_match"]) / len(structural_results)
        print(f"    Rho direction accuracy = {dir_acc:.1%}")

    print("\n  Individual categories:")
    for cat, metrics in sorted(by_category.items()):
        if metrics.get("insufficient_data"):
            print(f"    {cat}: n={metrics['n']} (insufficient)")
        else:
            print(f"    {cat}: n={metrics['n']}, r={metrics['spearman_r']:.3f}, rho_dir={metrics['rho_direction_accuracy']:.1%}")

    # Compare to v1
    print("\n--- Comparison to v1 ---")
    print("  | Metric | v1 (direct) | v2 (E2E) |")
    print("  |--------|-------------|----------|")
    print(f"  | Spearman r | -0.209 | {correlations['spearman_r']:.3f} |")
    print(f"  | Direction accuracy | 41.2% | {correlations['rho_direction_accuracy']:.1%} |")

    # Interpretation
    print("\n--- Interpretation ---")
    overall_r = correlations["spearman_r"]
    if overall_r >= 0.5:
        verdict = "YES - E2E works for VOI estimation"
    elif overall_r >= 0.3:
        verdict = "PARTIAL - E2E helps but not sufficient"
    elif overall_r > -0.1:
        verdict = "NO - E2E doesn't improve over v1"
    else:
        verdict = "NO - E2E makes things worse"
    print(f"  Verdict: {verdict}")

    # Save results
    output = {
        "metadata": {
            "model": MODEL,
            "n_pairs": len(results),
            "generated_at": datetime.now().isoformat(),
            "experiment": "Q2 v2 LLM VOI Estimation (E2E 1-call)",
            "prompt_type": "E2E 1-call with explicit rho estimation",
        },
        "summary": {
            "spearman_r_all": correlations["spearman_r"],
            "spearman_p_all": correlations["spearman_p"],
            "rho_direction_accuracy": correlations["rho_direction_accuracy"],
            "cond_direction_accuracy": correlations["cond_direction_accuracy"],
            "rho_estimation_r": correlations["rho_estimation_r"],
            "top_5_agreement": correlations["top5_agreement"],
            "verdict": verdict,
        },
        "comparison_to_v1": {
            "v1_spearman_r": -0.209,
            "v2_spearman_r": correlations["spearman_r"],
            "v1_direction_accuracy": 0.412,
            "v2_direction_accuracy": correlations["rho_direction_accuracy"],
        },
        "correlations": correlations,
        "by_category": by_category,
        "results": results,
    }

    # Add structural subset
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
