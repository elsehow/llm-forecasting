"""Q2 Experiment v3: "What Moves Prices" prompt framing.

Tests whether reframing from "logical relationships" to "price movements"
improves conditional estimation.

Key insight from failure analysis:
- LLMs reason about logical relationships (mutually exclusive → negative)
- Markets move on sentiment, shared drivers, information flow
- Need to shift the frame to "what news moves both prices"

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/q2_llm_voi_estimation_v3.py
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
OUTPUT_FILE = SCRIPT_DIR / "results" / "q2_llm_voi_estimation_v3_results.json"

# Model
MODEL = "claude-opus-4-5-20251101"

# v3 Prompt: Flip the frame from logical relationships to price movements
PRICE_MOVEMENT_PROMPT = """You are estimating how prediction market PRICES correlate, not how OUTCOMES logically relate.

Question X (target): "{question_x}"
Question Q (signal): "{question_q}"

Market data:
- P(X) = {p_x:.1%} (current market probability)
- P(Q) = {p_q:.1%} (current market probability)

IMPORTANT: Market correlations often DIFFER from logical relationships.

Example: "Trump nominates Warsh" and "Trump nominates Waller" are logically mutually exclusive (only one can be nominated). But markets show POSITIVE correlation because both prices move on shared sentiment ("Trump wants an unconventional Fed chair").

To estimate the correlation, ask yourself:
1. What NEWS would cause BOTH prices to move UP together?
2. What NEWS would cause BOTH prices to move DOWN together?
3. What NEWS would cause them to move in OPPOSITE directions?

If you can easily imagine news that moves both up or both down, the correlation is likely POSITIVE, even if the outcomes seem mutually exclusive or unrelated.

Common patterns:
- Alternative candidates (same race/position) often have POSITIVE ρ (shared "change" sentiment)
- Events in different domains can have ρ ≠ 0 if they share bettors or news cycles
- Logically causal relationships may have WEAKER market ρ than you'd expect

Now estimate:
- ρ: correlation coefficient (-1 to +1) based on how PRICES would co-move
- P(X|Q=yes): probability of X if Q resolves YES
- P(X|Q=no): probability of X if Q resolves NO

Respond with JSON only:
{{"rho_estimate": <float>, "p_x_given_q_yes": <float>, "p_x_given_q_no": <float>, "reasoning": "<what news moves both prices>"}}"""


def load_data() -> list[dict]:
    """Load curated pairs with classifications."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)
    return curated["curated_pairs"]


async def estimate_conditionals_v3(
    question_x: str,
    question_q: str,
    p_x: float,
    p_q: float,
    model: str = MODEL,
) -> tuple[float, float, float, str]:
    """Estimate conditionals using price-movement framing."""
    import litellm

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": PRICE_MOVEMENT_PROMPT.format(
                    question_x=question_x,
                    question_q=question_q,
                    p_x=p_x,
                    p_q=p_q,
                )
            }],
            max_tokens=600,
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
    """Run Q2 v3 experiment."""
    print(f"\nRunning Q2 v3 experiment (price-movement framing) on {len(pairs)} pairs...")
    print(f"Model: {MODEL}")

    results = []

    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Processing {pair['classification']['category']}...")

        resolved = pair["resolved"]
        if resolved == "A":
            question_q = pair["question_a"]
            question_x = pair["question_b"]
            p_x = pair["other_price_at_resolution"]
        else:
            question_q = pair["question_b"]
            question_x = pair["question_a"]
            p_x = pair["other_price_at_resolution"]

        p_q = 0.5

        # Ground truth
        rho = pair["rho"]
        ground_truth_voi = entropy_voi_from_rho(p_x, p_q, rho)
        gt_p_x_yes, gt_p_x_no = rho_to_posteriors(rho, p_x, p_q)

        # LLM estimate
        rho_est, p_x_yes, p_x_no, reasoning = await estimate_conditionals_v3(
            question_x=question_x,
            question_q=question_q,
            p_x=p_x,
            p_q=p_q,
            model=MODEL,
        )

        llm_voi = entropy_voi(p_x, p_q, p_x_yes, p_x_no)

        # Diagnostics
        llm_spread = abs(p_x_yes - p_x_no)
        gt_spread = abs(gt_p_x_yes - gt_p_x_no)

        llm_direction = 1 if rho_est > 0.05 else (-1 if rho_est < -0.05 else 0)
        gt_direction = 1 if rho > 0.05 else (-1 if rho < -0.05 else 0)
        direction_match = llm_direction == gt_direction

        cond_direction = 1 if p_x_yes > p_x_no else (-1 if p_x_yes < p_x_no else 0)
        cond_direction_match = cond_direction == gt_direction

        results.append({
            "question_x": question_x,
            "question_q": question_q,
            "category": pair["classification"]["category"],
            "resolved": resolved,
            "p_x_market": p_x,
            "p_q_market": p_q,
            "rho": rho,
            "ground_truth_voi": ground_truth_voi,
            "gt_p_x_yes": gt_p_x_yes,
            "gt_p_x_no": gt_p_x_no,
            "gt_spread": gt_spread,
            "llm_rho_estimate": rho_est,
            "llm_p_x_yes": p_x_yes,
            "llm_p_x_no": p_x_no,
            "llm_voi": llm_voi,
            "llm_spread": llm_spread,
            "llm_reasoning": reasoning,
            "rho_direction_match": direction_match,
            "cond_direction_match": cond_direction_match,
            "rho_error": rho_est - rho,
            "spread_ratio": llm_spread / gt_spread if gt_spread > 0.001 else None,
        })

    return results


def compute_correlations(results: list[dict]) -> dict:
    """Compute correlation metrics."""
    gt_voi = np.array([r["ground_truth_voi"] for r in results])
    llm_voi = np.array([r["llm_voi"] for r in results])

    spearman_r, spearman_p = stats.spearmanr(gt_voi, llm_voi)
    pearson_r, pearson_p = stats.pearsonr(gt_voi, llm_voi)

    rho_direction_matches = sum(1 for r in results if r["rho_direction_match"])
    rho_direction_accuracy = rho_direction_matches / len(results)

    cond_direction_matches = sum(1 for r in results if r["cond_direction_match"])
    cond_direction_accuracy = cond_direction_matches / len(results)

    gt_rhos = np.array([r["rho"] for r in results])
    llm_rhos = np.array([r["llm_rho_estimate"] for r in results])
    rho_corr, rho_p = stats.spearmanr(gt_rhos, llm_rhos)
    rho_mae = np.mean(np.abs(llm_rhos - gt_rhos))

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
    """Compute correlations by category."""
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

        gt_rhos = np.array([x["rho"] for x in cat_results])
        llm_rhos = np.array([x["llm_rho_estimate"] for x in cat_results])

        # Handle constant arrays
        if np.std(gt_rhos) > 0 and np.std(llm_rhos) > 0:
            rho_corr, _ = stats.spearmanr(gt_rhos, llm_rhos)
        else:
            rho_corr = 0.0

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
    """Run the Q2 v3 experiment."""
    print("=" * 70)
    print("Q2 v3 Experiment: Price-Movement Framing")
    print("=" * 70)

    pairs = load_data()
    print(f"\nLoaded {len(pairs)} curated pairs")

    results = await run_experiment(pairs)

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

    # Compare to v1 and v2
    print("\n--- Comparison to v1/v2 ---")
    print("  | Metric | v1 (direct) | v2 (E2E) | v3 (price-mvmt) |")
    print("  |--------|-------------|----------|-----------------|")
    print(f"  | Spearman r | -0.209 | -0.130 | {correlations['spearman_r']:.3f} |")
    print(f"  | Direction acc | 41.2% | 38.2% | {correlations['rho_direction_accuracy']:.1%} |")

    # Interpretation
    print("\n--- Interpretation ---")
    overall_r = correlations["spearman_r"]
    dir_acc = correlations["rho_direction_accuracy"]

    if overall_r >= 0.3 and dir_acc >= 0.5:
        verdict = "IMPROVED - Price framing helps"
    elif overall_r > -0.1 and dir_acc > 0.45:
        verdict = "MARGINAL - Slight improvement"
    else:
        verdict = "NO CHANGE - Price framing doesn't help"
    print(f"  Verdict: {verdict}")

    # Save results
    output = {
        "metadata": {
            "model": MODEL,
            "n_pairs": len(results),
            "generated_at": datetime.now().isoformat(),
            "experiment": "Q2 v3 LLM VOI Estimation (Price-Movement Framing)",
            "prompt_type": "Price movement framing with market-logic warning",
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
        "comparison": {
            "v1_spearman_r": -0.209,
            "v1_direction_accuracy": 0.412,
            "v2_spearman_r": -0.130,
            "v2_direction_accuracy": 0.382,
            "v3_spearman_r": correlations["spearman_r"],
            "v3_direction_accuracy": correlations["rho_direction_accuracy"],
        },
        "correlations": correlations,
        "by_category": by_category,
        "results": results,
    }

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
