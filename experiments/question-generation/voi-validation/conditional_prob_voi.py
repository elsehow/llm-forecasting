"""Conditional-Probability VOI Experiment on Polymarket Pairs.

Compare conditional-prob VOI (Russell-style direct elicitation) to ρ-based VOI
on the same 34 Polymarket pairs to enable apples-to-apples comparison.

Russell method:
    P(crux=yes), P(ultimate|crux=yes), P(ultimate|crux=no)
    VOI = p_yes × |p_ult_yes - p_ult| + p_no × |p_ult_no - p_ult|

ρ-based method:
    Convert ρ to posteriors, then compute VOI

This isolates whether differences come from domain (Polymarket vs Russell) or
method (direct conditional elicitation vs ρ conversion).

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/conditional_prob_voi.py
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

from scipy import stats
import numpy as np

from llm_forecasting.voi import (
    linear_voi,
    entropy_voi,
    linear_voi_from_rho,
    entropy_voi_from_rho,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
CURATED_PAIRS_FILE = SCRIPT_DIR / "curated_pairs_nontrivial.json"
VALIDATION_RESULTS_FILE = SCRIPT_DIR / "results" / "voi_validation_nontrivial.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "conditional_prob_voi.json"

# Model to use for LLM estimation
MODEL = "anthropic/claude-sonnet-4-20250514"

# Prompt for direct conditional probability elicitation
CONDITIONAL_PROB_PROMPT = """You are estimating probabilities for a forecasting scenario.

CRUX QUESTION (signal): "{crux_question}"
ULTIMATE QUESTION (target): "{ultimate_question}"

Estimate three probabilities:

1. P(crux = yes): What is the probability the crux question resolves YES?

2. P(ultimate | crux = yes): If the crux resolves YES, what is the probability the ultimate question resolves YES?

3. P(ultimate | crux = no): If the crux resolves NO, what is the probability the ultimate question resolves YES?

Think about:
- How are these questions related?
- What does the crux outcome tell us about the ultimate?
- Are there shared factors or causal relationships?

Respond with JSON only:
{{"p_crux": <float 0.0-1.0>, "p_ultimate_given_yes": <float 0.0-1.0>, "p_ultimate_given_no": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


def load_data() -> tuple[list[dict], dict]:
    """Load curated pairs and validation results."""
    with open(CURATED_PAIRS_FILE) as f:
        curated = json.load(f)

    # Try to load validation results (has actual_shift data)
    validation = {}
    if VALIDATION_RESULTS_FILE.exists():
        with open(VALIDATION_RESULTS_FILE) as f:
            validation = json.load(f)

    return curated["curated_pairs"], validation


async def estimate_conditional_probs(
    crux_question: str,
    ultimate_question: str,
    model: str = MODEL,
) -> tuple[float, float, float, str]:
    """Estimate P(crux), P(ultimate|crux=yes), P(ultimate|crux=no) using LLM.

    Returns:
        Tuple of (p_crux, p_ult_given_yes, p_ult_given_no, reasoning)
    """
    import litellm

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": CONDITIONAL_PROB_PROMPT.format(
                    crux_question=crux_question,
                    ultimate_question=ultimate_question,
                )
            }],
            max_tokens=400,
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
        p_crux = float(result.get("p_crux", 0.5))
        p_ult_yes = float(result.get("p_ultimate_given_yes", 0.5))
        p_ult_no = float(result.get("p_ultimate_given_no", 0.5))
        reasoning = result.get("reasoning", "")

        # Clamp to valid range
        p_crux = max(0.01, min(0.99, p_crux))
        p_ult_yes = max(0.01, min(0.99, p_ult_yes))
        p_ult_no = max(0.01, min(0.99, p_ult_no))

        return p_crux, p_ult_yes, p_ult_no, reasoning

    except Exception as e:
        return 0.5, 0.5, 0.5, f"Error: {e}"


async def run_experiment(pairs: list[dict], validation: dict) -> list[dict]:
    """Run conditional prob VOI estimation for all pairs."""
    print(f"\nEstimating conditional probabilities for {len(pairs)} pairs...")

    # Build lookup for validation data (has actual_shift)
    validation_lookup = {}
    if "pairs" in validation:
        for p in validation["pairs"]:
            key = (p["question_a"], p["question_b"])
            validation_lookup[key] = p

    results = []

    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Processing...")

        # Determine which market is "crux" and which is "ultimate"
        # The resolved market is the crux (we learn its outcome first)
        resolved = pair["resolved"]  # "A" or "B"

        if resolved == "A":
            crux_question = pair["question_a"]
            ultimate_question = pair["question_b"]
        else:
            crux_question = pair["question_b"]
            ultimate_question = pair["question_a"]

        # Get LLM estimates
        p_crux, p_ult_yes, p_ult_no, reasoning = await estimate_conditional_probs(
            crux_question,
            ultimate_question,
            model=MODEL,
        )

        # Compute unconditional P(ultimate) using total probability
        p_ultimate = p_crux * p_ult_yes + (1 - p_crux) * p_ult_no

        # Compute conditional-prob VOI (Russell formula)
        voi_conditional_linear = linear_voi(p_ultimate, p_crux, p_ult_yes, p_ult_no)
        voi_conditional_entropy = entropy_voi(p_ultimate, p_crux, p_ult_yes, p_ult_no)

        # Get ground-truth data for comparison
        key = (pair["question_a"], pair["question_b"])
        val_data = validation_lookup.get(key, {})
        actual_shift = val_data.get("actual_shift", None)
        p_before = val_data.get("p_before", pair.get("other_price_at_resolution", 0.5))

        # Compute ρ-based VOI (ground truth) for comparison
        ground_truth_rho = pair["rho"]
        voi_rho_linear = linear_voi_from_rho(ground_truth_rho, p_before, 0.5)
        voi_rho_entropy = entropy_voi_from_rho(ground_truth_rho, p_before, 0.5)

        results.append({
            "question_a": pair["question_a"],
            "question_b": pair["question_b"],
            "classification": pair["classification"]["category"],
            "resolved": resolved,
            "crux_question": crux_question,
            "ultimate_question": ultimate_question,
            # LLM estimates
            "p_crux_llm": p_crux,
            "p_ultimate_given_yes_llm": p_ult_yes,
            "p_ultimate_given_no_llm": p_ult_no,
            "p_ultimate_llm": p_ultimate,
            "reasoning": reasoning,
            # Conditional-prob VOI
            "voi_conditional_linear": voi_conditional_linear,
            "voi_conditional_entropy": voi_conditional_entropy,
            # Ground-truth data
            "ground_truth_rho": ground_truth_rho,
            "p_before": p_before,
            "actual_shift": actual_shift,
            # ρ-based VOI for comparison
            "voi_rho_linear": voi_rho_linear,
            "voi_rho_entropy": voi_rho_entropy,
        })

    return results


def compute_correlations(results: list[dict]) -> dict:
    """Compute correlation metrics."""
    # Filter to pairs with actual_shift data
    valid_results = [r for r in results if r["actual_shift"] is not None]

    if len(valid_results) < 5:
        return {"error": f"Only {len(valid_results)} pairs with actual_shift data"}

    # Extract arrays
    actual_shifts = np.array([r["actual_shift"] for r in valid_results])
    voi_conditional_linear = np.array([r["voi_conditional_linear"] for r in valid_results])
    voi_conditional_entropy = np.array([r["voi_conditional_entropy"] for r in valid_results])
    voi_rho_linear = np.array([r["voi_rho_linear"] for r in valid_results])
    voi_rho_entropy = np.array([r["voi_rho_entropy"] for r in valid_results])

    correlations = {}

    # Conditional-prob VOI vs actual shift (the main test)
    r, p = stats.pearsonr(voi_conditional_linear, actual_shifts)
    correlations["conditional_linear_voi_vs_shift"] = {"r": r, "p": p}

    r, p = stats.pearsonr(voi_conditional_entropy, actual_shifts)
    correlations["conditional_entropy_voi_vs_shift"] = {"r": r, "p": p}

    # Ground-truth ρ-based VOI vs actual shift (baseline)
    r, p = stats.pearsonr(voi_rho_linear, actual_shifts)
    correlations["rho_linear_voi_vs_shift"] = {"r": r, "p": p}

    r, p = stats.pearsonr(voi_rho_entropy, actual_shifts)
    correlations["rho_entropy_voi_vs_shift"] = {"r": r, "p": p}

    # Spearman for robustness
    rho, p = stats.spearmanr(voi_conditional_linear, actual_shifts)
    correlations["conditional_linear_voi_vs_shift_spearman"] = {"rho": rho, "p": p}

    return correlations


def compute_by_category(results: list[dict]) -> dict:
    """Compute correlations by classification category."""
    valid_results = [r for r in results if r["actual_shift"] is not None]

    by_category = {}
    categories = set(r["classification"] for r in valid_results)

    for cat in categories:
        cat_results = [r for r in valid_results if r["classification"] == cat]
        n = len(cat_results)

        if n < 3:
            by_category[cat] = {"n": n, "insufficient_data": True}
            continue

        actual_shifts = np.array([r["actual_shift"] for r in cat_results])
        voi_conditional = np.array([r["voi_conditional_linear"] for r in cat_results])
        voi_rho = np.array([r["voi_rho_linear"] for r in cat_results])

        r_cond, p_cond = stats.pearsonr(voi_conditional, actual_shifts)
        r_rho, p_rho = stats.pearsonr(voi_rho, actual_shifts)

        by_category[cat] = {
            "n": n,
            "conditional_voi_r": r_cond,
            "conditional_voi_p": p_cond,
            "rho_voi_r": r_rho,
            "rho_voi_p": p_rho,
            "mean_actual_shift": float(np.mean(actual_shifts)),
        }

    return by_category


async def main():
    """Run the conditional probability VOI experiment."""
    print("=" * 70)
    print("Conditional-Probability VOI Experiment on Polymarket")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    pairs, validation = load_data()
    print(f"  Curated pairs: {len(pairs)}")
    print(f"  Validation data available: {len(validation.get('pairs', []))}")

    # Run experiment
    results = await run_experiment(pairs, validation)

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- VOI vs Actual Shift Correlations ---")

    if "conditional_linear_voi_vs_shift" in correlations:
        r_cond = correlations["conditional_linear_voi_vs_shift"]["r"]
        p_cond = correlations["conditional_linear_voi_vs_shift"]["p"]
        print(f"Conditional-prob Linear VOI:  r = {r_cond:.3f} (p = {p_cond:.4f})")

    if "conditional_entropy_voi_vs_shift" in correlations:
        r_cond_ent = correlations["conditional_entropy_voi_vs_shift"]["r"]
        p_cond_ent = correlations["conditional_entropy_voi_vs_shift"]["p"]
        print(f"Conditional-prob Entropy VOI: r = {r_cond_ent:.3f} (p = {p_cond_ent:.4f})")

    if "rho_entropy_voi_vs_shift" in correlations:
        r_rho = correlations["rho_entropy_voi_vs_shift"]["r"]
        p_rho = correlations["rho_entropy_voi_vs_shift"]["p"]
        print(f"Ground-truth ρ Entropy VOI:   r = {r_rho:.3f} (p = {p_rho:.4f}) [baseline]")

    # By category analysis
    print("\n--- By Category ---")
    by_category = compute_by_category(results)
    for cat, metrics in sorted(by_category.items()):
        if metrics.get("insufficient_data"):
            print(f"  {cat}: n={metrics['n']} (insufficient data)")
        else:
            print(f"  {cat}: n={metrics['n']}, cond_r={metrics['conditional_voi_r']:.3f}, rho_r={metrics['rho_voi_r']:.3f}")

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\n| Method | r vs actual_shift |")
    print("|--------|-------------------|")
    if "rho_entropy_voi_vs_shift" in correlations:
        print(f"| Ground-truth ρ VOI | {correlations['rho_entropy_voi_vs_shift']['r']:.3f} |")
    if "conditional_entropy_voi_vs_shift" in correlations:
        print(f"| Conditional-prob VOI | {correlations['conditional_entropy_voi_vs_shift']['r']:.3f} |")
    print("| Calibrated LLM ρ VOI | 0.355 (from prior experiment) |")
    print("| Baseline LLM ρ VOI | 0.138 (from prior experiment) |")

    # Save results
    output = {
        "metadata": {
            "n_pairs": len(results),
            "n_with_shift": len([r for r in results if r["actual_shift"] is not None]),
            "model": MODEL,
            "generated_at": datetime.now().isoformat(),
        },
        "correlations": correlations,
        "by_category": by_category,
        "comparison": {
            "ground_truth_rho_voi_r": correlations.get("rho_entropy_voi_vs_shift", {}).get("r"),
            "conditional_prob_voi_r": correlations.get("conditional_entropy_voi_vs_shift", {}).get("r"),
            "calibrated_llm_rho_voi_r": 0.355,  # From prior experiment
            "baseline_llm_rho_voi_r": 0.138,    # From prior experiment
        },
        "pairs": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
