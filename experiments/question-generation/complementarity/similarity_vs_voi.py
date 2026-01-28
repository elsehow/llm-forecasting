#!/usr/bin/env python3
"""
Similarity vs VOI: Is semantic similarity a sufficient proxy for information flow?

Hypothesis: Semantic similarity may be a cheaper proxy for information flow than LLM rho estimates.

Evidence so far:
- Co-movement validation (n=78): similarity r=+0.41, VOI r=+0.17
- Cross-domain (n=168): VOI r=+0.43

Tests:
1. Head-to-head comparison: similarity vs VOI correlation with actual_shift
2. Incremental value: Does VOI add information beyond similarity?
3. Bin analysis: At what similarity level does VOI stop adding value?
4. Conditional update prediction: Does similarity predict |P(A|B) - P(A)|?

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/complementarity/similarity_vs_voi.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "within-category" / "data"
OUTPUT_DIR = SCRIPT_DIR / "data"
VALIDATION_FILE = DATA_DIR / "within_vs_cross_validation.json"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_validation_data() -> list[dict]:
    """Load resolved pairs from validation experiment."""
    with open(VALIDATION_FILE) as f:
        data = json.load(f)
    return data["pairs"]


def compute_embeddings(questions: list[str], model) -> dict[str, np.ndarray]:
    """Compute normalized embeddings for all questions."""
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    return {q: embeddings_norm[i] for i, q in enumerate(questions)}


def compute_similarity(q1: str, q2: str, embeddings: dict[str, np.ndarray]) -> float:
    """Compute cosine similarity between two questions."""
    if q1 not in embeddings or q2 not in embeddings:
        return 0.0
    return float(embeddings[q1] @ embeddings[q2])


def partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Compute partial correlation of x and y controlling for z.

    Returns (r, p-value).
    """
    # Regress x on z
    slope_x, intercept_x, *_ = stats.linregress(z, x)
    resid_x = x - (slope_x * z + intercept_x)

    # Regress y on z
    slope_y, intercept_y, *_ = stats.linregress(z, y)
    resid_y = y - (slope_y * z + intercept_y)

    # Correlation of residuals
    return stats.pearsonr(resid_x, resid_y)


def run_head_to_head(df: pd.DataFrame) -> dict:
    """Test 1: Compare similarity vs VOI correlation with actual_shift."""
    print("\n" + "=" * 70)
    print("TEST 1: HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    r_sim, p_sim = stats.pearsonr(df["similarity"], df["actual_shift"])
    r_voi, p_voi = stats.pearsonr(df["linear_voi"], df["actual_shift"])

    print(f"\n  Similarity vs actual_shift:  r={r_sim:+.4f} (p={p_sim:.4e})")
    print(f"  VOI vs actual_shift:         r={r_voi:+.4f} (p={p_voi:.4e})")

    if r_sim > r_voi:
        winner = "SIMILARITY"
        delta = r_sim - r_voi
    else:
        winner = "VOI"
        delta = r_voi - r_sim

    print(f"\n  Winner: {winner} (delta = {delta:+.4f})")

    return {
        "similarity_vs_shift": {"r": float(r_sim), "p": float(p_sim)},
        "voi_vs_shift": {"r": float(r_voi), "p": float(p_voi)},
        "winner": winner,
        "delta": float(delta),
    }


def run_incremental_value(df: pd.DataFrame) -> dict:
    """Test 2: Does VOI add information beyond similarity?"""
    print("\n" + "=" * 70)
    print("TEST 2: INCREMENTAL VALUE (VOI after controlling for similarity)")
    print("=" * 70)

    # Partial correlation: VOI vs shift | similarity
    r_partial, p_partial = partial_correlation(
        df["linear_voi"].values,
        df["actual_shift"].values,
        df["similarity"].values,
    )

    print(f"\n  Partial r(VOI, shift | similarity): r={r_partial:+.4f} (p={p_partial:.4e})")

    # Also do OLS regression
    from scipy.stats import zscore

    X = pd.DataFrame({
        "similarity": zscore(df["similarity"]),
        "voi": zscore(df["linear_voi"]),
    })
    y = zscore(df["actual_shift"])

    # Simple OLS
    X_with_const = np.column_stack([np.ones(len(X)), X["similarity"], X["voi"]])
    try:
        betas, resid, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
        beta_similarity = betas[1]
        beta_voi = betas[2]

        # Compute t-stats and p-values
        n = len(y)
        k = 3  # intercept + 2 predictors
        mse = np.sum(resid) / (n - k) if len(resid) > 0 else np.var(y - X_with_const @ betas) * n / (n - k)
        var_betas = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
        se_similarity = np.sqrt(var_betas[1])
        se_voi = np.sqrt(var_betas[2])
        t_similarity = beta_similarity / se_similarity if se_similarity > 0 else 0
        t_voi = beta_voi / se_voi if se_voi > 0 else 0
        p_similarity = 2 * (1 - stats.t.cdf(abs(t_similarity), n - k))
        p_voi_reg = 2 * (1 - stats.t.cdf(abs(t_voi), n - k))

        print(f"\n  Regression: actual_shift ~ similarity + VOI")
        print(f"    beta(similarity) = {beta_similarity:+.4f} (p={p_similarity:.4e})")
        print(f"    beta(VOI)        = {beta_voi:+.4f} (p={p_voi_reg:.4e})")

        regression_results = {
            "beta_similarity": float(beta_similarity),
            "p_similarity": float(p_similarity),
            "beta_voi": float(beta_voi),
            "p_voi": float(p_voi_reg),
        }
    except Exception as e:
        print(f"\n  Regression failed: {e}")
        regression_results = {"error": str(e)}

    # Interpretation
    voi_adds_value = r_partial > 0.1 and p_partial < 0.05
    print(f"\n  VOI adds incremental value: {voi_adds_value}")

    return {
        "partial_r_voi": {"r": float(r_partial), "p": float(p_partial)},
        "regression": regression_results,
        "voi_adds_value": bool(voi_adds_value),
    }


def run_bin_analysis(df: pd.DataFrame) -> dict:
    """Test 3: At what similarity level does VOI stop adding value?"""
    print("\n" + "=" * 70)
    print("TEST 3: BIN ANALYSIS BY SIMILARITY LEVEL")
    print("=" * 70)

    # Define bins
    bins = [0.0, 0.3, 0.5, 0.7, 1.0]
    bin_labels = ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7+"]
    df["sim_bin"] = pd.cut(df["similarity"], bins=bins, labels=bin_labels, include_lowest=True)

    print(f"\n  {'Bin':<10} {'n':>5} {'r(VOI)':<12} {'r(sim)':<12} {'VOI adds?'}")
    print(f"  {'-'*10} {'-'*5} {'-'*12} {'-'*12} {'-'*10}")

    results_by_bin = {}

    for bin_label in bin_labels:
        bin_df = df[df["sim_bin"] == bin_label]
        n = len(bin_df)

        if n < 5:
            print(f"  {bin_label:<10} {n:>5} {'N/A':<12} {'N/A':<12} {'N/A'}")
            results_by_bin[bin_label] = {"n": int(n), "insufficient_data": True}
            continue

        r_voi, p_voi = stats.pearsonr(bin_df["linear_voi"], bin_df["actual_shift"])
        r_sim, p_sim = stats.pearsonr(bin_df["similarity"], bin_df["actual_shift"])

        # Check if VOI adds value in this bin
        voi_better = r_voi > r_sim + 0.05
        voi_adds = "YES" if voi_better else "NO"

        print(f"  {bin_label:<10} {n:>5} {r_voi:+.3f} (p={p_voi:.2f}) {r_sim:+.3f} (p={p_sim:.2f}) {voi_adds}")

        results_by_bin[bin_label] = {
            "n": int(n),
            "voi_r": float(r_voi),
            "voi_p": float(p_voi),
            "sim_r": float(r_sim),
            "sim_p": float(p_sim),
            "voi_adds_value": bool(voi_better),
        }

    # Find transition point
    print("\n  Hypothesis: VOI adds value at low similarity, not at high similarity")

    return {"by_bin": results_by_bin}


def run_conditional_update_test(df: pd.DataFrame) -> dict:
    """Test 4: Does similarity predict conditional update magnitude?

    If similarity correlates with actual_shift, it might also correlate
    with |P(A|B) - P(A)|, which would make it useful for conditional forecasting.
    """
    print("\n" + "=" * 70)
    print("TEST 4: CONDITIONAL UPDATE MAGNITUDE PREDICTION")
    print("=" * 70)

    # Approximate conditional update from rho
    # |P(A|B=1) - P(A)| approximately equals |rho| * sqrt(p_a * (1-p_a))
    # For simplicity, use |rho| as proxy for conditional update magnitude

    df["abs_rho"] = df["rho"].abs()

    r_sim_rho, p_sim_rho = stats.pearsonr(df["similarity"], df["abs_rho"])
    r_sim_shift, p_sim_shift = stats.pearsonr(df["similarity"], df["actual_shift"])

    print(f"\n  Does similarity predict |rho| (correlation strength)?")
    print(f"    r(similarity, |rho|): r={r_sim_rho:+.4f} (p={p_sim_rho:.4e})")

    print(f"\n  Does similarity predict actual_shift (realized update)?")
    print(f"    r(similarity, actual_shift): r={r_sim_shift:+.4f} (p={p_sim_shift:.4e})")

    # Compare to VOI
    r_voi_rho, _ = stats.pearsonr(df["linear_voi"], df["abs_rho"])
    r_voi_shift, _ = stats.pearsonr(df["linear_voi"], df["actual_shift"])

    print(f"\n  Comparison:")
    print(f"    For predicting |rho|:        sim r={r_sim_rho:+.3f}, VOI r={r_voi_rho:+.3f}")
    print(f"    For predicting actual_shift: sim r={r_sim_shift:+.3f}, VOI r={r_voi_shift:+.3f}")

    # Interpretation for conditional forecasting
    if r_sim_shift > 0.3:
        interp = "Similarity is a useful signal for conditional relationships"
    elif r_sim_shift > 0.15:
        interp = "Similarity shows moderate signal for conditional relationships"
    else:
        interp = "Similarity shows weak signal for conditional relationships"

    print(f"\n  Interpretation: {interp}")

    return {
        "similarity_vs_abs_rho": {"r": float(r_sim_rho), "p": float(p_sim_rho)},
        "similarity_vs_actual_shift": {"r": float(r_sim_shift), "p": float(p_sim_shift)},
        "voi_vs_abs_rho": {"r": float(r_voi_rho)},
        "voi_vs_actual_shift": {"r": float(r_voi_shift)},
        "interpretation": interp,
    }


def compute_efficiency_comparison() -> dict:
    """Test 3 (renamed): Computational cost comparison."""
    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST COMPARISON")
    print("=" * 70)

    print(f"\n  {'Method':<20} {'Cost per pair':<20} {'Notes'}")
    print(f"  {'-'*20} {'-'*20} {'-'*40}")
    print(f"  {'Similarity':<20} {'~0.001s':<20} {'Embedding lookup + dot product'}")
    print(f"  {'VOI (LLM rho)':<20} {'~2s':<20} {'LLM API call'}")

    cost_ratio = 2000
    print(f"\n  Similarity is ~{cost_ratio}x more efficient than LLM VOI")

    return {"similarity_cost_seconds": 0.001, "voi_cost_seconds": 2.0, "cost_ratio": cost_ratio}


def main():
    print("=" * 70)
    print("SIMILARITY VS VOI: IS SIMILARITY A SUFFICIENT PROXY?")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading validation data...")
    pairs = load_validation_data()
    print(f"      Loaded {len(pairs)} resolved pairs")

    # Get unique questions
    print("\n[2/6] Computing embeddings...")
    all_questions = set()
    for p in pairs:
        all_questions.add(p["resolved_question"])
        all_questions.add(p["other_question"])

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = compute_embeddings(list(all_questions), model)
    print(f"      Computed embeddings for {len(embeddings)} questions")

    # Compute similarity for each pair
    print("\n[3/6] Computing pairwise similarities...")
    for p in pairs:
        p["similarity"] = compute_similarity(
            p["resolved_question"], p["other_question"], embeddings
        )

    # Create DataFrame
    df = pd.DataFrame(pairs)
    print(f"      Mean similarity: {df['similarity'].mean():.3f}")
    print(f"      Std similarity: {df['similarity'].std():.3f}")

    # Run tests
    print("\n[4/6] Running tests...")

    test1_results = run_head_to_head(df)
    test2_results = run_incremental_value(df)
    test3_results = run_bin_analysis(df)
    test4_results = run_conditional_update_test(df)
    cost_results = compute_efficiency_comparison()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  n = {len(df)} resolved pairs")
    print(f"\n  Head-to-head:")
    print(f"    Similarity r = {test1_results['similarity_vs_shift']['r']:+.3f}")
    print(f"    VOI r        = {test1_results['voi_vs_shift']['r']:+.3f}")
    print(f"    Winner: {test1_results['winner']}")

    print(f"\n  Incremental value of VOI:")
    print(f"    Partial r(VOI | similarity) = {test2_results['partial_r_voi']['r']:+.3f}")
    print(f"    VOI adds significant value: {test2_results['voi_adds_value']}")

    print(f"\n  Cost comparison:")
    print(f"    Similarity is {cost_results['cost_ratio']}x more efficient")

    # Implications for conditional forecasting
    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR CONDITIONAL FORECASTING")
    print("=" * 70)

    sim_r = test1_results["similarity_vs_shift"]["r"]
    voi_r = test1_results["voi_vs_shift"]["r"]
    ratio = sim_r / voi_r if voi_r > 0 else float("inf")

    print(f"\n  Similarity achieves {ratio:.0%} of VOI's predictive power")

    if ratio >= 0.9:
        recommendation = "Replace VOI with similarity for most use cases"
    elif ratio >= 0.7:
        recommendation = "Use similarity as first-pass filter, VOI for edge cases"
    else:
        recommendation = "VOI provides substantial value beyond similarity"

    print(f"  Recommendation: {recommendation}")

    if test2_results["voi_adds_value"]:
        print(f"\n  Market-Logic Gap detection:")
        print(f"    VOI adds incremental value, suggesting LLM captures")
        print(f"    information beyond semantic relatedness.")
    else:
        print(f"\n  Market-Logic Gap detection:")
        print(f"    VOI does not add incremental value beyond similarity.")
        print(f"    Similarity may be sufficient for most pairs.")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "experiment": "similarity_vs_voi",
            "n_pairs": len(df),
            "embedding_model": EMBEDDING_MODEL,
            "run_at": datetime.now().isoformat(),
        },
        "test1_head_to_head": test1_results,
        "test2_incremental_value": test2_results,
        "test3_bin_analysis": test3_results,
        "test4_conditional_update": test4_results,
        "cost_comparison": cost_results,
        "summary": {
            "similarity_r": float(test1_results["similarity_vs_shift"]["r"]),
            "voi_r": float(test1_results["voi_vs_shift"]["r"]),
            "winner": test1_results["winner"],
            "voi_adds_incremental_value": bool(test2_results["voi_adds_value"]),
            "recommendation": recommendation,
            "similarity_pct_of_voi": float(ratio) if ratio != float("inf") else None,
        },
        "pairs": [{
            "resolved_question": p["resolved_question"][:80],
            "other_question": p["other_question"][:80],
            "similarity": p["similarity"],
            "rho": p["rho"],
            "linear_voi": p["linear_voi"],
            "actual_shift": p["actual_shift"],
        } for p in pairs],
    }

    output_path = OUTPUT_DIR / "similarity_vs_voi_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
