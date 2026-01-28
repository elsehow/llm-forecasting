#!/usr/bin/env python3
"""
Complementarity Scoring Experiment.

Tests whether complementarity scoring—measuring what a crux *adds* beyond
already-selected cruxes—recovers within-domain ranking signal where raw VOI fails.

Formulation:
    Complementarity(q | S, T) = VOI(q, T) - lambda * max_{s in S} sim(q, s)

Datasets:
- Russell 2000 earnings: 80 events, 5 cruxes each, ground truth = |return|
- Polymarket pairs: 168 resolved pairs, ground truth = actual co-movement
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

# Paths
RUSSELL_DATA_DIR = Path(__file__).parent.parent.parent / "russell-2000-crux" / "data"
POLYMARKET_DATA_DIR = Path(__file__).parent.parent / "within-category" / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Model for semantic similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedder():
    """Load sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def compute_embedding_similarity_matrix(texts: list[str], embedder) -> np.ndarray:
    """Compute pairwise cosine similarity using embeddings."""
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    # Cosine similarity matrix
    return embeddings_norm @ embeddings_norm.T


def complementarity_score(
    idx: int,
    selected_indices: list[int],
    vois_norm: np.ndarray,
    sim_matrix: np.ndarray,
    lambda_param: float
) -> float:
    """
    Compute complementarity score for a candidate crux.

    Complementarity(q | S) = VOI(q) - lambda * max_{s in S} sim(q, s)

    This differs from MMR:
    - MMR: lambda * relevance - (1-lambda) * redundancy
    - Complementarity: relevance - lambda * redundancy

    The key difference: complementarity uses raw subtraction, treating
    similarity as a direct penalty rather than a weighted trade-off.
    """
    relevance = vois_norm[idx]

    if not selected_indices:
        redundancy = 0
    else:
        # Max similarity to any already-selected crux
        redundancy = max(sim_matrix[idx, s] for s in selected_indices)

    return relevance - lambda_param * redundancy


def select_naive_topk(vois: np.ndarray, k: int) -> list[int]:
    """Naive selection: top-K by VOI."""
    return list(np.argsort(vois)[::-1][:k])


def select_mmr(
    vois: np.ndarray,
    sim_matrix: np.ndarray,
    k: int,
    lambda_param: float = 0.7
) -> list[int]:
    """MMR selection: balance VOI with diversity."""
    n = len(vois)

    # Normalize VOI to [0, 1]
    voi_min, voi_max = vois.min(), vois.max()
    if voi_max > voi_min:
        vois_norm = (vois - voi_min) / (voi_max - voi_min)
    else:
        vois_norm = np.ones(n) * 0.5

    selected = []
    remaining = list(range(n))

    while len(selected) < k and remaining:
        best_score = -float("inf")
        best_idx = None

        for idx in remaining:
            relevance = vois_norm[idx]
            if selected:
                redundancy = max(sim_matrix[idx, s] for s in selected)
            else:
                redundancy = 0

            # MMR formula: lambda * relevance - (1-lambda) * redundancy
            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def select_complementarity(
    vois: np.ndarray,
    sim_matrix: np.ndarray,
    k: int,
    lambda_param: float = 0.5
) -> list[int]:
    """Complementarity selection: penalize similarity to selected set."""
    n = len(vois)

    # Normalize VOI to [0, 1]
    voi_min, voi_max = vois.min(), vois.max()
    if voi_max > voi_min:
        vois_norm = (vois - voi_min) / (voi_max - voi_min)
    else:
        vois_norm = np.ones(n) * 0.5

    selected = []
    remaining = list(range(n))

    while len(selected) < k and remaining:
        best_score = -float("inf")
        best_idx = None

        for idx in remaining:
            score = complementarity_score(idx, selected, vois_norm, sim_matrix, lambda_param)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def compute_diversity(selected_indices: list[int], sim_matrix: np.ndarray) -> float:
    """Compute diversity (1 - avg pairwise similarity) for a selection."""
    if len(selected_indices) < 2:
        return 1.0

    pairwise_sims = []
    for i, idx1 in enumerate(selected_indices):
        for idx2 in selected_indices[i+1:]:
            pairwise_sims.append(sim_matrix[idx1, idx2])

    return 1 - np.mean(pairwise_sims)


def run_russell_experiment(embedder) -> dict:
    """
    Run complementarity experiment on Russell 2000 earnings data.

    Ground truth: |return| on earnings day
    Test: Does complementarity-based selection correlate with returns?
    """
    print("\n" + "=" * 70)
    print("RUSSELL 2000 EARNINGS EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    cruxes_df = pd.read_parquet(RUSSELL_DATA_DIR / "cruxes_with_voi.parquet")
    returns_df = pd.read_parquet(RUSSELL_DATA_DIR / "stock_returns.parquet")

    # Filter to earnings days
    returns_earnings = returns_df[returns_df["is_earnings_day"]].copy()
    returns_earnings["abs_return"] = returns_earnings["return"].abs()

    print(f"      Cruxes: {len(cruxes_df)}")
    print(f"      Earnings days: {len(returns_earnings)}")

    # Group by stock-date
    print("\n[2/4] Computing embeddings and selection scores...")

    results_by_method = {"naive": [], "mmr_0.7": [], "complementarity_0.5": []}

    groups = list(cruxes_df.groupby(["ticker", "date"]))

    for i, ((ticker, date), group) in enumerate(groups):
        if len(group) < 3:
            continue

        # Get return for this stock-date
        ret_row = returns_earnings[
            (returns_earnings["ticker"] == ticker) &
            (returns_earnings["date"] == date)
        ]
        if len(ret_row) == 0:
            continue
        abs_return = ret_row["abs_return"].values[0]

        # Get cruxes and VOIs
        texts = group["crux"].tolist()
        vois = group["linear_voi"].values

        # Compute similarity matrix
        sim_matrix = compute_embedding_similarity_matrix(texts, embedder)

        k = min(3, len(group))

        # Naive selection
        naive_selected = select_naive_topk(vois, k)
        naive_voi = np.mean(vois[naive_selected])
        naive_div = compute_diversity(naive_selected, sim_matrix)
        results_by_method["naive"].append({
            "ticker": ticker, "date": str(date),
            "voi_mean": naive_voi, "diversity": naive_div,
            "abs_return": abs_return
        })

        # MMR selection
        mmr_selected = select_mmr(vois, sim_matrix, k, lambda_param=0.7)
        mmr_voi = np.mean(vois[mmr_selected])
        mmr_div = compute_diversity(mmr_selected, sim_matrix)
        results_by_method["mmr_0.7"].append({
            "ticker": ticker, "date": str(date),
            "voi_mean": mmr_voi, "diversity": mmr_div,
            "abs_return": abs_return
        })

        # Complementarity selection
        comp_selected = select_complementarity(vois, sim_matrix, k, lambda_param=0.5)
        comp_voi = np.mean(vois[comp_selected])
        comp_div = compute_diversity(comp_selected, sim_matrix)
        results_by_method["complementarity_0.5"].append({
            "ticker": ticker, "date": str(date),
            "voi_mean": comp_voi, "diversity": comp_div,
            "abs_return": abs_return
        })

    # Compute correlations
    print("\n[3/4] Computing correlations...")

    summary = {}
    for method, results in results_by_method.items():
        if len(results) < 5:
            continue

        df = pd.DataFrame(results)
        r, p = stats.pearsonr(df["voi_mean"], df["abs_return"])

        summary[method] = {
            "n": len(df),
            "r_voi_return": float(r),
            "p_voi_return": float(p),
            "mean_diversity": float(df["diversity"].mean()),
            "mean_voi": float(df["voi_mean"].mean()),
        }

        print(f"  {method:20}: r={r:+.3f} (p={p:.3f}), diversity={df['diversity'].mean():.3f}")

    # Analysis
    print("\n[4/4] Analysis...")

    if "naive" in summary and "complementarity_0.5" in summary:
        naive_r = summary["naive"]["r_voi_return"]
        comp_r = summary["complementarity_0.5"]["r_voi_return"]
        delta_r = comp_r - naive_r

        print(f"\n  Naive r:           {naive_r:+.3f}")
        print(f"  Complementarity r: {comp_r:+.3f}")
        print(f"  Delta:             {delta_r:+.3f}")

        if comp_r > 0 and naive_r < 0:
            verdict = "SUCCESS: Complementarity reverses negative correlation"
        elif comp_r > naive_r:
            verdict = "PARTIAL: Complementarity improves but still weak"
        else:
            verdict = "FAIL: Complementarity doesn't help"

        print(f"\n  Verdict: {verdict}")
        summary["verdict"] = verdict

    return {"russell": summary, "russell_raw": results_by_method}


def run_polymarket_experiment(embedder) -> dict:
    """
    Run complementarity experiment on Polymarket within-category pairs.

    Ground truth: actual co-movement (did high-VOI pairs actually move together?)
    Test: Does complementarity-based ranking predict co-movement better than VOI?
    """
    print("\n" + "=" * 70)
    print("POLYMARKET WITHIN-CATEGORY EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    validation_path = POLYMARKET_DATA_DIR / "within_vs_cross_validation.json"

    with open(validation_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    within_pairs = [p for p in pairs if p["is_within_category"]]

    print(f"      Total pairs: {len(pairs)}")
    print(f"      Within-category pairs: {len(within_pairs)}")

    if len(within_pairs) < 5:
        print("      Not enough within-category pairs for meaningful analysis")
        return {"polymarket": {"error": "insufficient_data", "n_within": len(within_pairs)}}

    # Extract questions and compute embeddings
    print("\n[2/4] Computing embeddings...")

    # Get unique questions for embedding
    all_questions = set()
    for p in within_pairs:
        all_questions.add(p["resolved_question"])
        all_questions.add(p["other_question"])

    question_list = list(all_questions)
    question_to_idx = {q: i for i, q in enumerate(question_list)}

    sim_matrix = compute_embedding_similarity_matrix(question_list, embedder)

    # For each pair, compute complementarity score
    print("\n[3/4] Computing scores...")

    results = []
    for p in within_pairs:
        q1_idx = question_to_idx[p["resolved_question"]]
        q2_idx = question_to_idx[p["other_question"]]

        # Semantic similarity between the pair
        semantic_sim = sim_matrix[q1_idx, q2_idx]

        # VOI for this pair
        voi = p["linear_voi"]

        # Complementarity: VOI penalized by similarity
        # (treating the resolved question as "already selected")
        complementarity = voi - 0.5 * semantic_sim

        # Ground truth: actual shift
        actual_shift = abs(p["actual_shift"])

        results.append({
            "q1": p["resolved_question"][:50],
            "q2": p["other_question"][:50],
            "voi": voi,
            "semantic_sim": semantic_sim,
            "complementarity": complementarity,
            "actual_shift": actual_shift,
            "category": p["category_pair"],
        })

    df = pd.DataFrame(results)

    # Correlations
    print("\n[4/4] Computing correlations...")

    r_voi, p_voi = stats.pearsonr(df["voi"], df["actual_shift"])
    r_comp, p_comp = stats.pearsonr(df["complementarity"], df["actual_shift"])
    r_sim, p_sim = stats.pearsonr(df["semantic_sim"], df["actual_shift"])

    summary = {
        "n": len(df),
        "voi_vs_shift": {"r": float(r_voi), "p": float(p_voi)},
        "complementarity_vs_shift": {"r": float(r_comp), "p": float(p_comp)},
        "similarity_vs_shift": {"r": float(r_sim), "p": float(p_sim)},
        "mean_semantic_sim": float(df["semantic_sim"].mean()),
    }

    print(f"  VOI vs actual_shift:           r={r_voi:+.3f} (p={p_voi:.3f})")
    print(f"  Complementarity vs shift:      r={r_comp:+.3f} (p={p_comp:.3f})")
    print(f"  Semantic similarity vs shift:  r={r_sim:+.3f} (p={p_sim:.3f})")

    # Verdict
    if r_comp > r_voi and r_comp > 0.1:
        verdict = "SUCCESS: Complementarity improves over raw VOI"
    elif r_comp > r_voi:
        verdict = "PARTIAL: Slight improvement but still weak"
    else:
        verdict = "FAIL: Complementarity doesn't help"

    print(f"\n  Verdict: {verdict}")
    summary["verdict"] = verdict

    return {"polymarket": summary, "polymarket_raw": results}


def main():
    print("=" * 70)
    print("COMPLEMENTARITY SCORING EXPERIMENT")
    print("=" * 70)
    print("\nHypothesis: Within-domain VOI fails because cruxes explain similar")
    print("variance. Complementarity scoring may recover ranking signal.")

    # Load embedder
    print("\n[0/3] Loading embedding model...")
    embedder = get_embedder()

    # Run Russell experiment
    russell_results = run_russell_experiment(embedder)

    # Run Polymarket experiment
    polymarket_results = run_polymarket_experiment(embedder)

    # Combine results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  RUSSELL 2000:")
    if "russell" in russell_results:
        for method, stats in russell_results["russell"].items():
            if isinstance(stats, dict) and "r_voi_return" in stats:
                print(f"    {method}: r={stats['r_voi_return']:+.3f}, div={stats['mean_diversity']:.3f}")

    print("\n  POLYMARKET:")
    if "polymarket" in polymarket_results:
        pm = polymarket_results["polymarket"]
        if "error" not in pm:
            print(f"    VOI:           r={pm['voi_vs_shift']['r']:+.3f}")
            print(f"    Complementarity: r={pm['complementarity_vs_shift']['r']:+.3f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "experiment": "complementarity_scoring",
            "embedding_model": EMBEDDING_MODEL,
            "run_at": datetime.now().isoformat(),
        },
        "russell": russell_results.get("russell", {}),
        "polymarket": polymarket_results.get("polymarket", {}),
    }

    output_path = OUTPUT_DIR / "complementarity_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
