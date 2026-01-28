#!/usr/bin/env python3
"""
MMR (Maximal Marginal Relevance) Selection for Question Generation.

Tests whether combining VOI with diversity constraints improves question
coverage compared to naive top-K by VOI.

Key insight: VOI is a category detector, not a within-category ranker.
So use VOI for category selection, add diversity to avoid redundancy.

MMR formula:
    score(q) = λ * VOI(q) - (1-λ) * max_similarity(q, selected)
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# Paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark-mvp"
OUTPUT_DIR = Path(__file__).parent / "data"


def load_benchmark_cruxes() -> list[dict]:
    """Load cruxes from benchmark results."""
    results_path = BENCHMARK_DIR / "results" / "benchmark_results.json"
    with open(results_path) as f:
        data = json.load(f)

    # Flatten all cruxes with their ultimates
    all_cruxes = []
    for result in data["results"]:
        if "error" in result:
            continue
        ultimate = result["ultimate"]
        for cs in result["crux_scores"]:
            all_cruxes.append({
                "ultimate": ultimate,
                "crux": cs["crux"],
                "voi": cs["voi_linear"],
                "rho": cs["rho_estimated"],
            })

    return all_cruxes


def compute_similarity_matrix(texts: list[str]) -> np.ndarray:
    """Compute pairwise cosine similarity using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


def select_top_k(cruxes: list[dict], k: int) -> list[dict]:
    """Naive selection: top-K by VOI."""
    sorted_cruxes = sorted(cruxes, key=lambda x: -x["voi"])
    return sorted_cruxes[:k]


def select_mmr(cruxes: list[dict], k: int, sim_matrix: np.ndarray,
               lambda_param: float = 0.7) -> list[dict]:
    """MMR selection: balance VOI with diversity."""
    n = len(cruxes)
    selected_indices = []
    remaining_indices = list(range(n))

    # Normalize VOI to [0, 1] for fair comparison with similarity
    vois = np.array([c["voi"] for c in cruxes])
    voi_min, voi_max = vois.min(), vois.max()
    if voi_max > voi_min:
        vois_norm = (vois - voi_min) / (voi_max - voi_min)
    else:
        vois_norm = np.ones(n) * 0.5

    while len(selected_indices) < k and remaining_indices:
        best_score = -float("inf")
        best_idx = None

        for idx in remaining_indices:
            relevance = vois_norm[idx]

            if selected_indices:
                # Max similarity to any already-selected crux
                redundancy = max(sim_matrix[idx, s] for s in selected_indices)
            else:
                redundancy = 0

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [cruxes[i] for i in selected_indices]


def compute_coverage_metrics(selected: list[dict], all_cruxes: list[dict],
                             sim_matrix: np.ndarray) -> dict:
    """Compute coverage and diversity metrics for a selection."""
    if not selected:
        return {}

    selected_indices = [all_cruxes.index(s) for s in selected]

    # Average pairwise similarity within selection (lower = more diverse)
    if len(selected_indices) > 1:
        pairwise_sims = []
        for i, idx1 in enumerate(selected_indices):
            for idx2 in selected_indices[i+1:]:
                pairwise_sims.append(sim_matrix[idx1, idx2])
        avg_similarity = np.mean(pairwise_sims)
    else:
        avg_similarity = 0

    # Number of unique ultimates covered
    ultimates_covered = len(set(s["ultimate"] for s in selected))

    # VOI statistics
    vois = [s["voi"] for s in selected]

    # Category diversity (using simple keyword heuristics)
    categories = defaultdict(int)
    for s in selected:
        text = s["crux"].lower()
        if any(w in text for w in ["fed", "rate", "interest", "fomc"]):
            categories["fed"] += 1
        elif any(w in text for w in ["trump", "biden", "election", "president"]):
            categories["politics"] += 1
        elif any(w in text for w in ["bitcoin", "crypto", "ethereum"]):
            categories["crypto"] += 1
        elif any(w in text for w in ["war", "russia", "ukraine", "iran"]):
            categories["geopolitics"] += 1
        else:
            categories["other"] += 1

    return {
        "n_selected": len(selected),
        "avg_similarity": float(avg_similarity),
        "diversity": 1 - float(avg_similarity),  # Higher = more diverse
        "ultimates_covered": ultimates_covered,
        "voi_mean": float(np.mean(vois)),
        "voi_sum": float(np.sum(vois)),
        "voi_min": float(np.min(vois)),
        "category_counts": dict(categories),
        "n_categories": len([c for c in categories.values() if c > 0]),
    }


def main():
    print("=" * 70)
    print("MMR SELECTION VS NAIVE TOP-K")
    print("=" * 70)

    # Load cruxes
    print("\n[1/4] Loading benchmark cruxes...")
    all_cruxes = load_benchmark_cruxes()
    print(f"      Loaded {len(all_cruxes)} cruxes")

    # Compute similarity matrix
    print("\n[2/4] Computing similarity matrix...")
    texts = [c["crux"] for c in all_cruxes]
    sim_matrix = compute_similarity_matrix(texts)
    print(f"      Matrix shape: {sim_matrix.shape}")

    # Test different K values and lambda values
    print("\n[3/4] Comparing selection methods...")

    k_values = [5, 10, 20]
    lambda_values = [0.5, 0.7, 0.9, 1.0]  # 1.0 = pure VOI (no diversity)

    results = []

    for k in k_values:
        print(f"\n  K = {k}:")

        # Naive top-K
        naive = select_top_k(all_cruxes, k)
        naive_metrics = compute_coverage_metrics(naive, all_cruxes, sim_matrix)
        naive_metrics["method"] = "naive_top_k"
        naive_metrics["k"] = k
        naive_metrics["lambda"] = None
        results.append(naive_metrics)

        print(f"    Naive Top-K:     diversity={naive_metrics['diversity']:.3f}, "
              f"categories={naive_metrics['n_categories']}, "
              f"voi_sum={naive_metrics['voi_sum']:.3f}")

        # MMR with different lambda values
        for lam in lambda_values:
            if lam == 1.0:
                continue  # Same as naive
            mmr = select_mmr(all_cruxes, k, sim_matrix, lambda_param=lam)
            mmr_metrics = compute_coverage_metrics(mmr, all_cruxes, sim_matrix)
            mmr_metrics["method"] = "mmr"
            mmr_metrics["k"] = k
            mmr_metrics["lambda"] = lam
            results.append(mmr_metrics)

            print(f"    MMR (λ={lam}):      diversity={mmr_metrics['diversity']:.3f}, "
                  f"categories={mmr_metrics['n_categories']}, "
                  f"voi_sum={mmr_metrics['voi_sum']:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Best lambda for each K
    print("\n  Best λ by diversity (for each K):")
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        best = max(k_results, key=lambda x: x["diversity"])
        print(f"    K={k}: λ={best['lambda']} → diversity={best['diversity']:.3f}")

    # Trade-off analysis
    print("\n  VOI vs Diversity Trade-off (K=10):")
    k10_results = [r for r in results if r["k"] == 10]
    for r in sorted(k10_results, key=lambda x: x.get("lambda") or 999):
        method = f"λ={r['lambda']}" if r["lambda"] else "naive"
        print(f"    {method:8}: voi_sum={r['voi_sum']:.2f}, diversity={r['diversity']:.3f}, "
              f"categories={r['n_categories']}")

    # Show example selections
    print("\n" + "=" * 70)
    print("EXAMPLE SELECTIONS (K=5)")
    print("=" * 70)

    print("\n  NAIVE TOP-5 BY VOI:")
    naive_5 = select_top_k(all_cruxes, 5)
    for i, c in enumerate(naive_5):
        print(f"    {i+1}. VOI={c['voi']:.3f}: {c['crux'][:60]}...")

    print("\n  MMR TOP-5 (λ=0.7):")
    mmr_5 = select_mmr(all_cruxes, 5, sim_matrix, lambda_param=0.7)
    for i, c in enumerate(mmr_5):
        print(f"    {i+1}. VOI={c['voi']:.3f}: {c['crux'][:60]}...")

    # Compute improvement
    naive_metrics = compute_coverage_metrics(naive_5, all_cruxes, sim_matrix)
    mmr_metrics = compute_coverage_metrics(mmr_5, all_cruxes, sim_matrix)

    print("\n  COMPARISON (K=5):")
    print(f"    Diversity:  Naive={naive_metrics['diversity']:.3f} → MMR={mmr_metrics['diversity']:.3f} "
          f"(+{(mmr_metrics['diversity']-naive_metrics['diversity'])*100:.0f}%)")
    print(f"    Categories: Naive={naive_metrics['n_categories']} → MMR={mmr_metrics['n_categories']}")
    print(f"    VOI sum:    Naive={naive_metrics['voi_sum']:.3f} → MMR={mmr_metrics['voi_sum']:.3f} "
          f"({(mmr_metrics['voi_sum']/naive_metrics['voi_sum']-1)*100:+.0f}%)")

    # Save results
    print("\n[4/4] Saving results...")
    output = {
        "metadata": {
            "experiment": "mmr_selection",
            "n_cruxes": len(all_cruxes),
            "k_values": k_values,
            "lambda_values": lambda_values,
            "run_at": datetime.now().isoformat(),
        },
        "results": results,
        "examples": {
            "naive_top_5": [{"crux": c["crux"], "voi": c["voi"]} for c in naive_5],
            "mmr_top_5": [{"crux": c["crux"], "voi": c["voi"]} for c in mmr_5],
        },
        "summary": {
            "naive_k5_diversity": naive_metrics["diversity"],
            "mmr_k5_diversity": mmr_metrics["diversity"],
            "diversity_improvement": mmr_metrics["diversity"] - naive_metrics["diversity"],
            "voi_retention": mmr_metrics["voi_sum"] / naive_metrics["voi_sum"],
        }
    }

    output_path = OUTPUT_DIR / "mmr_selection_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"      Saved to {output_path}")


if __name__ == "__main__":
    main()
