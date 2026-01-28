#!/usr/bin/env python3
"""
DML Clustering Experiment - Phase 1: Market Clustering

Build correlation matrix from pairs.json and apply hierarchical clustering
to discover market structure.

Usage:
    python cluster_markets.py --k 10 --output results/clusters_k10.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


def load_pairs(pairs_path: str) -> list[dict]:
    """Load market pairs with correlations."""
    with open(pairs_path) as f:
        return json.load(f)


def load_markets(markets_path: str) -> dict[str, dict]:
    """Load market metadata for labeling."""
    with open(markets_path) as f:
        markets = json.load(f)
    return {m["condition_id"]: m for m in markets}


def build_correlation_matrix(pairs: list[dict]) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """
    Build NxN correlation matrix from pairs.

    Returns:
        corr_matrix: NxN correlation matrix
        market_ids: list of market IDs (row/column order)
        id_to_question: mapping from market ID to question text
    """
    # Collect all unique market IDs and questions
    market_ids = set()
    id_to_question = {}
    for pair in pairs:
        # Handle both nested dict format and flat format
        if isinstance(pair["market_a"], dict):
            mid_a = pair["market_a"]["condition_id"]
            mid_b = pair["market_b"]["condition_id"]
            id_to_question[mid_a] = pair["market_a"].get("question", "")
            id_to_question[mid_b] = pair["market_b"].get("question", "")
        else:
            mid_a = pair["market_a"]
            mid_b = pair["market_b"]

        market_ids.add(mid_a)
        market_ids.add(mid_b)

    market_ids = sorted(list(market_ids))

    # Create ID -> index mapping
    id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
    n = len(market_ids)

    # Initialize with identity (self-correlation = 1)
    corr_matrix = np.eye(n)

    # Fill in correlations
    for pair in pairs:
        if isinstance(pair["market_a"], dict):
            mid_a = pair["market_a"]["condition_id"]
            mid_b = pair["market_b"]["condition_id"]
        else:
            mid_a = pair["market_a"]
            mid_b = pair["market_b"]

        i = id_to_idx[mid_a]
        j = id_to_idx[mid_b]
        rho = pair["rho"]
        if rho is not None and not np.isnan(rho):
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho  # symmetric

    return corr_matrix, market_ids, id_to_question


def correlation_to_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.

    Uses D = 1 - |ρ| so highly correlated (positive or negative)
    markets are close together.
    """
    return 1 - np.abs(corr_matrix)


def cluster_hierarchical(
    corr_matrix: np.ndarray,
    k: int,
    method: str = "ward"
) -> np.ndarray:
    """
    Apply hierarchical clustering and cut at k clusters.

    Args:
        corr_matrix: NxN correlation matrix
        k: number of clusters
        method: linkage method (ward, complete, average, single)

    Returns:
        cluster_labels: array of cluster assignments (1 to k)
    """
    # Convert to distance matrix
    dist_matrix = correlation_to_distance(corr_matrix)

    # Ensure diagonal is 0 (distance to self)
    np.fill_diagonal(dist_matrix, 0)

    # Convert to condensed form for scipy
    condensed = squareform(dist_matrix, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed, method=method)

    # Cut at k clusters
    labels = fcluster(Z, k, criterion="maxclust")

    return labels, Z


def get_cluster_members(
    labels: np.ndarray,
    market_ids: list[str],
    id_to_question: dict[str, str]
) -> dict[int, list[dict]]:
    """
    Get market details for each cluster.

    Returns:
        clusters: {cluster_id: [{market_id, question}, ...]}
    """
    clusters = defaultdict(list)

    for i, mid in enumerate(market_ids):
        cluster_id = int(labels[i])
        clusters[cluster_id].append({
            "market_id": mid,
            "question": id_to_question.get(mid, "Unknown"),
        })

    return dict(clusters)


def compute_cluster_centrality(
    corr_matrix: np.ndarray,
    labels: np.ndarray,
    market_ids: list[str]
) -> dict[int, list[tuple[str, float]]]:
    """
    Compute centrality (mean correlation to cluster members) for each market.

    Returns markets ranked by centrality within each cluster.
    """
    cluster_centrality = defaultdict(list)
    n = len(market_ids)

    for i, mid in enumerate(market_ids):
        cluster_id = int(labels[i])

        # Get indices of cluster members
        cluster_mask = labels == cluster_id

        # Mean correlation to other cluster members
        cluster_corrs = corr_matrix[i, cluster_mask]
        centrality = np.mean(np.abs(cluster_corrs))

        cluster_centrality[cluster_id].append((mid, centrality))

    # Sort by centrality (descending)
    for cluster_id in cluster_centrality:
        cluster_centrality[cluster_id].sort(key=lambda x: x[1], reverse=True)

    return dict(cluster_centrality)


def main():
    parser = argparse.ArgumentParser(description="Cluster markets by correlation")
    parser.add_argument("--pairs", default="../conditional-forecasting/data/pairs.json",
                        help="Path to pairs.json")
    parser.add_argument("--markets", default="../conditional-forecasting/data/markets.json",
                        help="Path to markets.json")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--method", default="ward",
                        choices=["ward", "complete", "average", "single"],
                        help="Linkage method")
    parser.add_argument("--output", default="results/clusters.json",
                        help="Output path for cluster assignments")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    pairs_path = script_dir / args.pairs
    markets_path = script_dir / args.markets
    output_path = script_dir / args.output

    print(f"Loading pairs from {pairs_path}...")
    pairs = load_pairs(pairs_path)
    print(f"  Loaded {len(pairs)} pairs")

    print(f"Loading markets from {markets_path}...")
    markets_metadata = load_markets(markets_path)
    print(f"  Loaded {len(markets_metadata)} markets")

    print("Building correlation matrix...")
    corr_matrix, market_ids, id_to_question = build_correlation_matrix(pairs)
    print(f"  Matrix size: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")

    # Merge question info from pairs and markets metadata
    for mid in market_ids:
        if mid in markets_metadata and "question" in markets_metadata[mid]:
            id_to_question[mid] = markets_metadata[mid]["question"]

    # Count non-zero correlations
    n_nonzero = np.sum(corr_matrix != 0) - len(market_ids)  # exclude diagonal
    n_possible = len(market_ids) * (len(market_ids) - 1)
    coverage = n_nonzero / n_possible * 100
    print(f"  Correlation coverage: {coverage:.1f}%")

    print(f"Clustering with k={args.k}, method={args.method}...")
    labels, linkage_matrix = cluster_hierarchical(corr_matrix, args.k, args.method)

    # Get cluster details
    clusters = get_cluster_members(labels, market_ids, id_to_question)
    centrality = compute_cluster_centrality(corr_matrix, labels, market_ids)

    # Print summary
    print(f"\nCluster summary (k={args.k}):")
    print("-" * 60)
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        top_markets = centrality[cluster_id][:5]
        print(f"\nCluster {cluster_id}: {len(members)} markets")
        print("  Top 5 by centrality:")
        for mid, cent in top_markets:
            question = id_to_question.get(mid, "Unknown")[:60]
            print(f"    [{cent:.3f}] {question}...")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "k": args.k,
        "method": args.method,
        "n_markets": len(market_ids),
        "n_pairs": len(pairs),
        "coverage_pct": coverage,
        "clusters": {
            str(cluster_id): {
                "n_members": len(members),
                "members": members,
                "top_by_centrality": [
                    {"market_id": mid, "centrality": cent,
                     "question": id_to_question.get(mid, "")}
                    for mid, cent in centrality[cluster_id][:10]
                ]
            }
            for cluster_id, members in clusters.items()
        },
        "market_ids": market_ids,
        "labels": labels.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
