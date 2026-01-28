#!/usr/bin/env python3
"""
DML Clustering Experiment - Visualization

Generate dendrogram and correlation heatmap from clustering results.

Usage:
    python visualize_clusters.py --input results/clusters_k10.json --output results/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def load_pairs(pairs_path: str) -> list[dict]:
    """Load market pairs with correlations."""
    with open(pairs_path) as f:
        return json.load(f)


def load_clusters(clusters_path: str) -> dict:
    """Load clustering results."""
    with open(clusters_path) as f:
        return json.load(f)


def load_markets(markets_path: str) -> dict[str, dict]:
    """Load market metadata."""
    with open(markets_path) as f:
        markets = json.load(f)
    return {m["condition_id"]: m for m in markets}


def build_correlation_matrix(pairs: list[dict], market_ids: list[str]) -> np.ndarray:
    """Build correlation matrix in the order of market_ids."""
    id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
    n = len(market_ids)
    corr_matrix = np.eye(n)

    for pair in pairs:
        # Handle both nested dict format and flat format
        if isinstance(pair["market_a"], dict):
            mid_a = pair["market_a"]["condition_id"]
            mid_b = pair["market_b"]["condition_id"]
        else:
            mid_a = pair["market_a"]
            mid_b = pair["market_b"]

        if mid_a in id_to_idx and mid_b in id_to_idx:
            i = id_to_idx[mid_a]
            j = id_to_idx[mid_b]
            rho = pair["rho"]
            if rho is not None and not np.isnan(rho):
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    return corr_matrix


def plot_dendrogram(
    corr_matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    method: str = "ward",
    truncate_mode: str = "lastp",
    p: int = 30
):
    """Plot hierarchical clustering dendrogram."""
    # Convert to distance
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)

    # Linkage
    Z = linkage(condensed, method=method)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    ax.set_title(f"Market Clustering Dendrogram ({method} linkage)")
    ax.set_xlabel("Market (truncated)")
    ax.set_ylabel("Distance (1 - |ρ|)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved dendrogram to {output_path}")


def plot_heatmap(
    corr_matrix: np.ndarray,
    cluster_labels: list[int],
    output_path: Path,
    k: int
):
    """Plot correlation heatmap sorted by cluster."""
    # Sort by cluster assignment
    sorted_idx = np.argsort(cluster_labels)
    sorted_matrix = corr_matrix[sorted_idx][:, sorted_idx]
    sorted_labels = [cluster_labels[i] for i in sorted_idx]

    # Find cluster boundaries
    boundaries = [0]
    for i in range(1, len(sorted_labels)):
        if sorted_labels[i] != sorted_labels[i-1]:
            boundaries.append(i)
    boundaries.append(len(sorted_labels))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sorted_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Add cluster boundary lines
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.5)
        ax.axvline(b - 0.5, color="black", linewidth=0.5)

    # Add cluster labels on margins
    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i+1]) / 2
        cluster_id = sorted_labels[boundaries[i]]
        ax.text(-10, mid, f"C{cluster_id}", va="center", ha="right", fontsize=8)
        ax.text(mid, -10, f"C{cluster_id}", va="bottom", ha="center", fontsize=8, rotation=90)

    ax.set_title(f"Correlation Matrix (sorted by {k} clusters)")
    ax.set_xlabel("Markets")
    ax.set_ylabel("Markets")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation (ρ)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {output_path}")


def plot_cluster_sizes(clusters: dict, output_path: Path, k: int):
    """Plot cluster size distribution."""
    cluster_ids = sorted([int(k) for k in clusters.keys()])
    sizes = [clusters[str(c)]["n_members"] for c in cluster_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_ids, sizes, color="steelblue", edgecolor="black")

    # Add count labels on bars
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(size), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Markets")
    ax.set_title(f"Cluster Size Distribution (k={k})")
    ax.set_xticks(cluster_ids)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved cluster sizes to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize market clusters")
    parser.add_argument("--input", required=True, help="Path to clusters JSON")
    parser.add_argument("--pairs", default="../conditional-forecasting/data/pairs.json",
                        help="Path to pairs.json")
    parser.add_argument("--markets", default="../conditional-forecasting/data/markets.json",
                        help="Path to markets.json")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    clusters_path = script_dir / args.input
    pairs_path = script_dir / args.pairs
    markets_path = script_dir / args.markets
    output_dir = script_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clusters from {clusters_path}...")
    clusters_data = load_clusters(clusters_path)
    k = clusters_data["k"]
    method = clusters_data["method"]
    market_ids = clusters_data["market_ids"]
    labels = clusters_data["labels"]

    print(f"Loading pairs from {pairs_path}...")
    pairs = load_pairs(pairs_path)

    print("Building correlation matrix...")
    corr_matrix = build_correlation_matrix(pairs, market_ids)

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_dendrogram(
        corr_matrix, market_ids,
        output_dir / f"dendrogram_k{k}.png",
        method=method
    )

    plot_heatmap(
        corr_matrix, labels,
        output_dir / f"heatmap_k{k}.png",
        k=k
    )

    plot_cluster_sizes(
        clusters_data["clusters"],
        output_dir / f"cluster_sizes_k{k}.png",
        k=k
    )

    # Print top markets per cluster for labeling
    print(f"\n{'='*70}")
    print("CLUSTER LABELING WORKSHEET")
    print(f"{'='*70}")
    print("\nFor each cluster, review the top markets and assign a label.")
    print("Success criterion: 3+ clusters labelable, >70% of markets fit label.\n")

    for cluster_id in sorted(clusters_data["clusters"].keys(), key=int):
        cluster = clusters_data["clusters"][cluster_id]
        print(f"\n--- Cluster {cluster_id} ({cluster['n_members']} markets) ---")
        print("Top 10 markets by centrality:")
        for i, m in enumerate(cluster["top_by_centrality"][:10], 1):
            q = m["question"][:70] + "..." if len(m["question"]) > 70 else m["question"]
            print(f"  {i:2}. [{m['centrality']:.3f}] {q}")
        print(f"\n  LABEL: _________________")
        print(f"  FIT SCORE (0-100%): ___")


if __name__ == "__main__":
    main()
