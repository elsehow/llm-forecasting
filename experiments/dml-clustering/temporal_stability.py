#!/usr/bin/env python3
"""
DML Clustering Experiment - Phase 2: Temporal Stability

Test whether market clusters are stable over time by comparing
clusters computed on different time windows.

RQ2 Success Criteria:
- Adjusted Rand Index > 0.5 between time periods
- Core cluster members (top 10 per cluster) are >60% stable

Usage:
    python temporal_stability.py --k 10 --output results/stability.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


def load_price_history(price_dir: Path) -> dict[str, list[dict]]:
    """Load price history for all markets."""
    histories = {}
    for f in price_dir.glob("*.json"):
        market_id = f.stem
        with open(f) as fp:
            data = json.load(fp)
            # Handle both raw list format and {candles: [...]} format
            if isinstance(data, dict) and "candles" in data:
                histories[market_id] = data["candles"]
            elif isinstance(data, list):
                histories[market_id] = data
            else:
                histories[market_id] = []
    return histories


def compute_returns(prices: list[dict]) -> tuple[list[int], list[float]]:
    """
    Compute daily returns from candlestick data.

    Returns timestamps and returns as separate lists.
    """
    if len(prices) < 2:
        return [], []

    timestamps = []
    returns = []

    for i in range(1, len(prices)):
        prev_close = prices[i-1].get("close", prices[i-1].get("c", 0))
        curr_close = prices[i].get("close", prices[i].get("c", 0))

        if prev_close > 0:
            ret = (curr_close - prev_close) / prev_close
        else:
            ret = 0

        timestamps.append(prices[i].get("timestamp", prices[i].get("t", i)))
        returns.append(ret)

    return timestamps, returns


def filter_by_time_window(
    timestamps: list[int],
    returns: list[float],
    start_day: int,
    end_day: int
) -> tuple[list[int], list[float]]:
    """Filter returns to a specific day range (0-indexed from start of data)."""
    filtered_ts = []
    filtered_ret = []
    for i, (ts, ret) in enumerate(zip(timestamps, returns)):
        if start_day <= i < end_day:
            filtered_ts.append(ts)
            filtered_ret.append(ret)
    return filtered_ts, filtered_ret


def align_series(
    ts1: list[int], vals1: list[float],
    ts2: list[int], vals2: list[float]
) -> tuple[list[float], list[float]]:
    """Align two time series by timestamp."""
    set1 = set(ts1)
    set2 = set(ts2)
    common = sorted(set1 & set2)

    if not common:
        return [], []

    idx1 = {t: i for i, t in enumerate(ts1)}
    idx2 = {t: i for i, t in enumerate(ts2)}

    aligned1 = [vals1[idx1[t]] for t in common]
    aligned2 = [vals2[idx2[t]] for t in common]

    return aligned1, aligned2


def compute_correlation(vals1: list[float], vals2: list[float]) -> float | None:
    """Compute Pearson correlation between two aligned series."""
    if len(vals1) < 5:  # Minimum data points
        return None

    arr1 = np.array(vals1)
    arr2 = np.array(vals2)

    # Check for constant series
    if np.std(arr1) == 0 or np.std(arr2) == 0:
        return None

    corr = np.corrcoef(arr1, arr2)[0, 1]
    return corr if not np.isnan(corr) else None


def compute_correlation_matrix_for_window(
    histories: dict[str, list[dict]],
    market_ids: list[str],
    start_day: int,
    end_day: int
) -> np.ndarray:
    """
    Compute correlation matrix for a specific time window.
    """
    n = len(market_ids)
    corr_matrix = np.eye(n)

    # Precompute returns for each market
    returns_cache = {}
    for mid in market_ids:
        if mid in histories:
            ts, rets = compute_returns(histories[mid])
            ts, rets = filter_by_time_window(ts, rets, start_day, end_day)
            returns_cache[mid] = (ts, rets)
        else:
            returns_cache[mid] = ([], [])

    # Compute pairwise correlations
    for i, mid_a in enumerate(market_ids):
        ts_a, rets_a = returns_cache[mid_a]
        if not rets_a:
            continue

        for j in range(i + 1, len(market_ids)):
            mid_b = market_ids[j]
            ts_b, rets_b = returns_cache[mid_b]
            if not rets_b:
                continue

            aligned_a, aligned_b = align_series(ts_a, rets_a, ts_b, rets_b)
            rho = compute_correlation(aligned_a, aligned_b)

            if rho is not None:
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    return corr_matrix


def cluster_from_matrix(
    corr_matrix: np.ndarray,
    k: int,
    method: str = "ward"
) -> np.ndarray:
    """Apply hierarchical clustering to correlation matrix."""
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, k, criterion="maxclust")
    return labels


def get_core_members(
    corr_matrix: np.ndarray,
    labels: np.ndarray,
    market_ids: list[str],
    top_n: int = 10
) -> dict[int, list[str]]:
    """Get top N most central members of each cluster."""
    core_members = defaultdict(list)
    n = len(market_ids)

    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Compute centrality for each member
        centralities = []
        for idx in cluster_indices:
            cluster_corrs = corr_matrix[idx, cluster_mask]
            centrality = np.mean(np.abs(cluster_corrs))
            centralities.append((market_ids[idx], centrality))

        # Sort by centrality and take top N
        centralities.sort(key=lambda x: x[1], reverse=True)
        core_members[int(cluster_id)] = [m[0] for m in centralities[:top_n]]

    return dict(core_members)


def compute_core_member_stability(
    core_members_a: dict[int, list[str]],
    core_members_b: dict[int, list[str]]
) -> float:
    """
    Compute stability of core members across two clusterings.

    Returns fraction of core members that appear in the same cluster
    in both time periods.
    """
    # Build mapping: market -> cluster for each period
    market_to_cluster_a = {}
    market_to_cluster_b = {}

    for cluster_id, members in core_members_a.items():
        for m in members:
            market_to_cluster_a[m] = cluster_id

    for cluster_id, members in core_members_b.items():
        for m in members:
            market_to_cluster_b[m] = cluster_id

    # Find markets that are core in both periods
    common_core = set(market_to_cluster_a.keys()) & set(market_to_cluster_b.keys())

    if not common_core:
        return 0.0

    # Check if they're in the same cluster with the same core neighbors
    stable_count = 0
    for m in common_core:
        cluster_a = market_to_cluster_a[m]
        cluster_b = market_to_cluster_b[m]

        # Get other core members in same cluster
        neighbors_a = set(core_members_a[cluster_a]) - {m}
        neighbors_b = set(core_members_b[cluster_b]) - {m}

        # Count as stable if >50% of neighbors are shared
        if neighbors_a and neighbors_b:
            overlap = len(neighbors_a & neighbors_b) / len(neighbors_a | neighbors_b)
            if overlap > 0.5:
                stable_count += 1

    return stable_count / len(common_core)


def main():
    parser = argparse.ArgumentParser(description="Test temporal stability of clusters")
    parser.add_argument("--price-dir", default="../conditional-forecasting/data/price_history",
                        help="Directory containing price history JSON files")
    parser.add_argument("--pairs", default="../conditional-forecasting/data/pairs.json",
                        help="Path to pairs.json (for market ID list)")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--method", default="ward", help="Linkage method")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Size of each time window in days")
    parser.add_argument("--output", default="results/stability.json",
                        help="Output path")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    price_dir = script_dir / args.price_dir
    pairs_path = script_dir / args.pairs
    output_path = script_dir / args.output

    print(f"Loading price histories from {price_dir}...")
    histories = load_price_history(price_dir)
    print(f"  Loaded {len(histories)} markets")

    # Get market IDs from pairs (to use same set as clustering experiment)
    with open(pairs_path) as f:
        pairs = json.load(f)

    # Handle both nested dict format and flat format
    def get_market_id(m):
        return m["condition_id"] if isinstance(m, dict) else m

    market_ids = sorted(set(
        [get_market_id(p["market_a"]) for p in pairs] +
        [get_market_id(p["market_b"]) for p in pairs]
    ))

    # Price history files use truncated IDs (first 40 chars)
    # Build mapping from full ID to truncated ID
    truncated_histories = {full_id[:40]: data for full_id, data in histories.items()}

    # Filter to markets with price history (using truncated match)
    market_ids = [m for m in market_ids if m[:40] in truncated_histories]
    print(f"  {len(market_ids)} markets with price history")

    # Update histories dict to use full IDs as keys
    histories = {m: truncated_histories[m[:40]] for m in market_ids}

    # Compute clusters for each time window
    window_size = args.window_size

    print(f"\nComputing correlations for window 1 (days 0-{window_size})...")
    corr_matrix_1 = compute_correlation_matrix_for_window(
        histories, market_ids, 0, window_size
    )
    labels_1 = cluster_from_matrix(corr_matrix_1, args.k, args.method)
    core_members_1 = get_core_members(corr_matrix_1, labels_1, market_ids)

    print(f"Computing correlations for window 2 (days {window_size}-{2*window_size})...")
    corr_matrix_2 = compute_correlation_matrix_for_window(
        histories, market_ids, window_size, 2 * window_size
    )
    labels_2 = cluster_from_matrix(corr_matrix_2, args.k, args.method)
    core_members_2 = get_core_members(corr_matrix_2, labels_2, market_ids)

    # Compute stability metrics
    print("\nComputing stability metrics...")

    # Adjusted Rand Index
    ari = adjusted_rand_score(labels_1, labels_2)
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"    (Success threshold: > 0.5)")
    print(f"    {'PASS' if ari > 0.5 else 'FAIL'}")

    # Core member stability
    core_stability = compute_core_member_stability(core_members_1, core_members_2)
    print(f"\n  Core member stability: {core_stability:.1%}")
    print(f"    (Success threshold: > 60%)")
    print(f"    {'PASS' if core_stability > 0.6 else 'FAIL'}")

    # Track which markets switched clusters
    switched = []
    for i, mid in enumerate(market_ids):
        if labels_1[i] != labels_2[i]:
            switched.append({
                "market_id": mid,
                "cluster_window_1": int(labels_1[i]),
                "cluster_window_2": int(labels_2[i])
            })

    switch_rate = len(switched) / len(market_ids)
    print(f"\n  Markets that switched clusters: {len(switched)}/{len(market_ids)} ({switch_rate:.1%})")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "k": args.k,
        "method": args.method,
        "window_size_days": window_size,
        "n_markets": len(market_ids),
        "metrics": {
            "adjusted_rand_index": ari,
            "ari_threshold": 0.5,
            "ari_pass": ari > 0.5,
            "core_member_stability": core_stability,
            "core_stability_threshold": 0.6,
            "core_stability_pass": core_stability > 0.6,
            "switch_rate": switch_rate
        },
        "core_members_window_1": core_members_1,
        "core_members_window_2": core_members_2,
        "switched_markets": switched[:50]  # Top 50 for brevity
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Overall RQ2 assessment
    print(f"\n{'='*50}")
    print("RQ2 ASSESSMENT")
    print(f"{'='*50}")
    if ari > 0.5 and core_stability > 0.6:
        print("PASS: Cluster structure is stable over time")
    elif ari > 0.3:
        print("PARTIAL: Some stability, but below threshold")
    else:
        print("FAIL: Clusters are regime-dependent noise")


if __name__ == "__main__":
    main()
