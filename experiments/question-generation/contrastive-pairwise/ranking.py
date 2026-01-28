"""Ranking algorithms from pairwise comparisons."""

import numpy as np
from collections import defaultdict


def compute_win_rates(comparisons: list[dict], crux_key: str = "crux") -> dict[str, float]:
    """Compute win rate for each crux from pairwise comparisons.

    Args:
        comparisons: List of dicts with 'crux_a', 'crux_b', 'winner' (0=A, 1=B)
        crux_key: Key to use for crux identification

    Returns:
        Dict mapping crux -> win rate [0, 1]
    """
    wins = defaultdict(int)
    total = defaultdict(int)

    for comp in comparisons:
        crux_a = comp["crux_a"]
        crux_b = comp["crux_b"]
        winner = comp["winner"]  # 0 = A wins, 1 = B wins

        total[crux_a] += 1
        total[crux_b] += 1

        if winner == 0:
            wins[crux_a] += 1
        else:
            wins[crux_b] += 1

    # Compute win rates
    win_rates = {}
    for crux in total:
        win_rates[crux] = wins[crux] / total[crux] if total[crux] > 0 else 0.5

    return win_rates


def bradley_terry_mle(
    comparisons: list[dict],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Fit Bradley-Terry model via maximum likelihood.

    Bradley-Terry model: P(A beats B) = exp(s_A) / (exp(s_A) + exp(s_B))

    Uses iterative algorithm from Zermelo (1929) / Hunter (2004).

    Args:
        comparisons: List of dicts with 'crux_a', 'crux_b', 'winner' (0=A, 1=B)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Dict mapping crux -> strength score (higher = stronger)
    """
    # Collect all items and build comparison matrix
    items = set()
    for comp in comparisons:
        items.add(comp["crux_a"])
        items.add(comp["crux_b"])

    items = sorted(items)
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}

    # Count wins and losses for each pair
    # wins[i, j] = number of times i beat j
    wins = np.zeros((n, n))
    for comp in comparisons:
        i = item_to_idx[comp["crux_a"]]
        j = item_to_idx[comp["crux_b"]]
        if comp["winner"] == 0:  # A wins
            wins[i, j] += 1
        else:  # B wins
            wins[j, i] += 1

    # Total comparisons involving each pair
    comparisons_matrix = wins + wins.T

    # Initialize scores uniformly
    scores = np.ones(n)

    # Iterative update (MM algorithm)
    for _ in range(max_iter):
        old_scores = scores.copy()

        for i in range(n):
            # Number of wins for i
            n_wins = wins[i, :].sum()

            # Denominator: sum over opponents j of (n_ij / (s_i + s_j))
            denom = 0
            for j in range(n):
                if i != j and comparisons_matrix[i, j] > 0:
                    denom += comparisons_matrix[i, j] / (old_scores[i] + old_scores[j])

            if denom > 0:
                scores[i] = n_wins / denom
            else:
                scores[i] = old_scores[i]

        # Normalize (sum to n)
        scores = scores * n / scores.sum()

        # Check convergence
        if np.max(np.abs(scores - old_scores)) < tol:
            break

    # Convert to log-scale for better interpretability
    log_scores = np.log(scores + 1e-10)

    return {item: float(log_scores[item_to_idx[item]]) for item in items}


def rank_by_scores(scores: dict[str, float], descending: bool = True) -> list[tuple[str, float, int]]:
    """Convert scores to ranks.

    Args:
        scores: Dict mapping item -> score
        descending: If True, highest score = rank 1

    Returns:
        List of (item, score, rank) tuples sorted by rank
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=descending)
    return [(item, score, rank + 1) for rank, (item, score) in enumerate(sorted_items)]
