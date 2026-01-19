"""Probability rollup from leaf signals to root.

This module computes an aggregate probability for a target question by
treating signal base_rates as "soft evidence" that updates our belief.

Key insight: Signal base_rates represent market/expert beliefs about
correlated events. If a positively-correlated signal is likely (high
base_rate), that's implicit evidence the target is likely.

Formula for each signal's evidence contribution:
    evidence_i = (base_rate_i - 0.5) * spread_i

Where:
    - base_rate: P(signal=YES) from market/estimate
    - spread: P(parent|signal=YES) - P(parent|signal=NO), computed via rho_to_posteriors

Interpretation:
    - (base_rate - 0.5): How certain we are about the signal, and in which direction
        - base_rate=0.9 → +0.4 (strong evidence signal is YES)
        - base_rate=0.1 → -0.4 (strong evidence signal is NO)
        - base_rate=0.5 → 0.0 (maximum uncertainty, no evidence)
    - spread: How much knowing the signal's outcome would move our belief
        - Large positive spread: signal=YES would increase parent probability
        - Large negative spread (rare): signal=YES would decrease parent probability

Example:
    Signal: "Will One Battle be nominated for Best Picture?"
    base_rate: 0.75 (market thinks likely)
    rho: +0.7 (positive correlation with winning)

    spread = P(win|nominated) - P(win|not_nominated) ≈ 0.65 - 0.35 = 0.30
    evidence = (0.75 - 0.5) * 0.30 = 0.25 * 0.30 = 0.075

    → This signal provides +0.075 evidence toward the target.

Aggregation:
    - Convert prior to log-odds
    - Sum evidence contributions (scaled)
    - Convert back to probability
"""

from __future__ import annotations

import math

from llm_forecasting.voi import rho_to_posteriors

from .tree import SignalNode, SignalTree


def compute_signal_evidence(
    signal: SignalNode,
    parent_prior: float,
) -> tuple[float, float, float]:
    """Compute a signal's evidence contribution to its parent.

    Uses rho_to_posteriors to compute actual conditional probabilities,
    then weights by how certain we are about the signal (base_rate).

    Args:
        signal: The signal node
        parent_prior: Prior probability of the parent

    Returns:
        Tuple of (evidence, spread, direction_multiplier) where:
        - evidence: The contribution to parent probability
        - spread: P(parent|YES) - P(parent|NO)
        - direction_multiplier: (base_rate - 0.5), indicates certainty and direction
    """
    if signal.rho is None or signal.base_rate is None:
        return 0.0, 0.0, 0.0

    # Compute conditional probabilities using the rho model
    p_parent_yes, p_parent_no = rho_to_posteriors(
        signal.rho, parent_prior, signal.base_rate
    )

    # Spread: how much would knowing the signal's outcome move our belief?
    spread = p_parent_yes - p_parent_no

    # Direction multiplier: how certain are we about the signal, and which way?
    # Ranges from -0.5 (certain NO) to +0.5 (certain YES)
    direction = signal.base_rate - 0.5

    # Evidence: certainty × impact
    evidence = direction * spread

    return evidence, spread, direction


def compute_node_probability(
    node: SignalNode,
    tree: SignalTree,
    prior: float = 0.5,
) -> float:
    """Compute P(node=YES) from its children using Bayesian-ish aggregation.

    Treats each child's base_rate as "soft evidence" about the node.
    Aggregates evidence in log-odds space for numerical stability.

    Args:
        node: The node to compute probability for
        tree: The full signal tree
        prior: Prior probability if no children

    Returns:
        Computed P(node=YES)
    """
    children = tree.get_children(node.id)

    if not children:
        # Leaf node - use base_rate
        return node.base_rate if node.base_rate is not None else prior

    # Clamp prior to valid range
    prior = max(0.01, min(0.99, prior))

    # Start with prior in log-odds space
    log_odds = math.log(prior / (1 - prior))

    # Accumulate evidence from children
    total_evidence = 0.0

    for child in children:
        evidence, spread, direction = compute_signal_evidence(child, prior)
        total_evidence += evidence

    # Scale factor for converting evidence to log-odds
    # Evidence ranges roughly in [-0.25, +0.25] per signal (spread × direction)
    # We want moderate signals to have meaningful impact
    # k=4.0 means a single signal with spread=0.5 and direction=0.5 shifts log-odds by 1.0
    k = 4.0
    log_odds += k * total_evidence

    # Convert back to probability
    prob = 1 / (1 + math.exp(-log_odds))

    # Clamp to avoid extremes
    return max(0.01, min(0.99, prob))


def rollup_tree(
    tree: SignalTree,
    target_prior: float = 0.5,
) -> float:
    """Compute P(target=YES) by rolling up from leaves to root.

    Process nodes bottom-up: leaves first, then their parents, etc.
    At each level, compute the node's probability from its children,
    then use that as the "base_rate" for computing its contribution
    to its own parent.

    Args:
        tree: The signal tree
        target_prior: Prior probability for target

    Returns:
        Computed P(target=YES)
    """
    # Build depth-ordered list (deepest first)
    max_depth = tree.max_depth
    nodes_by_depth: dict[int, list[SignalNode]] = {}

    for signal in tree.signals:
        depth = signal.depth
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(signal)

    # Process bottom-up
    computed_probs: dict[str, float] = {}

    for depth in range(max_depth, 0, -1):
        if depth not in nodes_by_depth:
            continue

        for node in nodes_by_depth[depth]:
            if node.is_leaf:
                # Leaf: use base_rate
                computed_probs[node.id] = (
                    node.base_rate if node.base_rate is not None else 0.5
                )
            else:
                # Internal node: compute from children
                computed_probs[node.id] = compute_node_probability(
                    node, tree, prior=0.5
                )

            # Compute and store conditional probabilities for analysis
            if node.rho is not None and node.base_rate is not None:
                parent = tree.get_node(node.parent_id) if node.parent_id else tree.target
                parent_prior = target_prior if parent.id == tree.target.id else 0.5

                p_yes, p_no = rho_to_posteriors(
                    node.rho, parent_prior, node.base_rate
                )
                node.p_parent_given_yes = p_yes
                node.p_parent_given_no = p_no

    # Compute target probability
    target_prob = compute_node_probability(tree.target, tree, prior=target_prior)
    tree.computed_probability = target_prob

    return target_prob


def compute_signal_contribution(
    signal: SignalNode,
    tree: SignalTree,
    parent_prior: float = 0.5,
) -> dict:
    """Compute detailed metrics about a signal's contribution.

    Returns comprehensive metrics for understanding the signal's impact:
    - evidence: Actual contribution to parent probability
    - spread: P(parent|YES) - P(parent|NO)
    - direction: "enhances", "suppresses", or "neutral"
    - certainty: |base_rate - 0.5|, how certain we are about the signal

    Args:
        signal: The signal to analyze
        tree: The signal tree
        parent_prior: Prior probability of the parent

    Returns:
        Dict with contribution metrics
    """
    base_rate = signal.base_rate if signal.base_rate is not None else 0.5
    rho = signal.rho if signal.rho is not None else 0.0

    # Compute evidence using the proper formula
    evidence, spread, direction_mult = compute_signal_evidence(signal, parent_prior)

    # Direction based on rho sign (does signal=YES help or hurt parent?)
    direction = "enhances" if rho > 0 else "suppresses" if rho < 0 else "neutral"

    # Certainty: how far from 50% is the signal?
    certainty = abs(base_rate - 0.5)

    # Use conditional probs if already computed, otherwise compute them
    p_yes = signal.p_parent_given_yes
    p_no = signal.p_parent_given_no
    if p_yes is None or p_no is None:
        p_yes, p_no = rho_to_posteriors(rho, parent_prior, base_rate)

    return {
        "signal_id": signal.id,
        "text": signal.text,
        "base_rate": base_rate,
        "rho": rho,
        "evidence": evidence,  # New: actual contribution
        "spread": spread,  # P(parent|YES) - P(parent|NO)
        "certainty": certainty,  # |base_rate - 0.5|
        "direction": direction,
        "p_parent_given_yes": p_yes,
        "p_parent_given_no": p_no,
        # Legacy field for backwards compatibility
        "contribution": abs(evidence),
    }


def analyze_tree(tree: SignalTree, target_prior: float = 0.5) -> dict:
    """Analyze the full tree and return summary statistics.

    Args:
        tree: The signal tree
        target_prior: Prior probability for target

    Returns:
        Dict with tree analysis including:
        - computed_probability: Aggregate P(target=YES)
        - top_contributors: Signals with highest |evidence|
        - evidence_breakdown: Positive vs negative evidence totals
    """
    # Rollup probabilities
    computed_prob = rollup_tree(tree, target_prior)

    # Analyze leaf contributions
    leaves = tree.get_leaves()
    leaf_contributions = [
        compute_signal_contribution(leaf, tree, target_prior) for leaf in leaves
    ]

    # Sort by absolute evidence (most impactful first)
    leaf_contributions.sort(key=lambda x: abs(x["evidence"]), reverse=True)

    # Compute evidence breakdown
    positive_evidence = sum(c["evidence"] for c in leaf_contributions if c["evidence"] > 0)
    negative_evidence = sum(c["evidence"] for c in leaf_contributions if c["evidence"] < 0)

    return {
        "target": tree.target.text,
        "computed_probability": computed_prob,
        "target_prior": target_prior,
        "max_depth": tree.max_depth,
        "total_signals": len(tree.signals),
        "leaf_count": tree.leaf_count,
        "top_contributors": leaf_contributions[:10],
        "all_contributions": leaf_contributions,
        "evidence_breakdown": {
            "positive": positive_evidence,
            "negative": negative_evidence,
            "net": positive_evidence + negative_evidence,
        },
    }
