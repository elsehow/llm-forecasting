"""Probability rollup from leaf signals to root.

This module computes an aggregate probability for a target question by
treating signal base_rates as "soft evidence" that updates our belief.

## Relationship Types

Signals can have three relationship types to their parent:

1. **correlation** (default): Statistical relationship captured by rho
   - Evidence formula: (base_rate - 0.5) * spread
   - Example: "Golden Globe win" correlates with "Oscar win"

2. **necessity**: Signal=NO implies Parent=0 (logical constraint)
   - P(parent|signal=NO) = 0
   - Example: "Must be nominated to win" - if not nominated, can't win
   - Effect: Parent probability capped at signal's base_rate

3. **sufficiency**: Signal=YES implies Parent=1 (rare)
   - P(parent|signal=YES) = 1
   - Example: Winning a required qualifier
   - Effect: Parent probability floored at signal's base_rate

## Correlation Evidence Formula

For correlation signals:
    evidence_i = (base_rate_i - 0.5) * spread_i

Where:
    - base_rate: P(signal=YES) from market/estimate
    - spread: P(parent|signal=YES) - P(parent|signal=NO), computed via rho_to_posteriors

Interpretation:
    - (base_rate - 0.5): How certain we are about the signal, and in which direction
    - spread: How much knowing the signal's outcome would move our belief

Aggregation:
    - Apply necessity/sufficiency constraints first (hard bounds)
    - Convert prior to log-odds
    - Sum correlation evidence contributions (scaled)
    - Convert back to probability
    - Enforce constraint bounds
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from llm_forecasting.voi import rho_to_posteriors

from .tree import SignalNode, SignalTree

if TYPE_CHECKING:
    from .registry import TreeRegistry

# Load rollup config
_config_path = Path(__file__).parent / "rollup_config.json"
with open(_config_path) as f:
    ROLLUP_CONFIG = json.load(f)

K = ROLLUP_CONFIG["evidence_scale_factor"]
PROB_CLAMP_MIN = ROLLUP_CONFIG["probability_clamp_min"]
PROB_CLAMP_MAX = ROLLUP_CONFIG["probability_clamp_max"]
DEFAULT_PRIOR = ROLLUP_CONFIG["default_prior"]


def compute_signal_evidence(
    signal: SignalNode,
    parent_prior: float,
    registry: TreeRegistry | None = None,
) -> tuple[float, float, float]:
    """Compute a signal's evidence contribution to its parent.

    For correlation signals: Uses rho_to_posteriors to compute actual
    conditional probabilities, then weights by certainty (base_rate).

    For necessity/sufficiency signals: Returns 0 evidence (constraints
    are applied separately in compute_node_probability).

    Args:
        signal: The signal node
        parent_prior: Prior probability of the parent
        registry: Optional TreeRegistry for resolving cross-tree refs

    Returns:
        Tuple of (evidence, spread, direction_multiplier) where:
        - evidence: The contribution to parent probability
        - spread: P(parent|YES) - P(parent|NO)
        - direction_multiplier: (base_rate - 0.5), indicates certainty and direction
    """
    # Necessity and sufficiency are handled as constraints, not evidence
    if signal.relationship_type in ("necessity", "sufficiency"):
        return 0.0, 0.0, 0.0

    # Resolve effective base_rate (ref > market > base_rate)
    effective_base_rate = _get_effective_base_rate(signal, registry)

    if signal.rho is None or effective_base_rate is None:
        return 0.0, 0.0, 0.0

    # Compute conditional probabilities using the rho model
    p_parent_yes, p_parent_no = rho_to_posteriors(
        signal.rho, parent_prior, effective_base_rate
    )

    # Spread: how much would knowing the signal's outcome move our belief?
    spread = p_parent_yes - p_parent_no

    # Direction multiplier: how certain are we about the signal, and which way?
    # Ranges from -0.5 (certain NO) to +0.5 (certain YES)
    direction = effective_base_rate - 0.5

    # Evidence: certainty × impact
    evidence = direction * spread

    return evidence, spread, direction


def _get_effective_base_rate(
    signal: SignalNode,
    registry: TreeRegistry | None = None,
    prefer_market: bool = True,
) -> float | None:
    """Get effective base_rate, prioritizing market price for leaves.

    Priority order:
    1. Cross-tree ref (if registry provided)
    2. Market price (if prefer_market=True and signal is a leaf with market_price)
    3. base_rate

    Args:
        signal: The signal node
        registry: Optional TreeRegistry for resolving cross-tree refs
        prefer_market: If True, prefer market_price over base_rate for leaves

    Returns:
        Effective probability, or None if not available
    """
    # Cross-tree ref takes precedence
    if signal.ref and registry:
        return registry.get_probability(signal.ref)

    # For leaves: prefer market_price if available and flagged
    if prefer_market and signal.is_leaf and signal.market_price is not None:
        return signal.market_price

    return signal.base_rate


def compute_node_gap(node: SignalNode, computed_prob: float) -> float | None:
    """Compute gap between computed and market price (if available).

    Args:
        node: The signal node
        computed_prob: Computed probability from rollup

    Returns:
        Gap in percentage points (computed - market), or None if no market price
    """
    if node.market_price is None:
        return None
    return (computed_prob - node.market_price) * 100


def compute_node_probability(
    node: SignalNode,
    tree: SignalTree,
    prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> float:
    """Compute P(node=YES) from its children using Bayesian-ish aggregation.

    Handles three relationship types:
    - correlation: Treats base_rate as "soft evidence" (aggregated in log-odds)
    - necessity: Caps probability at signal's base_rate (can't win if not nominated)
    - sufficiency: Floors probability at signal's base_rate (rare)

    Args:
        node: The node to compute probability for
        tree: The full signal tree
        prior: Prior probability if no children
        registry: Optional TreeRegistry for resolving cross-tree refs

    Returns:
        Computed P(node=YES)
    """
    children = tree.get_children(node.id)

    if not children:
        # Leaf node - use base_rate (or resolved ref)
        effective_rate = _get_effective_base_rate(node, registry)
        return effective_rate if effective_rate is not None else prior

    # Clamp prior to valid range
    prior = max(0.01, min(0.99, prior))

    # Collect constraints from necessity/sufficiency signals
    upper_bound = 1.0  # From necessity signals
    lower_bound = 0.0  # From sufficiency signals

    for child in children:
        effective_rate = _get_effective_base_rate(child, registry)
        if effective_rate is None:
            continue

        if child.relationship_type == "necessity":
            # P(parent|signal=NO) = 0, so P(parent) ≤ P(signal)
            # If nomination is 98% likely, win probability capped at 98%
            upper_bound = min(upper_bound, effective_rate)

        elif child.relationship_type == "sufficiency":
            # P(parent|signal=YES) = 1, so P(parent) ≥ P(signal)
            lower_bound = max(lower_bound, effective_rate)

    # Start with prior in log-odds space
    log_odds = math.log(prior / (1 - prior))

    # Accumulate evidence from correlation children only
    total_evidence = 0.0

    for child in children:
        if child.relationship_type == "correlation":
            evidence, spread, direction = compute_signal_evidence(child, prior, registry)
            total_evidence += evidence

    # Scale factor for converting evidence to log-odds
    log_odds += K * total_evidence

    # Convert back to probability
    prob = 1 / (1 + math.exp(-log_odds))

    # Apply constraint bounds from necessity/sufficiency signals
    prob = max(lower_bound, min(upper_bound, prob))

    # Final clamp to avoid exact 0 or 1
    return max(PROB_CLAMP_MIN, min(PROB_CLAMP_MAX, prob))


def rollup_tree(
    tree: SignalTree,
    target_prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> float:
    """Compute P(target=YES) by rolling up from leaves to root.

    Process nodes bottom-up: leaves first, then their parents, etc.
    At each level, compute the node's probability from its children,
    then use that as the "base_rate" for computing its contribution
    to its own parent.

    Args:
        tree: The signal tree
        target_prior: Prior probability for target
        registry: Optional TreeRegistry for resolving cross-tree refs

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
                # Leaf: use base_rate (or resolved ref)
                effective_rate = _get_effective_base_rate(node, registry)
                computed_probs[node.id] = (
                    effective_rate if effective_rate is not None else DEFAULT_PRIOR
                )
            else:
                # Internal node: compute from children
                computed_probs[node.id] = compute_node_probability(
                    node, tree, prior=0.5, registry=registry
                )

            # Compute and store conditional probabilities for analysis
            effective_rate = _get_effective_base_rate(node, registry)
            if node.rho is not None and effective_rate is not None:
                parent = tree.get_node(node.parent_id) if node.parent_id else tree.target
                parent_prior = target_prior if parent.id == tree.target.id else 0.5

                p_yes, p_no = rho_to_posteriors(
                    node.rho, parent_prior, effective_rate
                )
                node.p_parent_given_yes = p_yes
                node.p_parent_given_no = p_no

    # Compute target probability
    target_prob = compute_node_probability(tree.target, tree, prior=target_prior, registry=registry)
    tree.computed_probability = target_prob

    return target_prob


def compute_signal_contribution(
    signal: SignalNode,
    tree: SignalTree,
    parent_prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> dict:
    """Compute detailed metrics about a signal's contribution.

    Returns comprehensive metrics for understanding the signal's impact:
    - evidence: Actual contribution to parent probability (0 for necessity/sufficiency)
    - spread: P(parent|YES) - P(parent|NO)
    - direction: "enhances", "suppresses", "neutral", "necessity", or "sufficiency"
    - certainty: |base_rate - 0.5|, how certain we are about the signal
    - relationship_type: "correlation", "necessity", or "sufficiency"

    Args:
        signal: The signal to analyze
        tree: The signal tree
        parent_prior: Prior probability of the parent
        registry: Optional TreeRegistry for resolving cross-tree refs

    Returns:
        Dict with contribution metrics
    """
    # Resolve effective base_rate
    effective_rate = _get_effective_base_rate(signal, registry)
    base_rate = effective_rate if effective_rate is not None else 0.5
    rho = signal.rho if signal.rho is not None else 0.0
    relationship_type = signal.relationship_type

    # Compute evidence using the proper formula
    evidence, spread, direction_mult = compute_signal_evidence(signal, parent_prior, registry)

    # Direction based on relationship type and rho sign
    if relationship_type == "necessity":
        direction = "necessity"
        # For necessity, spread is P(parent|YES) - 0
        p_yes = parent_prior  # Approximate: if nominated, prior applies
        p_no = 0.0  # Hard constraint
        spread = p_yes - p_no
    elif relationship_type == "sufficiency":
        direction = "sufficiency"
        p_yes = 1.0  # Hard constraint
        p_no = parent_prior  # Approximate
        spread = p_yes - p_no
    else:
        # Correlation: direction based on rho sign
        direction = "enhances" if rho > 0 else "suppresses" if rho < 0 else "neutral"
        # Use conditional probs if already computed, otherwise compute them
        p_yes = signal.p_parent_given_yes
        p_no = signal.p_parent_given_no
        if p_yes is None or p_no is None:
            p_yes, p_no = rho_to_posteriors(rho, parent_prior, base_rate)

    # Certainty: how far from 50% is the signal?
    certainty = abs(base_rate - 0.5)

    return {
        "signal_id": signal.id,
        "text": signal.text,
        "base_rate": base_rate,
        "ref": signal.ref,  # Include ref info if present
        "rho": rho,
        "relationship_type": relationship_type,
        "evidence": evidence,  # 0 for necessity/sufficiency (they're constraints)
        "spread": spread,  # P(parent|YES) - P(parent|NO)
        "certainty": certainty,  # |base_rate - 0.5|
        "direction": direction,
        "p_parent_given_yes": p_yes,
        "p_parent_given_no": p_no,
        # Legacy field for backwards compatibility
        "contribution": abs(evidence),
    }


def analyze_tree(
    tree: SignalTree,
    target_prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> dict:
    """Analyze the full tree and return summary statistics.

    Args:
        tree: The signal tree
        target_prior: Prior probability for target
        registry: Optional TreeRegistry for resolving cross-tree refs

    Returns:
        Dict with tree analysis including:
        - computed_probability: Aggregate P(target=YES)
        - top_contributors: Signals with highest |evidence|
        - evidence_breakdown: Positive vs negative evidence totals
    """
    # Rollup probabilities
    computed_prob = rollup_tree(tree, target_prior, registry)

    # Analyze leaf contributions
    leaves = tree.get_leaves()
    leaf_contributions = [
        compute_signal_contribution(leaf, tree, target_prior, registry) for leaf in leaves
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
