"""Probability propagation through causal scenario graph.

This module provides functions to:
1. Derive upstream relationships from the Relationship model
2. Propagate probability updates through the causal graph
3. Compute unconditional forecasts as weighted sums over scenarios
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
        BinaryForecast,
        CategoricalForecast,
        ConditionalForecast,
        ContinuousForecast,
        ForecastTree,
        GlobalScenario,
        Relationship,
    )


def derive_upstream_graph(
    scenarios: list[GlobalScenario],
    relationships: list[Relationship],
) -> dict[str, list[str]]:
    """Compute upstream scenarios for each scenario from Relationships.

    Derivation rules:
    - hierarchical: scenario_a (parent) is upstream of scenario_b (child)
    - correlated: scenario_a is upstream of scenario_b
    - orthogonal: no upstream link (independent by definition)
    - mutually_exclusive: no upstream link (alternatives, not causal)

    Args:
        scenarios: List of global scenarios
        relationships: List of pairwise relationships

    Returns:
        Dict mapping scenario_id -> list of upstream scenario_ids
    """
    scenario_ids = {s.id for s in scenarios}
    upstream: dict[str, list[str]] = defaultdict(list)

    for rel in relationships:
        # Skip relationships with unknown scenarios
        if rel.scenario_a not in scenario_ids or rel.scenario_b not in scenario_ids:
            continue

        if rel.type == "hierarchical":
            # Parent (a) is upstream of child (b)
            upstream[rel.scenario_b].append(rel.scenario_a)

        elif rel.type == "correlated":
            # a influences b
            upstream[rel.scenario_b].append(rel.scenario_a)

        # orthogonal and mutually_exclusive: no upstream links

    # Convert to regular dict with deduplicated lists
    return {
        sid: list(dict.fromkeys(upstreams))  # Preserve order, remove duplicates
        for sid, upstreams in upstream.items()
    }


def get_downstream_graph(upstream_graph: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert the upstream graph to get downstream relationships.

    Args:
        upstream_graph: Dict mapping scenario_id -> upstream scenario_ids

    Returns:
        Dict mapping scenario_id -> downstream scenario_ids
    """
    downstream: dict[str, list[str]] = defaultdict(list)

    for scenario_id, upstreams in upstream_graph.items():
        for upstream_id in upstreams:
            downstream[upstream_id].append(scenario_id)

    return dict(downstream)


def topological_sort(
    scenario_ids: list[str],
    upstream_graph: dict[str, list[str]],
) -> list[str]:
    """Sort scenarios so upstreams come before downstreams (Kahn's algorithm).

    Args:
        scenario_ids: All scenario IDs to sort
        upstream_graph: Dict mapping scenario_id -> upstream scenario_ids

    Returns:
        Topologically sorted list of scenario IDs

    Raises:
        ValueError: If the graph contains a cycle
    """
    # Build in-degree count (number of upstreams)
    in_degree: dict[str, int] = {sid: 0 for sid in scenario_ids}
    for sid in scenario_ids:
        in_degree[sid] = len(upstream_graph.get(sid, []))

    # Start with scenarios that have no upstreams
    queue = [sid for sid in scenario_ids if in_degree[sid] == 0]
    result = []

    # Build downstream graph for traversal
    downstream = get_downstream_graph(upstream_graph)

    while queue:
        # Process in stable order
        queue.sort()
        current = queue.pop(0)
        result.append(current)

        # Reduce in-degree of downstream scenarios
        for downstream_id in downstream.get(current, []):
            if downstream_id in in_degree:
                in_degree[downstream_id] -= 1
                if in_degree[downstream_id] == 0:
                    queue.append(downstream_id)

    if len(result) != len(scenario_ids):
        # Some scenarios weren't reached - indicates a cycle
        missing = set(scenario_ids) - set(result)
        raise ValueError(f"Cycle detected in upstream graph involving: {missing}")

    return result


def propagate_update(
    tree: ForecastTree,
    updated_scenario_id: str,
    new_probability: float,
    renormalize: bool = True,
) -> ForecastTree:
    """Update scenario probability and propagate to downstream scenarios.

    This function:
    1. Sets the new probability for the updated scenario
    2. Identifies downstream scenarios that depend on it
    3. Adjusts downstream probabilities proportionally
    4. Renormalizes all probabilities to sum to 1.0

    Note: This is a simple proportional adjustment. More sophisticated
    Bayesian updating would require additional model assumptions.

    Args:
        tree: The ForecastTree to update
        updated_scenario_id: ID of the scenario whose probability changed
        new_probability: New probability value (0-1)
        renormalize: Whether to renormalize probabilities to sum to 1.0

    Returns:
        New ForecastTree with updated probabilities (immutable pattern)
    """
    # Create mutable copies
    scenarios = [s.model_copy() for s in tree.global_scenarios]
    scenario_map = {s.id: s for s in scenarios}

    if updated_scenario_id not in scenario_map:
        raise ValueError(f"Unknown scenario: {updated_scenario_id}")

    # Get the old probability and compute the change ratio
    old_prob = scenario_map[updated_scenario_id].probability
    scenario_map[updated_scenario_id].probability = new_probability

    # If renormalizing, adjust other scenarios proportionally
    if renormalize and old_prob != new_probability:
        prob_delta = new_probability - old_prob
        other_prob_sum = sum(
            s.probability for s in scenarios if s.id != updated_scenario_id
        )

        if other_prob_sum > 0:
            # Distribute the delta proportionally among other scenarios
            scale_factor = (other_prob_sum - prob_delta) / other_prob_sum
            for s in scenarios:
                if s.id != updated_scenario_id:
                    s.probability *= scale_factor

    # Create new tree with updated scenarios
    return tree.model_copy(update={"global_scenarios": scenarios})


def compute_unconditional(
    tree: ForecastTree,
    question_id: str,
) -> float | dict[str, float]:
    """Compute unconditional forecast as weighted sum over scenarios.

    E[outcome] = Σ P(scenario) × E[outcome | scenario]

    For continuous questions: returns weighted median
    For binary questions: returns weighted probability
    For categorical questions: returns weighted probability distribution

    Args:
        tree: The ForecastTree containing scenarios and conditionals
        question_id: ID of the question to compute unconditional forecast for

    Returns:
        float for continuous/binary, dict[str, float] for categorical

    Raises:
        ValueError: If question not found or no conditionals available
    """
    # Get all conditionals for this question
    conditionals = [c for c in tree.conditionals if c.question_id == question_id]

    if not conditionals:
        raise ValueError(f"No conditionals found for question: {question_id}")

    # Build scenario probability lookup
    scenario_probs = {s.id: s.probability for s in tree.global_scenarios}

    # Check the type of the first conditional to determine return type
    first = conditionals[0]

    # Import here to avoid circular imports
    from .models import BinaryForecast, CategoricalForecast, ContinuousForecast

    if isinstance(first, ContinuousForecast):
        # Weighted median
        weighted_sum = 0.0
        total_weight = 0.0
        for c in conditionals:
            if not isinstance(c, ContinuousForecast):
                continue
            weight = scenario_probs.get(c.scenario_id, 0.0)
            weighted_sum += weight * c.median
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    elif isinstance(first, BinaryForecast):
        # Weighted probability
        weighted_sum = 0.0
        total_weight = 0.0
        for c in conditionals:
            if not isinstance(c, BinaryForecast):
                continue
            weight = scenario_probs.get(c.scenario_id, 0.0)
            weighted_sum += weight * c.probability
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    elif isinstance(first, CategoricalForecast):
        # Weighted probability distribution
        result: dict[str, float] = defaultdict(float)
        total_weight = 0.0
        for c in conditionals:
            if not isinstance(c, CategoricalForecast):
                continue
            weight = scenario_probs.get(c.scenario_id, 0.0)
            total_weight += weight
            for option, prob in c.probabilities.items():
                result[option] += weight * prob

        # Normalize
        if total_weight > 0:
            return {k: v / total_weight for k, v in result.items()}
        return dict(result)

    else:
        raise ValueError(f"Unknown conditional type: {type(first)}")


def compute_all_unconditionals(
    tree: ForecastTree,
) -> dict[str, float | dict[str, float]]:
    """Compute unconditional forecasts for all questions.

    Args:
        tree: The ForecastTree

    Returns:
        Dict mapping question_id -> unconditional forecast value
    """
    question_ids = {q.id for q in tree.questions}
    return {qid: compute_unconditional(tree, qid) for qid in question_ids}
