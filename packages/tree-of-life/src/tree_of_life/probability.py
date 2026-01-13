"""Probability normalization with tiered handling for scenario probabilities."""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityResult:
    """Result of probability normalization with diagnostic info."""

    raw_probabilities: dict[str, float]
    raw_sum: float
    normalized: dict[str, float] | None
    status: Literal["ok", "warning", "retry_needed", "suspect"]
    action_taken: str


def handle_probability_sum(
    scenarios: dict[str, float], tolerance: float = 0.05
) -> ProbabilityResult:
    """
    Handle probability sum with tiered approach.

    Tiers:
    - 95-105% (within tolerance): Silent normalize
    - 70-130%: Warn + normalize
    - Outside 70-130%: Request retry

    Args:
        scenarios: Dict mapping scenario_id to probability
        tolerance: Acceptable deviation from 1.0 for silent normalization

    Returns:
        ProbabilityResult with normalization status and diagnostics
    """
    total = sum(scenarios.values())

    # Tier 1: Within tolerance (95-105%) — silent normalize
    if abs(total - 1.0) <= tolerance:
        normalized = {k: v / total for k, v in scenarios.items()}
        return ProbabilityResult(
            raw_probabilities=scenarios,
            raw_sum=total,
            normalized=normalized,
            status="ok",
            action_taken="normalized",
        )

    # Tier 2: 70-130% — warn + normalize
    if 0.7 <= total <= 1.3:
        logger.warning(
            f"Probability sum {total:.1%} outside tolerance, normalizing anyway"
        )
        normalized = {k: v / total for k, v in scenarios.items()}
        return ProbabilityResult(
            raw_probabilities=scenarios,
            raw_sum=total,
            normalized=normalized,
            status="warning",
            action_taken="normalized_with_warning",
        )

    # Tier 3: Outside 70-130% — request retry
    logger.error(
        f"Probability sum {total:.1%} far outside expected range, retry needed"
    )
    return ProbabilityResult(
        raw_probabilities=scenarios,
        raw_sum=total,
        normalized=None,
        status="retry_needed",
        action_taken="requesting_retry",
    )


def force_normalize(scenarios: dict[str, float]) -> dict[str, float]:
    """
    Force normalization even when retry failed.

    Used as fallback when retry still produces bad probabilities.
    Marks result as 'suspect' in the calling code.

    Args:
        scenarios: Dict mapping scenario_id to probability

    Returns:
        Normalized probabilities (always sums to 1.0)
    """
    total = sum(scenarios.values())
    if total == 0:
        # Avoid division by zero - assign equal probabilities
        n = len(scenarios)
        return {k: 1.0 / n for k in scenarios}
    return {k: v / total for k, v in scenarios.items()}


def format_probabilities_for_retry(scenarios: dict[str, float]) -> str:
    """Format probabilities for the retry prompt."""
    lines = []
    for scenario_id, prob in sorted(scenarios.items()):
        lines.append(f"  - {scenario_id}: {prob:.1%}")
    return "\n".join(lines)
