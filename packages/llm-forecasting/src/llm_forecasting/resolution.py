"""Resolution logic for determining outcomes.

This module handles:
1. Determining resolution values for different source types (market vs data)
2. Handling edge cases (ambiguous resolutions, early resolutions, etc.)

Scoring functions are in scoring.py.
"""

from dataclasses import dataclass
from datetime import date

from llm_forecasting.models import Forecast, Question, Resolution, SourceType
from llm_forecasting.eval.scoring import compute_brier_score


@dataclass
class ResolutionResult:
    """Result of attempting to resolve a forecast."""

    resolved: bool
    resolution_value: float | None = None
    resolution_date: date | None = None
    reason: str | None = None  # Why resolution failed or was skipped


def is_valid_resolution(value: float | None) -> bool:
    """Check if a resolution value is valid for scoring.

    Valid values are 0 or 1 for binary questions.
    Values like 0.5 (ambiguous) or None are invalid.
    """
    if value is None:
        return False
    # Allow some floating point tolerance
    return abs(value - 0.0) < 0.001 or abs(value - 1.0) < 0.001


def resolve_market_question(
    question: Question,
    resolution: Resolution | None,
    forecast_due_date: date,
) -> ResolutionResult:
    """Resolve a market-based question (Manifold, Metaculus, Polymarket, etc.).

    For markets:
    - If the question has officially resolved (0 or 1), use that value
    - If resolution happened before forecast_due_date, skip (question was stale)
    - If question resolved to ambiguous value (e.g., 0.5 for "MKT"), skip
    - Otherwise, use the current market probability as interim resolution

    Args:
        question: The question being resolved
        resolution: The resolution data (market value at resolution date)
        forecast_due_date: When forecasts were due

    Returns:
        ResolutionResult with resolution status and value
    """
    if resolution is None:
        return ResolutionResult(
            resolved=False,
            reason="No resolution data available",
        )

    # Check if question officially resolved
    if question.resolved:
        # Validate resolution value
        if not is_valid_resolution(question.resolution_value):
            return ResolutionResult(
                resolved=False,
                reason=f"Ambiguous resolution: {question.resolution_value}",
            )

        # Check if resolved before forecast due date
        if question.resolution_date and question.resolution_date <= forecast_due_date:
            return ResolutionResult(
                resolved=False,
                reason=f"Resolved {question.resolution_date} before forecast due {forecast_due_date}",
            )

        return ResolutionResult(
            resolved=True,
            resolution_value=question.resolution_value,
            resolution_date=question.resolution_date,
        )

    # Not officially resolved - use market probability as interim resolution
    return ResolutionResult(
        resolved=True,
        resolution_value=resolution.value,
        resolution_date=resolution.date,
        reason="Using market probability as interim resolution",
    )


def resolve_data_question(
    question: Question,
    resolution_at_due_date: Resolution | None,
    resolution_at_horizon: Resolution | None,
) -> ResolutionResult:
    """Resolve a data-based question (FRED, Yahoo Finance, etc.).

    For data questions:
    - Compare value at forecast_due_date vs value at resolution_date
    - Resolution = 1 if value increased, 0 if decreased
    - Skip if either value is missing

    Args:
        question: The question being resolved
        resolution_at_due_date: Value snapshot at forecast due date
        resolution_at_horizon: Value snapshot at resolution horizon

    Returns:
        ResolutionResult with resolution status and value
    """
    if resolution_at_due_date is None:
        return ResolutionResult(
            resolved=False,
            reason="Missing value at forecast due date",
        )

    if resolution_at_horizon is None:
        return ResolutionResult(
            resolved=False,
            reason="Missing value at resolution horizon",
        )

    # Compare values: did it increase?
    value_at_due = resolution_at_due_date.value
    value_at_horizon = resolution_at_horizon.value

    if value_at_horizon > value_at_due:
        resolution_value = 1.0
    elif value_at_horizon < value_at_due:
        resolution_value = 0.0
    else:
        # Exactly equal - ambiguous
        return ResolutionResult(
            resolved=False,
            reason=f"Value unchanged: {value_at_due}",
        )

    return ResolutionResult(
        resolved=True,
        resolution_value=resolution_value,
        resolution_date=resolution_at_horizon.date,
    )


def resolve_question(
    question: Question,
    resolution: Resolution | None,
    forecast_due_date: date,
    resolution_at_due_date: Resolution | None = None,
) -> ResolutionResult:
    """Resolve a question based on its source type.

    Args:
        question: The question to resolve
        resolution: Resolution data at the evaluation horizon
        forecast_due_date: When forecasts were due
        resolution_at_due_date: For data questions, the value at forecast due date

    Returns:
        ResolutionResult with resolution status and value
    """
    if question.source_type == SourceType.MARKET:
        return resolve_market_question(question, resolution, forecast_due_date)
    else:
        return resolve_data_question(question, resolution_at_due_date, resolution)


def score_forecast(
    forecast: Forecast,
    resolution_result: ResolutionResult,
) -> float | None:
    """Score a forecast given a resolution result.

    Args:
        forecast: The forecast to score
        resolution_result: The resolution result

    Returns:
        Brier score if scorable, None otherwise
    """
    if not resolution_result.resolved:
        return None

    if forecast.probability is None:
        return None

    if resolution_result.resolution_value is None:
        return None

    return compute_brier_score(forecast.probability, resolution_result.resolution_value)
