"""Visualization utilities for evaluation results.

This module provides:
- Calibration plots
- Score distributions
- Leaderboard rendering

TODO: Implement
- plot_calibration(forecasts, resolutions) -> matplotlib figure
- plot_score_distribution(scores) -> matplotlib figure
- render_leaderboard_html(entries) -> str
"""


def plot_calibration(
    probabilities: list[float],
    outcomes: list[float],
    bins: int = 10,
) -> None:
    """Plot calibration curve.

    TODO: Implement with matplotlib
    """
    raise NotImplementedError("Calibration plot not yet implemented")


def plot_score_distribution(
    scores_by_forecaster: dict[str, list[float]],
) -> None:
    """Plot score distributions for multiple forecasters.

    TODO: Implement with matplotlib
    """
    raise NotImplementedError("Score distribution plot not yet implemented")
