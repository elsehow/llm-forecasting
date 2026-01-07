"""Evaluation module for LLM forecasting.

This module provides:
- scoring: Brier score, RMSE, CRPS, log score, statistical significance
- runner: Run agents against question sets (TODO)
- viz: Leaderboards and calibration plots (TODO)
"""

from llm_forecasting.eval.scoring import (
    LeaderboardEntry,
    PairwiseComparison,
    build_leaderboard,
    compute_brier_score,
    compute_confidence_interval,
    compute_log_score,
    compute_mae,
    compute_pairwise_significance,
    compute_rmse,
    compute_std_error,
    format_leaderboard,
    paired_t_test,
)

__all__ = [
    # Scoring functions
    "compute_brier_score",
    "compute_rmse",
    "compute_mae",
    "compute_log_score",
    # Statistics
    "compute_std_error",
    "compute_confidence_interval",
    "paired_t_test",
    # Leaderboard
    "LeaderboardEntry",
    "PairwiseComparison",
    "build_leaderboard",
    "compute_pairwise_significance",
    "format_leaderboard",
]
