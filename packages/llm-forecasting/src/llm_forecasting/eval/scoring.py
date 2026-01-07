"""Scoring functions and statistical significance for leaderboard.

This module provides:
1. Pure scoring functions (Brier, RMSE, CRPS)
2. Aggregate score calculations
3. Statistical significance testing between forecasters
4. Confidence intervals for scores

Scores are computed on-the-fly from stored forecasts and resolutions,
not persisted to the database. This allows easy addition of new scoring
methods without schema migrations.
"""

import math
from dataclasses import dataclass


# =============================================================================
# Core Scoring Functions (Pure)
# =============================================================================


def compute_brier_score(probability: float, outcome: float) -> float:
    """Compute Brier score for a binary forecast.

    Brier score = (probability - outcome)^2

    Args:
        probability: Forecasted probability (0-1)
        outcome: Actual outcome (0 or 1)

    Returns:
        Brier score (0-1, lower is better)
            - 0.0 = perfect prediction
            - 0.25 = maximum uncertainty (p=0.5)
            - 1.0 = completely wrong
    """
    return (probability - outcome) ** 2


def compute_rmse(predictions: list[float], actuals: list[float]) -> float:
    """Compute Root Mean Square Error for continuous predictions.

    RMSE = sqrt(mean((prediction - actual)^2))

    Args:
        predictions: List of predicted values
        actuals: List of actual values (same length as predictions)

    Returns:
        RMSE (lower is better, 0 = perfect)

    Raises:
        ValueError: If lists have different lengths or are empty
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")
    if len(predictions) == 0:
        raise ValueError("Cannot compute RMSE on empty lists")

    squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compute_mae(predictions: list[float], actuals: list[float]) -> float:
    """Compute Mean Absolute Error for continuous predictions.

    MAE = mean(|prediction - actual|)

    Args:
        predictions: List of predicted values
        actuals: List of actual values (same length as predictions)

    Returns:
        MAE (lower is better, 0 = perfect)

    Raises:
        ValueError: If lists have different lengths or are empty
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")
    if len(predictions) == 0:
        raise ValueError("Cannot compute MAE on empty lists")

    absolute_errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    return sum(absolute_errors) / len(absolute_errors)


def compute_log_score(probability: float, outcome: float, epsilon: float = 1e-15) -> float:
    """Compute log score (negative log likelihood) for a binary forecast.

    Log score = -log(p) if outcome=1, -log(1-p) if outcome=0

    Args:
        probability: Forecasted probability (0-1)
        outcome: Actual outcome (0 or 1)
        epsilon: Small value to avoid log(0)

    Returns:
        Log score (lower is better, 0 = perfect certainty)
    """
    # Clamp probability to avoid log(0)
    p = max(epsilon, min(1 - epsilon, probability))

    if outcome >= 0.5:  # Treat as 1
        return -math.log(p)
    else:  # Treat as 0
        return -math.log(1 - p)


# =============================================================================
# Leaderboard and Statistics
# =============================================================================


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""

    forecaster: str
    mean_brier_score: float
    num_forecasts: int
    std_error: float | None = None
    confidence_interval_95: tuple[float, float] | None = None


@dataclass
class PairwiseComparison:
    """Statistical comparison between two forecasters."""

    forecaster_a: str
    forecaster_b: str
    score_diff: float  # a - b (negative means a is better)
    p_value: float | None
    is_significant: bool  # at alpha=0.05


def compute_std_error(scores: list[float]) -> float:
    """Compute standard error of the mean for a list of scores.

    SE = std_dev / sqrt(n)

    Args:
        scores: List of Brier scores

    Returns:
        Standard error of the mean
    """
    n = len(scores)
    if n < 2:
        return 0.0

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)  # Sample variance
    std_dev = math.sqrt(variance)
    return std_dev / math.sqrt(n)


def compute_confidence_interval(
    mean: float, std_error: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval for the mean.

    Uses normal approximation (valid for n >= 30).

    Args:
        mean: Sample mean
        std_error: Standard error of the mean
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Z-score for 95% confidence
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    else:
        # Approximate using normal distribution
        z = 1.96

    margin = z * std_error
    return (mean - margin, mean + margin)


def paired_t_test(scores_a: list[float], scores_b: list[float]) -> float | None:
    """Perform paired t-test on two sets of scores.

    Requires scores to be paired (same questions, same order).

    Args:
        scores_a: Brier scores for forecaster A
        scores_b: Brier scores for forecaster B

    Returns:
        p-value, or None if test cannot be performed
    """
    if len(scores_a) != len(scores_b):
        return None

    n = len(scores_a)
    if n < 2:
        return None

    # Compute differences
    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    # Mean and std of differences
    mean_diff = sum(diffs) / n
    if n < 2:
        return None

    variance_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if variance_diff == 0:
        return 1.0 if mean_diff == 0 else 0.0

    std_diff = math.sqrt(variance_diff)
    se_diff = std_diff / math.sqrt(n)

    # t-statistic
    t_stat = mean_diff / se_diff

    # Approximate p-value using normal distribution (valid for large n)
    # For small n, would need t-distribution
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    return p_value


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF.

    Uses the error function approximation.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def build_leaderboard(
    scores_by_forecaster: dict[str, list[float]],
    with_confidence: bool = True,
) -> list[LeaderboardEntry]:
    """Build leaderboard entries from scores.

    Args:
        scores_by_forecaster: Dict mapping forecaster name to list of Brier scores
        with_confidence: Whether to compute confidence intervals

    Returns:
        List of LeaderboardEntry sorted by mean Brier score (best first)
    """
    entries = []

    for forecaster, scores in scores_by_forecaster.items():
        if not scores:
            continue

        mean = sum(scores) / len(scores)
        std_error = compute_std_error(scores) if with_confidence else None
        ci = None
        if std_error is not None and len(scores) >= 2:
            ci = compute_confidence_interval(mean, std_error)

        entries.append(
            LeaderboardEntry(
                forecaster=forecaster,
                mean_brier_score=mean,
                num_forecasts=len(scores),
                std_error=std_error,
                confidence_interval_95=ci,
            )
        )

    # Sort by mean Brier score (lower is better)
    entries.sort(key=lambda e: e.mean_brier_score)
    return entries


def compute_pairwise_significance(
    scores_by_forecaster: dict[str, list[float]],
    question_ids: list[str],
    alpha: float = 0.05,
) -> list[PairwiseComparison]:
    """Compute pairwise statistical significance between forecasters.

    For valid comparison, forecasts must be on the same questions.

    Args:
        scores_by_forecaster: Dict mapping forecaster name to list of Brier scores
        question_ids: List of question IDs in the same order as scores
        alpha: Significance level

    Returns:
        List of PairwiseComparison for each pair
    """
    forecasters = list(scores_by_forecaster.keys())
    comparisons = []

    for i, fa in enumerate(forecasters):
        for fb in forecasters[i + 1 :]:
            scores_a = scores_by_forecaster[fa]
            scores_b = scores_by_forecaster[fb]

            # Compute mean difference
            if not scores_a or not scores_b:
                continue

            mean_a = sum(scores_a) / len(scores_a)
            mean_b = sum(scores_b) / len(scores_b)
            score_diff = mean_a - mean_b

            # Paired t-test (if same length, assumed to be paired)
            p_value = None
            if len(scores_a) == len(scores_b):
                p_value = paired_t_test(scores_a, scores_b)

            is_significant = p_value is not None and p_value < alpha

            comparisons.append(
                PairwiseComparison(
                    forecaster_a=fa,
                    forecaster_b=fb,
                    score_diff=score_diff,
                    p_value=p_value,
                    is_significant=is_significant,
                )
            )

    return comparisons


def format_leaderboard(
    entries: list[LeaderboardEntry],
    comparisons: list[PairwiseComparison] | None = None,
) -> str:
    """Format leaderboard for display.

    Args:
        entries: Leaderboard entries
        comparisons: Optional pairwise comparisons

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("")
    lines.append("🏆 Leaderboard")
    lines.append("")

    # Header
    header = f"{'Rank':<6}{'Forecaster':<35}{'Mean Brier':<12}{'95% CI':<20}{'N':<8}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, entry in enumerate(entries, 1):
        ci_str = ""
        if entry.confidence_interval_95:
            low, high = entry.confidence_interval_95
            ci_str = f"[{low:.4f}, {high:.4f}]"

        line = (
            f"{i:<6}{entry.forecaster:<35}{entry.mean_brier_score:<12.4f}"
            f"{ci_str:<20}{entry.num_forecasts:<8}"
        )
        lines.append(line)

    # Add significance notes if available
    if comparisons:
        sig_comparisons = [c for c in comparisons if c.is_significant]
        if sig_comparisons:
            lines.append("")
            lines.append("Significant differences (p < 0.05):")
            for c in sig_comparisons:
                better = c.forecaster_a if c.score_diff < 0 else c.forecaster_b
                worse = c.forecaster_b if c.score_diff < 0 else c.forecaster_a
                lines.append(f"  • {better} beats {worse} (p={c.p_value:.4f})")

    return "\n".join(lines)
