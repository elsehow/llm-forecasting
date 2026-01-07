"""Unit tests for scoring and statistical significance."""

import math

import pytest

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


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_prediction_yes(self):
        """Perfect prediction of YES outcome."""
        assert compute_brier_score(1.0, 1.0) == 0.0

    def test_perfect_prediction_no(self):
        """Perfect prediction of NO outcome."""
        assert compute_brier_score(0.0, 0.0) == 0.0

    def test_worst_prediction_yes(self):
        """Completely wrong prediction of YES outcome."""
        assert compute_brier_score(0.0, 1.0) == 1.0

    def test_worst_prediction_no(self):
        """Completely wrong prediction of NO outcome."""
        assert compute_brier_score(1.0, 0.0) == 1.0

    def test_uncertain_prediction(self):
        """50% prediction scores 0.25 regardless of outcome."""
        assert compute_brier_score(0.5, 0.0) == 0.25
        assert compute_brier_score(0.5, 1.0) == 0.25

    def test_typical_prediction(self):
        """Typical prediction scenario."""
        # 70% confidence, outcome YES
        assert compute_brier_score(0.7, 1.0) == pytest.approx(0.09)
        # 70% confidence, outcome NO
        assert compute_brier_score(0.7, 0.0) == pytest.approx(0.49)


class TestRMSE:
    """Tests for Root Mean Square Error calculation."""

    def test_perfect_predictions(self):
        """Perfect predictions have RMSE of 0."""
        predictions = [1.0, 2.0, 3.0]
        actuals = [1.0, 2.0, 3.0]
        assert compute_rmse(predictions, actuals) == 0.0

    def test_single_value(self):
        """RMSE of single value equals absolute error."""
        assert compute_rmse([5.0], [3.0]) == 2.0
        assert compute_rmse([3.0], [5.0]) == 2.0

    def test_known_calculation(self):
        """RMSE calculation with known values."""
        # errors: [1, 2, 3], squared: [1, 4, 9], mean: 14/3, sqrt: ~2.16
        predictions = [1.0, 2.0, 3.0]
        actuals = [0.0, 0.0, 0.0]
        expected = math.sqrt((1 + 4 + 9) / 3)
        assert compute_rmse(predictions, actuals) == pytest.approx(expected)

    def test_different_length_raises(self):
        """Different length lists raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_rmse([1.0, 2.0], [1.0])

    def test_empty_lists_raises(self):
        """Empty lists raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_rmse([], [])


class TestMAE:
    """Tests for Mean Absolute Error calculation."""

    def test_perfect_predictions(self):
        """Perfect predictions have MAE of 0."""
        predictions = [1.0, 2.0, 3.0]
        actuals = [1.0, 2.0, 3.0]
        assert compute_mae(predictions, actuals) == 0.0

    def test_single_value(self):
        """MAE of single value equals absolute error."""
        assert compute_mae([5.0], [3.0]) == 2.0

    def test_known_calculation(self):
        """MAE calculation with known values."""
        # errors: [1, 2, 3], mean: 2.0
        predictions = [1.0, 2.0, 3.0]
        actuals = [0.0, 0.0, 0.0]
        assert compute_mae(predictions, actuals) == 2.0

    def test_symmetric(self):
        """MAE is symmetric (under vs over prediction same)."""
        assert compute_mae([5.0], [3.0]) == compute_mae([3.0], [5.0])

    def test_different_length_raises(self):
        """Different length lists raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_mae([1.0, 2.0], [1.0])


class TestLogScore:
    """Tests for log score calculation."""

    def test_perfect_prediction_yes(self):
        """Perfect prediction of YES has log score near 0."""
        # p=1.0 should give -log(1) = 0
        score = compute_log_score(0.9999, 1.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_perfect_prediction_no(self):
        """Perfect prediction of NO has log score near 0."""
        # p=0.0 should give -log(1) = 0
        score = compute_log_score(0.0001, 0.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_worst_prediction_yes(self):
        """Wrong prediction of YES has high log score."""
        # p=0.01 when outcome=1 should be high
        score = compute_log_score(0.01, 1.0)
        assert score > 4.0  # -log(0.01) ≈ 4.6

    def test_worst_prediction_no(self):
        """Wrong prediction of NO has high log score."""
        # p=0.99 when outcome=0 should be high
        score = compute_log_score(0.99, 0.0)
        assert score > 4.0

    def test_uncertain_prediction(self):
        """50% prediction has log score of ln(2) ≈ 0.693."""
        score_yes = compute_log_score(0.5, 1.0)
        score_no = compute_log_score(0.5, 0.0)
        assert score_yes == pytest.approx(0.693, rel=0.01)
        assert score_no == pytest.approx(0.693, rel=0.01)


class TestStdError:
    """Tests for standard error calculation."""

    def test_single_value_returns_zero(self):
        """Single value has zero standard error."""
        assert compute_std_error([0.5]) == 0.0

    def test_empty_list_returns_zero(self):
        """Empty list handled gracefully."""
        assert compute_std_error([]) == 0.0

    def test_identical_values(self):
        """Identical values have zero standard error."""
        assert compute_std_error([0.5, 0.5, 0.5, 0.5]) == 0.0

    def test_basic_calculation(self):
        """Standard error calculation with known values."""
        # scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        # mean = 0.3, std = 0.1581, se = 0.1581 / sqrt(5) = 0.0707
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        se = compute_std_error(scores)
        assert se == pytest.approx(0.0707, rel=0.01)

    def test_decreases_with_more_samples(self):
        """Standard error decreases with more samples."""
        scores_small = [0.1, 0.3, 0.5]
        scores_large = [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4]

        se_small = compute_std_error(scores_small)
        se_large = compute_std_error(scores_large)

        assert se_large < se_small


class TestConfidenceInterval:
    """Tests for confidence interval calculation."""

    def test_zero_error_gives_point_estimate(self):
        """Zero standard error gives CI of just the mean."""
        low, high = compute_confidence_interval(0.5, 0.0)
        assert low == 0.5
        assert high == 0.5

    def test_symmetric_around_mean(self):
        """CI is symmetric around mean."""
        mean = 0.5
        se = 0.1
        low, high = compute_confidence_interval(mean, se)
        assert (high - mean) == pytest.approx(mean - low)

    def test_95_confidence(self):
        """95% CI uses ~1.96 standard errors."""
        mean = 0.5
        se = 0.1
        low, high = compute_confidence_interval(mean, se, confidence=0.95)
        # margin should be 1.96 * 0.1 = 0.196
        assert (high - low) == pytest.approx(0.392, rel=0.01)


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_identical_scores_p_value_one(self):
        """Identical scores have p-value of 1."""
        scores_a = [0.1, 0.2, 0.3, 0.4]
        scores_b = [0.1, 0.2, 0.3, 0.4]
        p = paired_t_test(scores_a, scores_b)
        assert p == 1.0

    def test_different_length_returns_none(self):
        """Different length lists return None."""
        scores_a = [0.1, 0.2, 0.3]
        scores_b = [0.1, 0.2]
        assert paired_t_test(scores_a, scores_b) is None

    def test_very_different_scores_low_p_value(self):
        """Very different scores have low p-value."""
        # A is much worse (higher Brier)
        scores_a = [0.8, 0.9, 0.7, 0.8, 0.9, 0.85]
        scores_b = [0.1, 0.2, 0.15, 0.1, 0.2, 0.15]
        p = paired_t_test(scores_a, scores_b)
        assert p is not None
        assert p < 0.01  # Should be highly significant

    def test_similar_scores_high_p_value(self):
        """Similar scores have high p-value."""
        scores_a = [0.3, 0.35, 0.32, 0.28, 0.31]
        scores_b = [0.32, 0.33, 0.30, 0.29, 0.34]
        p = paired_t_test(scores_a, scores_b)
        assert p is not None
        assert p > 0.1  # Not significant

    def test_single_pair_returns_none(self):
        """Single pair of scores cannot compute t-test."""
        assert paired_t_test([0.5], [0.3]) is None


class TestBuildLeaderboard:
    """Tests for building leaderboard."""

    def test_empty_scores(self):
        """Empty scores dict gives empty leaderboard."""
        entries = build_leaderboard({})
        assert entries == []

    def test_single_forecaster(self):
        """Single forecaster gets an entry."""
        scores = {"model-a": [0.1, 0.2, 0.3]}
        entries = build_leaderboard(scores)
        assert len(entries) == 1
        assert entries[0].forecaster == "model-a"
        assert entries[0].mean_brier_score == pytest.approx(0.2)
        assert entries[0].num_forecasts == 3

    def test_sorted_by_score(self):
        """Leaderboard sorted by mean Brier score (lower is better)."""
        scores = {
            "bad-model": [0.5, 0.6, 0.7],
            "good-model": [0.1, 0.1, 0.1],
            "mid-model": [0.3, 0.3, 0.3],
        }
        entries = build_leaderboard(scores)
        assert entries[0].forecaster == "good-model"
        assert entries[1].forecaster == "mid-model"
        assert entries[2].forecaster == "bad-model"

    def test_includes_confidence_intervals(self):
        """Confidence intervals computed when requested."""
        scores = {"model-a": [0.1, 0.2, 0.3, 0.4, 0.5]}
        entries = build_leaderboard(scores, with_confidence=True)
        assert entries[0].std_error is not None
        assert entries[0].confidence_interval_95 is not None

    def test_skips_confidence_when_disabled(self):
        """No confidence when disabled."""
        scores = {"model-a": [0.1, 0.2, 0.3]}
        entries = build_leaderboard(scores, with_confidence=False)
        assert entries[0].std_error is None


class TestPairwiseSignificance:
    """Tests for pairwise significance testing."""

    def test_single_forecaster_no_comparisons(self):
        """Single forecaster has no pairwise comparisons."""
        scores = {"model-a": [0.1, 0.2, 0.3]}
        comps = compute_pairwise_significance(scores, ["q1", "q2", "q3"])
        assert comps == []

    def test_two_forecasters_one_comparison(self):
        """Two forecasters have one comparison."""
        scores = {
            "model-a": [0.1, 0.2, 0.3],
            "model-b": [0.4, 0.5, 0.6],
        }
        comps = compute_pairwise_significance(scores, ["q1", "q2", "q3"])
        assert len(comps) == 1
        assert comps[0].forecaster_a == "model-a"
        assert comps[0].forecaster_b == "model-b"

    def test_three_forecasters_three_comparisons(self):
        """Three forecasters have three pairwise comparisons."""
        scores = {
            "a": [0.1, 0.2],
            "b": [0.2, 0.3],
            "c": [0.3, 0.4],
        }
        comps = compute_pairwise_significance(scores, ["q1", "q2"])
        assert len(comps) == 3

    def test_detects_significant_difference(self):
        """Detects statistically significant differences."""
        # Model A is much better
        scores = {
            "good": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "bad": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        }
        comps = compute_pairwise_significance(scores, ["q1", "q2", "q3", "q4", "q5", "q6"])
        assert comps[0].is_significant is True
        assert comps[0].p_value < 0.05


class TestFormatLeaderboard:
    """Tests for leaderboard formatting."""

    def test_empty_leaderboard(self):
        """Empty leaderboard formats without error."""
        output = format_leaderboard([])
        assert "Leaderboard" in output

    def test_includes_forecaster_names(self):
        """Output includes forecaster names."""
        entries = [
            LeaderboardEntry(
                forecaster="claude-3-opus",
                mean_brier_score=0.15,
                num_forecasts=100,
                std_error=0.02,
                confidence_interval_95=(0.11, 0.19),
            )
        ]
        output = format_leaderboard(entries)
        assert "claude-3-opus" in output
        assert "0.15" in output

    def test_includes_significance_notes(self):
        """Output includes significance when provided."""
        entries = [
            LeaderboardEntry("a", 0.1, 10, None, None),
            LeaderboardEntry("b", 0.3, 10, None, None),
        ]
        comps = [
            PairwiseComparison("a", "b", -0.2, 0.01, True),
        ]
        output = format_leaderboard(entries, comps)
        assert "Significant" in output
        assert "a beats b" in output
