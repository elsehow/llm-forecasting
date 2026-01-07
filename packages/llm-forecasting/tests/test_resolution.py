"""Unit tests for resolution logic."""

from datetime import date, datetime, timezone

import pytest

from llm_forecasting.models import Forecast, Question, QuestionType, Resolution, SourceType
from llm_forecasting.resolution import (
    ResolutionResult,
    is_valid_resolution,
    resolve_data_question,
    resolve_market_question,
    resolve_question,
    score_forecast,
)
from llm_forecasting.eval.scoring import compute_brier_score


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


class TestValidResolution:
    """Tests for resolution value validation."""

    def test_valid_zero(self):
        assert is_valid_resolution(0.0) is True

    def test_valid_one(self):
        assert is_valid_resolution(1.0) is True

    def test_none_invalid(self):
        assert is_valid_resolution(None) is False

    def test_ambiguous_invalid(self):
        """Values like 0.5 (market resolved at market price) are invalid."""
        assert is_valid_resolution(0.5) is False
        assert is_valid_resolution(0.73) is False


class TestResolveMarketQuestion:
    """Tests for market question resolution."""

    @pytest.fixture
    def market_question(self) -> Question:
        return Question(
            id="market-1",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Will X happen?",
            created_at=datetime.now(timezone.utc),
            resolved=False,
        )

    @pytest.fixture
    def resolved_question(self) -> Question:
        return Question(
            id="market-2",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Will Y happen?",
            created_at=datetime.now(timezone.utc),
            resolved=True,
            resolution_value=1.0,
            resolution_date=date(2025, 2, 15),
        )

    def test_no_resolution_data(self, market_question: Question):
        """Missing resolution data returns unresolved."""
        result = resolve_market_question(
            market_question,
            resolution=None,
            forecast_due_date=date(2025, 1, 15),
        )
        assert result.resolved is False
        assert "No resolution data" in result.reason

    def test_unresolved_uses_market_probability(self, market_question: Question):
        """Unresolved market uses current probability as interim resolution."""
        resolution = Resolution(
            question_id="market-1",
            source="manifold",
            date=date(2025, 1, 20),
            value=0.73,
        )
        result = resolve_market_question(
            market_question,
            resolution=resolution,
            forecast_due_date=date(2025, 1, 15),
        )
        assert result.resolved is True
        assert result.resolution_value == 0.73
        assert "interim resolution" in result.reason

    def test_resolved_question_uses_final_value(self, resolved_question: Question):
        """Resolved market uses final resolution value."""
        resolution = Resolution(
            question_id="market-2",
            source="manifold",
            date=date(2025, 2, 15),
            value=1.0,
        )
        result = resolve_market_question(
            resolved_question,
            resolution=resolution,
            forecast_due_date=date(2025, 1, 15),
        )
        assert result.resolved is True
        assert result.resolution_value == 1.0
        assert result.resolution_date == date(2025, 2, 15)

    def test_resolved_before_forecast_due_date_skipped(self):
        """Question that resolved before forecasts were due is skipped."""
        question = Question(
            id="early-resolve",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Already resolved",
            created_at=datetime.now(timezone.utc),
            resolved=True,
            resolution_value=1.0,
            resolution_date=date(2025, 1, 10),  # Before due date
        )
        resolution = Resolution(
            question_id="early-resolve",
            source="manifold",
            date=date(2025, 1, 10),
            value=1.0,
        )
        result = resolve_market_question(
            question,
            resolution=resolution,
            forecast_due_date=date(2025, 1, 15),  # Due date is after resolution
        )
        assert result.resolved is False
        assert "before forecast due" in result.reason

    def test_ambiguous_resolution_skipped(self):
        """Ambiguous resolution (e.g., MKT at 0.5) is skipped."""
        question = Question(
            id="ambiguous",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Resolved ambiguously",
            created_at=datetime.now(timezone.utc),
            resolved=True,
            resolution_value=0.5,  # Market resolved at MKT
            resolution_date=date(2025, 2, 15),
        )
        resolution = Resolution(
            question_id="ambiguous",
            source="manifold",
            date=date(2025, 2, 15),
            value=0.5,
        )
        result = resolve_market_question(
            question,
            resolution=resolution,
            forecast_due_date=date(2025, 1, 15),
        )
        assert result.resolved is False
        assert "Ambiguous" in result.reason


class TestResolveDataQuestion:
    """Tests for data-based question resolution."""

    @pytest.fixture
    def data_question(self) -> Question:
        return Question(
            id="UNRATE",
            source="fred",
            source_type=SourceType.DATA,
            text="Will unemployment rate increase?",
            created_at=datetime.now(timezone.utc),
        )

    def test_value_increased(self, data_question: Question):
        """Value increased -> resolution = 1."""
        at_due = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 1, 15), value=4.0
        )
        at_horizon = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 2, 15), value=4.5
        )
        result = resolve_data_question(data_question, at_due, at_horizon)
        assert result.resolved is True
        assert result.resolution_value == 1.0

    def test_value_decreased(self, data_question: Question):
        """Value decreased -> resolution = 0."""
        at_due = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 1, 15), value=4.5
        )
        at_horizon = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 2, 15), value=4.0
        )
        result = resolve_data_question(data_question, at_due, at_horizon)
        assert result.resolved is True
        assert result.resolution_value == 0.0

    def test_value_unchanged_ambiguous(self, data_question: Question):
        """Value unchanged -> ambiguous, not resolved."""
        at_due = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 1, 15), value=4.0
        )
        at_horizon = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 2, 15), value=4.0
        )
        result = resolve_data_question(data_question, at_due, at_horizon)
        assert result.resolved is False
        assert "unchanged" in result.reason

    def test_missing_due_date_value(self, data_question: Question):
        """Missing value at due date -> not resolved."""
        at_horizon = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 2, 15), value=4.5
        )
        result = resolve_data_question(data_question, None, at_horizon)
        assert result.resolved is False
        assert "due date" in result.reason

    def test_missing_horizon_value(self, data_question: Question):
        """Missing value at horizon -> not resolved."""
        at_due = Resolution(
            question_id="UNRATE", source="fred", date=date(2025, 1, 15), value=4.0
        )
        result = resolve_data_question(data_question, at_due, None)
        assert result.resolved is False
        assert "horizon" in result.reason


class TestResolveQuestion:
    """Tests for the unified resolve_question function."""

    def test_dispatches_to_market(self):
        """Market questions use market resolution logic."""
        question = Question(
            id="m1",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Market question",
            created_at=datetime.now(timezone.utc),
        )
        resolution = Resolution(
            question_id="m1", source="manifold", date=date(2025, 1, 20), value=0.8
        )
        result = resolve_question(
            question, resolution, forecast_due_date=date(2025, 1, 15)
        )
        assert result.resolved is True
        assert result.resolution_value == 0.8

    def test_dispatches_to_data(self):
        """Data questions use data resolution logic."""
        question = Question(
            id="AAPL",
            source="yfinance",
            source_type=SourceType.DATA,
            text="Will AAPL increase?",
            created_at=datetime.now(timezone.utc),
        )
        at_due = Resolution(
            question_id="AAPL", source="yfinance", date=date(2025, 1, 15), value=150.0
        )
        at_horizon = Resolution(
            question_id="AAPL", source="yfinance", date=date(2025, 2, 15), value=160.0
        )
        result = resolve_question(
            question,
            at_horizon,
            forecast_due_date=date(2025, 1, 15),
            resolution_at_due_date=at_due,
        )
        assert result.resolved is True
        assert result.resolution_value == 1.0


class TestScoreForecast:
    """Tests for forecast scoring."""

    def test_scores_resolved_forecast(self):
        """Successfully scores a resolved forecast."""
        forecast = Forecast(
            question_id="q1",
            source="test",
            forecaster="model",
            probability=0.7,
        )
        result = ResolutionResult(resolved=True, resolution_value=1.0)
        score = score_forecast(forecast, result)
        assert score == pytest.approx(0.09)

    def test_returns_none_for_unresolved(self):
        """Returns None for unresolved forecasts."""
        forecast = Forecast(
            question_id="q1",
            source="test",
            forecaster="model",
            probability=0.7,
        )
        result = ResolutionResult(resolved=False, reason="Not resolved")
        score = score_forecast(forecast, result)
        assert score is None

    def test_returns_none_for_missing_probability(self):
        """Returns None when forecast has no probability."""
        forecast = Forecast(
            question_id="q1",
            source="test",
            forecaster="model",
            point_estimate=100.0,  # Continuous, not binary
        )
        result = ResolutionResult(resolved=True, resolution_value=1.0)
        score = score_forecast(forecast, result)
        assert score is None
