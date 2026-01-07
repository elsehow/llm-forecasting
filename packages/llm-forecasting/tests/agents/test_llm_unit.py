"""Unit tests for LLM forecaster (no API calls)."""

import pytest

from llm_forecasting.agents.llm import (
    BinaryForecastResponse,
    ContinuousForecastResponse,
    LLMForecaster,
    QuantileForecastResponse,
)
from llm_forecasting.models import Question, QuestionType


class TestForecastResponses:
    """Tests for the structured forecast response models."""

    def test_binary_valid_probability(self):
        """Create binary response with valid probability."""
        resp = BinaryForecastResponse(probability=0.75, reasoning="Test reasoning")
        assert resp.probability == 0.75

    def test_binary_probability_bounds(self):
        """Probability must be between 0 and 1."""
        with pytest.raises(ValueError):
            BinaryForecastResponse(probability=1.5, reasoning="Too high")
        with pytest.raises(ValueError):
            BinaryForecastResponse(probability=-0.1, reasoning="Too low")

    def test_binary_parse_json(self):
        """Parse JSON response."""
        json_str = '{"probability": 0.65, "reasoning": "Based on analysis..."}'
        resp = BinaryForecastResponse.model_validate_json(json_str)
        assert resp.probability == 0.65
        assert "analysis" in resp.reasoning

    def test_continuous_response(self):
        """Create continuous forecast response."""
        resp = ContinuousForecastResponse(
            point_estimate=4950.5,
            confidence_low=4800.0,
            confidence_high=5100.0,
            reasoning="Market trend analysis",
        )
        assert resp.point_estimate == 4950.5

    def test_quantile_response(self):
        """Create quantile forecast response."""
        resp = QuantileForecastResponse(
            quantile_values=[4700.0, 4850.0, 4950.0, 5050.0, 5200.0],
            reasoning="Distribution estimate",
        )
        assert len(resp.quantile_values) == 5
        assert resp.quantile_values[2] == 4950.0  # Median


class TestLLMForecasterPrompts:
    """Tests for LLM forecaster prompt building."""

    @pytest.fixture
    def binary_question(self) -> Question:
        return Question(
            id="test-q",
            source="test",
            text="Will the S&P 500 close above 5000 by end of January 2025?",
            background="The S&P 500 is currently trading around 4900.",
            question_type=QuestionType.BINARY,
        )

    @pytest.fixture
    def continuous_question(self) -> Question:
        return Question(
            id="test-cont",
            source="test",
            text="What will be the S&P 500 closing price on January 31, 2025?",
            background="The S&P 500 is currently trading around 4900.",
            question_type=QuestionType.CONTINUOUS,
            value_range=(4000.0, 6000.0),
        )

    @pytest.fixture
    def quantile_question(self) -> Question:
        return Question(
            id="test-quant",
            source="test",
            text="What will be the S&P 500 closing price on January 31, 2025?",
            background="The S&P 500 is currently trading around 4900.",
            question_type=QuestionType.QUANTILE,
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            value_range=(4000.0, 6000.0),
        )

    def test_forecaster_name(self):
        """Forecaster name matches model."""
        forecaster = LLMForecaster(model="openai/gpt-4o-mini")
        assert forecaster.name == "openai/gpt-4o-mini"

    def test_build_prompt_binary(self, binary_question: Question):
        """Verify binary prompt is built correctly."""
        forecaster = LLMForecaster()
        prompt, schema = forecaster._build_prompt(binary_question)

        assert "S&P 500" in prompt
        assert "5000" in prompt
        assert "superforecaster" in prompt
        assert "probability" in prompt.lower()
        assert schema == BinaryForecastResponse

    def test_build_prompt_continuous(self, continuous_question: Question):
        """Verify continuous prompt is built correctly."""
        forecaster = LLMForecaster()
        prompt, schema = forecaster._build_prompt(continuous_question)

        assert "S&P 500" in prompt
        assert "point estimate" in prompt.lower()
        assert "4000" in prompt  # Value range
        assert schema == ContinuousForecastResponse

    def test_build_prompt_quantile(self, quantile_question: Question):
        """Verify quantile prompt is built correctly."""
        forecaster = LLMForecaster()
        prompt, schema = forecaster._build_prompt(quantile_question)

        assert "S&P 500" in prompt
        assert "quantile" in prompt.lower()
        assert "10%" in prompt  # Quantile values
        assert "50%" in prompt
        assert schema == QuantileForecastResponse
