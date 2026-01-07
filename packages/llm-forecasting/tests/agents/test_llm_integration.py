"""Integration tests for LLM forecaster (requires API keys).

Run all providers:
    pytest tests/forecasters/test_llm_integration.py -v --integration

Run specific provider:
    pytest tests/forecasters/test_llm_integration.py -v --integration -k anthropic
    pytest tests/forecasters/test_llm_integration.py -v --integration -k openai
    pytest tests/forecasters/test_llm_integration.py -v --integration -k google
    pytest tests/forecasters/test_llm_integration.py -v --integration -k mistral
"""

import os

import pytest

from llm_forecasting.agents.llm import LLMForecaster
from llm_forecasting.models import Question, QuestionType

# Provider configurations: (env_var, model_id, display_name)
PROVIDERS = [
    ("ANTHROPIC_API_KEY", "anthropic/claude-3-haiku-20240307", "anthropic"),
    ("OPENAI_API_KEY", "openai/gpt-4o-mini", "openai"),
    ("GOOGLE_API_KEY", "gemini/gemini-2.0-flash", "google"),
    ("MISTRAL_API_KEY", "mistral/mistral-small-latest", "mistral"),
    ("TOGETHERAI_API_KEY", "together_ai/meta-llama/Llama-3-8b-chat-hf", "togetherai"),
    ("XAI_API_KEY", "xai/grok-beta", "xai"),
]


def get_available_providers() -> list[tuple[str, str, str]]:
    """Get list of providers with available API keys."""
    return [(env, model, name) for env, model, name in PROVIDERS if os.environ.get(env)]


@pytest.fixture
def binary_question() -> Question:
    return Question(
        id="test-binary",
        source="test",
        text="Will artificial general intelligence (AGI) be achieved by 2030?",
        background="AGI refers to AI systems that can perform any intellectual task a human can.",
        question_type=QuestionType.BINARY,
    )


@pytest.fixture
def continuous_question() -> Question:
    return Question(
        id="test-continuous",
        source="test",
        text="What will be the global average temperature anomaly in 2025 (in Celsius)?",
        background="The 2024 anomaly was approximately 1.45°C above pre-industrial levels.",
        question_type=QuestionType.CONTINUOUS,
        value_range=(1.0, 2.0),
    )


@pytest.fixture
def quantile_question() -> Question:
    return Question(
        id="test-quantile",
        source="test",
        text="What will be the US unemployment rate in December 2025?",
        background="Current unemployment is around 4.1%.",
        question_type=QuestionType.QUANTILE,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        value_range=(2.0, 10.0),
    )


# =============================================================================
# Per-provider tests - each provider gets its own test class
# =============================================================================


@pytest.mark.integration
class TestAnthropicForecaster:
    """Integration tests for Anthropic Claude."""

    MODEL = "anthropic/claude-3-haiku-20240307"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_binary_forecast(self, binary_question: Question):
        """Test binary forecast with Anthropic."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(binary_question)

        assert forecast.question_id == binary_question.id
        assert forecast.forecaster == self.MODEL
        assert forecast.probability is not None
        assert 0 <= forecast.probability <= 1
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_continuous_forecast(self, continuous_question: Question):
        """Test continuous forecast with Anthropic."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(continuous_question)

        assert forecast.point_estimate is not None
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_quantile_forecast(self, quantile_question: Question):
        """Test quantile forecast with Anthropic."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(quantile_question)

        assert forecast.quantile_values is not None
        assert len(forecast.quantile_values) == 5


@pytest.mark.integration
class TestOpenAIForecaster:
    """Integration tests for OpenAI GPT."""

    MODEL = "openai/gpt-4o-mini"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_binary_forecast(self, binary_question: Question):
        """Test binary forecast with OpenAI."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(binary_question)

        assert forecast.question_id == binary_question.id
        assert forecast.forecaster == self.MODEL
        assert forecast.probability is not None
        assert 0 <= forecast.probability <= 1
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_continuous_forecast(self, continuous_question: Question):
        """Test continuous forecast with OpenAI."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(continuous_question)

        assert forecast.point_estimate is not None
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_quantile_forecast(self, quantile_question: Question):
        """Test quantile forecast with OpenAI."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(quantile_question)

        assert forecast.quantile_values is not None
        assert len(forecast.quantile_values) == 5


@pytest.mark.integration
class TestGoogleForecaster:
    """Integration tests for Google Gemini."""

    MODEL = "gemini/gemini-2.0-flash"

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    )
    async def test_binary_forecast(self, binary_question: Question):
        """Test binary forecast with Google."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(binary_question)

        assert forecast.question_id == binary_question.id
        assert forecast.forecaster == self.MODEL
        assert forecast.probability is not None
        assert 0 <= forecast.probability <= 1
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    )
    async def test_continuous_forecast(self, continuous_question: Question):
        """Test continuous forecast with Google."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(continuous_question)

        assert forecast.point_estimate is not None
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    )
    async def test_quantile_forecast(self, quantile_question: Question):
        """Test quantile forecast with Google."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(quantile_question)

        assert forecast.quantile_values is not None
        assert len(forecast.quantile_values) == 5


@pytest.mark.integration
class TestMistralForecaster:
    """Integration tests for Mistral AI."""

    MODEL = "mistral/mistral-small-latest"

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set",
    )
    async def test_binary_forecast(self, binary_question: Question):
        """Test binary forecast with Mistral."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(binary_question)

        assert forecast.question_id == binary_question.id
        assert forecast.forecaster == self.MODEL
        assert forecast.probability is not None
        assert 0 <= forecast.probability <= 1
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set",
    )
    async def test_continuous_forecast(self, continuous_question: Question):
        """Test continuous forecast with Mistral."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(continuous_question)

        assert forecast.point_estimate is not None
        assert forecast.reasoning is not None

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set",
    )
    async def test_quantile_forecast(self, quantile_question: Question):
        """Test quantile forecast with Mistral."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(quantile_question)

        assert forecast.quantile_values is not None
        assert len(forecast.quantile_values) == 5


@pytest.mark.integration
class TestTogetherAIForecaster:
    """Integration tests for Together AI."""

    MODEL = "together_ai/meta-llama/Llama-3-8b-chat-hf"

    @pytest.mark.skipif(
        not os.environ.get("TOGETHERAI_API_KEY"),
        reason="TOGETHERAI_API_KEY not set",
    )
    async def test_binary_forecast(self, binary_question: Question):
        """Test binary forecast with Together AI."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(binary_question)

        assert forecast.question_id == binary_question.id
        assert forecast.forecaster == self.MODEL
        assert forecast.probability is not None
        assert 0 <= forecast.probability <= 1
        assert forecast.reasoning is not None


@pytest.mark.integration
class TestXAIForecaster:
    """Integration tests for xAI Grok."""

    MODEL = "xai/grok-beta"

    @pytest.mark.skipif(
        not os.environ.get("XAI_API_KEY"),
        reason="XAI_API_KEY not set",
    )
    async def test_binary_forecast(self, binary_question: Question):
        """Test binary forecast with xAI."""
        forecaster = LLMForecaster(model=self.MODEL)
        forecast = await forecaster.forecast(binary_question)

        assert forecast.question_id == binary_question.id
        assert forecast.forecaster == self.MODEL
        assert forecast.probability is not None
        assert 0 <= forecast.probability <= 1
        assert forecast.reasoning is not None


# =============================================================================
# Cross-provider tests
# =============================================================================


@pytest.mark.integration
class TestCrossProviderConsistency:
    """Tests that run across all available providers."""

    @pytest.mark.skipif(
        len(get_available_providers()) == 0,
        reason="No LLM API keys available",
    )
    async def test_all_providers_return_valid_binary_forecast(self, binary_question: Question):
        """All available providers should return valid binary forecasts."""
        providers = get_available_providers()
        results = {}

        for env_var, model, name in providers:
            forecaster = LLMForecaster(model=model)
            forecast = await forecaster.forecast(binary_question)

            assert 0 <= forecast.probability <= 1, f"{name} returned invalid probability"
            assert forecast.reasoning, f"{name} returned no reasoning"
            results[name] = forecast.probability

        # Log results for comparison
        print(f"\nBinary forecast results across {len(results)} providers:")
        for name, prob in results.items():
            print(f"  {name}: {prob:.2%}")

    @pytest.mark.skipif(
        len(get_available_providers()) == 0,
        reason="No LLM API keys available",
    )
    async def test_extreme_question_calibration(self):
        """Providers with structured output support should give high probability to near-certain events."""
        extreme_question = Question(
            id="extreme-test",
            source="test",
            text="Will the sun rise tomorrow?",
            background="The sun has risen every day for billions of years.",
            question_type=QuestionType.BINARY,
        )

        providers = get_available_providers()
        results = {}
        skipped = []

        for env_var, model, name in providers:
            forecaster = LLMForecaster(model=model)
            forecast = await forecaster.forecast(extreme_question)

            # Skip providers that errored (e.g., don't support structured outputs)
            if forecast.reasoning and forecast.reasoning.startswith("Error:"):
                skipped.append((name, forecast.reasoning))
                continue

            assert 0 <= forecast.probability <= 1, f"{name} returned invalid probability"
            # All models should give > 90% for sun rising
            assert forecast.probability > 0.9, f"{name} gave surprisingly low probability: {forecast.probability}"
            results[name] = forecast.probability

        print(f"\nExtreme question results across {len(results)} providers:")
        for name, prob in results.items():
            print(f"  {name}: {prob:.2%}")
        if skipped:
            print(f"\nSkipped {len(skipped)} providers (no structured output support):")
            for name, reason in skipped:
                print(f"  {name}: {reason[:80]}...")
