"""LLM-based forecaster using LiteLLM with structured outputs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import litellm
from pydantic import BaseModel, Field

from llm_forecasting.config import settings
from llm_forecasting.agents.base import Forecaster
from llm_forecasting.models import Forecast, Question, QuestionType

if TYPE_CHECKING:
    from llm_forecasting.registry import ModelInfo

logger = logging.getLogger(__name__)

# Import prompts from shared module
from llm_forecasting.prompts import (
    BINARY_FORECAST_PROMPT as BINARY_PROMPT,
    CONTINUOUS_FORECAST_PROMPT as CONTINUOUS_PROMPT,
    QUANTILE_FORECAST_PROMPT as QUANTILE_PROMPT,
)


# Structured output schemas for each question type


class BinaryForecastResponse(BaseModel):
    """Structured output for binary forecast responses."""

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability estimate between 0 and 1",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class ContinuousForecastResponse(BaseModel):
    """Structured output for continuous forecast responses."""

    point_estimate: float = Field(
        description="Point estimate for the numeric value",
    )
    confidence_low: float | None = Field(
        default=None,
        description="Lower bound of 80% confidence interval (optional)",
    )
    confidence_high: float | None = Field(
        default=None,
        description="Upper bound of 80% confidence interval (optional)",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class QuantileForecastResponse(BaseModel):
    """Structured output for quantile forecast responses."""

    quantile_values: list[float] = Field(
        description="Predicted values for each quantile, in the same order as requested",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class LLMForecaster(Forecaster):
    """Forecaster that uses LLMs via LiteLLM with structured outputs.

    Supports binary, continuous, and quantile question types.

    Can be initialized with either:
    - A LiteLLM model string (e.g., "anthropic/claude-opus-4-5-20251101")
    - A ModelInfo object from the registry

    Example:
        # Using LiteLLM string directly
        forecaster = LLMForecaster(model="anthropic/claude-opus-4-5-20251101")

        # Using registry
        from llm_forecasting.registry import get_model
        model_info = get_model("claude-opus-4-5-20251101")
        forecaster = LLMForecaster.from_registry(model_info)
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        *,
        model_info: ModelInfo | None = None,
    ):
        """Initialize the LLM forecaster.

        Args:
            model: LiteLLM model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-opus").
                  If not provided, uses settings.default_model.
            temperature: Sampling temperature (0.0 for deterministic).
            model_info: Optional ModelInfo from registry. If provided, uses its litellm_id
                       and stores metadata for reference.
        """
        self._model_info = model_info
        if model_info is not None:
            self.model = model_info.litellm_id
        else:
            self.model = model or settings.default_model
        self.temperature = temperature

    @classmethod
    def from_registry(cls, model_info: ModelInfo, temperature: float = 0.0) -> LLMForecaster:
        """Create a forecaster from a registry ModelInfo.

        Args:
            model_info: ModelInfo object from the registry
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            LLMForecaster configured with the model's settings
        """
        return cls(model_info=model_info, temperature=temperature)

    @property
    def name(self) -> str:
        """Return the model identifier used for this forecaster."""
        return self.model

    @property
    def model_info(self) -> ModelInfo | None:
        """Return the ModelInfo if this forecaster was created from the registry."""
        return self._model_info

    @property
    def supports_structured_output(self) -> bool:
        """Check if this model supports structured output.

        Returns True if unknown (no model_info) to maintain backwards compatibility.
        """
        if self._model_info is not None:
            return self._model_info.supports_structured_output
        return True  # Assume yes if unknown

    def _build_prompt(self, question: Question) -> tuple[str, type[BaseModel]]:
        """Build the appropriate prompt and response schema for a question.

        Returns:
            Tuple of (prompt_string, response_schema_class)
        """
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        resolution_date = question.resolution_date or "Not specified"
        background = question.background or "No additional background provided."

        if question.question_type == QuestionType.BINARY:
            prompt = BINARY_PROMPT.format(
                question=question.text,
                background=background,
                today_date=today_date,
                resolution_date=resolution_date,
            )
            return prompt, BinaryForecastResponse

        elif question.question_type == QuestionType.CONTINUOUS:
            value_range_context = ""
            if question.value_range:
                low, high = question.value_range
                value_range_context = f"\nExpected range: {low} to {high}"

            prompt = CONTINUOUS_PROMPT.format(
                question=question.text,
                background=background,
                value_range_context=value_range_context,
                today_date=today_date,
                resolution_date=resolution_date,
            )
            return prompt, ContinuousForecastResponse

        elif question.question_type == QuestionType.QUANTILE:
            value_range_context = ""
            if question.value_range:
                low, high = question.value_range
                value_range_context = f"\nExpected range: {low} to {high}"

            quantiles = question.quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
            quantiles_str = ", ".join(f"{q:.0%}" for q in quantiles)

            prompt = QUANTILE_PROMPT.format(
                question=question.text,
                background=background,
                value_range_context=value_range_context,
                today_date=today_date,
                resolution_date=resolution_date,
                quantiles=quantiles_str,
            )
            return prompt, QuantileForecastResponse

        else:
            # Default to binary
            prompt = BINARY_PROMPT.format(
                question=question.text,
                background=background,
                today_date=today_date,
                resolution_date=resolution_date,
            )
            return prompt, BinaryForecastResponse

    async def forecast(self, question: Question) -> Forecast:
        """Generate a forecast for a question using the LLM with structured output.

        Automatically selects the appropriate prompt and response format based
        on the question type (binary, continuous, or quantile).

        Args:
            question: The question to forecast.

        Returns:
            A Forecast with the appropriate prediction fields populated.
        """
        prompt, response_schema = self._build_prompt(question)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=response_schema,
            )

            content = response.choices[0].message.content
            parsed = response_schema.model_validate_json(content)

            # Build forecast based on question type
            if isinstance(parsed, BinaryForecastResponse):
                return Forecast(
                    question_id=question.id,
                    source=question.source,
                    forecaster=self.name,
                    probability=parsed.probability,
                    reasoning=parsed.reasoning,
                )
            elif isinstance(parsed, ContinuousForecastResponse):
                return Forecast(
                    question_id=question.id,
                    source=question.source,
                    forecaster=self.name,
                    point_estimate=parsed.point_estimate,
                    reasoning=parsed.reasoning,
                )
            elif isinstance(parsed, QuantileForecastResponse):
                return Forecast(
                    question_id=question.id,
                    source=question.source,
                    forecaster=self.name,
                    quantile_values=parsed.quantile_values,
                    reasoning=parsed.reasoning,
                )
            else:
                raise ValueError(f"Unknown response type: {type(parsed)}")

        except Exception as e:
            logger.error(f"Error forecasting question {question.id}: {e}")
            # Return a default forecast on error
            return Forecast(
                question_id=question.id,
                source=question.source,
                forecaster=self.name,
                probability=0.5 if question.question_type == QuestionType.BINARY else None,
                reasoning=f"Error: {e}",
            )
