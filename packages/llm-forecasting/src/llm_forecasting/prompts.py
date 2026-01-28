"""Shared prompts for LLM forecasting.

These prompts are used across the codebase for probability estimation,
scenario generation, and other forecasting tasks.
"""

import logging
from datetime import datetime, timezone

import litellm
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Superforecaster prompts
# =============================================================================

SUPERFORECASTER_SYSTEM = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
You have a strong track record of making well-calibrated probability estimates."""

BINARY_FORECAST_PROMPT = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Today's Date: {today_date}

Resolution Date: {resolution_date}

Provide your probability estimate."""


CONTINUOUS_FORECAST_PROMPT = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction for the numeric value that this question will resolve to. \
You MUST give a point estimate UNDER ALL CIRCUMSTANCES.

Question:
{question}

Question Background:
{background}
{value_range_context}

Today's Date: {today_date}

Resolution Date: {resolution_date}

Provide your point estimate and reasoning."""


QUANTILE_FORECAST_PROMPT = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction for the distribution of possible values for this question. \
Provide your estimates for each of the requested quantiles.

Question:
{question}

Question Background:
{background}
{value_range_context}

Today's Date: {today_date}

Resolution Date: {resolution_date}

Quantiles to predict: {quantiles}

For each quantile, provide the value X such that there is that probability the true value is less than X."""


# =============================================================================
# Simple probability estimation (for fallback when market data unavailable)
# =============================================================================

SIMPLE_PROBABILITY_PROMPT = """\
You are an expert superforecaster. Estimate the probability that this event occurs.

Question: {question}

Background: {background}

Return only a number between 0 and 1."""


# =============================================================================
# Probability estimation helper
# =============================================================================


class ProbabilityEstimate(BaseModel):
    """Structured output for probability estimation."""

    probability: float = Field(ge=0.0, le=1.0, description="Probability between 0 and 1")
    reasoning: str = Field(description="Brief reasoning for the estimate")


DEFAULT_MODEL = "claude-sonnet-4-20250514"


async def estimate_probability(
    question: str,
    background: str = "",
    model: str = DEFAULT_MODEL,
) -> float | None:
    """Estimate probability for a binary question using LLM.

    This is a lightweight fallback for when market data is unavailable.
    Uses the superforecaster prompt to get a probability estimate.

    Args:
        question: The question text
        background: Optional background/context
        model: LLM model to use

    Returns:
        Probability between 0 and 1, or None if estimation fails
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = BINARY_FORECAST_PROMPT.format(
        question=question,
        background=background or "No additional background.",
        today_date=today,
        resolution_date="Not specified",
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format=ProbabilityEstimate,
        )

        content = response.choices[0].message.content
        parsed = ProbabilityEstimate.model_validate_json(content)
        return parsed.probability

    except Exception as e:
        logger.warning(f"Failed to estimate probability for '{question[:50]}...': {e}")
        return None
