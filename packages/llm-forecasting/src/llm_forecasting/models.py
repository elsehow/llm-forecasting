"""Core data models for ForecastBench."""

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class QuestionType(str, Enum):
    """Type of forecasting question."""

    BINARY = "binary"  # Yes/No, resolves to 0 or 1
    CONTINUOUS = "continuous"  # Numeric value (e.g., stock price, temperature)
    QUANTILE = "quantile"  # Predict quantiles of a distribution


class SourceType(str, Enum):
    """Type of question source."""

    MARKET = "market"  # Prediction markets (Manifold, Metaculus, Polymarket, etc.)
    DATA = "data"  # Data-based questions (FRED, ACLED, Yahoo Finance, etc.)


class Question(BaseModel):
    """A forecasting question from any source."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    source_type: SourceType = SourceType.MARKET
    text: str
    background: str | None = None
    url: str | None = None
    question_type: QuestionType = QuestionType.BINARY
    created_at: datetime = Field(default_factory=_utc_now)
    resolution_date: date | None = None
    category: str | None = None
    resolved: bool = False
    resolution_value: float | None = None

    # For quantile questions: the quantiles to predict (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
    quantiles: list[float] | None = None

    # For continuous questions: expected range for context
    value_range: tuple[float, float] | None = None

    # Sampling metadata - used for balancing question sets
    base_rate: float | None = None  # Historical base rate if known


class Forecast(BaseModel):
    """A forecast made by a model or human on a question."""

    model_config = ConfigDict(frozen=True)

    id: int | None = None  # Database ID, set after saving
    question_set_id: int | None = None  # Which question set this forecast belongs to
    question_id: str
    source: str  # Question source (for namespacing)
    forecaster: str  # Model name or human identifier
    created_at: datetime = Field(default_factory=_utc_now)
    reasoning: str | None = None

    # For binary questions
    probability: float | None = Field(default=None, ge=0, le=1)

    # For continuous questions: point estimate
    point_estimate: float | None = None

    # For quantile questions: predicted values at each quantile
    quantile_values: list[float] | None = None


class Resolution(BaseModel):
    """A resolution value for a question at a point in time.

    For binary questions, value is 0 or 1.
    For numeric/continuous questions, value is the actual number.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    source: str
    date: date
    value: float


class ForecastScore(BaseModel):
    """Scored forecast after resolution."""

    model_config = ConfigDict(frozen=True)

    forecast: Forecast
    resolution: Resolution
    brier_score: float | None = Field(default=None, ge=0, le=1)  # For binary
    crps: float | None = None  # Continuous Ranked Probability Score (for quantile/continuous)
    log_score: float | None = None  # Log probability score

    @classmethod
    def from_forecast_and_resolution(
        cls, forecast: Forecast, resolution: Resolution
    ) -> "ForecastScore":
        """Calculate appropriate score based on question type."""
        brier = None
        if forecast.probability is not None:
            brier = (forecast.probability - resolution.value) ** 2
        return cls(forecast=forecast, resolution=resolution, brier_score=brier)
