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
    CATEGORICAL = "categorical"  # Multiple choice (e.g., "Under blockade", "Active conflict")


class SourceType(str, Enum):
    """Type of question source."""

    MARKET = "market"  # Prediction markets (Manifold, Metaculus, Polymarket, etc.)
    DATA = "data"  # Data-based questions (FRED, ACLED, Yahoo Finance, etc.)


class Unit(BaseModel):
    """Unit metadata for formatting question values.

    The type field is a free-form string to allow arbitrary units
    (e.g., "percent", "usd_trillions", "per_cow_slaughtered").
    """

    model_config = ConfigDict(frozen=True)

    type: str  # Free-form unit type identifier
    label: str  # Full label for display (e.g., "US dollars (trillions)")
    short_label: str  # Compact label (e.g., "$T", "%")

    @classmethod
    def from_type(cls, unit_type: str) -> "Unit":
        """Create a Unit with standard label/short_label for common types.

        For unknown types, uses the type string as both labels.
        """
        labels = {
            "percent": ("percent", "%"),
            "usd": ("US dollars", "$"),
            "usd_trillions": ("US dollars (trillions)", "$T"),
            "population": ("people (millions)", "M"),
            "population_billions": ("people (billions)", "B"),
            "rate": ("per 1,000", "per 1k"),
            "years": ("years", "yrs"),
            "count": ("count", ""),
            "ratio": ("ratio", "x"),
        }
        if unit_type in labels:
            label, short = labels[unit_type]
        else:
            # For custom types, use the type as the label
            label, short = unit_type, unit_type
        return cls(type=unit_type, label=label, short_label=short)


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

    # For categorical questions: the options to choose from
    options: list[str] | None = None

    # Domain/category for tree questions (e.g., "Macro", "AI/Labor")
    domain: str | None = None

    # Where to look for resolution (for tree questions, distinct from source)
    resolution_source: str | None = None

    # Unit metadata for formatting values in visualization
    unit: Unit | None = None


class Signal(Question):
    """A Question with VOI metadata for scenario construction.

    Extends Question to add value-of-information fields used in scenario
    generation pipelines. Signals can be either observed (from prediction
    markets/data sources) or synthetic (LLM-generated).
    """

    model_config = ConfigDict(frozen=False)  # Allow mutation for VOI enrichment

    # VOI-specific fields
    voi: float  # Value of Information for target question
    rho: float  # Correlation coefficient with target question
    rho_reasoning: str | None = None  # Explanation of correlation estimate
    uncertainty_source: str | None = None  # Which uncertainty axis (for hybrid/topdown)

    # Conditional probability fields (for binary targets, computed from rho)
    p_target_given_yes: float | None = None  # P(target | signal=YES)
    p_target_given_no: float | None = None  # P(target | signal=NO)
    cruxiness_spread: float | None = None  # |P(target|YES) - P(target|NO)|

    # Conditional expectation fields (for continuous targets, LLM-estimated)
    e_target_given_yes: float | None = None  # E[target | signal=YES]
    e_target_given_no: float | None = None  # E[target | signal=NO]

    # Probability metadata - tracks how the probability was obtained
    probability_source: str | None = None  # "market", "cache", "llm_estimate"
    probability_at: datetime | None = None  # When the probability was obtained

    @property
    def is_synthetic(self) -> bool:
        """True if signal is LLM-generated (not from a market/data source)."""
        return self.source == "llm"

    @property
    def probability(self) -> float | None:
        """Current P(signal=yes). Uses base_rate from Question."""
        return self.base_rate


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
