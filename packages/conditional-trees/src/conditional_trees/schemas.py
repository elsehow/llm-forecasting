"""Pydantic response schemas for structured LLM outputs.

These schemas define the expected response format for each pipeline phase.
Used with Claude's structured outputs to guarantee valid JSON responses.
"""

from typing import Literal

from pydantic import BaseModel, Field


def make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert Pydantic schema to strict schema for Claude structured outputs.

    Claude's structured outputs require:
    - additionalProperties: false on all objects
    - No minimum/maximum constraints on numbers (not supported)
    """
    schema = model.model_json_schema()

    # Properties not supported by Claude's structured outputs
    unsupported_props = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}

    def process_schema(obj):
        if isinstance(obj, dict):
            # Add additionalProperties: false to objects
            if obj.get("type") == "object":
                obj["additionalProperties"] = False

            # Remove unsupported properties
            for prop in unsupported_props:
                obj.pop(prop, None)

            # Recurse into nested structures
            for v in obj.values():
                process_schema(v)
        elif isinstance(obj, list):
            for item in obj:
                process_schema(item)

    process_schema(schema)

    # Handle $defs
    if "$defs" in schema:
        for defn in schema["$defs"].values():
            process_schema(defn)

    return schema


# =============================================================================
# Phase 0: Base Rates
# =============================================================================

class BaseRateItem(BaseModel):
    """A single base rate fetched via web search."""
    name: str
    description: str
    value: float
    unit: str
    as_of: str
    source: str


class BaseRatesResponse(BaseModel):
    """Response from Phase 0: Base Rates."""
    base_rates: list[BaseRateItem]


# =============================================================================
# Phase 1: Diverge
# =============================================================================

class ScenarioItem(BaseModel):
    """A scenario generated for a question."""
    name: str
    description: str
    key_assumptions: list[str]


class DivergeResponse(BaseModel):
    """Response from Phase 1: Diverge."""
    scenarios: list[ScenarioItem]


# =============================================================================
# Phase 2: Converge
# =============================================================================

class GlobalScenarioItem(BaseModel):
    """A consolidated global scenario."""
    id: str
    name: str
    description: str
    key_drivers: list[str]
    member_scenarios: list[str]


class ConvergeResponse(BaseModel):
    """Response from Phase 2: Converge."""
    global_scenarios: list[GlobalScenarioItem]


# =============================================================================
# Phase 3: Structure
# =============================================================================

class RelationshipItem(BaseModel):
    """A relationship between two scenarios."""
    scenario_a: str
    scenario_b: str
    type: Literal["orthogonal", "correlated", "hierarchical", "mutually_exclusive"]
    strength: float | None = None
    notes: str | None = None


class StructureResponse(BaseModel):
    """Response from Phase 3: Structure."""
    relationships: list[RelationshipItem]


# =============================================================================
# Phase 4: Quantify
# =============================================================================

class ProbabilityItem(BaseModel):
    """Probability assignment for a scenario."""
    scenario_id: str
    probability: float = Field(ge=0, le=1)
    reasoning: str


class QuantifyResponse(BaseModel):
    """Response from Phase 4: Quantify."""
    probabilities: list[ProbabilityItem]


# =============================================================================
# Phase 5: Condition (individual - not used in batched mode)
# =============================================================================

class ContinuousForecastItem(BaseModel):
    """Continuous forecast with median and confidence interval."""
    median: float
    ci_80_low: float
    ci_80_high: float
    reasoning: str


class CategoricalForecastItem(BaseModel):
    """Categorical forecast with probability distribution."""
    probabilities: dict[str, float]
    reasoning: str


class BinaryForecastItem(BaseModel):
    """Binary forecast with single probability."""
    probability: float = Field(ge=0, le=1)
    reasoning: str


# =============================================================================
# Phase 5: Condition (batched - all scenarios for one question)
# =============================================================================

class ConditionBatchContinuousResponse(BaseModel):
    """Response from batched continuous condition phase."""
    forecasts: dict[str, ContinuousForecastItem]


class ConditionBatchCategoricalResponse(BaseModel):
    """Response from batched categorical condition phase."""
    forecasts: dict[str, CategoricalForecastItem]


class ConditionBatchBinaryResponse(BaseModel):
    """Response from batched binary condition phase."""
    forecasts: dict[str, BinaryForecastItem]


# =============================================================================
# Phase 6: Signals
# =============================================================================

class SignalItem(BaseModel):
    """An early warning signal for a scenario."""
    text: str
    resolves_by: str
    direction: Literal["increases", "decreases"]
    magnitude: Literal["small", "medium", "large"]
    current_probability: float | None = Field(default=None, ge=0, le=1)
    update_cadence: Literal["event", "daily", "weekly", "monthly", "quarterly", "annual"] = "event"
    causal_priority: int = Field(default=50, ge=0, le=100)


class SignalsResponse(BaseModel):
    """Response from Phase 6: Signals."""
    signals: list[SignalItem]
