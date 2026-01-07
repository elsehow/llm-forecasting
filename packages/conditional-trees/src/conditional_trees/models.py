"""Pydantic models for conditional forecasting trees."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Import Question from core library - no duplicate types
from llm_forecasting.models import Question, QuestionType


class Scenario(BaseModel):
    """Raw scenario from Phase 1 (per-question)."""

    name: str
    description: str
    key_assumptions: list[str]
    source_question_id: str | None = None


class GlobalScenario(BaseModel):
    """Consolidated scenario from Phase 2."""

    id: str
    name: str
    description: str
    probability: float = 0.0
    raw_probability: float | None = None  # Pre-normalization value for diagnostics
    key_drivers: list[str] = Field(default_factory=list)
    member_scenarios: list[str] = Field(default_factory=list)
    # Explicit override for upstream scenarios; None = derive from Relationships
    upstream_scenarios: list[str] | None = None


class Relationship(BaseModel):
    """Pairwise relationship from Phase 3.

    Convention for directional relationships:
    - hierarchical: scenario_a (parent) is upstream of scenario_b (child)
    - correlated: scenario_a influences scenario_b
    """

    scenario_a: str
    scenario_b: str
    type: Literal["orthogonal", "correlated", "hierarchical", "mutually_exclusive"]
    strength: float | None = None  # 0-1 for correlated relationships
    notes: str | None = None


class ContinuousForecast(BaseModel):
    """Conditional forecast for continuous questions."""

    question_id: str
    scenario_id: str
    median: float
    ci_80_low: float
    ci_80_high: float
    reasoning: str | None = None


class CategoricalForecast(BaseModel):
    """Conditional forecast for categorical questions."""

    question_id: str
    scenario_id: str
    probabilities: dict[str, float]  # option -> probability
    reasoning: str | None = None


class BinaryForecast(BaseModel):
    """Conditional forecast for binary questions."""

    question_id: str
    scenario_id: str
    probability: float
    reasoning: str | None = None


ConditionalForecast = ContinuousForecast | CategoricalForecast | BinaryForecast


class Signal(BaseModel):
    """Early signal from Phase 6."""

    id: str
    text: str
    resolves_by: date
    scenario_id: str
    direction: Literal["increases", "decreases"]
    magnitude: Literal["small", "medium", "large"]
    current_probability: float | None = None  # P(signal occurs)
    # Tiered signal fields
    update_cadence: Literal["event", "monthly", "quarterly", "annual"] = "event"
    causal_priority: int = 50  # 0-100, lower resolves first

    def is_past_dated(self) -> bool:
        """Check if signal resolution date is in the past."""
        return self.resolves_by < date.today()


class ForecastTree(BaseModel):
    """Complete output of the pipeline."""

    questions: list[Question]
    raw_scenarios: list[Scenario]
    global_scenarios: list[GlobalScenario]
    relationships: list[Relationship]
    conditionals: list[ConditionalForecast]
    signals: list[Signal]
    created_at: str | None = None

    # Probability diagnostics
    raw_probability_sum: float | None = None  # Diagnostic: 1.91 means overlap assumed
    probability_status: str | None = None  # "ok" | "warning" | "suspect"

    # Base rates used at generation time (for reproducibility)
    base_rates_snapshot: dict | None = None

    def past_dated_signals(self) -> list[Signal]:
        """Return signals with resolution dates in the past."""
        return [s for s in self.signals if s.is_past_dated()]

    def scenarios_missing_signals(self) -> list[str]:
        """Return scenario IDs that have no signals."""
        scenario_ids = {s.id for s in self.global_scenarios}
        signal_scenario_ids = {s.scenario_id for s in self.signals}
        return list(scenario_ids - signal_scenario_ids)

    def derive_upstream_graph(self) -> dict[str, list[str]]:
        """Compute upstream graph from relationships.

        Returns:
            Dict mapping scenario_id -> list of upstream scenario_ids
        """
        from .propagation import derive_upstream_graph

        return derive_upstream_graph(self.global_scenarios, self.relationships)

    def get_upstream(self, scenario_id: str) -> list[str]:
        """Get upstream scenarios for a given scenario.

        Uses explicit override if set, otherwise derives from relationships.

        Args:
            scenario_id: The scenario to get upstreams for

        Returns:
            List of upstream scenario IDs
        """
        scenario = next(
            (s for s in self.global_scenarios if s.id == scenario_id), None
        )
        if scenario is None:
            return []
        if scenario.upstream_scenarios is not None:
            return scenario.upstream_scenarios
        return self.derive_upstream_graph().get(scenario_id, [])

    def unconditional_forecast(
        self, question_id: str
    ) -> float | dict[str, float]:
        """Compute unconditional forecast for a question.

        E[outcome] = Σ P(scenario) × E[outcome | scenario]

        Args:
            question_id: The question to compute forecast for

        Returns:
            float for continuous/binary, dict for categorical
        """
        from .propagation import compute_unconditional

        return compute_unconditional(self, question_id)

    def signals_by_cadence(
        self, cadence: Literal["event", "monthly", "quarterly", "annual"]
    ) -> list[Signal]:
        """Filter signals by update cadence.

        Args:
            cadence: The cadence to filter by

        Returns:
            List of signals with the specified cadence
        """
        return [s for s in self.signals if s.update_cadence == cadence]

    def signals_by_priority(self, max_priority: int = 100) -> list[Signal]:
        """Get signals sorted by causal priority, filtered by max.

        Args:
            max_priority: Only include signals with priority <= this value

        Returns:
            List of signals sorted by causal_priority (ascending)
        """
        filtered = [s for s in self.signals if s.causal_priority <= max_priority]
        return sorted(filtered, key=lambda s: s.causal_priority)

    def hydrate_upstream(self) -> "ForecastTree":
        """Populate upstream_scenarios from relationships where not explicitly set.

        This fills in the derived upstream relationships so they're visible
        in serialized JSON output. Only populates scenarios where
        upstream_scenarios is None (preserves explicit overrides).

        Returns:
            New ForecastTree with upstream_scenarios populated
        """
        derived = self.derive_upstream_graph()
        scenarios = []
        for s in self.global_scenarios:
            if s.upstream_scenarios is None:
                upstreams = derived.get(s.id, [])
                s = s.model_copy(
                    update={"upstream_scenarios": upstreams if upstreams else None}
                )
            scenarios.append(s)
        return self.model_copy(update={"global_scenarios": scenarios})
