"""Data models for scenario construction.

These models define the structure of scenario generation outputs,
designed to hydrate a scenario explorer UI.
"""

from pydantic import BaseModel, Field

from llm_forecasting.models import Question, Signal

from .scenarios import MECEScenario
from .uncertainties import Uncertainty


class UncertaintyModel(BaseModel):
    """Pydantic version of Uncertainty for serialization."""

    name: str
    description: str
    search_query: str

    @classmethod
    def from_uncertainty(cls, u: Uncertainty) -> "UncertaintyModel":
        """Convert from dataclass to pydantic model."""
        return cls(
            name=u.name,
            description=u.description,
            search_query=u.search_query,
        )


class ScenarioSet(BaseModel):
    """Top-level container for scenario generation output.

    This is the primary output format for all three approaches
    (bottom-up, hybrid, top-down). Designed to hydrate a scenario
    explorer UI.
    """

    id: str = Field(description="Unique ID for this scenario set")
    name: str = Field(description="Human-readable name, e.g., 'GDP 2050 (hybrid)'")
    approach: str = Field(description="Generation approach: 'bottomup', 'hybrid', or 'topdown'")
    created_at: str = Field(description="ISO timestamp of generation")

    # Target question
    question: Question = Field(description="The target forecasting question")

    # Entities
    signals: list[Signal] = Field(description="Signals used for scenario generation")
    scenarios: list[MECEScenario] = Field(description="Generated MECE scenarios")

    # Metadata
    uncertainties: list[UncertaintyModel] | None = Field(
        default=None,
        description="Uncertainty axes (for hybrid/topdown approaches)",
    )
    voi_floor: float = Field(description="VOI floor used for filtering signals")

    # Generation metadata
    knowledge_cutoff: str | None = Field(
        default=None,
        description="Model knowledge cutoff date",
    )
    mece_reasoning: str | None = Field(
        default=None,
        description="Explanation of why scenarios are MECE",
    )
    coverage_gaps: list[str] | None = Field(
        default=None,
        description="Any coverage gaps identified",
    )
