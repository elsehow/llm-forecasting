"""Target question configuration for scenario generation.

Reuses Question and Unit from llm_forecasting.models for consistency.
"""

from dataclasses import dataclass

from llm_forecasting.models import Question, QuestionType, Unit


@dataclass
class TargetConfig:
    """Configuration for scenario generation target question."""

    question: Question  # Reuse existing Question model
    context: str  # Additional context for LLM prompts
    cruxiness_normalizer: float  # Spread that equals 1.0 cruxiness score


# =============================================================================
# Presets
# =============================================================================

GDP_2040 = TargetConfig(
    question=Question(
        id="gdp_2040",
        source="scenario_construction",
        text="What will US real GDP be in 2040 (in 2024 dollars)?",
        question_type=QuestionType.CONTINUOUS,
        value_range=(20.0, 80.0),  # $20T to $80T plausible
        base_rate=29.0,  # Current GDP
        unit=Unit.from_type("usd_trillions"),
    ),
    context="Current US GDP: ~$29 trillion (2024). Historical growth: 2-3% annually. Key uncertainties: AI/automation, geopolitics, fiscal policy, demographics, energy.",
    cruxiness_normalizer=20.0,  # $20T spread = 1.0
)

GDP_2050 = TargetConfig(
    question=Question(
        id="gdp_2050",
        source="scenario_construction",
        text="What will US real GDP be in 2050 (in 2024 dollars)?",
        question_type=QuestionType.CONTINUOUS,
        value_range=(25.0, 150.0),  # $25T to $150T plausible (longer horizon = wider range)
        base_rate=29.0,  # Current GDP
        unit=Unit.from_type("usd_trillions"),
    ),
    context="Current US GDP: ~$29 trillion (2024). 26-year horizon allows for transformative changes. Key uncertainties: AI/automation impact, transformative tech (quantum, fusion), geopolitical realignment, climate adaptation costs, demographic shifts, institutional stability.",
    cruxiness_normalizer=40.0,  # $40T spread = 1.0 (wider for longer horizon)
)

RENEWABLE_2050 = TargetConfig(
    question=Question(
        id="renewable_2050",
        source="scenario_construction",
        text="What share of global energy will come from renewable sources in 2050?",
        question_type=QuestionType.CONTINUOUS,
        value_range=(10.0, 90.0),  # 10% to 90% plausible
        base_rate=15.0,  # Current renewable share
        unit=Unit.from_type("percent"),
    ),
    context="Current renewable share: ~15% (2024). Paris targets: 60-70% by 2050. Key uncertainties: policy ambition, battery/storage costs, grid infrastructure, fossil fuel phase-out pace, nuclear expansion.",
    cruxiness_normalizer=30.0,  # 30 percentage point spread = 1.0
)


# Registry for CLI access
TARGETS = {
    "gdp_2040": GDP_2040,
    "gdp_2050": GDP_2050,
    "renewable_2050": RENEWABLE_2050,
}


def get_target(name: str) -> TargetConfig:
    """Get target config by name, with helpful error message."""
    if name not in TARGETS:
        available = ", ".join(TARGETS.keys())
        raise ValueError(f"Unknown target '{name}'. Available: {available}")
    return TARGETS[name]
