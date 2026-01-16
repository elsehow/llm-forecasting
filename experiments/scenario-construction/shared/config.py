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

# =============================================================================
# New targets with empty context (to avoid LLM-written context bias)
# NOTE: base_rate and value_range are not used in scenario generation.
# We set them to None to avoid confusion (they're metadata only).
# =============================================================================

GDP_2050_NOCONTEXT = TargetConfig(
    question=Question(
        id="gdp_2050_nocontext",
        source="scenario_construction",
        text="What will US real GDP be in 2050 (in 2024 dollars)?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,  # Not used in scenario generation
        base_rate=None,  # Not used in scenario generation
        unit=Unit.from_type("usd_trillions"),
    ),
    context="",  # Empty to compare with GDP_2050 which has context
    cruxiness_normalizer=40.0,
)

POPULATION_2050 = TargetConfig(
    question=Question(
        id="population_2050",
        source="scenario_construction",
        text="What will the world population be in 2050?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,  # Not used in scenario generation
        base_rate=None,  # Not used in scenario generation
        unit=Unit.from_type("population_billions"),
    ),
    context="",  # Empty to avoid bias
    cruxiness_normalizer=1.0,
)

LIFE_EXPECTANCY_2050 = TargetConfig(
    question=Question(
        id="life_expectancy_2050",
        source="scenario_construction",
        text="What will global life expectancy at birth be in 2050?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,  # Not used in scenario generation
        base_rate=None,  # Not used in scenario generation
        unit=Unit.from_type("years"),
    ),
    context="",  # Empty to avoid bias
    cruxiness_normalizer=5.0,
)

DEMOCRACY_2050 = TargetConfig(
    question=Question(
        id="democracy_2050",
        source="scenario_construction",
        text="What proportion of the global population will live in free democracies (per Freedom House) in 2050?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,  # Not used in scenario generation
        base_rate=None,  # Not used in scenario generation
        unit=Unit.from_type("percent"),
    ),
    context="",  # Empty to avoid bias
    cruxiness_normalizer=15.0,
)

RENEWABLE_2050 = TargetConfig(
    question=Question(
        id="renewable_2050",
        source="scenario_construction",
        text="What share of global energy will come from renewable sources in 2050?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,  # Not used in scenario generation
        base_rate=None,  # Not used in scenario generation
        unit=Unit.from_type("percent"),
    ),
    context="",  # Empty to avoid bias
    cruxiness_normalizer=30.0,
)

# =============================================================================
# Near-term targets (2027-2030)
# =============================================================================

CARNEY_PM_2027 = TargetConfig(
    question=Question(
        id="carney_pm_2027",
        source="scenario_construction",
        text="Will Mark Carney be Prime Minister of Canada on December 31, 2027?",
        question_type=QuestionType.BINARY,
        value_range=None,
        base_rate=None,
        unit=None,
    ),
    context="Mark Carney became Prime Minister of Canada on March 14, 2025, after winning the Liberal Party leadership following Justin Trudeau's resignation in January 2025. Carney, former Governor of the Bank of Canada and Bank of England, won the leadership on the first ballot with strong support from the party establishment. He has not yet faced a general election as PM. The next federal election must be held by October 2025 at the latest. Current polling (January 2026) shows the Liberals with a slight lead over the Conservatives (recent polls: Nanos L39-C36, Ipsos L40-C37, Pallas L41-C38), a reversal from the significant Conservative lead before Carney took office. Key issues: cost of living, housing affordability, US-Canada trade relations under the Trump administration.",
    cruxiness_normalizer=0.3,  # 30pp probability spread = 1.0 cruxiness
)

DEMOCRAT_WHITEHOUSE_2028 = TargetConfig(
    question=Question(
        id="democrat_whitehouse_2028",
        source="scenario_construction",
        text="Will a Democrat win the White House in 2028?",
        question_type=QuestionType.BINARY,
        value_range=None,
        base_rate=None,
        unit=None,
    ),
    context="",
    cruxiness_normalizer=0.3,  # 30pp probability spread = 1.0 cruxiness
)

US_GDP_2029 = TargetConfig(
    question=Question(
        id="us_gdp_2029",
        source="scenario_construction",
        text="What will US real GDP be in 2029 (in 2024 dollars)?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,
        base_rate=None,
        unit=Unit.from_type("usd_trillions"),
    ),
    context="",
    cruxiness_normalizer=5.0,  # Shorter horizon = tighter range
)

CANADA_GDP_2030 = TargetConfig(
    question=Question(
        id="canada_gdp_2030",
        source="scenario_construction",
        text="What will Canada real GDP be in 2030 (in 2024 dollars)?",
        question_type=QuestionType.CONTINUOUS,
        value_range=None,
        base_rate=None,
        unit=Unit.from_type("usd_trillions"),
    ),
    context="",
    cruxiness_normalizer=0.5,  # Canada is ~$2T economy
)


# Registry for CLI access
TARGETS = {
    "gdp_2040": GDP_2040,
    "gdp_2050": GDP_2050,
    "gdp_2050_nocontext": GDP_2050_NOCONTEXT,
    "population_2050": POPULATION_2050,
    "life_expectancy_2050": LIFE_EXPECTANCY_2050,
    "democracy_2050": DEMOCRACY_2050,
    "renewable_2050": RENEWABLE_2050,
    # Near-term targets
    "carney_pm_2027": CARNEY_PM_2027,
    "democrat_whitehouse_2028": DEMOCRAT_WHITEHOUSE_2028,
    "us_gdp_2029": US_GDP_2029,
    "canada_gdp_2030": CANADA_GDP_2030,
}


def get_target(name: str) -> TargetConfig:
    """Get target config by name, with helpful error message."""
    if name not in TARGETS:
        available = ", ".join(TARGETS.keys())
        raise ValueError(f"Unknown target '{name}'. Available: {available}")
    return TARGETS[name]
