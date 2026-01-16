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
# Active Targets (2027-2029)
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
    context="Donald Trump won the 2024 presidential election and began his second term in January 2025. He is constitutionally barred from seeking a third term in 2028, making the 2028 election an open race on the Republican side. Current polling (January 2026): Trump approval 41.9%, disapproval 55% (Silver Bulletin). Generic congressional ballot: Democrats 42.6%, Republicans 38.8% (Decision Desk HQ) - a reversal from the 2024 election environment. Democratic field: No clear frontrunner. Potential candidates include governors Gavin Newsom (CA), Gretchen Whitmer (MI), Josh Shapiro (PA), and JB Pritzker (IL). Vice President Kamala Harris's 2024 loss may complicate another run. Republican field: Wide open with no obvious Trump successor. JD Vance (current VP) is positioned but untested. Other potential candidates include Ron DeSantis, Nikki Haley, and figures from the 2024 primary. Key factors: Trump administration performance over next 2 years, economic conditions, tariff/trade policy outcomes, Democratic party rebuilding post-2024.",
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
    context="Current US real GDP: $31.1 trillion (BEA Q3 2025, seasonally adjusted annual rate). Historical growth averages 2-3% annually. Key uncertainties: Trump administration economic policies, tariff impacts, Fed monetary policy, labor market conditions, AI productivity effects.",
    cruxiness_normalizer=5.0,  # Shorter horizon = tighter range
)


# Registry for CLI access
TARGETS = {
    "carney_pm_2027": CARNEY_PM_2027,
    "democrat_whitehouse_2028": DEMOCRAT_WHITEHOUSE_2028,
    "us_gdp_2029": US_GDP_2029,
}


def get_target(name: str) -> TargetConfig:
    """Get target config by name, with helpful error message."""
    if name not in TARGETS:
        available = ", ".join(TARGETS.keys())
        raise ValueError(f"Unknown target '{name}'. Available: {available}")
    return TARGETS[name]
