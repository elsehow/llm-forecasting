"""Uncertainty identification for scenario construction.

Used by top-down and hybrid approaches to identify key uncertainty axes
that structure the forecasting problem.
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field
import litellm


# Default model for uncertainty identification
DEFAULT_MODEL = "claude-sonnet-4-20250514"


@dataclass
class Uncertainty:
    """An uncertainty axis that affects the forecast."""
    name: str  # Short name, e.g., "AI Productivity Impact"
    description: str  # 1-2 sentence description
    search_query: str  # Query for semantic search


class UncertaintyResponse(BaseModel):
    """Pydantic model for LLM response."""
    name: str = Field(description="2-4 word name for this uncertainty")
    description: str = Field(description="1-2 sentence description of this uncertainty")
    search_query: str = Field(description="Semantic search query to find signals related to this uncertainty")


class UncertaintiesResponse(BaseModel):
    """Response containing multiple uncertainties."""
    uncertainties: list[UncertaintyResponse]
    reasoning: str = Field(description="Why these uncertainties are the most important")


UNCERTAINTY_PROMPT = """You are identifying the key uncertainties for a long-range forecast.

TARGET QUESTION: {question}
CONTEXT: {context}

Identify {n} KEY UNCERTAINTIES that most affect this forecast.

Requirements:
1. Each uncertainty should be a distinct axis of variation (not overlapping)
2. Together they should cover the major drivers of the outcome
3. Each should have resolution possible within 5-10 years (for trackability)
4. They should be at an appropriate level of abstraction (not too broad, not too narrow)

For each uncertainty, provide:
- **name**: A memorable 2-4 word label
- **description**: 1-2 sentences explaining what this uncertainty covers
- **search_query**: A query we could use to find prediction market signals related to this uncertainty

Examples of good uncertainty names:
- "AI Productivity Impact"
- "Geopolitical Stability"
- "Energy Transition Speed"
- "Fiscal Policy Direction"
- "Demographic Shifts"

Think carefully about which {n} uncertainties matter MOST for the specific target question."""


async def identify_uncertainties(
    question: str,
    context: str,
    n: int = 3,
    model: str | None = None,
) -> list[Uncertainty]:
    """
    Identify key uncertainties for a forecasting question.

    Used by top-down and hybrid approaches to structure the problem space.

    Args:
        question: The target forecasting question
        context: Additional context about the question
        n: Number of uncertainties to identify (default 3)
        model: LLM model to use

    Returns:
        List of Uncertainty objects with name, description, and search_query
    """
    model = model or DEFAULT_MODEL

    response = await litellm.acompletion(
        model=model,
        messages=[{
            "role": "user",
            "content": UNCERTAINTY_PROMPT.format(
                question=question,
                context=context,
                n=n,
            )
        }],
        response_format=UncertaintiesResponse,
    )

    result = UncertaintiesResponse.model_validate_json(
        response.choices[0].message.content
    )

    return [
        Uncertainty(
            name=u.name,
            description=u.description,
            search_query=u.search_query,
        )
        for u in result.uncertainties
    ]
