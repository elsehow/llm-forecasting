"""Phase 1: Logical structure generation.

Uses LLM to identify the logical decomposition of a target question:
- Necessity constraints: What MUST happen for target to be true
- Exclusivity constraints: Competitors that preclude target
- Causal pathways: Earlier events that inform the outcome
"""

from __future__ import annotations

from dataclasses import dataclass

import litellm
from pydantic import BaseModel, Field


@dataclass
class NecessityConstraint:
    """A prerequisite that must happen for target to be true.

    Example: "To win Best Picture, One Battle must be nominated"
    """

    prerequisite: str
    reasoning: str


@dataclass
class ExclusivityConstraint:
    """A competitor that precludes the target if they win.

    Example: "If Sinners wins Best Picture, One Battle loses"
    """

    competitor: str
    prize: str  # What they're competing for
    reasoning: str


@dataclass
class CausalPathway:
    """An upstream event that informs the target outcome.

    Example: "Golden Globe win → industry momentum → Oscar win"
    """

    upstream_event: str
    mechanism: str
    effect_on_target: str  # "positive", "negative", or description


@dataclass
class LogicalStructure:
    """The logical decomposition of a target question."""

    target: str
    necessity_constraints: list[NecessityConstraint]
    exclusivity_constraints: list[ExclusivityConstraint]
    causal_pathways: list[CausalPathway]


class NecessityItem(BaseModel):
    """Pydantic model for LLM response parsing."""

    prerequisite: str = Field(description="What must happen for target to be true")
    reasoning: str = Field(description="Why this is a necessity constraint")


class ExclusivityItem(BaseModel):
    """Pydantic model for LLM response parsing."""

    competitor: str = Field(description="The competing entity/outcome")
    prize: str = Field(description="What they're competing for")
    reasoning: str = Field(description="Why this is mutually exclusive")


class CausalItem(BaseModel):
    """Pydantic model for LLM response parsing."""

    upstream_event: str = Field(description="The earlier event")
    mechanism: str = Field(description="How it connects to target")
    effect_on_target: str = Field(description="positive, negative, or description")


class LogicalStructureResponse(BaseModel):
    """Structured LLM response for logical decomposition."""

    necessity: list[NecessityItem] = Field(default_factory=list)
    exclusivity: list[ExclusivityItem] = Field(default_factory=list)
    causal: list[CausalItem] = Field(default_factory=list)


STRUCTURE_PROMPT = """Analyze the logical structure of this forecasting question:

Target: {target}

{context_section}

Identify THREE types of logical relationships:

## 1. NECESSITY CONSTRAINTS
What MUST happen for the target to be true? These are hard prerequisites.
- If the prerequisite is false, the target CANNOT be true
- Format: "For [target], [prerequisite] must happen"
- Example: "For One Battle to win Best Picture, it must be nominated"

## 2. EXCLUSIVITY CONSTRAINTS
What outcomes are mutually exclusive with the target? These are competitors.
- If a competitor wins the same prize, the target loses
- Format: "[competitor] winning [prize] means [target entity] loses"
- Example: "If Sinners wins Best Picture, One Battle cannot win"

## 3. CAUSAL PATHWAYS
What earlier events would inform this outcome? These are correlational signals.
- Events that provide evidence about the target's likelihood
- Format: "[upstream event] -> [mechanism] -> [effect on target]"
- Example: "Golden Globe win -> industry momentum -> Oscar win (positive)"

Be specific about:
- Entity names (people, films, companies)
- Dates when events resolve
- The direction of causal effects (positive/negative)

Return your analysis as structured JSON."""


async def generate_logical_structure(
    target: str,
    context: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> LogicalStructure:
    """Generate logical decomposition of a target question using LLM.

    Args:
        target: The target question to decompose
        context: Optional background context (current status, key players, etc.)
        model: LLM model to use for generation

    Returns:
        LogicalStructure with necessity, exclusivity, and causal constraints
    """
    context_section = f"CONTEXT:\n{context}\n" if context else ""

    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": STRUCTURE_PROMPT.format(
                    target=target,
                    context_section=context_section,
                ),
            }
        ],
        response_format=LogicalStructureResponse,
    )

    result = LogicalStructureResponse.model_validate_json(
        response.choices[0].message.content
    )

    # Convert to dataclasses
    necessity_constraints = [
        NecessityConstraint(
            prerequisite=item.prerequisite,
            reasoning=item.reasoning,
        )
        for item in result.necessity
    ]

    exclusivity_constraints = [
        ExclusivityConstraint(
            competitor=item.competitor,
            prize=item.prize,
            reasoning=item.reasoning,
        )
        for item in result.exclusivity
    ]

    causal_pathways = [
        CausalPathway(
            upstream_event=item.upstream_event,
            mechanism=item.mechanism,
            effect_on_target=item.effect_on_target,
        )
        for item in result.causal
    ]

    return LogicalStructure(
        target=target,
        necessity_constraints=necessity_constraints,
        exclusivity_constraints=exclusivity_constraints,
        causal_pathways=causal_pathways,
    )


def structure_to_dict(structure: LogicalStructure) -> dict:
    """Convert LogicalStructure to a dictionary for serialization."""
    return {
        "target": structure.target,
        "necessity_constraints": [
            {"prerequisite": c.prerequisite, "reasoning": c.reasoning}
            for c in structure.necessity_constraints
        ],
        "exclusivity_constraints": [
            {"competitor": c.competitor, "prize": c.prize, "reasoning": c.reasoning}
            for c in structure.exclusivity_constraints
        ],
        "causal_pathways": [
            {
                "upstream_event": p.upstream_event,
                "mechanism": p.mechanism,
                "effect_on_target": p.effect_on_target,
            }
            for p in structure.causal_pathways
        ],
    }
