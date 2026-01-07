"""Phase 3: Structure - Identify relationships between global scenarios."""

import json

from conditional_trees.config import MODEL_STRUCTURE
from ..llm import llm_call
from ..models import GlobalScenario, Relationship
from ..prompts import STRUCTURE_SYSTEM, STRUCTURE_USER
from ..schemas import StructureResponse


def _global_scenarios_to_json(scenarios: list[GlobalScenario]) -> str:
    """Convert global scenarios to JSON for prompt."""
    data = []
    for s in scenarios:
        data.append(
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "key_drivers": s.key_drivers,
            }
        )
    return json.dumps(data, indent=2)


async def structure(scenarios: list[GlobalScenario]) -> list[Relationship]:
    """Phase 3: Identify relationships between all scenario pairs."""
    scenarios_json = _global_scenarios_to_json(scenarios)

    user_prompt = STRUCTURE_USER.format(scenarios_json=scenarios_json)

    result = await llm_call(
        STRUCTURE_SYSTEM,
        user_prompt,
        model=MODEL_STRUCTURE,
        response_model=StructureResponse,
    )

    relationships = []
    for r in result["relationships"]:
        relationships.append(
            Relationship(
                scenario_a=r["scenario_a"],
                scenario_b=r["scenario_b"],
                type=r["type"],
                strength=r.get("strength"),
                notes=r.get("notes"),
            )
        )

    return relationships
