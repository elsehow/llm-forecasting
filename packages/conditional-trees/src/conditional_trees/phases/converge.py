"""Phase 2: Converge - Cluster raw scenarios into global scenarios using LLM."""

import json

from conditional_trees.config import MAX_GLOBAL_SCENARIOS, MODEL_CONVERGE
from ..llm import llm_call
from ..models import GlobalScenario, Scenario
from ..prompts import CONVERGE_SYSTEM, CONVERGE_USER
from ..schemas import ConvergeResponse


def _scenarios_to_json(scenarios: list[Scenario]) -> str:
    """Convert scenarios to JSON for prompt."""
    data = []
    for s in scenarios:
        data.append(
            {
                "name": s.name,
                "description": s.description,
                "key_assumptions": s.key_assumptions,
                "source_question": s.source_question_id,
            }
        )
    return json.dumps(data, indent=2)


async def converge(
    scenarios: list[Scenario], max_scenarios: int = MAX_GLOBAL_SCENARIOS
) -> list[GlobalScenario]:
    """Phase 2: Consolidate raw scenarios into global scenarios."""
    scenarios_json = _scenarios_to_json(scenarios)

    user_prompt = CONVERGE_USER.format(
        max_scenarios=max_scenarios, scenarios_json=scenarios_json
    )

    result = await llm_call(
        CONVERGE_SYSTEM,
        user_prompt,
        model=MODEL_CONVERGE,
        response_model=ConvergeResponse,
    )

    global_scenarios = []
    for gs in result["global_scenarios"]:
        global_scenarios.append(
            GlobalScenario(
                id=gs["id"],
                name=gs["name"],
                description=gs["description"],
                key_drivers=gs.get("key_drivers", []),
                member_scenarios=gs.get("member_scenarios", []),
            )
        )

    return global_scenarios
