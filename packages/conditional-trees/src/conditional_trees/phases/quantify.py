"""Phase 4: Quantify - Assign probabilities to global scenarios."""

import json
import logging

from conditional_trees.config import MODEL_QUANTIFY
from ..llm import llm_call
from ..models import GlobalScenario, Relationship
from ..probability import (
    ProbabilityResult,
    force_normalize,
    format_probabilities_for_retry,
    handle_probability_sum,
)
from ..prompts import QUANTIFY_RETRY, QUANTIFY_SYSTEM, QUANTIFY_USER
from ..schemas import QuantifyResponse

logger = logging.getLogger(__name__)


def _global_scenarios_to_json(scenarios: list[GlobalScenario]) -> str:
    """Convert global scenarios to JSON for prompt."""
    data = []
    for s in scenarios:
        data.append(
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
            }
        )
    return json.dumps(data, indent=2)


def _relationships_to_json(relationships: list[Relationship]) -> str:
    """Convert relationships to JSON for prompt."""
    data = []
    for r in relationships:
        data.append(
            {
                "scenario_a": r.scenario_a,
                "scenario_b": r.scenario_b,
                "type": r.type,
                "strength": r.strength,
            }
        )
    return json.dumps(data, indent=2)


def _parse_probabilities(result: dict) -> dict[str, float]:
    """Extract scenario_id -> probability map from LLM response."""
    return {p["scenario_id"]: p["probability"] for p in result["probabilities"]}


async def quantify(
    scenarios: list[GlobalScenario],
    relationships: list[Relationship],
    verbose: bool = True,
) -> tuple[list[GlobalScenario], ProbabilityResult]:
    """Phase 4: Assign probabilities to global scenarios.

    Returns:
        Tuple of (updated scenarios, probability result with diagnostics)
    """
    scenarios_json = _global_scenarios_to_json(scenarios)
    relationships_json = _relationships_to_json(relationships)

    user_prompt = QUANTIFY_USER.format(
        scenarios_json=scenarios_json, relationships_json=relationships_json
    )

    # First attempt
    result = await llm_call(
        QUANTIFY_SYSTEM,
        user_prompt,
        model=MODEL_QUANTIFY,
        response_model=QuantifyResponse,
    )
    prob_map = _parse_probabilities(result)

    # Check probability sum with tiered handling
    prob_result = handle_probability_sum(prob_map)

    if prob_result.status == "retry_needed":
        # Make retry call with correction prompt
        logger.info(
            f"Probability sum {prob_result.raw_sum:.1%} requires retry, "
            "sending correction prompt"
        )

        if verbose:
            print(f"  Probability sum {prob_result.raw_sum:.1%} outside range, retrying...")

        retry_prompt = QUANTIFY_RETRY.format(
            sum=prob_result.raw_sum,
            previous=format_probabilities_for_retry(prob_map),
        )

        retry_result = await llm_call(
            QUANTIFY_SYSTEM,
            retry_prompt,
            model=MODEL_QUANTIFY,
            response_model=QuantifyResponse,
        )
        retry_prob_map = _parse_probabilities(retry_result)

        # Check retry result
        prob_result = handle_probability_sum(retry_prob_map)

        if prob_result.status == "retry_needed":
            # Retry still failed, force normalize and mark as suspect
            logger.warning(
                f"Retry still produced sum of {prob_result.raw_sum:.1%}, "
                "force normalizing as suspect"
            )
            if verbose:
                print(f"  Retry still off ({prob_result.raw_sum:.1%}), forcing normalization")

            prob_result = ProbabilityResult(
                raw_probabilities=retry_prob_map,
                raw_sum=sum(retry_prob_map.values()),
                normalized=force_normalize(retry_prob_map),
                status="suspect",
                action_taken="force_normalized_after_retry",
            )

    # Use normalized probabilities
    normalized = prob_result.normalized
    if normalized is None:
        # This shouldn't happen after our handling, but be safe
        normalized = force_normalize(prob_map)
        prob_result = ProbabilityResult(
            raw_probabilities=prob_map,
            raw_sum=sum(prob_map.values()),
            normalized=normalized,
            status="suspect",
            action_taken="fallback_normalization",
        )

    # Log the result
    logger.info(
        f"Probability normalization: raw_sum={prob_result.raw_sum:.1%}, "
        f"status={prob_result.status}, action={prob_result.action_taken}"
    )

    # Update scenarios with normalized probabilities
    updated_scenarios = []
    for s in scenarios:
        updated = s.model_copy()
        updated.probability = normalized.get(s.id, 0.0)
        # Store raw probability for diagnostics
        updated.raw_probability = prob_result.raw_probabilities.get(s.id)
        updated_scenarios.append(updated)

    return updated_scenarios, prob_result
