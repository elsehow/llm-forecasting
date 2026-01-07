"""Phase 5: Condition - Generate conditional forecasts for each question-scenario pair."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from conditional_trees.config import MODEL_CONDITION
from ..llm import llm_call_many
from ..models import (
    BinaryForecast,
    CategoricalForecast,
    ConditionalForecast,
    ContinuousForecast,
    GlobalScenario,
    Question,
    QuestionType,
)
from ..prompts import (
    CONDITION_BATCH_BINARY_USER,
    CONDITION_BATCH_CATEGORICAL_USER,
    CONDITION_BATCH_CONTINUOUS_USER,
    CONDITION_BATCH_SYSTEM,
)
from ..schemas import (
    ConditionBatchBinaryResponse,
    ConditionBatchCategoricalResponse,
    ConditionBatchContinuousResponse,
)

logger = logging.getLogger(__name__)


def _scenarios_to_json(scenarios: list[GlobalScenario]) -> str:
    """Convert scenarios to JSON for batched prompt."""
    data = []
    for s in scenarios:
        data.append({
            "id": s.id,
            "name": s.name,
            "description": s.description,
        })
    return json.dumps(data, indent=2)


def _build_batched_condition_call(
    question: Question,
    scenarios: list[GlobalScenario],
    base_rate_context: str = "",
) -> tuple[str, str, str]:
    """Build the LLM call for a question with all scenarios."""
    call_id = question.id
    scenarios_json = _scenarios_to_json(scenarios)

    if question.question_type == QuestionType.CONTINUOUS:
        user_prompt = CONDITION_BATCH_CONTINUOUS_USER.format(
            question_text=question.text,
            resolution_source=question.resolution_source or "Official data source",
            base_rate_context=base_rate_context,
            scenarios_json=scenarios_json,
        )
    elif question.question_type == QuestionType.CATEGORICAL:
        user_prompt = CONDITION_BATCH_CATEGORICAL_USER.format(
            question_text=question.text,
            options=", ".join(question.options or []),
            scenarios_json=scenarios_json,
        )
    else:  # binary
        user_prompt = CONDITION_BATCH_BINARY_USER.format(
            question_text=question.text,
            scenarios_json=scenarios_json,
        )

    return (call_id, CONDITION_BATCH_SYSTEM, user_prompt)


def _parse_batched_result(
    question: Question,
    scenarios: list[GlobalScenario],
    result: dict,
) -> list[ConditionalForecast]:
    """Parse batched LLM result into list of forecasts."""
    forecasts = []

    # Result format: {"forecasts": {"scenario_id": {...}, ...}}
    forecasts_data = result.get("forecasts", {})

    # Debug: log what keys we got vs what we expected
    if forecasts_data:
        expected_ids = {s.id for s in scenarios}
        received_keys = set(forecasts_data.keys())
        if expected_ids != received_keys:
            print(f"  DEBUG: Key mismatch for {question.id}")
            print(f"    Expected: {sorted(expected_ids)[:3]}...")
            print(f"    Got: {sorted(received_keys)[:3]}...")

    for scenario in scenarios:
        scenario_result = forecasts_data.get(scenario.id)
        if not scenario_result:
            logger.warning(f"Missing forecast for {question.id}/{scenario.id}")
            continue

        if question.question_type == QuestionType.CONTINUOUS:
            forecasts.append(ContinuousForecast(
                question_id=question.id,
                scenario_id=scenario.id,
                median=scenario_result["median"],
                ci_80_low=scenario_result["ci_80_low"],
                ci_80_high=scenario_result["ci_80_high"],
                reasoning=scenario_result.get("reasoning"),
            ))
        elif question.question_type == QuestionType.CATEGORICAL:
            forecasts.append(CategoricalForecast(
                question_id=question.id,
                scenario_id=scenario.id,
                probabilities=scenario_result["probabilities"],
                reasoning=scenario_result.get("reasoning"),
            ))
        else:  # binary
            forecasts.append(BinaryForecast(
                question_id=question.id,
                scenario_id=scenario.id,
                probability=scenario_result["probability"],
                reasoning=scenario_result.get("reasoning"),
            ))

    return forecasts


async def condition(
    questions: list[Question],
    scenarios: list[GlobalScenario],
    base_rate_context: str = "",
    verbose: bool = True,
) -> list[ConditionalForecast]:
    """Phase 5: Generate conditional forecasts for all question-scenario pairs.

    Uses batched prompts - one call per question with all scenarios together.
    This ensures cross-scenario coherence for each question.

    Groups questions by type to apply correct structured output schema.

    Args:
        questions: List of forecasting questions
        scenarios: List of global scenarios
        base_rate_context: Formatted base rate string for prompt injection
        verbose: Print progress messages
    """
    # Group questions by type for schema-specific batching
    questions_by_type: dict[QuestionType, list[Question]] = defaultdict(list)
    for q in questions:
        questions_by_type[q.question_type].append(q)

    # Schema mapping by question type
    schema_map = {
        QuestionType.CONTINUOUS: ConditionBatchContinuousResponse,
        QuestionType.CATEGORICAL: ConditionBatchCategoricalResponse,
        QuestionType.BINARY: ConditionBatchBinaryResponse,
    }

    if verbose:
        type_counts = {t: len(qs) for t, qs in questions_by_type.items()}
        print(f"  Processing questions by type: {type_counts}")

    all_forecasts = []

    # Process each question type separately with its schema
    for q_type, type_questions in questions_by_type.items():
        # Build calls for this type
        calls = []
        for question in type_questions:
            calls.append(_build_batched_condition_call(question, scenarios, base_rate_context))

        if verbose:
            print(f"  Making {len(calls)} {q_type} calls ({len(scenarios)} scenarios each)")

        # Execute with type-specific schema
        results = await llm_call_many(
            calls,
            model=MODEL_CONDITION,
            verbose=verbose,
            response_model=schema_map.get(q_type),
        )

        # Parse results
        for question in type_questions:
            result = results.get(question.id, {})
            if result and "error" not in result:
                forecasts = _parse_batched_result(question, scenarios, result)
                all_forecasts.extend(forecasts)
            else:
                logger.error(f"Failed to get forecasts for question {question.id}: {result}")

    return all_forecasts
