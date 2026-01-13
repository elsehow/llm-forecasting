"""Phase 5: Condition - Generate conditional forecasts for each question-scenario pair.

Uses Bracket-style direction commitment to ensure logical coherence across scenarios.
The model first declares which scenarios should produce higher/lower probabilities,
then provides probabilities that must respect that ordering.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from tree_of_life.config import MODEL_CONDITION
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


def _validate_direction_consistency(
    directions: dict[str, str],
    forecasts: dict[str, dict],
    question_type: QuestionType,
    options: list[str] | None = None,
) -> list[str]:
    """Validate that probabilities respect direction commitments.

    Returns list of violations (empty if consistent).
    """
    violations = []

    def get_value(sid: str, target_option: str | None = None) -> float | None:
        forecast = forecasts.get(sid)
        if not forecast:
            return None
        if question_type == QuestionType.BINARY:
            return forecast.get("probability")
        if question_type == QuestionType.CONTINUOUS:
            return forecast.get("median")
        if question_type == QuestionType.CATEGORICAL and target_option:
            return forecast.get("probabilities", {}).get(target_option)
        return None

    def check_ordering(high_ids: list, low_ids: list, high_label: str, low_label: str):
        """Check that all values in high_ids > all values in low_ids."""
        for h_id in high_ids:
            h_val = get_value(h_id)
            if h_val is None:
                continue
            for l_id in low_ids:
                l_val = get_value(l_id)
                if l_val is not None and h_val <= l_val:
                    violations.append(
                        f"{h_id} ({high_label}: {h_val:.3f}) should be > "
                        f"{l_id} ({low_label}: {l_val:.3f})"
                    )

    if question_type in (QuestionType.BINARY, QuestionType.CONTINUOUS):
        by_dir = {d: [s for s, v in directions.items() if v == d] for d in ["increases", "neutral", "decreases"]}
        check_ordering(by_dir["increases"], by_dir["neutral"], "increases", "neutral")
        check_ordering(by_dir["neutral"], by_dir["decreases"], "neutral", "decreases")
        check_ordering(by_dir["increases"], by_dir["decreases"], "increases", "decreases")

    elif question_type == QuestionType.CATEGORICAL and options:
        neutral_ids = [s for s, d in directions.items() if d == "neutral"]
        for sid, direction in directions.items():
            if direction == "neutral":
                continue
            if direction not in options:
                violations.append(f"{sid}: direction '{direction}' not in options {options}")
                continue
            s_val = get_value(sid, direction)
            if s_val is None:
                continue
            for n_id in neutral_ids:
                n_val = get_value(n_id, direction)
                if n_val is not None and s_val <= n_val:
                    violations.append(
                        f"{sid} (toward {direction}: {s_val:.3f}) should be > {n_id} (neutral: {n_val:.3f})"
                    )

    return violations


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
    validate_directions: bool = True,
) -> list[ConditionalForecast]:
    """Phase 5: Generate conditional forecasts for all question-scenario pairs.

    Uses Bracket-style direction commitment: the model first declares which
    scenarios should produce higher/lower probabilities, then provides values
    that must respect that ordering. This eliminates impossible forecasts.

    Uses batched prompts - one call per question with all scenarios together.
    This ensures cross-scenario coherence for each question.

    Groups questions by type to apply correct structured output schema.

    Args:
        questions: List of forecasting questions
        scenarios: List of global scenarios
        base_rate_context: Formatted base rate string for prompt injection
        verbose: Print progress messages
        validate_directions: If True, validate that probabilities respect
            direction commitments and log warnings for violations.
            Does not block pipeline - just warns.
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

        # Parse results and validate directions
        for question in type_questions:
            result = results.get(question.id, {})
            if result and "error" not in result:
                # Validate direction consistency if enabled
                if validate_directions and "directions" in result:
                    directions = result["directions"]
                    forecasts_data = result.get("forecasts", {})
                    violations = _validate_direction_consistency(
                        directions,
                        forecasts_data,
                        q_type,
                        options=question.options,
                    )
                    if violations:
                        logger.warning(
                            f"Direction violations for {question.id}: {violations}"
                        )
                        if verbose:
                            print(f"  WARNING: Direction violations for {question.id}:")
                            for v in violations[:3]:  # Show first 3
                                print(f"    - {v}")
                            if len(violations) > 3:
                                print(f"    ... and {len(violations) - 3} more")

                forecasts = _parse_batched_result(question, scenarios, result)
                all_forecasts.extend(forecasts)
            else:
                logger.error(f"Failed to get forecasts for question {question.id}: {result}")

    return all_forecasts
