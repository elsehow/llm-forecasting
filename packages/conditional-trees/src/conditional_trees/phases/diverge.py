"""Phase 1: Diverge - Generate scenarios for each question."""

from __future__ import annotations

from conditional_trees.config import FORECAST_HORIZON, MODEL_DIVERGE, N_SCENARIOS_PER_QUESTION, START_DATE
from ..llm import llm_call_many
from ..models import Question, Scenario
from ..prompts import DIVERGE_SYSTEM, DIVERGE_USER
from ..schemas import DivergeResponse


async def diverge(
    questions: list[Question],
    n_scenarios: int = N_SCENARIOS_PER_QUESTION,
    start_date: str | None = None,
    forecast_horizon: str | None = None,
    base_rate_context: str = "",
    verbose: bool = True,
) -> list[Scenario]:
    """Phase 1: Generate scenarios for all questions using batch API.

    Args:
        questions: List of forecasting questions
        n_scenarios: Number of scenarios to generate per question
        start_date: Reference date for "today" (default: config.START_DATE)
        forecast_horizon: End of forecast window (default: config.FORECAST_HORIZON)
        base_rate_context: Formatted base rate string for prompt injection
        verbose: Print progress messages
    """
    # Use defaults from config if not provided
    start = start_date or START_DATE
    horizon = forecast_horizon or FORECAST_HORIZON

    # Format system prompt with date and base rate info
    system_prompt = DIVERGE_SYSTEM.format(
        base_rate_context=base_rate_context,
        start_date=start,
        forecast_horizon=horizon,
    )

    # Build all calls
    calls = []
    for q in questions:
        user_prompt = DIVERGE_USER.format(
            n_scenarios=n_scenarios,
            question_text=q.text,
            question_type=q.type,
            domain=q.domain or "General",
        )
        calls.append((q.id, system_prompt, user_prompt))

    # Execute (batch or parallel sync based on config)
    results = await llm_call_many(
        calls,
        model=MODEL_DIVERGE,
        verbose=verbose,
        response_model=DivergeResponse,
    )

    # Parse results
    all_scenarios = []
    for question in questions:
        result = results.get(question.id, {})
        for s in result.get("scenarios", []):
            all_scenarios.append(
                Scenario(
                    name=s["name"],
                    description=s["description"],
                    key_assumptions=s["key_assumptions"],
                    source_question_id=question.id,
                )
            )

    return all_scenarios
