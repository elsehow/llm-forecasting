"""Phase 6: Signals - Generate early warning signals for each scenario."""

from datetime import date

from tree_of_life.config import MODEL_SIGNALS, SIGNAL_HORIZON_DATE
from ..llm import llm_call_many
from ..models import GlobalScenario, Signal
from ..prompts import SIGNALS_SYSTEM, SIGNALS_USER
from ..schemas import SignalsResponse


async def signals(
    scenarios: list[GlobalScenario],
    horizon_date: str = SIGNAL_HORIZON_DATE,
    verbose: bool = True,
) -> list[Signal]:
    """Phase 6: Generate early warning signals for all scenarios."""
    # Build all calls
    calls = []
    for scenario in scenarios:
        user_prompt = SIGNALS_USER.format(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            probability=scenario.probability,
            horizon_date=horizon_date,
        )
        calls.append((scenario.id, SIGNALS_SYSTEM, user_prompt))

    # Execute (batch or parallel sync based on config)
    results = await llm_call_many(
        calls,
        model=MODEL_SIGNALS,
        verbose=verbose,
        response_model=SignalsResponse,
    )

    # Parse results
    all_signals = []
    for scenario in scenarios:
        result = results.get(scenario.id, {})
        for i, s in enumerate(result.get("signals", [])):
            all_signals.append(
                Signal(
                    id=f"{scenario.id}_signal_{i+1}",
                    text=s["text"],
                    resolves_by=date.fromisoformat(s["resolves_by"]),
                    scenario_id=scenario.id,
                    direction=s["direction"],
                    magnitude=s["magnitude"],
                    current_probability=s.get("current_probability"),
                    update_cadence=s.get("update_cadence", "event"),
                    causal_priority=s.get("causal_priority", 50),
                )
            )

    return all_signals
