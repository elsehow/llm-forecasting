"""Unified MECE scenario generation for all approaches.

This module provides a single generate_mece_scenarios() function that all three
approaches (bottom-up, hybrid, top-down) share. Structure emerges from signals,
not imposed by uncertainty grid.
"""

import json
from pydantic import BaseModel, Field
import litellm


# Default model
DEFAULT_MODEL = "claude-sonnet-4-20250514"


# ============================================================
# Pydantic models for structured output
# ============================================================

class MECEScenario(BaseModel):
    """A single MECE scenario."""
    name: str = Field(description="2-4 word memorable name")
    description: str = Field(description="2-3 sentences describing this world state")
    outcome_range: str = Field(description="Expected outcome range, e.g., '$45-55T' for GDP or '40-50%' for renewable share")
    key_drivers: list[str] = Field(description="3-5 factors that define this scenario")
    why_exclusive: str = Field(description="Why this scenario CANNOT co-occur with the others")
    mapped_signals: list[str] = Field(description="Which input signals map to this scenario")
    indicator_bundle: dict[str, str] = Field(description="3-5 measurable indicators with thresholds")


class MECEScenariosResponse(BaseModel):
    """Response from MECE scenario generation."""
    scenarios: list[MECEScenario]
    mece_reasoning: str = Field(description="Explanation of why these scenarios are MECE")
    coverage_gaps: list[str] = Field(description="Any outcomes not covered (should be empty or minimal)")


# ============================================================
# Core generation function
# ============================================================

MECE_SCENARIO_PROMPT = """You are constructing MECE scenarios for forecasting.

TARGET QUESTION: {question}
CONTEXT: {context}

I have {n_signals} signals that may affect this forecast:

{signals_json}

YOUR TASK: Generate 3-5 MECE SCENARIOS based on these signals.

CRITICAL REQUIREMENTS:

1. MUTUALLY EXCLUSIVE: Each scenario must be a distinct world-state that CANNOT co-occur with the others.
   - Bad: "AI Boom" and "Green Revolution" (these can happen together)
   - Good: "AI-Led Growth" vs "AI Winter" (opposite states of same variable)
   - Good: "High Growth" vs "Stagnation" vs "Crisis" (different outcome ranges)

2. COLLECTIVELY EXHAUSTIVE: Together, the scenarios should cover ALL plausible outcomes.
   - Don't leave gaps in the possibility space
   - Include baseline/moderate outcomes, not just extremes

3. OUTCOME ANCHORING:
   - Use outcome ranges as the primary differentiator
   - Scenarios should partition the outcome space (non-overlapping ranges)

4. SIGNAL GROUNDING:
   - Map signals to the scenario(s) they would indicate
   - Use the signals to inform the causal story for each scenario

5. For each scenario, explain WHY it cannot co-occur with the others.

6. Acknowledge any coverage gaps honestly (ideally there are none).

Think step by step:
- What are the key axes of uncertainty implied by these signals?
- How can we partition the outcome space so scenarios don't overlap?
- What outcome ranges correspond to each scenario?
"""


async def generate_mece_scenarios(
    signals: list[dict],
    question: str,
    context: str,
    voi_floor: float = 0.0,
    model: str | None = None,
) -> MECEScenariosResponse:
    """
    Generate MECE scenarios from signals.

    This is the unified function used by all three approaches (bottom-up,
    hybrid, top-down). Structure emerges from signals, not imposed by
    uncertainty grid.

    Args:
        signals: List of dicts with "text" and optional "source", "reasoning" keys
        question: The target forecasting question
        context: Additional context about the question
        voi_floor: Minimum VOI threshold for signals to be included (default 0.0 = no filter)
        model: LLM model to use

    Returns:
        MECEScenariosResponse with scenarios, reasoning, and coverage gaps
    """
    model = model or DEFAULT_MODEL

    # Filter signals by VOI floor (if signals have VOI values)
    if voi_floor > 0:
        filtered_signals = [s for s in signals if s.get("voi", 0) >= voi_floor]
        print(f"  VOI floor {voi_floor}: {len(filtered_signals)}/{len(signals)} signals pass")
    else:
        filtered_signals = signals

    # Format signals for prompt
    signals_formatted = []
    for i, s in enumerate(filtered_signals):
        signal_info = {
            "idx": i + 1,
            "text": s.get("text") or s.get("question", ""),
            "source": s.get("source", "unknown"),
        }
        if "reasoning" in s:
            signal_info["reasoning"] = s["reasoning"][:100]
        if "voi" in s:
            signal_info["voi"] = round(s["voi"], 3)
        signals_formatted.append(signal_info)

    signals_json = json.dumps(signals_formatted, indent=2)

    response = await litellm.acompletion(
        model=model,
        messages=[{
            "role": "user",
            "content": MECE_SCENARIO_PROMPT.format(
                question=question,
                context=context,
                n_signals=len(filtered_signals),
                signals_json=signals_json,
            )
        }],
        response_format=MECEScenariosResponse,
    )

    return MECEScenariosResponse.model_validate_json(
        response.choices[0].message.content
    )
