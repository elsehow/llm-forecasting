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

from typing import Literal


class SignalImpact(BaseModel):
    """How a signal affects a scenario's probability."""
    signal_index: int = Field(description="Index of signal (0-based, from input array)")
    effect: Literal["enhances", "suppresses"] = Field(
        description="'enhances' = signal YES makes scenario more likely; 'suppresses' = signal YES makes scenario less likely"
    )


class MECEScenario(BaseModel):
    """A single MECE scenario."""
    name: str = Field(description="2-4 word memorable name")
    description: str = Field(description="2-3 sentences describing this world state")

    # LLM-estimated probability of this scenario occurring (E2E approach)
    scenario_probability: float = Field(
        description="Estimated probability of this scenario occurring (0.0-1.0). Must sum to 1.0 across all scenarios."
    )

    # For continuous questions (GDP, population, etc.)
    outcome_range: str | None = Field(default=None, description="Expected outcome range, e.g., '$45-55T' for GDP")
    outcome_low: float | None = Field(default=None, description="Numeric lower bound (e.g., 45.0 for $45T)")
    outcome_high: float | None = Field(default=None, description="Numeric upper bound (e.g., 55.0 for $55T)")

    # For binary questions (will X happen?)
    probability_range: str | None = Field(default=None, description="Probability estimate, e.g., '60-75% YES'")
    probability_low: float | None = Field(default=None, description="Lower probability bound (e.g., 0.60)")
    probability_high: float | None = Field(default=None, description="Upper probability bound (e.g., 0.75)")

    # Common fields
    key_drivers: list[str] = Field(description="3-5 factors that define this scenario")
    why_exclusive: str = Field(description="Why this scenario CANNOT co-occur with the others")
    signal_impacts: list[SignalImpact] = Field(
        description="Which signals affect this scenario and how (enhances or suppresses)"
    )
    indicator_bundle: dict[str, str] = Field(description="3-5 measurable indicators with thresholds")


class MECEScenariosResponse(BaseModel):
    """Response from MECE scenario generation."""
    scenarios: list[MECEScenario]
    mece_reasoning: str = Field(description="Explanation of why these scenarios are MECE")
    coverage_gaps: list[str] | str | None = Field(
        default=None,
        description="Any outcomes not covered (should be empty or minimal)"
    )


# ============================================================
# Core generation function
# ============================================================

MECE_SCENARIO_PROMPT_CONTINUOUS = """You are constructing MECE scenarios for forecasting.

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
   - Provide both human-readable outcome_range (e.g., "$45-55T") AND numeric bounds:
     * outcome_low: numeric lower bound (e.g., 45.0)
     * outcome_high: numeric upper bound (e.g., 55.0)

4. SIGNAL IMPACTS:
   For each scenario, identify which signals affect it and HOW:
   - ENHANCES: If this signal resolves YES, this scenario becomes MORE likely
   - SUPPRESSES: If this signal resolves YES, this scenario becomes LESS likely

   Use the signal's "idx" value (0-based index) in signal_impacts.

   Guidelines:
   - Aim for at least 2 enhancing AND 2 suppressing signals per scenario
   - A signal can enhance one scenario while suppressing another
   - Only include signals with clear directional impact (skip neutral ones)
   - Think about the causal mechanism: WHY does the signal affect this scenario?

5. For each scenario, explain WHY it cannot co-occur with the others.

6. Acknowledge any coverage gaps honestly (ideally there are none).

7. SCENARIO PROBABILITY (scenario_probability):
   - For each scenario, estimate P(scenario) - the probability that THIS scenario occurs
   - Consider all signals holistically when making this estimate
   - Think about: current trends, signal base rates, geopolitical context, expert forecasts
   - Probabilities across ALL scenarios MUST sum to 1.0 (they are MECE)
   - Provide scenario_probability as a decimal (e.g., 0.35 for 35%)

Think step by step:
- What are the key axes of uncertainty implied by these signals?
- How can we partition the outcome space so scenarios don't overlap?
- For each scenario, which signals would make it more/less likely?
- Given current evidence, what is the probability of each scenario?
"""

MECE_SCENARIO_PROMPT_BINARY = """You are constructing MECE scenarios for a BINARY forecasting question.

TARGET QUESTION: {question}
CONTEXT: {context}

I have {n_signals} signals that may affect this forecast:

{signals_json}

YOUR TASK: Generate 3-5 MECE SCENARIOS based on these signals.

CRITICAL: This is a BINARY question (YES/NO outcome). Instead of outcome ranges, each scenario
has a PROBABILITY ESTIMATE for the target question.

CRITICAL REQUIREMENTS:

1. MUTUALLY EXCLUSIVE: Each scenario must be a distinct world-state that CANNOT co-occur with the others.
   - Scenarios represent different "worlds" that could lead to the outcome
   - Bad: "Blue Wave" and "High Turnout" (these can happen together)
   - Good: "Landslide Victory" vs "Narrow Win" vs "Close Loss" vs "Major Defeat" (different states)

2. COLLECTIVELY EXHAUSTIVE: Together, the scenarios should cover ALL plausible world-states.
   - Don't leave gaps in the possibility space
   - Include moderate/baseline scenarios, not just extremes

3. PROBABILITY ANCHORING:
   - Each scenario gets a PROBABILITY RANGE for the target question's YES outcome
   - Example: "Blue Wave" scenario → 75-85% YES
   - Provide both human-readable probability_range (e.g., "65-75% YES") AND numeric bounds:
     * probability_low: lower bound as decimal (e.g., 0.65)
     * probability_high: upper bound as decimal (e.g., 0.75)
   - DO NOT use outcome_range, outcome_low, outcome_high (those are for continuous questions)

4. SIGNAL IMPACTS:
   For each scenario, identify which signals affect it and HOW:
   - ENHANCES: If this signal resolves YES, this scenario becomes MORE likely
   - SUPPRESSES: If this signal resolves YES, this scenario becomes LESS likely

   Use the signal's "idx" value (0-based index) in signal_impacts.

   Guidelines:
   - Aim for at least 2 enhancing AND 2 suppressing signals per scenario
   - A signal can enhance one scenario while suppressing another
   - Only include signals with clear directional impact (skip neutral ones)
   - Think about the causal mechanism: WHY does the signal affect this scenario?

5. For each scenario, explain WHY it cannot co-occur with the others.

6. Acknowledge any coverage gaps honestly (ideally there are none).

7. SCENARIO PROBABILITY (scenario_probability):
   - For each scenario, estimate P(scenario) - the probability that THIS scenario occurs
   - Consider all signals holistically when making this estimate
   - Think about: current trends, signal base rates, geopolitical context, expert forecasts
   - Probabilities across ALL scenarios MUST sum to 1.0 (they are MECE)
   - Provide scenario_probability as a decimal (e.g., 0.35 for 35%)

Think step by step:
- What are the key axes of uncertainty implied by these signals?
- How can we partition the world-state space so scenarios don't overlap?
- For each scenario, what probability would we assign to the target question's YES outcome?
- For each scenario, which signals would make it more/less likely?
- Given current evidence, what is the probability of each scenario?
"""


async def generate_mece_scenarios(
    signals: list[dict],
    question: str,
    context: str,
    question_type: str = "continuous",
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
        question_type: "continuous" (GDP, population) or "binary" (yes/no)
        voi_floor: Minimum VOI threshold for signals to be included (default 0.0 = no filter)
        model: LLM model to use

    Returns:
        MECEScenariosResponse with scenarios, reasoning, and coverage gaps
    """
    model = model or DEFAULT_MODEL

    # Select prompt based on question type
    if question_type == "binary":
        prompt_template = MECE_SCENARIO_PROMPT_BINARY
    else:
        prompt_template = MECE_SCENARIO_PROMPT_CONTINUOUS

    # Filter signals by VOI floor (if signals have VOI values)
    if voi_floor > 0:
        filtered_signals = [s for s in signals if s.get("voi", 0) >= voi_floor]
        print(f"  VOI floor {voi_floor}: {len(filtered_signals)}/{len(signals)} signals pass")
    else:
        filtered_signals = signals

    # Format signals for prompt (0-indexed)
    signals_formatted = []
    for i, s in enumerate(filtered_signals):
        signal_info = {
            "idx": i,  # 0-based index for signal_impacts reference
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
            "content": prompt_template.format(
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


async def generate_and_print_scenarios(
    signals: list[dict],
    question: str,
    context: str,
    question_type: str = "continuous",
    voi_floor: float = 0.0,
    model: str | None = None,
) -> MECEScenariosResponse:
    """Generate MECE scenarios and print results.

    Wrapper around generate_mece_scenarios that also:
    - Restructures signals for the prompt
    - Prints results using print_results()

    Args:
        signals: List of signal dicts (with text, source, voi keys)
        question: Target question text
        context: Question context
        question_type: "continuous" or "binary"
        voi_floor: Minimum VOI for signals to include
        model: LLM model to use

    Returns:
        MECEScenariosResponse with scenarios
    """
    from .output import print_results

    print("Generating MECE scenarios...")

    # Restructure signals for the prompt
    signals_for_prompt = [
        {"text": s.get("text") or s.get("question", ""), "source": s["source"], "voi": s.get("voi", 0)}
        for s in signals
    ]

    result = await generate_mece_scenarios(
        signals=signals_for_prompt,
        question=question,
        context=context,
        question_type=question_type,
        voi_floor=voi_floor,
        model=model,
    )

    print_results(result)
    return result
