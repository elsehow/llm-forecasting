#!/usr/bin/env python3
"""
Outcome-anchored MECE scenario generation.

Core insight: For a specific outcome variable, the only way to guarantee
mutually exclusive scenarios is to partition the outcome space itself.

Usage:
    from shared.scenario_mece import (
        define_outcome_buckets,
        generate_canonical_stories,
        validate_mece,
    )
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
import litellm

# LLM config
MODEL = "claude-sonnet-4-20250514"


# ============================================================
# Data classes
# ============================================================

@dataclass
class OutcomeBucket:
    """A partition of the outcome space."""
    name: str
    description: str
    lower_bound: float | None  # None = negative infinity
    upper_bound: float | None  # None = positive infinity
    cagr_range: str  # Human-readable, e.g., "1-2% CAGR"

    def contains(self, value: float) -> bool:
        """Check if value falls in this bucket."""
        lower_ok = self.lower_bound is None or value >= self.lower_bound
        upper_ok = self.upper_bound is None or value < self.upper_bound
        return lower_ok and upper_ok


@dataclass
class Scenario:
    """An outcome-anchored scenario with causal story."""
    name: str
    bucket: OutcomeBucket
    story: str  # Causal narrative for how we get to this outcome
    key_drivers: list[str]  # 3-5 factors that drive this outcome
    indicator_bundle: dict[str, str]  # Measurable indicators with thresholds
    resolution_criteria: str  # How experts would know we're here
    signals: list[str] = field(default_factory=list)  # Assigned later


@dataclass
class MECEReport:
    """Validation report for scenario set."""
    is_mece: bool
    coverage: float  # What % of outcome space is covered
    warnings: list[str]
    suggestions: list[str]


# ============================================================
# Pydantic models for LLM structured output
# ============================================================

class StoryResponse(BaseModel):
    """LLM response for a single scenario story."""
    name: str = Field(description="2-4 word memorable scenario name")
    story: str = Field(description="3-5 sentence causal narrative explaining how we reach this outcome")
    key_drivers: list[str] = Field(description="3-5 key factors that drive this outcome")
    indicator_bundle: dict[str, str] = Field(description="3-5 measurable indicators with specific thresholds")
    resolution_criteria: str = Field(description="How experts in 2030 would know we're in this scenario")


class SignalAssignment(BaseModel):
    """Assignment of a signal to scenarios it discriminates."""
    signal: str
    primary_scenario: str = Field(description="Scenario this signal most strongly indicates")
    direction: Literal["increases", "decreases"] = Field(description="Whether observing this signal increases or decreases probability of primary_scenario")
    reasoning: str = Field(description="Why this signal discriminates for this scenario")


class SignalAssignmentsResponse(BaseModel):
    """LLM response for signal assignments."""
    assignments: list[SignalAssignment]


# Models for unified signal-scenario mapping (multi-scenario, directional)

class ScenarioMapping(BaseModel):
    """How a signal relates to one scenario."""
    scenario: str = Field(description="Name of the scenario")
    high_resolution_effect: Literal["increases", "decreases", "neutral"] = Field(
        description="Effect on scenario probability if signal resolves HIGH/YES"
    )
    discrimination_strength: float = Field(
        ge=0, le=1,
        description="0=no discrimination, 1=decisive signal for this scenario"
    )
    reasoning: str = Field(description="Why this signal matters for this scenario")


class SignalMappings(BaseModel):
    """Complete mapping for one signal across all scenarios."""
    signal: str = Field(description="The signal text")
    scenario_mappings: list[ScenarioMapping] = Field(
        description="How this signal relates to each scenario"
    )


class UnifiedMappingsResponse(BaseModel):
    """LLM response for batch of signal mappings."""
    mappings: list[SignalMappings]


# ============================================================
# Core functions
# ============================================================

def define_outcome_buckets(
    variable: str,
    current_value: float,
    horizon_years: int,
    unit: str = "",
    n_buckets: int = 4,
) -> list[OutcomeBucket]:
    """
    Partition outcome space into MECE buckets based on CAGR.

    Default buckets for economic variables:
    - Stagnation: <1% CAGR
    - Baseline: 1-2% CAGR
    - Acceleration: 2-3.5% CAGR
    - Transformation: >3.5% CAGR
    """
    # CAGR thresholds (annualized growth rates)
    cagr_thresholds = [0.01, 0.02, 0.035]  # 1%, 2%, 3.5%

    bucket_configs = [
        ("Stagnation", "<1% CAGR", None, cagr_thresholds[0]),
        ("Baseline", "1-2% CAGR", cagr_thresholds[0], cagr_thresholds[1]),
        ("Acceleration", "2-3.5% CAGR", cagr_thresholds[1], cagr_thresholds[2]),
        ("Transformation", ">3.5% CAGR", cagr_thresholds[2], None),
    ]

    buckets = []
    for name, cagr_range, cagr_low, cagr_high in bucket_configs:
        # Convert CAGR to absolute values
        lower = current_value * ((1 + cagr_low) ** horizon_years) if cagr_low else None
        upper = current_value * ((1 + cagr_high) ** horizon_years) if cagr_high else None

        # Human-readable description
        if lower is None:
            desc = f"{variable} below {upper:.1f}{unit} by {2024 + horizon_years}"
        elif upper is None:
            desc = f"{variable} above {lower:.1f}{unit} by {2024 + horizon_years}"
        else:
            desc = f"{variable} between {lower:.1f}-{upper:.1f}{unit} by {2024 + horizon_years}"

        buckets.append(OutcomeBucket(
            name=name,
            description=desc,
            lower_bound=lower,
            upper_bound=upper,
            cagr_range=cagr_range,
        ))

    return buckets


async def generate_canonical_stories(
    buckets: list[OutcomeBucket],
    variable: str,
    current_value: float,
    unit: str,
    horizon_year: int,
    cruxy_factors: list[str] | None = None,
) -> list[Scenario]:
    """
    For each outcome bucket, generate the most plausible causal story.

    The story explains HOW we get to that outcome, not just WHAT the outcome is.
    """
    if cruxy_factors is None:
        cruxy_factors = [
            "AI/automation impact on productivity",
            "Geopolitical stability (especially US-China, Taiwan)",
            "Fiscal/monetary policy and debt dynamics",
            "Energy transition and costs",
            "Demographics and labor force",
        ]

    scenarios = []

    for bucket in buckets:
        prompt = f"""You are a superforecaster constructing scenarios for {variable} in {horizon_year}.

Current {variable}: {current_value}{unit} (2024)
Target outcome bucket: {bucket.name}
Outcome range: {bucket.description}
Growth trajectory: {bucket.cagr_range}

Key factors to consider:
{json.dumps(cruxy_factors, indent=2)}

Generate a CAUSAL STORY for how the US economy reaches this outcome.

Requirements:
1. The story should be a coherent narrative, not a list of factors
2. Explain the causal chain: what happens first, what follows, how factors interact
3. Be specific about timing (e.g., "by 2028", "in the early 2030s")
4. The story must be PLAUSIBLE - this is how things COULD unfold, not science fiction
5. Indicators should be measurable by 2030 (mid-point check)

The name should capture the essence of this world in 2-4 words.
"""

        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=StoryResponse,
        )

        result = StoryResponse.model_validate_json(response.choices[0].message.content)

        scenarios.append(Scenario(
            name=result.name,
            bucket=bucket,
            story=result.story,
            key_drivers=result.key_drivers,
            indicator_bundle=result.indicator_bundle,
            resolution_criteria=result.resolution_criteria,
        ))

    return scenarios


def validate_mece(scenarios: list[Scenario]) -> MECEReport:
    """
    Validate that scenarios are truly MECE.

    Since we anchor on outcome buckets, MECE is guaranteed by construction.
    This function checks for other issues.
    """
    warnings = []
    suggestions = []

    # Check bucket coverage
    buckets = [s.bucket for s in scenarios]
    has_lower_unbounded = any(b.lower_bound is None for b in buckets)
    has_upper_unbounded = any(b.upper_bound is None for b in buckets)

    if not has_lower_unbounded:
        warnings.append("No scenario covers extreme downside (unbounded lower)")
    if not has_upper_unbounded:
        warnings.append("No scenario covers extreme upside (unbounded upper)")

    # Check for gaps between buckets
    sorted_buckets = sorted(
        [b for b in buckets if b.lower_bound is not None],
        key=lambda b: b.lower_bound
    )
    for i in range(len(sorted_buckets) - 1):
        if sorted_buckets[i].upper_bound != sorted_buckets[i + 1].lower_bound:
            warnings.append(
                f"Gap between {sorted_buckets[i].name} and {sorted_buckets[i+1].name}"
            )

    # Check story distinctiveness (basic heuristic)
    stories = [s.story.lower() for s in scenarios]
    for i, s1 in enumerate(stories):
        for j, s2 in enumerate(stories[i+1:], i+1):
            # Simple word overlap check
            words1 = set(s1.split())
            words2 = set(s2.split())
            overlap = len(words1 & words2) / min(len(words1), len(words2))
            if overlap > 0.7:
                suggestions.append(
                    f"Stories for {scenarios[i].name} and {scenarios[j].name} "
                    f"have high overlap ({overlap:.0%}) - consider differentiating"
                )

    coverage = 1.0 if (has_lower_unbounded and has_upper_unbounded) else 0.9
    is_mece = len(warnings) == 0

    return MECEReport(
        is_mece=is_mece,
        coverage=coverage,
        warnings=warnings,
        suggestions=suggestions,
    )


async def assign_signals_to_scenarios(
    scenarios: list[Scenario],
    signals: list[dict],  # List of {"text": str, "reasoning": str}
) -> dict[str, list[dict]]:
    """
    Map signals to scenarios they discriminate.

    A signal discriminates for scenario S if observing it significantly
    updates P(S) relative to other scenarios.
    """
    scenario_descriptions = [
        {
            "name": s.name,
            "bucket": s.bucket.name,
            "outcome_range": s.bucket.description,
            "story_summary": s.story[:200] + "...",
        }
        for s in scenarios
    ]

    signal_texts = [s["text"] for s in signals]

    prompt = f"""You are assigning early warning signals to GDP scenarios.

SCENARIOS (mutually exclusive outcome buckets):
{json.dumps(scenario_descriptions, indent=2)}

SIGNALS to assign:
{json.dumps(signal_texts, indent=2)}

For each signal, determine:
1. Which scenario does this signal MOST discriminate for?
2. Does observing this signal INCREASE or DECREASE the probability of that scenario?
3. Why does this signal matter for distinguishing between scenarios?

A signal discriminates for a scenario if observing it would cause a significant
update to the probability of that scenario relative to others.
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SignalAssignmentsResponse,
    )

    result = SignalAssignmentsResponse.model_validate_json(response.choices[0].message.content)

    # Group by scenario
    assignments: dict[str, list[dict]] = {s.name: [] for s in scenarios}

    for assignment in result.assignments:
        if assignment.primary_scenario in assignments:
            # Find the original signal dict
            for sig in signals:
                if sig["text"] == assignment.signal:
                    assignments[assignment.primary_scenario].append({
                        **sig,
                        "direction": assignment.direction,
                        "discrimination_reasoning": assignment.reasoning,
                    })
                    break

    return assignments


async def map_signals_to_all_scenarios(
    signals: list[dict],  # [{"text": str, "reasoning": str, "source": str}, ...]
    scenarios: list[dict],  # [{"name": str, "description": str, "gdp_range": str}, ...]
    target_question: str,
    batch_size: int = 10,
) -> list[dict]:
    """
    Map each signal to ALL scenarios with direction and discrimination strength.

    Unlike assign_signals_to_scenarios which picks ONE primary scenario per signal,
    this captures how each signal discriminates for/against ALL scenarios.

    Returns list of signal mappings with full scenario coverage.
    """
    all_mappings = []

    # Process signals in batches
    for i in range(0, len(signals), batch_size):
        batch = signals[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(signals) + batch_size - 1) // batch_size
        print(f"  Processing signal batch {batch_num}/{total_batches}...")

        signal_summaries = [
            {"idx": j, "text": s["text"][:200], "source": s.get("source", "unknown")}
            for j, s in enumerate(batch)
        ]

        scenario_summaries = [
            {"name": s["name"], "gdp_range": s.get("gdp_range", ""), "description": s.get("description", "")[:150]}
            for s in scenarios
        ]

        prompt = f"""You are mapping forecasting signals to scenarios to understand what each signal tells us about future outcomes.

TARGET QUESTION: {target_question}

SCENARIOS (mutually exclusive GDP outcome ranges):
{json.dumps(scenario_summaries, indent=2)}

SIGNALS to map:
{json.dumps(signal_summaries, indent=2)}

For EACH signal, assess its relationship to EACH scenario:

1. HIGH RESOLUTION EFFECT: If this signal resolves HIGH/YES/positive, does it:
   - "increases" probability of this scenario?
   - "decreases" probability of this scenario?
   - "neutral" - minimal effect on this scenario?

2. DISCRIMINATION STRENGTH (0.0 to 1.0):
   - 0.0-0.2: Minimal - this signal tells us little about this scenario
   - 0.3-0.5: Moderate - noticeable probability update
   - 0.6-0.8: Strong - significant probability update
   - 0.9-1.0: Decisive - observing this nearly confirms or rules out the scenario

3. REASONING: Brief explanation of why this signal matters (or doesn't) for this scenario.

IMPORTANT:
- Most signals will discriminate for MULTIPLE scenarios with OPPOSITE directions
- Example: "AI breakthrough by 2027" HIGH increases "Tech Acceleration" but decreases "Stagnation"
- Don't just pick one scenario - assess relationships to ALL scenarios
- LOW resolution effect is typically the opposite of HIGH (but can be neutral for one-sided signals)
"""

        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=UnifiedMappingsResponse,
            )

            result = UnifiedMappingsResponse.model_validate_json(response.choices[0].message.content)

            # Add source info back to mappings
            for mapping in result.mappings:
                # Find original signal to get source
                for sig in batch:
                    if sig["text"][:100] in mapping.signal or mapping.signal[:100] in sig["text"]:
                        all_mappings.append({
                            "signal": sig["text"],
                            "source": sig.get("source", "unknown"),
                            "scenario_mappings": [
                                {
                                    "scenario": sm.scenario,
                                    "high_resolution_effect": sm.high_resolution_effect,
                                    "discrimination_strength": sm.discrimination_strength,
                                    "reasoning": sm.reasoning,
                                }
                                for sm in mapping.scenario_mappings
                            ]
                        })
                        break

        except Exception as e:
            print(f"  Warning: Batch {batch_num} failed: {e}")
            # Add placeholder mappings for failed batch
            for sig in batch:
                all_mappings.append({
                    "signal": sig["text"],
                    "source": sig.get("source", "unknown"),
                    "scenario_mappings": [],
                    "error": str(e),
                })

    return all_mappings


# ============================================================
# Convenience function for full pipeline
# ============================================================

async def build_mece_scenarios(
    variable: str,
    current_value: float,
    unit: str,
    horizon_year: int,
    signals: list[dict] | None = None,
    cruxy_factors: list[str] | None = None,
) -> tuple[list[Scenario], MECEReport]:
    """
    Full pipeline: buckets → stories → validation → signal assignment.

    Returns scenarios with signals assigned and a validation report.
    """
    horizon_years = horizon_year - 2024

    # Step 1: Define outcome buckets
    buckets = define_outcome_buckets(
        variable=variable,
        current_value=current_value,
        horizon_years=horizon_years,
        unit=unit,
    )

    # Step 2: Generate stories
    scenarios = await generate_canonical_stories(
        buckets=buckets,
        variable=variable,
        current_value=current_value,
        unit=unit,
        horizon_year=horizon_year,
        cruxy_factors=cruxy_factors,
    )

    # Step 3: Validate
    report = validate_mece(scenarios)

    # Step 4: Assign signals if provided
    if signals:
        assignments = await assign_signals_to_scenarios(scenarios, signals)
        for scenario in scenarios:
            scenario.signals = [s["text"] for s in assignments.get(scenario.name, [])]

    return scenarios, report
