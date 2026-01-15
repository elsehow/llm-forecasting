#!/usr/bin/env python3
"""
Flexible MECE scenario framework.

Key insight: MECE is a PROPERTY to validate, not a STRUCTURE to impose.

Approaches can use different partitioning strategies:
- Outcome bins (partition by target variable ranges)
- Uncertainty axes (2×2 or n×m matrix)
- Causal pathways (distinct mechanisms)
- Hierarchical (decision tree)

The framework validates MECE and helps fix violations.

Usage:
    from shared.scenario_framework import (
        Scenario,
        validate_mece,
        check_pairwise_exclusivity,
        check_collective_exhaustiveness,
        suggest_fixes,
    )
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
import litellm

# LLM config
MODEL = "claude-sonnet-4-20250514"


# ============================================================
# Core data structures
# ============================================================

@dataclass
class Scenario:
    """
    A scenario with flexible structure.

    The framework doesn't impose what fields a scenario must have.
    Approaches can add their own metadata (uncertainty states,
    outcome ranges, causal mechanisms, etc.)
    """
    id: str
    name: str
    description: str

    # Optional fields - approaches can use whichever are relevant
    outcome_range: tuple[float | None, float | None] | None = None  # (lower, upper)
    uncertainty_states: dict[str, str] | None = None  # {uncertainty_name: state}
    causal_pathway: str | None = None
    probability: float | None = None

    # Signals and indicators
    signals: list[str] = field(default_factory=list)
    indicator_bundle: dict[str, str] = field(default_factory=dict)
    resolution_criteria: str = ""

    # For tracking MECE validation
    _exclusivity_violations: list[str] = field(default_factory=list)


@dataclass
class MECEReport:
    """Validation report for a scenario set."""
    is_mece: bool

    # Mutual exclusivity
    exclusivity_score: float  # 0-1, 1 = perfectly exclusive
    overlapping_pairs: list[tuple[str, str, str]]  # (scenario_a, scenario_b, reasoning)

    # Collective exhaustiveness
    exhaustiveness_score: float  # 0-1, 1 = fully exhaustive
    gaps: list[str]  # Descriptions of uncovered possibility space

    # Suggested fixes
    suggested_merges: list[tuple[str, str, str]]  # (a, b, reason)
    suggested_splits: list[tuple[str, str]]  # (scenario, reason)
    suggested_additions: list[str]  # New scenarios to add


# ============================================================
# Pydantic models for LLM validation
# ============================================================

class PairwiseExclusivityCheck(BaseModel):
    """LLM assessment of whether two scenarios can co-occur."""
    scenario_a: str
    scenario_b: str
    can_cooccur: bool = Field(description="Can these two scenarios both be true simultaneously?")
    overlap_degree: Literal["none", "partial", "substantial", "complete"] = Field(
        description="Degree of overlap: none (mutually exclusive), partial (edge cases overlap), substantial (significant overlap), complete (one contains the other)"
    )
    reasoning: str = Field(description="Why these scenarios can or cannot co-occur")
    suggested_fix: str | None = Field(description="If overlapping, how to fix it")


class ExclusivityResponse(BaseModel):
    """Response from pairwise exclusivity check."""
    checks: list[PairwiseExclusivityCheck]


class ExhaustivenessCheck(BaseModel):
    """LLM assessment of whether scenarios cover the possibility space."""
    covers_full_space: bool = Field(description="Do the scenarios collectively cover all plausible outcomes?")
    coverage_estimate: float = Field(description="Estimated % of possibility space covered (0-100)")
    gaps: list[str] = Field(description="Descriptions of plausible outcomes not covered by any scenario")
    suggested_additions: list[str] = Field(description="New scenarios that would fill gaps")


# ============================================================
# Validation functions
# ============================================================

async def check_pairwise_exclusivity(
    scenarios: list[Scenario],
    context: str,
) -> tuple[float, list[tuple[str, str, str]]]:
    """
    Check if each pair of scenarios is mutually exclusive.

    Returns:
        (exclusivity_score, list of overlapping pairs with reasoning)
    """
    if len(scenarios) < 2:
        return 1.0, []

    # Build pairs
    pairs = []
    for i, a in enumerate(scenarios):
        for b in scenarios[i+1:]:
            pairs.append({
                "scenario_a": a.id,
                "scenario_a_name": a.name,
                "scenario_a_description": a.description,
                "scenario_b": b.id,
                "scenario_b_name": b.name,
                "scenario_b_description": b.description,
            })

    prompt = f"""You are validating whether scenarios are mutually exclusive.

CONTEXT: {context}

SCENARIO PAIRS TO CHECK:
{json.dumps(pairs, indent=2)}

For each pair, determine:
1. Can both scenarios be true at the same time? (co-occur)
2. What is the degree of overlap?
3. If they overlap, how might we fix it?

Two scenarios are mutually exclusive if they CANNOT both be true simultaneously.
Be rigorous - look for edge cases where both could partially apply.
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=ExclusivityResponse,
    )

    result = ExclusivityResponse.model_validate_json(response.choices[0].message.content)

    # Calculate score and collect violations
    overlapping = []
    exclusive_count = 0

    for check in result.checks:
        if check.can_cooccur and check.overlap_degree in ["partial", "substantial", "complete"]:
            overlapping.append((check.scenario_a, check.scenario_b, check.reasoning))
        else:
            exclusive_count += 1

    total_pairs = len(pairs)
    score = exclusive_count / total_pairs if total_pairs > 0 else 1.0

    return score, overlapping


async def check_collective_exhaustiveness(
    scenarios: list[Scenario],
    context: str,
    target_question: str,
) -> tuple[float, list[str], list[str]]:
    """
    Check if scenarios collectively cover the full possibility space.

    Returns:
        (exhaustiveness_score, gaps, suggested_additions)
    """
    scenario_summaries = [
        {"id": s.id, "name": s.name, "description": s.description}
        for s in scenarios
    ]

    prompt = f"""You are validating whether scenarios are collectively exhaustive.

TARGET QUESTION: {target_question}
CONTEXT: {context}

SCENARIOS:
{json.dumps(scenario_summaries, indent=2)}

Assess:
1. Do these scenarios collectively cover ALL plausible outcomes for the target question?
2. What percentage of the possibility space is covered?
3. What gaps exist? What plausible outcomes are NOT covered by any scenario?
4. What scenarios would you add to fill gaps?

Be thorough - consider edge cases, tail risks, and unexpected combinations.
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=ExhaustivenessCheck,
    )

    result = ExhaustivenessCheck.model_validate_json(response.choices[0].message.content)

    score = result.coverage_estimate / 100.0
    return score, result.gaps, result.suggested_additions


async def validate_mece(
    scenarios: list[Scenario],
    context: str,
    target_question: str,
) -> MECEReport:
    """
    Full MECE validation: exclusivity + exhaustiveness.
    """
    # Run both checks
    exclusivity_score, overlapping = await check_pairwise_exclusivity(scenarios, context)
    exhaustiveness_score, gaps, additions = await check_collective_exhaustiveness(
        scenarios, context, target_question
    )

    # Determine if MECE
    # Strict: require high scores on both
    is_mece = exclusivity_score >= 0.9 and exhaustiveness_score >= 0.8

    # Suggest fixes
    suggested_merges = []
    for a, b, reason in overlapping:
        suggested_merges.append((a, b, f"Overlap detected: {reason}"))

    return MECEReport(
        is_mece=is_mece,
        exclusivity_score=exclusivity_score,
        overlapping_pairs=overlapping,
        exhaustiveness_score=exhaustiveness_score,
        gaps=gaps,
        suggested_merges=suggested_merges,
        suggested_splits=[],  # Would need more analysis
        suggested_additions=additions,
    )


def print_mece_report(report: MECEReport) -> None:
    """Pretty-print a MECE validation report."""
    print("\n" + "=" * 50)
    print("MECE VALIDATION REPORT")
    print("=" * 50)

    status = "✓ PASS" if report.is_mece else "✗ FAIL"
    print(f"\nOverall: {status}")

    print(f"\nMutual Exclusivity: {report.exclusivity_score:.0%}")
    if report.overlapping_pairs:
        print("  Overlapping pairs:")
        for a, b, reason in report.overlapping_pairs:
            print(f"    - {a} ↔ {b}")
            print(f"      {reason[:80]}...")
    else:
        print("  No overlaps detected")

    print(f"\nCollective Exhaustiveness: {report.exhaustiveness_score:.0%}")
    if report.gaps:
        print("  Gaps in coverage:")
        for gap in report.gaps:
            print(f"    - {gap[:80]}...")
    else:
        print("  Full coverage")

    if report.suggested_merges:
        print("\nSuggested merges:")
        for a, b, reason in report.suggested_merges:
            print(f"  - Merge {a} + {b}: {reason[:60]}...")

    if report.suggested_additions:
        print("\nSuggested additions:")
        for addition in report.suggested_additions:
            print(f"  - {addition[:80]}...")


# ============================================================
# Helper: Different partitioning strategies
# ============================================================

def partition_by_outcome_bins(
    variable: str,
    current_value: float,
    horizon_years: int,
    n_bins: int = 4,
    cagr_thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Generate scenario skeletons by partitioning outcome space.

    Returns list of dicts with outcome_range that approaches can fill in.
    This is ONE strategy - approaches can use others.
    """
    if cagr_thresholds is None:
        cagr_thresholds = [0.01, 0.02, 0.035]  # Default: 1%, 2%, 3.5%

    bins = []
    prev_bound = None

    for i, threshold in enumerate(cagr_thresholds + [None]):
        if threshold is not None:
            upper = current_value * ((1 + threshold) ** horizon_years)
        else:
            upper = None

        bins.append({
            "outcome_range": (prev_bound, upper),
            "cagr_range": f"{cagr_thresholds[i-1]*100 if i > 0 else 0}-{threshold*100 if threshold else '∞'}%",
        })

        prev_bound = upper

    return bins


def partition_by_uncertainties(
    uncertainties: list[dict],  # [{name, low_state, high_state}, ...]
) -> list[dict]:
    """
    Generate scenario skeletons from uncertainty cross-product.

    For 2 uncertainties with 2 states each → 4 scenarios.
    Returns list of dicts with uncertainty_states that approaches can fill in.
    """
    from itertools import product

    if len(uncertainties) < 1:
        return []

    # Generate all combinations
    states = []
    for u in uncertainties:
        states.append([(u["name"], "low", u["low_state"]),
                       (u["name"], "high", u["high_state"])])

    scenarios = []
    for combo in product(*states):
        uncertainty_states = {name: state for name, level, state in combo}
        state_levels = {name: level for name, level, state in combo}
        scenarios.append({
            "uncertainty_states": uncertainty_states,
            "state_levels": state_levels,
        })

    return scenarios
