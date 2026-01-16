"""Shared output formatting for scenario construction approaches.

Consolidates duplicated output patterns across approach_*.py files.
"""

import json
from datetime import datetime
from uuid import uuid4


def build_scenario_dicts(scenarios) -> list[dict]:
    """Convert scenario result objects to JSON-serializable dicts.

    Args:
        scenarios: List of scenario objects with name, description, outcome_range, etc.

    Returns:
        List of dicts ready for JSON serialization
    """
    result = []
    for s in scenarios:
        d = {
            "name": s.name,
            "description": s.description,
            "key_drivers": s.key_drivers,
            "why_exclusive": s.why_exclusive,
            "signal_impacts": [
                {"signal_index": si.signal_index, "effect": si.effect}
                for si in s.signal_impacts
            ],
            "indicator_bundle": s.indicator_bundle,
        }
        # Continuous question fields
        if s.outcome_range is not None:
            d["outcome_range"] = s.outcome_range
            d["outcome_low"] = s.outcome_low
            d["outcome_high"] = s.outcome_high
        # Binary question fields
        if s.probability_range is not None:
            d["probability_range"] = s.probability_range
            d["probability_low"] = s.probability_low
            d["probability_high"] = s.probability_high
        result.append(d)
    return result


def build_question_dict(config) -> dict:
    """Build the question section of the output JSON.

    Args:
        config: Target config object with question attribute

    Returns:
        Dict with question metadata
    """
    q = config.question
    return {
        "id": q.id,
        "source": q.source,
        "text": q.text,
        "question_type": q.question_type.value,
        "unit": q.unit.type if q.unit else None,
        "base_rate": q.base_rate,
        "value_range": list(q.value_range) if q.value_range else None,
    }


def build_base_results(
    approach: str,
    target: str,
    config,
    signals_v7: list,
    scenarios_v7: list[dict],
    mece_reasoning: str,
    coverage_gaps: str | None,
    voi_floor: float,
    max_horizon_days: int | None = None,
) -> dict:
    """Build the common base structure for results JSON.

    Args:
        approach: Approach name (e.g., "bottomup", "hybrid")
        target: Target question key
        config: Target config object
        signals_v7: List of Signal model instances
        scenarios_v7: List of scenario dicts (from build_scenario_dicts)
        mece_reasoning: MECE reasoning from scenario generation
        coverage_gaps: Coverage gaps from scenario generation
        voi_floor: VOI floor threshold used
        max_horizon_days: Max days until resolution for actionable signals (optional)

    Returns:
        Dict with common result structure (approaches can extend with extras)
    """
    results = {
        "id": f"{approach}_{target}_{uuid4().hex[:8]}",
        "name": f"{target} ({approach})",
        "target": target,
        "approach": f"{approach}_v7",
        "question": build_question_dict(config),
        "config": {
            "context": config.context,
            "cruxiness_normalizer": config.cruxiness_normalizer,
            "voi_floor": voi_floor,
        },
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "signals": [s.model_dump(mode="json", exclude_none=True) for s in signals_v7],
        "scenarios": scenarios_v7,
        "mece_reasoning": mece_reasoning,
        "coverage_gaps": coverage_gaps,
        "created_at": datetime.now().isoformat(),
    }

    if max_horizon_days is not None:
        results["max_horizon_days"] = max_horizon_days

    return results


def print_results(result) -> None:
    """Print formatted results to console.

    Args:
        result: Scenario generation result with scenarios, mece_reasoning, coverage_gaps
    """
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMECE Reasoning: {result.mece_reasoning}")

    if result.coverage_gaps:
        print(f"\nCoverage Gaps: {result.coverage_gaps}")
    else:
        print("\nCoverage Gaps: None")

    print("\n" + "-" * 40)
    for s in result.scenarios:
        print(f"\n### {s.name}")
        # Handle both continuous (outcome_range) and binary (probability_range)
        if s.outcome_range is not None:
            print(f"  Outcome Range: {s.outcome_range} (low={s.outcome_low}, high={s.outcome_high})")
        elif s.probability_range is not None:
            print(f"  Probability: {s.probability_range} (low={s.probability_low}, high={s.probability_high})")
        print(f"  {s.description}")
        print(f"\n  Why Exclusive: {s.why_exclusive[:80]}...")
        print(f"\n  Key Drivers: {', '.join(s.key_drivers[:3])}")


def save_results(results: dict, output_file) -> None:
    """Save results dict to JSON file.

    Args:
        results: Results dict to save
        output_file: Path to output file
    """
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")


def count_by_field(items: list[dict], field: str, default: str = "unknown") -> dict:
    """Count items by a field value.

    Args:
        items: List of dicts
        field: Field name to count by
        default: Default value if field is missing

    Returns:
        Dict mapping field values to counts
    """
    counts = {}
    for item in items:
        value = item.get(field, default)
        counts[value] = counts.get(value, 0) + 1
    return counts
