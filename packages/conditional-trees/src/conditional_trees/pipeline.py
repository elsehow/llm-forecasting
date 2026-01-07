"""Pipeline orchestrator for conditional forecasting trees."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import FORECAST_HORIZON, START_DATE
from .models import ForecastTree, Question, QuestionType
from .phases.base_rates import fetch_base_rates, format_base_rates_context
from .phases.condition import condition
from .phases.converge import converge
from .phases.diverge import diverge
from .phases.quantify import quantify
from .phases.signals import signals
from .phases.structure import structure


async def build_forecast_tree(
    questions: list[Question],
    start_date: str | None = None,
    forecast_horizon: str | None = None,
    skip_base_rates: bool = False,
    verbose: bool = True,
) -> ForecastTree:
    """Build a complete forecast tree from a list of questions.

    Pipeline:
    0. Base Rates: Fetch current reference values via web search
    1. Diverge: Generate scenarios for each question
    2. Converge: Cluster into global scenarios
    3. Structure: Identify relationships
    4. Quantify: Assign probabilities
    5. Condition: Generate conditional forecasts
    6. Signals: Generate early warning signals

    Args:
        questions: List of forecasting questions
        start_date: Reference date for "today" (default: config.START_DATE)
        forecast_horizon: End of forecast window (default: config.FORECAST_HORIZON)
        skip_base_rates: Skip Phase 0 base rate fetching (default: False)
        verbose: Print progress messages
    """
    # Use defaults from config if not provided
    start = start_date or START_DATE
    horizon = forecast_horizon or FORECAST_HORIZON

    if verbose:
        print(f"Starting pipeline with {len(questions)} questions...")
        print(f"  Start date: {start}, Forecast horizon: {horizon}")

    # Phase 0: Base Rates
    base_rates_snapshot = None
    base_rate_context = ""
    if not skip_base_rates:
        if verbose:
            print("\n[Phase 0] Base Rates: Fetching current reference values...")
        base_rates_snapshot = await fetch_base_rates(questions, verbose=verbose)
        base_rate_context = format_base_rates_context(base_rates_snapshot)
        if verbose and base_rates_snapshot:
            print(f"  Fetched {len(base_rates_snapshot)} base rates")
    elif verbose:
        print("\n[Phase 0] Base Rates: Skipped")

    # Phase 1: Diverge
    if verbose:
        print("\n[Phase 1] Diverge: Generating scenarios...")
    raw_scenarios = await diverge(
        questions,
        start_date=start,
        forecast_horizon=horizon,
        base_rate_context=base_rate_context,
        verbose=verbose,
    )
    if verbose:
        print(f"  Generated {len(raw_scenarios)} raw scenarios")

    # Phase 2: Converge
    if verbose:
        print("\n[Phase 2] Converge: Clustering into global scenarios...")
    global_scenarios = await converge(raw_scenarios)
    if verbose:
        print(f"  Created {len(global_scenarios)} global scenarios:")
        for s in global_scenarios:
            print(f"    - {s.name}")

    # Phase 3: Structure
    if verbose:
        print("\n[Phase 3] Structure: Analyzing relationships...")
    relationships = await structure(global_scenarios)
    if verbose:
        print(f"  Identified {len(relationships)} relationships")

    # Phase 4: Quantify
    if verbose:
        print("\n[Phase 4] Quantify: Assigning probabilities...")
    global_scenarios, prob_result = await quantify(global_scenarios, relationships, verbose=verbose)
    if verbose:
        print(f"  Probability status: {prob_result.status} (raw sum: {prob_result.raw_sum:.1%})")
        for s in global_scenarios:
            print(f"    - {s.name}: {s.probability:.1%}")

    # Phase 5: Condition
    if verbose:
        print("\n[Phase 5] Condition: Generating conditional forecasts...")
    conditionals = await condition(
        questions,
        global_scenarios,
        base_rate_context=base_rate_context,
        verbose=verbose,
    )
    if verbose:
        print(f"  Generated {len(conditionals)} conditional forecasts")

    # Phase 6: Signals
    if verbose:
        print("\n[Phase 6] Signals: Generating early warning signals...")
    all_signals = await signals(global_scenarios, verbose=verbose)
    if verbose:
        print(f"  Generated {len(all_signals)} signals")

    # Build final tree
    tree = ForecastTree(
        questions=questions,
        raw_scenarios=raw_scenarios,
        global_scenarios=global_scenarios,
        relationships=relationships,
        conditionals=conditionals,
        signals=all_signals,
        created_at=datetime.now().isoformat(),
        raw_probability_sum=prob_result.raw_sum,
        probability_status=prob_result.status,
        base_rates_snapshot=base_rates_snapshot,
    )

    if verbose:
        print("\nPipeline complete!")

    return tree


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a JSON file.

    Transforms tree question format (with 'type' field) to core Question format
    (with 'question_type' enum and required 'source' field).
    """
    with open(path) as f:
        data = json.load(f)

    questions = []
    for q in data:
        # Transform tree format to core format
        q_data = {
            "id": q["id"],
            "text": q["text"],
            "source": "tree",  # Tree questions don't come from a market source
            "question_type": QuestionType(q["type"]),  # Map 'type' string to enum
            "options": q.get("options"),
            "resolution_source": q.get("resolution_source"),
            "domain": q.get("domain"),
        }
        questions.append(Question(**q_data))
    return questions


def load_tree(path: str | Path) -> ForecastTree:
    """Load a forecast tree from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return ForecastTree(**data)


def save_tree(tree: ForecastTree, path: str | Path) -> None:
    """Save forecast tree to a JSON file.

    Hydrates upstream_scenarios from relationships before saving,
    so the serialized JSON includes derived causal dependencies.
    """
    hydrated = tree.hydrate_upstream()
    with open(path, "w") as f:
        f.write(hydrated.model_dump_json(indent=2))


async def resume_from_phase(
    tree: ForecastTree,
    from_phase: int,
    skip_base_rates: bool = False,
    verbose: bool = True,
) -> ForecastTree:
    """Resume pipeline from a specific phase using existing tree data.

    Args:
        tree: Existing forecast tree with prior phase outputs
        from_phase: Phase number to start from (0-6)
        skip_base_rates: Skip Phase 0 base rate fetching (default: False)
        verbose: Print progress messages

    Returns:
        Updated ForecastTree with re-run phases
    """
    if from_phase < 0 or from_phase > 6:
        raise ValueError(f"from_phase must be 0-6, got {from_phase}")

    questions = tree.questions
    raw_scenarios = tree.raw_scenarios
    global_scenarios = tree.global_scenarios
    relationships = tree.relationships
    conditionals = tree.conditionals
    all_signals = tree.signals
    prob_result = None
    base_rates_snapshot = tree.base_rates_snapshot
    base_rate_context = format_base_rates_context(base_rates_snapshot) if base_rates_snapshot else ""

    if verbose:
        print(f"Resuming pipeline from phase {from_phase}...")

    # Phase 0: Base Rates
    if from_phase <= 0 and not skip_base_rates:
        if verbose:
            print("\n[Phase 0] Base Rates: Fetching current reference values...")
        base_rates_snapshot = await fetch_base_rates(questions, verbose=verbose)
        base_rate_context = format_base_rates_context(base_rates_snapshot)
        if verbose and base_rates_snapshot:
            print(f"  Fetched {len(base_rates_snapshot)} base rates")

    # Phase 1: Diverge
    if from_phase <= 1:
        if verbose:
            print("\n[Phase 1] Diverge: Generating scenarios...")
        raw_scenarios = await diverge(
            questions,
            base_rate_context=base_rate_context,
            verbose=verbose,
        )
        if verbose:
            print(f"  Generated {len(raw_scenarios)} raw scenarios")

    # Phase 2: Converge
    if from_phase <= 2:
        if verbose:
            print("\n[Phase 2] Converge: Clustering into global scenarios...")
        global_scenarios = await converge(raw_scenarios)
        if verbose:
            print(f"  Created {len(global_scenarios)} global scenarios:")
            for s in global_scenarios:
                print(f"    - {s.name}")

    # Phase 3: Structure
    if from_phase <= 3:
        if verbose:
            print("\n[Phase 3] Structure: Analyzing relationships...")
        relationships = await structure(global_scenarios)
        if verbose:
            print(f"  Identified {len(relationships)} relationships")

    # Phase 4: Quantify
    if from_phase <= 4:
        if verbose:
            print("\n[Phase 4] Quantify: Assigning probabilities...")
        global_scenarios, prob_result = await quantify(global_scenarios, relationships, verbose=verbose)
        if verbose:
            print(f"  Probability status: {prob_result.status} (raw sum: {prob_result.raw_sum:.1%})")
            for s in global_scenarios:
                print(f"    - {s.name}: {s.probability:.1%}")

    # Phase 5: Condition
    if from_phase <= 5:
        if verbose:
            print("\n[Phase 5] Condition: Generating conditional forecasts...")
        conditionals = await condition(
            questions,
            global_scenarios,
            base_rate_context=base_rate_context,
            verbose=verbose,
        )
        if verbose:
            print(f"  Generated {len(conditionals)} conditional forecasts")

    # Phase 6: Signals
    if from_phase <= 6:
        if verbose:
            print("\n[Phase 6] Signals: Generating early warning signals...")
        all_signals = await signals(global_scenarios, verbose=verbose)
        if verbose:
            print(f"  Generated {len(all_signals)} signals")

    # Build updated tree
    updated_tree = ForecastTree(
        questions=questions,
        raw_scenarios=raw_scenarios,
        global_scenarios=global_scenarios,
        relationships=relationships,
        conditionals=conditionals,
        signals=all_signals,
        created_at=datetime.now().isoformat(),
        raw_probability_sum=prob_result.raw_sum if prob_result else tree.raw_probability_sum,
        probability_status=prob_result.status if prob_result else tree.probability_status,
        base_rates_snapshot=base_rates_snapshot,
    )

    if verbose:
        print("\nPipeline complete!")

    return updated_tree


async def main():
    """Run the pipeline on the FRI questions."""
    questions = load_questions("examples/fri_questions.json")
    tree = await build_forecast_tree(questions)
    save_tree(tree, "output/forecast_tree.json")
    return tree


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
