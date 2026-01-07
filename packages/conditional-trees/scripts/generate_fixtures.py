#!/usr/bin/env python3
"""Generate test fixtures by running each phase with real API calls."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Force sync mode for faster generation
os.environ["USE_BATCH_API"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))

from conditional_trees.pipeline import load_questions
from conditional_trees.phases.diverge import diverge
from conditional_trees.phases.converge import converge
from conditional_trees.phases.structure import structure
from conditional_trees.phases.quantify import quantify
from conditional_trees.phases.condition import condition
from conditional_trees.phases.signals import signals

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def save_fixture(name: str, data: any):
    """Save data as JSON fixture."""
    path = FIXTURES_DIR / f"{name}.json"

    # Handle Pydantic models
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], "model_dump"):
        data = [item.model_dump() for item in data]

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {path}")


async def main():
    print("Generating test fixtures...\n")

    # Load questions
    questions = load_questions("examples/fri_questions.json")
    save_fixture("questions", questions)
    print(f"Loaded {len(questions)} questions\n")

    # Phase 1: Diverge
    print("=" * 50)
    print("Phase 1: Diverge")
    print("=" * 50)
    raw_scenarios = await diverge(questions, verbose=True)
    save_fixture("phase1_raw_scenarios", raw_scenarios)
    print(f"Generated {len(raw_scenarios)} raw scenarios\n")

    # Phase 2: Converge
    print("=" * 50)
    print("Phase 2: Converge")
    print("=" * 50)
    global_scenarios = await converge(raw_scenarios)
    save_fixture("phase2_global_scenarios", global_scenarios)
    print(f"Created {len(global_scenarios)} global scenarios\n")

    # Phase 3: Structure
    print("=" * 50)
    print("Phase 3: Structure")
    print("=" * 50)
    relationships = await structure(global_scenarios)
    save_fixture("phase3_relationships", relationships)
    print(f"Identified {len(relationships)} relationships\n")

    # Phase 4: Quantify
    print("=" * 50)
    print("Phase 4: Quantify")
    print("=" * 50)
    global_scenarios = await quantify(global_scenarios, relationships)
    save_fixture("phase4_quantified_scenarios", global_scenarios)
    print("Assigned probabilities:\n")
    for s in global_scenarios:
        print(f"  {s.name}: {s.probability:.1%}")
    print()

    # Phase 5: Condition
    print("=" * 50)
    print("Phase 5: Condition")
    print("=" * 50)
    conditionals = await condition(questions, global_scenarios, verbose=True)
    save_fixture("phase5_conditionals", conditionals)
    print(f"Generated {len(conditionals)} conditional forecasts\n")

    # Phase 6: Signals
    print("=" * 50)
    print("Phase 6: Signals")
    print("=" * 50)
    all_signals = await signals(global_scenarios, verbose=True)
    save_fixture("phase6_signals", all_signals)
    print(f"Generated {len(all_signals)} signals\n")

    print("=" * 50)
    print("All fixtures generated!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
