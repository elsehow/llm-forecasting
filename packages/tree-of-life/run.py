#!/usr/bin/env python3
"""CLI entry point for running the forecast tree pipeline."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from tree_of_life.config import setup_logging
from tree_of_life.models import ForecastTree
from tree_of_life.pipeline import (
    build_forecast_tree,
    load_questions,
    load_tree,
    resume_from_phase,
    save_tree,
)


def generate_timestamp() -> str:
    """Generate timestamp string for file naming."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def timestamped_path(path: str, timestamp: str) -> str:
    """Add timestamp to path: foo.json -> foo_2026-01-06T14-30-00.json"""
    p = Path(path)
    return str(p.parent / f"{p.stem}_{timestamp}{p.suffix}")


async def run(
    questions_path: str = "examples/fri_questions.json",
    output_path: str = "output/forecast_tree.json",
    start_date: str | None = None,
    forecast_horizon: str | None = None,
    skip_base_rates: bool = False,
    from_phase: int | None = None,
    input_tree_path: str | None = None,
):
    """Run the pipeline."""
    if from_phase and input_tree_path:
        # Resume from existing tree
        print(f"Loading existing tree from {input_tree_path}")
        tree = load_tree(input_tree_path)
        tree = await resume_from_phase(
            tree,
            from_phase=from_phase,
            skip_base_rates=skip_base_rates,
        )
    else:
        # Full run from scratch
        print(f"Loading questions from {questions_path}")
        questions = load_questions(questions_path)

        tree = await build_forecast_tree(
            questions,
            start_date=start_date,
            forecast_horizon=forecast_horizon,
            skip_base_rates=skip_base_rates,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_tree(tree, output_path)
    print(f"\nSaved forecast tree to {output_path}")

    return tree


def run_diagnostics(tree_path: str):
    """Run diagnostics on a forecast tree."""
    tree = load_tree(tree_path)

    print("=== Tree Diagnostics ===")
    print(f"Created: {tree.created_at}")
    print()

    # Probability diagnostics
    if tree.raw_probability_sum is not None:
        print(f"Probability sum (raw): {tree.raw_probability_sum:.0%}")
    else:
        print("Probability sum: not recorded")
    print(f"Probability status: {tree.probability_status or 'not recorded'}")
    print()

    # Counts
    print(f"Questions: {len(tree.questions)}")
    print(f"Global scenarios: {len(tree.global_scenarios)}")
    print(f"Conditionals: {len(tree.conditionals)}")
    print(f"Signals: {len(tree.signals)}")
    print()

    # Base rates snapshot
    if tree.base_rates_snapshot:
        print(f"Base rates: {len(tree.base_rates_snapshot)} fetched")
        for name, data in tree.base_rates_snapshot.items():
            print(f"  - {name}: {data.get('value')} {data.get('unit', '')} (as of {data.get('as_of', '?')})")
    else:
        print("Base rates: none")
    print()

    # Past-dated signals
    past_signals = tree.past_dated_signals()
    if past_signals:
        print(f"Past-dated signals: {len(past_signals)}")
        for s in past_signals[:3]:
            print(f"  - {s.id}: resolves_by {s.resolves_by}")
        if len(past_signals) > 3:
            print(f"  ... and {len(past_signals) - 3} more")
    else:
        print("Past-dated signals: 0")

    # Scenarios without signals
    missing = tree.scenarios_missing_signals()
    if missing:
        print(f"Scenarios without signals: {missing}")
    else:
        print("Scenarios without signals: 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run conditional forecast tree pipeline"
    )
    parser.add_argument(
        "-q", "--questions",
        default="examples/fri_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        default="output/forecast_tree.json",
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Reference date for 'today' (default: actual today)",
    )
    parser.add_argument(
        "--forecast-horizon",
        default=None,
        help="End of forecast window (default: 2040-12-31)",
    )
    parser.add_argument(
        "--no-base-rates",
        action="store_true",
        help="Disable base rate injection into prompts",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run diagnostics on existing tree instead of generating new one",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help="Resume from this phase (requires --input). Phase 0 = base rates.",
    )
    parser.add_argument(
        "-i", "--input",
        help="Input tree file for --from-phase",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't add timestamp to output filename (overwrites existing)",
    )
    args = parser.parse_args()

    if args.diagnostics:
        setup_logging()  # Use default log file for diagnostics
        run_diagnostics(args.output)
        sys.exit(0)

    if args.from_phase and not args.input:
        parser.error("--from-phase requires --input")

    # Generate timestamp once for both output and log files
    if args.no_timestamp:
        output_path = args.output
        log_path = "output/pipeline.log"
    else:
        ts = generate_timestamp()
        output_path = timestamped_path(args.output, ts)
        log_path = f"output/pipeline_{ts}.log"

    # Set up logging with matched timestamp
    setup_logging(log_path)
    print(f"Logging to {log_path}")

    asyncio.run(run(
        args.questions,
        output_path,
        start_date=args.start_date,
        forecast_horizon=args.forecast_horizon,
        skip_base_rates=args.no_base_rates,
        from_phase=args.from_phase,
        input_tree_path=args.input,
    ))
