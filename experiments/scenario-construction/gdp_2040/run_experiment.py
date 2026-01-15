#!/usr/bin/env python3
"""
Run all scenario generation approaches for a target question.

Usage:
    uv run python experiments/scenario-construction/gdp_2040/run_experiment.py --target renewable_2050
    uv run python experiments/scenario-construction/gdp_2040/run_experiment.py --target gdp_2040

Options:
    --target: Target question (gdp_2040, renewable_2050)
    --skip-eval: Skip evaluation after generation
    --approaches: Comma-separated list of approaches to run (default: all)
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import TARGETS, get_target

APPROACHES = ["hybrid", "topdown", "bottomup"]


async def run_approach(approach: str, target: str) -> dict:
    """Run one approach and return results."""
    script = Path(__file__).parent / f"approach_{approach}_v4.py"

    print(f"\n{'='*60}")
    print(f"RUNNING: {approach.upper()} for {target}")
    print(f"{'='*60}")

    process = await asyncio.create_subprocess_exec(
        "uv", "run", "python", str(script), "--target", target,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout, _ = await process.communicate()
    output = stdout.decode()

    # Print last part of output
    lines = output.strip().split("\n")
    print("\n".join(lines[-30:]))  # Last 30 lines

    return {
        "approach": approach,
        "returncode": process.returncode,
        "success": process.returncode == 0,
    }


async def run_evaluation(target: str) -> dict:
    """Run evaluation on all results for a target."""
    script = Path(__file__).parent / "evaluate.py"

    print(f"\n{'='*60}")
    print(f"EVALUATING: {target}")
    print(f"{'='*60}")

    process = await asyncio.create_subprocess_exec(
        "uv", "run", "python", str(script), "--target", target,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout, _ = await process.communicate()
    output = stdout.decode()

    print(output)

    return {
        "returncode": process.returncode,
        "success": process.returncode == 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run scenario generation experiment")
    parser.add_argument(
        "--target",
        choices=list(TARGETS.keys()),
        required=True,
        help="Target question to generate scenarios for",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after generation",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default=",".join(APPROACHES),
        help="Comma-separated list of approaches to run",
    )
    args = parser.parse_args()

    # Load config for display
    config = get_target(args.target)
    approaches = [a.strip() for a in args.approaches.split(",")]

    print("=" * 60)
    print("SCENARIO GENERATION EXPERIMENT")
    print("=" * 60)
    print(f"\nTarget: {config.question.text}")
    print(f"Approaches: {', '.join(approaches)}")
    print(f"Context: {config.context[:100]}...")

    # Run all approaches in parallel
    print(f"\n\nRunning {len(approaches)} approaches in parallel...")
    results = await asyncio.gather(*[
        run_approach(approach, args.target)
        for approach in approaches
    ])

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  {r['approach']}: {status}")

    all_success = all(r["success"] for r in results)

    # Run evaluation if all succeeded
    if all_success and not args.skip_eval:
        eval_result = await run_evaluation(args.target)
        if not eval_result["success"]:
            print("\nEvaluation FAILED")
            return 1
    elif not all_success:
        print("\nSome approaches failed. Skipping evaluation.")
        return 1

    print("\n\nExperiment complete!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
