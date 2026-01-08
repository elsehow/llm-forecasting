#!/usr/bin/env python3
"""Run the full conditional forecasting evaluation pipeline.

This script runs:
1. Basic conditional elicitation (P(A), P(A|B=1), P(A|B=0))
2. Bayesian consistency check (adds P(B), P(B|A=1), P(B|A=0))
3. Analysis of all metrics (direction, LOTP, Bayes consistency)

Usage:
    # Run all models on all pairs
    uv run python experiments/fb-conditional/run_full_eval.py

    # Run specific models
    uv run python experiments/fb-conditional/run_full_eval.py --models claude-sonnet-4-20250514 gpt-5.2

    # Quick test (3 pairs)
    uv run python experiments/fb-conditional/run_full_eval.py --test

    # Use fresh pairs (after resolving new questions)
    uv run python experiments/fb-conditional/run_full_eval.py --pairs pairs_new.json
"""

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Models we care about
DEFAULT_MODELS = [
    ("claude-sonnet-4-20250514", True),   # Sonnet 4 + thinking
    ("claude-sonnet-4-20250514", False),  # Sonnet 4 no thinking
    ("gpt-5.2", False),                   # GPT-5.2
    ("claude-opus-4-5-20251101", True),   # Opus 4.5 + thinking
    ("claude-opus-4-5-20251101", False),  # Opus 4.5 no thinking
    ("gpt-4o", False),                    # GPT-4o
]


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full conditional forecasting evaluation")
    parser.add_argument("--pairs", type=str, default="experiments/fb-conditional/pairs_filtered.json",
                        help="Path to pairs file")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to run (default: all)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run (3 pairs, 1 model)")
    parser.add_argument("--skip-basic", action="store_true",
                        help="Skip basic elicitation (run_experiment.py)")
    parser.add_argument("--skip-bayesian", action="store_true",
                        help="Skip Bayesian consistency check")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run analysis on existing results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine which models to run
    if args.test:
        models = [("claude-sonnet-4-20250514", True)]
        limit = 3
        print("\n*** TEST MODE: 3 pairs, 1 model ***\n")
    else:
        if args.models:
            # Parse model specs like "claude-sonnet-4-20250514:thinking"
            models = []
            for m in args.models:
                if ":thinking" in m:
                    models.append((m.replace(":thinking", ""), True))
                else:
                    models.append((m, False))
        else:
            models = DEFAULT_MODELS
        limit = None

    if args.analyze_only:
        print("\n*** ANALYZE ONLY MODE ***\n")
        # Just run analysis on most recent results
        run_command(
            ["uv", "run", "python", "experiments/fb-conditional/analyze_direction.py"],
            "Analyzing most recent results"
        )
        return

    # Run evaluations for each model
    for model, thinking in models:
        model_desc = f"{model}{' + thinking' if thinking else ''}"

        # 1. Basic elicitation
        if not args.skip_basic:
            cmd = [
                "uv", "run", "python", "experiments/fb-conditional/run_experiment.py",
                "--model", model,
                "--pairs", args.pairs,
                "--prompt", "skeptical",
            ]
            if thinking:
                cmd.append("--thinking")
            if limit:
                cmd.extend(["--limit", str(limit)])

            if not run_command(cmd, f"Basic elicitation: {model_desc}"):
                print(f"WARNING: Basic elicitation failed for {model_desc}")

        # 2. Bayesian consistency check
        if not args.skip_bayesian:
            cmd = [
                "uv", "run", "python", "experiments/fb-conditional/bayesian_check.py",
                "--model", model,
                "--pairs", args.pairs,
            ]
            if thinking:
                cmd.append("--thinking")
            if limit:
                cmd.extend(["--limit", str(limit)])

            if not run_command(cmd, f"Bayesian check: {model_desc}"):
                print(f"WARNING: Bayesian check failed for {model_desc}")

    # 3. Run analysis on all results
    print("\n" + "="*60)
    print("RUNNING ANALYSIS")
    print("="*60)

    results_dir = Path("experiments/fb-conditional/results")
    if results_dir.exists():
        # Analyze most recent result for each model
        for model, thinking in models:
            model_short = model.split("/")[-1]
            thinking_suffix = "_thinking" if thinking else ""

            # Find most recent result file for this model
            pattern = f"*{model_short}*{thinking_suffix}*.json"
            files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            if files:
                run_command(
                    ["uv", "run", "python", "experiments/fb-conditional/analyze_direction.py", str(files[0])],
                    f"Analysis: {model}{' + thinking' if thinking else ''}"
                )

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nResults saved to: experiments/fb-conditional/results/")
    print(f"Run timestamp: {timestamp}")


if __name__ == "__main__":
    main()
