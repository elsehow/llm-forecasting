#!/usr/bin/env python3
"""Generate a signal tree for a target question.

Usage:
    uv run python experiments/signal-tree/generate.py --target "one_battle_best_picture"
    uv run python experiments/signal-tree/generate.py --target "democrat_whitehouse_2028" --max-depth 3
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from shared.decomposition import build_signal_tree
from shared.rollup import analyze_tree
from shared.tree import TreeGenerationConfig


# Predefined targets (can also pass custom question via --question)
# Note: Trees now terminate based on signal resolution time (minimum_resolution_days)
# and max_signals budget, not max_depth.
TARGETS = {
    "one_battle_best_picture": {
        "question": "Will One Battle After Another win Best Picture at the 98th Academy Awards?",
        "horizon_days": 60,
        "minimum_resolution_days": 7,
        "max_signals": 30,  # Short horizon, smaller tree
    },
    "democrat_whitehouse_2028": {
        "question": "Will a Democrat win the White House in 2028?",
        "horizon_days": 365,
        "minimum_resolution_days": 14,
        "max_signals": 100,  # Long horizon, larger tree
    },
    "carney_pm_2027": {
        "question": "Will Mark Carney be Prime Minister of Canada on December 31, 2027?",
        "horizon_days": 365,
        "minimum_resolution_days": 14,
        "max_signals": 50,
    },
    "us_gdp_2029": {
        "question": "What will US real GDP be in 2029 (in 2024 dollars)?",
        "horizon_days": 365,
        "minimum_resolution_days": 14,
        "max_signals": 100,
    },
}


async def main():
    parser = argparse.ArgumentParser(description="Generate a signal tree")
    parser.add_argument(
        "--target",
        type=str,
        help="Target name (one of: " + ", ".join(TARGETS.keys()) + ")",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Custom target question (alternative to --target)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="Actionable horizon in days (default: from target config or 365)",
    )
    parser.add_argument(
        "--minimum-resolution-days",
        type=int,
        default=None,
        help="Stop decomposing when signal resolves within this many days (default: 7)",
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=None,
        help="Maximum total signals in tree (default: 100)",
    )
    parser.add_argument(
        "--signals-per-node",
        type=int,
        default=5,
        help="Number of signals to generate per node (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results/<target>/<timestamp>.json)",
    )

    args = parser.parse_args()

    # Determine target question and settings
    if args.target:
        if args.target not in TARGETS:
            print(f"Unknown target: {args.target}")
            print(f"Available targets: {', '.join(TARGETS.keys())}")
            return
        target_config = TARGETS[args.target]
        question = target_config["question"]
        horizon_days = args.horizon_days or target_config.get("horizon_days", 365)
        minimum_resolution_days = args.minimum_resolution_days or target_config.get("minimum_resolution_days", 7)
        max_signals = args.max_signals or target_config.get("max_signals", 100)
        target_name = args.target
    elif args.question:
        question = args.question
        horizon_days = args.horizon_days or 365
        minimum_resolution_days = args.minimum_resolution_days or 7
        max_signals = args.max_signals or 100
        target_name = "custom"
    else:
        print("Either --target or --question must be provided")
        return

    print(f"=" * 60)
    print(f"Signal Tree Generation")
    print(f"=" * 60)
    print(f"Target: {question}")
    print(f"Horizon: {horizon_days} days")
    print(f"Minimum resolution: {minimum_resolution_days} days")
    print(f"Max signals: {max_signals}")
    print(f"Signals per node: {args.signals_per_node}")
    print()

    # Build config
    config = TreeGenerationConfig(
        target_question=question,
        target_id=target_name,
        actionable_horizon_days=horizon_days,
        minimum_resolution_days=minimum_resolution_days,
        max_signals=max_signals,
        signals_per_node=args.signals_per_node,
    )

    # Generate tree
    print("Generating signal tree...")
    tree = await build_signal_tree(config, today=date.today())

    print(f"  Generated {len(tree.signals)} signals")
    print(f"  Max depth: {tree.max_depth}")
    print(f"  Leaf count: {tree.leaf_count}")
    print()

    # Analyze tree
    print("Analyzing tree...")
    analysis = analyze_tree(tree, target_prior=0.5)

    print(f"  Computed P(target): {analysis['computed_probability']:.1%}")
    print()

    # Print top contributors
    print("Top contributing signals:")
    for i, contrib in enumerate(analysis["top_contributors"][:5], 1):
        direction = "↑" if contrib["direction"] == "enhances" else "↓"
        print(
            f"  {i}. {contrib['text'][:60]}..."
        )
        print(
            f"     p={contrib['base_rate']:.0%}, ρ={contrib['rho']:+.2f}, "
            f"contribution={contrib['contribution']:.3f} {direction}"
        )
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results" / target_name
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().strftime("%Y%m%d")
        output_path = results_dir / f"tree_{timestamp}.json"

    # Serialize
    output_data = {
        "config": config.model_dump(),
        "tree": tree.model_dump(),
        "analysis": analysis,
        "generated_at": date.today().isoformat(),
    }

    # Custom serializer for dates
    def date_serializer(obj):
        if isinstance(obj, date):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=date_serializer)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
