#!/usr/bin/env python3
"""
Top-down approach v7: LLM-generated signals with enhanced data model.

Pipeline:
1. identify_uncertainties(question, n=3) — shared with hybrid
2. LLM generates signals per uncertainty (with resolution_date, base_rate)
3. rank_signals_by_voi(signals, target) — shared with bottom-up/hybrid
4. generate_mece_scenarios(signals, voi_floor=0.1) — shared with bottom-up/hybrid

v7 changes: Enhanced signal/scenario output for UI hydration.
- LLM-generated signals include resolution_date, base_rate
- Scenarios include outcome_low, outcome_high (numeric)

Usage:
    uv run python experiments/scenario-construction/gdp_2040/approach_topdown_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/gdp_2040/approach_topdown_v7.py --target renewable_2050
"""

import argparse
import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv

from llm_forecasting.models import Signal

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.signals import rank_signals_by_voi, parse_date
from shared.generation import generate_signals_for_uncertainty
from shared.uncertainties import identify_uncertainties
from shared.scenarios import generate_mece_scenarios
from shared.config import get_target, TARGETS

load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description="Top-down v7 scenario generation")
parser.add_argument(
    "--target",
    choices=list(TARGETS.keys()),
    default="gdp_2050",
    help="Target question to generate scenarios for",
)
parser.add_argument(
    "--n-uncertainties",
    type=int,
    default=3,
    help="Number of uncertainty axes to identify",
)
parser.add_argument(
    "--voi-floor",
    type=float,
    default=0.1,
    help="VOI floor for scenario generation (default 0.1)",
)
args = parser.parse_args()

# Load config
config = get_target(args.target)
TARGET_QUESTION = config.question.text
CONTEXT = config.context

# Paths
OUTPUT_DIR = Path(__file__).parent / "results" / args.target
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    print("=" * 60)
    print("TOP-DOWN APPROACH v7: Enhanced Data Model")
    print("=" * 60)
    print(f"\nTarget: {TARGET_QUESTION}")
    print(f"Uncertainty axes: {args.n_uncertainties}")
    print(f"VOI floor: {args.voi_floor}")

    # Step 1: Identify key uncertainties (shared with hybrid)
    print("\n[1/4] Identifying key uncertainties...")
    uncertainties = await identify_uncertainties(
        question=TARGET_QUESTION,
        context=CONTEXT,
        n=args.n_uncertainties,
    )

    for i, u in enumerate(uncertainties):
        print(f"\n  {i+1}. {u.name}")
        print(f"     {u.description}")

    # Step 2: Generate signals per uncertainty (LLM-only, no market search)
    print("\n[2/4] Generating signals per uncertainty...")
    all_signals = []
    uncertainty_signals = {}

    for u in uncertainties:
        print(f"\n  Generating for: {u.name}")

        signals = await generate_signals_for_uncertainty(
            question=TARGET_QUESTION,
            uncertainty_name=u.name,
            uncertainty_description=u.description,
            n=10,
        )

        print(f"    Generated {len(signals)} signals")
        print(f"    Sample: {signals[0]['text'][:60]}...")
        if signals[0].get('resolution_date'):
            print(f"    Resolution date: {signals[0]['resolution_date']}")

        uncertainty_signals[u.name] = signals
        all_signals.extend(signals)

    print(f"\n  Total signals: {len(all_signals)}")

    # Step 3: Rank signals by VOI (shared with bottom-up/hybrid)
    print("\n[3/4] Ranking signals by VOI...")
    ranked_signals = await rank_signals_by_voi(
        signals=all_signals,
        target=TARGET_QUESTION,
    )

    # Count by VOI threshold
    voi_counts = {
        ">=0.3": sum(1 for s in ranked_signals if s.get("voi", 0) >= 0.3),
        ">=0.2": sum(1 for s in ranked_signals if s.get("voi", 0) >= 0.2),
        ">=0.1": sum(1 for s in ranked_signals if s.get("voi", 0) >= 0.1),
        "<0.1": sum(1 for s in ranked_signals if s.get("voi", 0) < 0.1),
    }
    print(f"\n  VOI distribution:")
    for k, v in voi_counts.items():
        print(f"    {k}: {v}")

    print(f"\n  Top 3 by VOI:")
    for s in ranked_signals[:3]:
        print(f"    VOI={s.get('voi', 0):.2f} [{s['uncertainty_source'][:15]}] {s['text'][:50]}...")

    # Step 4: Generate scenarios (shared with bottom-up/hybrid)
    print("\n[4/4] Generating MECE scenarios...")
    result = await generate_mece_scenarios(
        signals=ranked_signals,
        question=TARGET_QUESTION,
        context=CONTEXT,
        voi_floor=args.voi_floor,
    )

    # Output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMECE Reasoning: {result.mece_reasoning}")

    if result.coverage_gaps:
        print(f"\nCoverage Gaps: {result.coverage_gaps}")
    else:
        print(f"\nCoverage Gaps: None")

    print("\n" + "-" * 40)
    for s in result.scenarios:
        print(f"\n### {s.name}")
        print(f"  Outcome Range: {s.outcome_range} (low={s.outcome_low}, high={s.outcome_high})")
        print(f"  {s.description}")
        print(f"\n  Why Exclusive: {s.why_exclusive[:80]}...")
        print(f"\n  Key Drivers: {', '.join(s.key_drivers[:3])}")

    # Save results
    output_file = OUTPUT_DIR / f"topdown_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Breakdown by uncertainty
    by_uncertainty = {k: len(v) for k, v in uncertainty_signals.items()}

    # Build Signal instances
    signals_v7 = [
        Signal(
            id=s["id"],
            source=s["source"],
            text=s["text"],
            background=s.get("background"),
            resolution_date=parse_date(s.get("resolution_date")),
            base_rate=s.get("base_rate"),
            voi=s.get("voi", 0.0),
            rho=s.get("rho", 0.0),
            rho_reasoning=s.get("rho_reasoning"),
            uncertainty_source=s.get("uncertainty_source"),
        )
        for s in ranked_signals
    ]

    # Build v7 scenario format (includes outcome_low, outcome_high)
    scenarios_v7 = [
        {
            "name": s.name,
            "description": s.description,
            "outcome_range": s.outcome_range,
            "outcome_low": s.outcome_low,
            "outcome_high": s.outcome_high,
            "key_drivers": s.key_drivers,
            "why_exclusive": s.why_exclusive,
            "signal_impacts": [{"signal_index": si.signal_index, "effect": si.effect} for si in s.signal_impacts],
            "indicator_bundle": s.indicator_bundle,
        }
        for s in result.scenarios
    ]

    results = {
        "id": f"topdown_{args.target}_{uuid4().hex[:8]}",
        "name": f"{args.target} (top-down)",
        "target": args.target,
        "approach": "topdown_v7",
        "question": {
            "id": config.question.id,
            "source": config.question.source,
            "text": config.question.text,
            "question_type": config.question.question_type.value,
            "unit": config.question.unit.type if config.question.unit else None,
            "base_rate": config.question.base_rate,
            "value_range": list(config.question.value_range) if config.question.value_range else None,
        },
        "config": {
            "context": config.context,
            "cruxiness_normalizer": config.cruxiness_normalizer,
            "voi_floor": args.voi_floor,
        },
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "uncertainties": [
            {
                "name": u.name,
                "description": u.description,
                "search_query": u.search_query,
            }
            for u in uncertainties
        ],
        "signals_per_uncertainty": by_uncertainty,
        "total_signals": len(all_signals),
        "signals_above_floor": sum(1 for s in ranked_signals if s.get("voi", 0) >= args.voi_floor),
        "voi_distribution": voi_counts,
        "signals": [s.model_dump(mode="json", exclude_none=True) for s in signals_v7],
        "scenarios": scenarios_v7,
        "mece_reasoning": result.mece_reasoning,
        "coverage_gaps": result.coverage_gaps,
        "created_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
