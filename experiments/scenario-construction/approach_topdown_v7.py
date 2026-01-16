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
    uv run python experiments/scenario-construction/approach_topdown_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/approach_topdown_v7.py --target renewable_2050
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import rank_signals_by_voi, build_signal_models
from shared.generation import generate_signals_for_uncertainty
from shared.uncertainties import identify_uncertainties
from shared.scenarios import generate_mece_scenarios
from shared.setup import (
    create_base_parser,
    add_uncertainty_args,
    load_config,
    print_header,
)
from shared.output import (
    build_scenario_dicts,
    build_base_results,
    print_results,
    save_results,
)

load_dotenv()

# Parse arguments
parser = create_base_parser("Top-down v7 scenario generation")
add_uncertainty_args(parser)
args = parser.parse_args()

# Load config
cfg = load_config(args, Path(__file__))


async def main():
    print_header(
        "TOP-DOWN",
        cfg.question_text,
        n_uncertainties=cfg.n_uncertainties,
        voi_floor=cfg.voi_floor,
    )

    # Step 1: Identify key uncertainties (shared with hybrid)
    print("\n[1/4] Identifying key uncertainties...")
    uncertainties = await identify_uncertainties(
        question=cfg.question_text,
        context=cfg.context,
        n=cfg.n_uncertainties,
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
            question=cfg.question_text,
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
        target=cfg.question_text,
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
        question=cfg.question_text,
        context=cfg.context,
        question_type=cfg.question_type,
        voi_floor=cfg.voi_floor,
    )

    # Print results
    print_results(result)

    # Save results
    output_file = cfg.output_dir / f"topdown_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    signals_v7 = build_signal_models(
        ranked_signals,
        text_key="text",
        include_background=True,
        include_uncertainty_source=True,
    )
    scenarios_v7 = build_scenario_dicts(result.scenarios)

    results = build_base_results(
        approach="topdown",
        target=cfg.target,
        config=cfg.config,
        signals_v7=signals_v7,
        scenarios_v7=scenarios_v7,
        mece_reasoning=result.mece_reasoning,
        coverage_gaps=result.coverage_gaps,
        voi_floor=cfg.voi_floor,
    )

    # Add approach-specific fields
    by_uncertainty = {k: len(v) for k, v in uncertainty_signals.items()}
    results.update({
        "uncertainties": [
            {"name": u.name, "description": u.description, "search_query": u.search_query}
            for u in uncertainties
        ],
        "signals_per_uncertainty": by_uncertainty,
        "total_signals": len(all_signals),
        "signals_above_floor": sum(1 for s in ranked_signals if s.get("voi", 0) >= cfg.voi_floor),
        "voi_distribution": voi_counts,
    })

    save_results(results, output_file)


if __name__ == "__main__":
    asyncio.run(main())
