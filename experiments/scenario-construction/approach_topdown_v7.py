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

from dotenv import load_dotenv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import rank_and_report_signals
from shared.generation import generate_signals_for_uncertainty
from shared.uncertainties import identify_and_report_uncertainties
from shared.scenarios import generate_and_print_scenarios
from shared.setup import (
    create_base_parser,
    add_uncertainty_args,
    load_config,
    print_header,
)
from shared.output import save_approach_results

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

    # Step 1: Identify key uncertainties
    print("\n[1/4] Identifying key uncertainties...")
    uncertainties = await identify_and_report_uncertainties(
        question=cfg.question_text,
        context=cfg.context,
        n=cfg.n_uncertainties,
    )

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
            max_horizon_days=cfg.max_horizon_days,
        )

        print(f"    Generated {len(signals)} signals")
        print(f"    Sample: {signals[0]['text'][:60]}...")
        if signals[0].get('resolution_date'):
            print(f"    Resolution date: {signals[0]['resolution_date']}")

        uncertainty_signals[u.name] = signals
        all_signals.extend(signals)

    print(f"\n  Total signals: {len(all_signals)}")

    # Step 3: Rank signals by VOI
    print("\n[3/4] Ranking signals by VOI...")
    is_continuous = cfg.question_type == "continuous"
    ranked_signals = await rank_and_report_signals(
        signals=all_signals,
        target=cfg.question_text,
        voi_floor=cfg.voi_floor,
        is_continuous=is_continuous,
    )

    # Step 4: Generate scenarios
    print("\n[4/4] Generating scenarios...")
    result = await generate_and_print_scenarios(
        signals=ranked_signals,
        question=cfg.question_text,
        context=cfg.context,
        question_type=cfg.question_type,
        voi_floor=cfg.voi_floor,
    )

    # Save results
    by_uncertainty = {k: len(v) for k, v in uncertainty_signals.items()}
    save_approach_results(
        approach="topdown",
        cfg=cfg,
        signals=ranked_signals,
        result=result,
        text_key="text",
        include_uncertainty_source=True,
        extra_fields={
            "uncertainties": [
                {"name": u.name, "description": u.description, "search_query": u.search_query}
                for u in uncertainties
            ],
            "signals_per_uncertainty": by_uncertainty,
            "total_signals": len(all_signals),
            "signals_above_floor": sum(1 for s in ranked_signals if s.get("voi", 0) >= cfg.voi_floor),
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
