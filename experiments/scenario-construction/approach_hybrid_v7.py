#!/usr/bin/env python3
"""
Hybrid approach v7: Uncertainty-guided VOI-based signal ranking.

Pipeline:
1. identify_uncertainties(question, n=3) — LLM identifies key uncertainty axes
2. For each uncertainty: semantic search + rank_signals_by_voi
3. Combine signals across uncertainties, deduplicate
4. generate_mece_scenarios(signals, voi_floor=0.1)

Usage:
    uv run python experiments/scenario-construction/approach_hybrid_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/approach_hybrid_v7.py --target renewable_2050
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import (
    load_market_signals_semantic,
    deduplicate_market_signals,
    rank_signals_by_voi,
    enrich_with_resolution_data,
)
from shared.uncertainties import identify_uncertainties
from shared.scenarios import generate_and_print_scenarios
from shared.refresh import refresh_if_stale
from shared.setup import (
    create_base_parser,
    add_uncertainty_args,
    load_config,
    print_header,
    DEFAULT_SOURCES,
)
from shared.output import count_by_field, save_approach_results

load_dotenv()

# Parse arguments
parser = create_base_parser("Hybrid v7 scenario generation")
add_uncertainty_args(parser)
args = parser.parse_args()

# Load config
cfg = load_config(args, Path(__file__))
SOURCES = DEFAULT_SOURCES


async def main():
    print_header(
        "HYBRID",
        cfg.question_text,
        sources=SOURCES,
        max_horizon_days=cfg.max_horizon_days,
        n_uncertainties=cfg.n_uncertainties,
        voi_floor=cfg.voi_floor,
    )

    # Step 0: Refresh market data if stale (>24h old)
    print("\n[0/5] Checking data freshness...")
    await refresh_if_stale(str(cfg.db_path), SOURCES)

    # Step 1: Identify key uncertainties
    print("\n[1/5] Identifying key uncertainties...")
    uncertainties = await identify_uncertainties(
        question=cfg.question_text,
        context=cfg.context,
        n=cfg.n_uncertainties,
    )

    for i, u in enumerate(uncertainties):
        print(f"\n  {i+1}. {u.name}")
        print(f"     {u.description}")
        print(f'     Search: "{u.search_query}"')

    # Step 2: For each uncertainty, search and rank by VOI
    print("\n[2/5] Searching and ranking signals per uncertainty...")
    all_signals = []
    uncertainty_signals = {}

    for u in uncertainties:
        print(f"\n  Processing: {u.name}")

        # Semantic search for this uncertainty
        raw_signals = load_market_signals_semantic(
            db_path=cfg.db_path,
            query=u.search_query,
            sources=SOURCES,
            top_k=100,
        )
        print(f"    Found {len(raw_signals)} signals")

        # Enrich with resolution data and URL
        enrich_with_resolution_data(raw_signals, cfg.db_path, cfg.max_horizon_days)

        # Filter pre-cutoff
        raw_signals = [s for s in raw_signals if s.get("signal_category") != "exclude"]
        print(f"    After filtering: {len(raw_signals)}")

        # Rank by VOI relative to this uncertainty
        uncertainty_target = f"{u.name}: {u.description}"
        ranked = await rank_signals_by_voi(
            signals=raw_signals,
            target=uncertainty_target,
            target_prior=0.5,
        )

        # Take top signals for this uncertainty
        top_for_uncertainty = ranked[:30]
        if top_for_uncertainty:
            print(f"    Top signal: VOI={top_for_uncertainty[0]['voi']:.3f} {top_for_uncertainty[0]['question'][:40]}...")

        # Track uncertainty source
        for s in top_for_uncertainty:
            s["uncertainty_source"] = u.name

        uncertainty_signals[u.name] = top_for_uncertainty
        all_signals.extend(top_for_uncertainty)

    print(f"\n  Total signals before dedup: {len(all_signals)}")

    # Step 3: Deduplicate across all uncertainties
    print("\n[3/5] Deduplicating across uncertainties...")
    before_dedup = len(all_signals)
    deduped_signals = deduplicate_market_signals(all_signals, threshold=0.45)
    print(f"  {before_dedup} → {len(deduped_signals)} (removed {before_dedup - len(deduped_signals)} duplicates)")

    by_uncertainty = count_by_field(deduped_signals, "uncertainty_source")
    print(f"  By uncertainty: {by_uncertainty}")

    # Step 4: Generate scenarios
    print("\n[4/5] Generating scenarios...")
    result = await generate_and_print_scenarios(
        signals=deduped_signals[:50],
        question=cfg.question_text,
        context=cfg.context,
        question_type=cfg.question_type,
        voi_floor=cfg.voi_floor,
    )

    # Save results
    print("\n[5/5] Saving results...")
    save_approach_results(
        approach="hybrid",
        cfg=cfg,
        signals=deduped_signals[:50],
        result=result,
        text_key="question",
        include_uncertainty_source=True,
        extra_fields={
            "sources_used": SOURCES,
            "uncertainties": [
                {"name": u.name, "description": u.description, "search_query": u.search_query}
                for u in uncertainties
            ],
            "signals_per_uncertainty": {k: len(v) for k, v in uncertainty_signals.items()},
            "signals_after_dedup": len(deduped_signals),
            "signals_above_floor": sum(1 for s in deduped_signals if s.get("voi", 0) >= cfg.voi_floor),
            "source_breakdown": count_by_field(deduped_signals, "source"),
            "category_breakdown": count_by_field(deduped_signals, "signal_category"),
            "uncertainty_breakdown": by_uncertainty,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
