#!/usr/bin/env python3
"""
Bottom-up approach v7: VOI-based signal ranking with enhanced data model.

Pipeline:
1. Semantic search → signals related to target
2. rank_signals_by_voi(signals, target) — uses Anthropic batch API
3. Deduplicate redundant signals
4. generate_mece_scenarios(signals, voi_floor=0.1) — shared with hybrid/top-down

Usage:
    uv run python experiments/scenario-construction/approach_bottomup_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/approach_bottomup_v7.py --target renewable_2050
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import (
    load_market_signals_semantic,
    deduplicate_market_signals,
    rank_signals_by_voi,
    enrich_with_resolution_data,
    build_signal_models,
)
from shared.scenarios import generate_mece_scenarios
from shared.refresh import refresh_if_stale
from shared.setup import (
    create_base_parser,
    load_config,
    print_header,
    DEFAULT_SOURCES,
)
from shared.output import (
    build_scenario_dicts,
    build_base_results,
    print_results,
    save_results,
    count_by_field,
)

load_dotenv()

# Parse arguments
parser = create_base_parser("Bottom-up v7 scenario generation")
args = parser.parse_args()

# Load config
cfg = load_config(args, Path(__file__))
SOURCES = DEFAULT_SOURCES


async def main():
    print_header(
        "BOTTOM-UP",
        cfg.question_text,
        sources=SOURCES,
        max_horizon_days=cfg.max_horizon_days,
        voi_floor=cfg.voi_floor,
    )

    # Step 0: Refresh market data if stale (>24h old)
    print("\n[0/4] Checking data freshness...")
    await refresh_if_stale(str(cfg.db_path), SOURCES)

    # Step 1: Semantic search for signals
    print("\n[1/4] Loading signals via semantic search...")
    raw_signals = load_market_signals_semantic(
        db_path=cfg.db_path,
        query=f"What signals affect {cfg.question_text}",
        sources=SOURCES,
        top_k=200,
    )
    print(f"  Retrieved {len(raw_signals)} semantically relevant signals")

    # Enrich with resolution data and URL
    enrich_with_resolution_data(raw_signals, cfg.db_path, cfg.max_horizon_days)

    # Filter out pre-cutoff resolved signals
    before_filter = len(raw_signals)
    raw_signals = [s for s in raw_signals if s["signal_category"] != "exclude"]
    print(f"  After resolution filtering: {len(raw_signals)} (excluded {before_filter - len(raw_signals)} pre-cutoff)")

    # Show category breakdown
    by_category = count_by_field(raw_signals, "signal_category")
    print(f"  Categories: {by_category}")

    # Step 2: Rank by VOI
    print("\n[2/4] Ranking signals by VOI (batch API)...")
    ranked_signals = await rank_signals_by_voi(
        signals=raw_signals,
        target=cfg.question_text,
        target_prior=0.5,
    )
    print(f"  Ranked {len(ranked_signals)} signals by VOI")

    # Show top signals
    print("\n  Top 5 by VOI:")
    for s in ranked_signals[:5]:
        print(f"    VOI={s['voi']:.3f} ρ={s['rho']:.2f} [{s['source']}] {s['question'][:50]}...")

    # Step 3: Deduplicate
    print("\n[3/4] Deduplicating...")
    top_signals = ranked_signals[:100]
    before_dedup = len(top_signals)
    deduped_signals = deduplicate_market_signals(top_signals, threshold=0.45)
    print(f"  {before_dedup} → {len(deduped_signals)} (removed {before_dedup - len(deduped_signals)} duplicates)")

    # Step 4: Generate scenarios
    print("\n[4/4] Generating MECE scenarios...")
    result = await generate_mece_scenarios(
        signals=[{"text": s["question"], "source": s["source"], "voi": s.get("voi", 0)} for s in deduped_signals[:50]],
        question=cfg.question_text,
        context=cfg.context,
        question_type=cfg.question_type,
        voi_floor=cfg.voi_floor,
    )

    # Print results
    print_results(result)

    # Build output
    output_file = cfg.output_dir / f"bottomup_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    signals_v7 = build_signal_models(deduped_signals[:50], text_key="question")
    scenarios_v7 = build_scenario_dicts(result.scenarios)

    results = build_base_results(
        approach="bottomup",
        target=cfg.target,
        config=cfg.config,
        signals_v7=signals_v7,
        scenarios_v7=scenarios_v7,
        mece_reasoning=result.mece_reasoning,
        coverage_gaps=result.coverage_gaps,
        voi_floor=cfg.voi_floor,
        max_horizon_days=cfg.max_horizon_days,
    )

    # Add approach-specific fields
    results.update({
        "sources_used": SOURCES,
        "signals_retrieved": len(raw_signals),
        "signals_after_dedup": len(deduped_signals),
        "signals_above_floor": sum(1 for s in deduped_signals if s.get("voi", 0) >= cfg.voi_floor),
        "source_breakdown": count_by_field(deduped_signals, "source"),
        "category_breakdown": by_category,
    })

    save_results(results, output_file)


if __name__ == "__main__":
    asyncio.run(main())
