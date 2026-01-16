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

from dotenv import load_dotenv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import (
    load_market_signals_semantic,
    deduplicate_market_signals,
    rank_and_report_signals,
    enrich_with_resolution_data,
)
from shared.scenarios import generate_and_print_scenarios
from shared.refresh import refresh_if_stale
from shared.setup import (
    create_base_parser,
    load_config,
    print_header,
    DEFAULT_SOURCES,
)
from shared.output import count_by_field, save_approach_results

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
    print("\n[2/4] Ranking signals by VOI...")
    is_continuous = cfg.question_type == "continuous"
    ranked_signals = await rank_and_report_signals(
        signals=raw_signals,
        target=cfg.question_text,
        voi_floor=cfg.voi_floor,
        is_continuous=is_continuous,
    )

    # Step 3: Deduplicate
    print("\n[3/4] Deduplicating...")
    top_signals = ranked_signals[:100]
    before_dedup = len(top_signals)
    deduped_signals = deduplicate_market_signals(top_signals, threshold=0.45)
    print(f"  {before_dedup} → {len(deduped_signals)} (removed {before_dedup - len(deduped_signals)} duplicates)")

    # Step 4: Generate scenarios
    print("\n[4/4] Generating scenarios...")
    result = await generate_and_print_scenarios(
        signals=deduped_signals[:50],
        question=cfg.question_text,
        context=cfg.context,
        question_type=cfg.question_type,
        voi_floor=cfg.voi_floor,
    )

    # Save results
    save_approach_results(
        approach="bottomup",
        cfg=cfg,
        signals=deduped_signals[:50],
        result=result,
        text_key="question",
        extra_fields={
            "sources_used": SOURCES,
            "signals_retrieved": len(raw_signals),
            "signals_after_dedup": len(deduped_signals),
            "signals_above_floor": sum(1 for s in deduped_signals if s.get("voi", 0) >= cfg.voi_floor),
            "source_breakdown": count_by_field(deduped_signals, "source"),
            "category_breakdown": by_category,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
