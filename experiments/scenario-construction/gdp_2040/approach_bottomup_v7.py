#!/usr/bin/env python3
"""
Bottom-up approach v7: VOI-based signal ranking with enhanced data model.

Pipeline:
1. Semantic search → signals related to target
2. rank_signals_by_voi(signals, target) — uses Anthropic batch API
3. Deduplicate redundant signals
4. generate_mece_scenarios(signals, voi_floor=0.1) — shared with hybrid/top-down

v7 changes: Enhanced signal/scenario output for UI hydration.
- Signals include id, url, resolution_date, base_rate
- Scenarios include outcome_low, outcome_high (numeric)

Usage:
    uv run python experiments/scenario-construction/gdp_2040/approach_bottomup_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/gdp_2040/approach_bottomup_v7.py --target renewable_2050
"""

import argparse
import json
import asyncio
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, date
from uuid import uuid4

from dotenv import load_dotenv

from llm_forecasting.models import Signal

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.signals import (
    load_market_signals_semantic,
    deduplicate_market_signals,
    rank_signals_by_voi,
    DEFAULT_KNOWLEDGE_CUTOFF,
    categorize_signal,
)
from shared.scenarios import generate_mece_scenarios
from shared.config import get_target, TARGETS
from shared.refresh import refresh_if_stale

load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description="Bottom-up v7 scenario generation")
parser.add_argument(
    "--target",
    choices=list(TARGETS.keys()),
    default="gdp_2050",
    help="Target question to generate scenarios for",
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
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = REPO_ROOT / "data" / "forecastbench.db"
OUTPUT_DIR = Path(__file__).parent / "results" / args.target
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge cutoff
KNOWLEDGE_CUTOFF = DEFAULT_KNOWLEDGE_CUTOFF

# Sources to search
# Question sources only (prediction markets) - not data sources like FRED
# See: Obsidian/projects/Scenario Generation.md#Source Architecture
SOURCES = ["polymarket", "metaculus", "kalshi", "infer", "manifold"]


async def main():
    print("=" * 60)
    print("BOTTOM-UP APPROACH v7: Enhanced Data Model")
    print("=" * 60)
    print(f"\nTarget: {TARGET_QUESTION}")
    print(f"Sources: {', '.join(SOURCES)}")
    print(f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}")
    print(f"VOI floor: {args.voi_floor}")

    # Step 0: Refresh market data if stale (>24h old)
    print("\n[0/4] Checking data freshness...")
    await refresh_if_stale(str(DB_PATH), SOURCES)

    # Step 1: Semantic search for signals
    print("\n[1/4] Loading signals via semantic search...")
    raw_signals = load_market_signals_semantic(
        db_path=DB_PATH,
        query=f"What signals affect {TARGET_QUESTION}",
        sources=SOURCES,
        top_k=200,
    )
    print(f"  Retrieved {len(raw_signals)} semantically relevant signals")

    # Enrich with resolution data and URL
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    for s in raw_signals:
        cursor.execute(
            "SELECT resolution_date, resolved, resolution_value, base_rate, url FROM questions WHERE id = ? AND source = ?",
            (s["id"], s["source"])
        )
        row = cursor.fetchone()
        if row:
            res_date = row["resolution_date"]
            resolved = bool(row["resolved"])
            s["resolution_date"] = str(res_date) if res_date else None
            s["resolved"] = resolved
            s["resolution_value"] = row["resolution_value"]
            s["base_rate"] = row["base_rate"]
            s["url"] = row["url"]
            s["signal_category"] = categorize_signal(s["resolution_date"], resolved, KNOWLEDGE_CUTOFF)
        else:
            s["resolution_date"] = None
            s["resolved"] = False
            s["url"] = None
            s["signal_category"] = "unknown"
    conn.close()

    # Filter out pre-cutoff resolved signals
    before_filter = len(raw_signals)
    raw_signals = [s for s in raw_signals if s["signal_category"] != "exclude"]
    print(f"  After resolution filtering: {len(raw_signals)} (excluded {before_filter - len(raw_signals)} pre-cutoff)")

    # Show category breakdown
    by_category = {}
    for s in raw_signals:
        cat = s.get("signal_category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    print(f"  Categories: {by_category}")

    # Step 2: Rank by VOI
    print("\n[2/4] Ranking signals by VOI (batch API)...")
    ranked_signals = await rank_signals_by_voi(
        signals=raw_signals,
        target=TARGET_QUESTION,
        target_prior=0.5,
    )
    print(f"  Ranked {len(ranked_signals)} signals by VOI")

    # Show top signals
    print("\n  Top 5 by VOI:")
    for s in ranked_signals[:5]:
        print(f"    VOI={s['voi']:.3f} ρ={s['rho']:.2f} [{s['source']}] {s['question'][:50]}...")

    # Step 3: Deduplicate
    print("\n[3/4] Deduplicating...")
    # Take top signals by VOI before deduplication
    top_signals = ranked_signals[:100]
    before_dedup = len(top_signals)
    deduped_signals = deduplicate_market_signals(top_signals, threshold=0.45)
    print(f"  {before_dedup} → {len(deduped_signals)} (removed {before_dedup - len(deduped_signals)} duplicates)")

    # Step 4: Generate scenarios
    print("\n[4/4] Generating MECE scenarios...")
    result = await generate_mece_scenarios(
        signals=[{"text": s["question"], "source": s["source"], "voi": s.get("voi", 0)} for s in deduped_signals[:50]],
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
    output_file = OUTPUT_DIR / f"bottomup_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Source breakdown
    source_counts = {}
    for s in deduped_signals:
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1

    # Build Signal instances
    def parse_date(d: str | None) -> date | None:
        if d is None:
            return None
        try:
            return date.fromisoformat(str(d)[:10])
        except ValueError:
            return None

    signals_v7 = [
        Signal(
            id=s["id"],
            source=s["source"],
            text=s["question"],
            url=s.get("url"),
            resolution_date=parse_date(s.get("resolution_date")),
            base_rate=s.get("base_rate"),
            voi=s.get("voi", 0.0),
            rho=s.get("rho", 0.0),
            rho_reasoning=s.get("rho_reasoning"),
        )
        for s in deduped_signals[:50]
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
        "id": f"bottomup_{args.target}_{uuid4().hex[:8]}",
        "name": f"{args.target} (bottom-up)",
        "target": args.target,
        "approach": "bottomup_v7",
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
        "knowledge_cutoff": KNOWLEDGE_CUTOFF,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "sources_used": SOURCES,
        "signals_retrieved": len(raw_signals),
        "signals_after_dedup": len(deduped_signals),
        "signals_above_floor": sum(1 for s in deduped_signals if s.get("voi", 0) >= args.voi_floor),
        "source_breakdown": source_counts,
        "category_breakdown": by_category,
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
