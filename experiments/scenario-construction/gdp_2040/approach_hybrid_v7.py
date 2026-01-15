#!/usr/bin/env python3
"""
Hybrid approach v7: Uncertainty-guided VOI-based signal ranking with enhanced data model.

Pipeline:
1. identify_uncertainties(question, n=3) — LLM identifies key uncertainty axes
2. For each uncertainty:
   - Semantic search for related signals
   - rank_signals_by_voi(signals, uncertainty) — uses Anthropic batch API
3. Combine signals across uncertainties, deduplicate
4. generate_mece_scenarios(signals, voi_floor=0.1) — shared with bottom-up/top-down

v7 changes: Enhanced signal/scenario output for UI hydration.
- Signals include id, url, resolution_date, base_rate
- Scenarios include outcome_low, outcome_high (numeric)

Usage:
    uv run python experiments/scenario-construction/gdp_2040/approach_hybrid_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/gdp_2040/approach_hybrid_v7.py --target renewable_2050
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
from shared.uncertainties import identify_uncertainties
from shared.scenarios import generate_mece_scenarios
from shared.config import get_target, TARGETS

load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description="Hybrid v7 scenario generation")
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
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = REPO_ROOT / "data" / "forecastbench.db"
OUTPUT_DIR = Path(__file__).parent / "results" / args.target
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge cutoff
KNOWLEDGE_CUTOFF = DEFAULT_KNOWLEDGE_CUTOFF

# Sources to search
SOURCES = ["polymarket", "metaculus", "fred", "infer", "manifold"]


def enrich_with_resolution_data(signals: list[dict], db_path: Path) -> list[dict]:
    """Add resolution metadata and URL to signals from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    for s in signals:
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
    return signals


async def main():
    print("=" * 60)
    print("HYBRID APPROACH v7: Enhanced Data Model")
    print("=" * 60)
    print(f"\nTarget: {TARGET_QUESTION}")
    print(f"Sources: {', '.join(SOURCES)}")
    print(f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}")
    print(f"Uncertainty axes: {args.n_uncertainties}")
    print(f"VOI floor: {args.voi_floor}")

    # Step 1: Identify key uncertainties
    print("\n[1/5] Identifying key uncertainties...")
    uncertainties = await identify_uncertainties(
        question=TARGET_QUESTION,
        context=CONTEXT,
        n=args.n_uncertainties,
    )

    for i, u in enumerate(uncertainties):
        print(f"\n  {i+1}. {u.name}")
        print(f"     {u.description}")
        print(f"     Search: \"{u.search_query}\"")

    # Step 2: For each uncertainty, search and rank by VOI
    print("\n[2/5] Searching and ranking signals per uncertainty...")
    all_signals = []
    uncertainty_signals = {}  # Track which signals came from which uncertainty

    for u in uncertainties:
        print(f"\n  Processing: {u.name}")

        # Semantic search for this uncertainty
        raw_signals = load_market_signals_semantic(
            db_path=DB_PATH,
            query=u.search_query,
            sources=SOURCES,
            top_k=100,
        )
        print(f"    Found {len(raw_signals)} signals")

        # Enrich with resolution data and URL
        raw_signals = enrich_with_resolution_data(raw_signals, DB_PATH)

        # Filter pre-cutoff
        raw_signals = [s for s in raw_signals if s.get("signal_category") != "exclude"]
        print(f"    After filtering: {len(raw_signals)}")

        # Rank by VOI relative to this uncertainty
        # Use the uncertainty description as the "target" for VOI calculation
        uncertainty_target = f"{u.name}: {u.description}"
        ranked = await rank_signals_by_voi(
            signals=raw_signals,
            target=uncertainty_target,
            target_prior=0.5,
        )

        # Take top signals for this uncertainty
        top_for_uncertainty = ranked[:30]
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

    # Show breakdown by uncertainty source after dedup
    by_uncertainty = {}
    for s in deduped_signals:
        u_src = s.get("uncertainty_source", "unknown")
        by_uncertainty[u_src] = by_uncertainty.get(u_src, 0) + 1
    print(f"  By uncertainty: {by_uncertainty}")

    # Step 4: Generate scenarios
    print("\n[4/5] Generating MECE scenarios...")
    result = await generate_mece_scenarios(
        signals=[
            {
                "text": s["question"],
                "source": s["source"],
                "voi": s.get("voi", 0),
                "uncertainty_source": s.get("uncertainty_source"),
            }
            for s in deduped_signals[:50]
        ],
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
    print("\n[5/5] Saving results...")
    output_file = OUTPUT_DIR / f"hybrid_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Source breakdown
    source_counts = {}
    for s in deduped_signals:
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1

    # Category breakdown
    category_counts = {}
    for s in deduped_signals:
        cat = s.get("signal_category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

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
            uncertainty_source=s.get("uncertainty_source"),
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
            "mapped_signals": s.mapped_signals,
            "indicator_bundle": s.indicator_bundle,
        }
        for s in result.scenarios
    ]

    results = {
        "id": f"hybrid_{args.target}_{uuid4().hex[:8]}",
        "name": f"{args.target} (hybrid)",
        "target": args.target,
        "approach": "hybrid_v7",
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
        "uncertainties": [
            {
                "name": u.name,
                "description": u.description,
                "search_query": u.search_query,
            }
            for u in uncertainties
        ],
        "signals_per_uncertainty": {k: len(v) for k, v in uncertainty_signals.items()},
        "signals_after_dedup": len(deduped_signals),
        "signals_above_floor": sum(1 for s in deduped_signals if s.get("voi", 0) >= args.voi_floor),
        "source_breakdown": source_counts,
        "category_breakdown": category_counts,
        "uncertainty_breakdown": by_uncertainty,
        "signals": [s.model_dump(mode="json", exclude_none=True) for s in signals_v7],
        "scenarios": scenarios_v7,
        "mece_reasoning": result.mece_reasoning,
        "coverage_gaps": result.coverage_gaps,
        "created_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
