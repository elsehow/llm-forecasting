#!/usr/bin/env python3
"""
Dual approach v7: Independent top-down + bottom-up, then merge with gap detection.

Pipeline:
1. Top-down: LLM generates cruxy signals per uncertainty (independent)
2. Bottom-up: Semantic search for market signals (independent)
3. Match: For each top-down signal, find semantic matches in market data
4. Merge: Market signals + unmatched top-down signals (gaps)
5. Generate scenarios from merged signal set
6. Report: Gaps reveal resolution sources to track (no adapter needed)

Value proposition:
- Hybrid scopes market search using LLM uncertainties (limited by market coverage)
- Dual generates signals that may not exist in any market
- Gaps reveal what resolution sources to track, not what adapters to build

See: Obsidian/projects/Scenario Generation.md#Approach Comparison

Usage:
    uv run python experiments/scenario-construction/approach_dual_v7.py --target gdp_2050
    uv run python experiments/scenario-construction/approach_dual_v7.py --target renewable_2050
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
from llm_forecasting.semantic_search import SemanticSignalSearcher

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import (
    load_market_signals_semantic,
    deduplicate_signals,
    rank_signals_by_voi,
    enrich_with_resolution_data,
    parse_date,
    DEFAULT_KNOWLEDGE_CUTOFF,
)
from shared.generation import generate_signals_for_uncertainty
from shared.uncertainties import identify_uncertainties
from shared.scenarios import generate_mece_scenarios
from shared.config import get_target, TARGETS
from shared.refresh import refresh_if_stale

load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description="Dual v7 scenario generation")
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
parser.add_argument(
    "--match-threshold",
    type=float,
    default=0.6,
    help="Similarity threshold for matching top-down to market signals (default 0.6)",
)
args = parser.parse_args()

# Load config
config = get_target(args.target)
TARGET_QUESTION = config.question.text
CONTEXT = config.context

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
DB_PATH = REPO_ROOT / "data" / "forecastbench.db"
OUTPUT_DIR = Path(__file__).parent / "results" / args.target
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge cutoff
KNOWLEDGE_CUTOFF = DEFAULT_KNOWLEDGE_CUTOFF

# Sources to search (question sources only)
SOURCES = ["polymarket", "metaculus", "kalshi", "infer", "manifold"]


# ============================================================
# Signal matching (top-down to market)
# ============================================================

def match_topdown_to_market(
    topdown_signals: list[dict],
    market_signals: list[dict],
    threshold: float = 0.6,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Match top-down signals to market signals by semantic similarity.

    Returns:
        matched: Top-down signals with market matches (includes market_match info)
        gaps: Top-down signals with NO market match (resolution source only)
        market_only: Market signals that don't match any top-down signal
    """
    if not topdown_signals or not market_signals:
        return [], topdown_signals, market_signals

    searcher = SemanticSignalSearcher()

    # Embed all signals
    topdown_texts = [s["text"] for s in topdown_signals]
    market_texts = [s.get("question") or s.get("text", "") for s in market_signals]

    from llm_forecasting.semantic_search.embeddings import embed_texts, cosine_similarity
    topdown_embeddings = embed_texts(topdown_texts)
    market_embeddings = embed_texts(market_texts)

    matched = []
    gaps = []
    matched_market_indices = set()

    for i, td_signal in enumerate(topdown_signals):
        # Find best market match
        best_sim = 0.0
        best_idx = -1

        for j, market_signal in enumerate(market_signals):
            sim = cosine_similarity(topdown_embeddings[i], market_embeddings[j])
            if sim > best_sim:
                best_sim = sim
                best_idx = j

        if best_sim >= threshold:
            # Found a match - use market signal but note the top-down origin
            market_match = market_signals[best_idx]
            matched.append({
                **td_signal,
                "has_market_match": True,
                "market_match": {
                    "id": market_match.get("id"),
                    "source": market_match.get("source"),
                    "text": market_match.get("question") or market_match.get("text"),
                    "similarity": best_sim,
                    "base_rate": market_match.get("base_rate"),
                    "url": market_match.get("url"),
                    "resolution_date": market_match.get("resolution_date"),
                },
            })
            matched_market_indices.add(best_idx)
        else:
            # No match - this is a gap (trackable via resolution source only)
            gaps.append({
                **td_signal,
                "has_market_match": False,
                "best_market_similarity": best_sim,
            })

    # Market signals that don't match any top-down signal
    market_only = [
        market_signals[j]
        for j in range(len(market_signals))
        if j not in matched_market_indices
    ]

    return matched, gaps, market_only


async def main():
    print("=" * 60)
    print("DUAL APPROACH v7: Independent Top-Down + Bottom-Up + Merge")
    print("=" * 60)
    print(f"\nTarget: {TARGET_QUESTION}")
    print(f"Uncertainty axes: {args.n_uncertainties}")
    print(f"VOI floor: {args.voi_floor}")
    print(f"Match threshold: {args.match_threshold}")

    # ============================================================
    # PHASE 1: Top-Down (LLM-generated signals)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: TOP-DOWN (LLM-generated signals)")
    print("=" * 60)

    # Step 1.1: Identify key uncertainties
    print("\n[1.1] Identifying key uncertainties...")
    uncertainties = await identify_uncertainties(
        question=TARGET_QUESTION,
        context=CONTEXT,
        n=args.n_uncertainties,
    )

    for i, u in enumerate(uncertainties):
        print(f"\n  {i+1}. {u.name}")
        print(f"     {u.description}")

    # Step 1.2: Generate signals per uncertainty
    print("\n[1.2] Generating signals per uncertainty...")
    topdown_signals = []

    for u in uncertainties:
        print(f"\n  Generating for: {u.name}")
        signals = await generate_signals_for_uncertainty(
            question=TARGET_QUESTION,
            uncertainty_name=u.name,
            uncertainty_description=u.description,
            n=10,
        )
        print(f"    Generated {len(signals)} signals")
        topdown_signals.extend(signals)

    print(f"\n  Total top-down signals: {len(topdown_signals)}")

    # ============================================================
    # PHASE 2: Bottom-Up (Market signals)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: BOTTOM-UP (Market signals)")
    print("=" * 60)

    # Step 2.1: Refresh market data if stale
    print("\n[2.1] Checking data freshness...")
    await refresh_if_stale(str(DB_PATH), SOURCES)

    # Step 2.2: Semantic search for market signals
    print("\n[2.2] Loading signals via semantic search...")
    market_signals = load_market_signals_semantic(
        db_path=DB_PATH,
        query=f"What signals affect {TARGET_QUESTION}",
        sources=SOURCES,
        top_k=200,
    )
    print(f"  Retrieved {len(market_signals)} semantically relevant signals")

    # Enrich with resolution data
    enrich_with_resolution_data(market_signals, DB_PATH, KNOWLEDGE_CUTOFF)

    # Filter out pre-cutoff resolved signals
    before_filter = len(market_signals)
    market_signals = [s for s in market_signals if s["signal_category"] != "exclude"]
    print(f"  After resolution filtering: {len(market_signals)} (excluded {before_filter - len(market_signals)} pre-cutoff)")

    # ============================================================
    # PHASE 3: Match and Merge
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: MATCH AND MERGE")
    print("=" * 60)

    # Step 3.1: Match top-down signals to market signals
    print(f"\n[3.1] Matching top-down to market (threshold={args.match_threshold})...")
    matched, gaps, market_only = match_topdown_to_market(
        topdown_signals,
        market_signals,
        threshold=args.match_threshold,
    )

    print(f"\n  Matching results:")
    print(f"    Top-down with market match: {len(matched)}")
    print(f"    Top-down gaps (no market):  {len(gaps)}")
    print(f"    Market-only signals:        {len(market_only)}")

    # Step 3.2: Show gaps (these reveal what resolution sources to track)
    if gaps:
        print(f"\n  GAPS (cruxy signals without market coverage):")
        for g in gaps[:5]:
            print(f"    - {g['text'][:60]}...")
            print(f"      Resolution source: {g.get('resolution_source', 'unspecified')}")
            print(f"      Best market similarity: {g.get('best_market_similarity', 0):.2f}")

    # Step 3.3: Build merged signal set
    print("\n[3.2] Building merged signal set...")

    # For matched signals, use market data but keep top-down metadata
    merged_signals = []

    # Add matched signals (use market signal with top-down context)
    for m in matched:
        market = m["market_match"]
        merged_signals.append({
            "id": market["id"],
            "source": market["source"],
            "text": market["text"],
            "base_rate": market.get("base_rate"),
            "url": market.get("url"),
            "resolution_date": market.get("resolution_date"),
            "origin": "matched",
            "topdown_text": m["text"],
            "uncertainty_source": m.get("uncertainty_source"),
            "resolution_source": m.get("resolution_source"),
        })

    # Add gap signals (LLM-generated, track via resolution source)
    for g in gaps:
        merged_signals.append({
            "id": g["id"],
            "source": "llm",
            "text": g["text"],
            "base_rate": g.get("base_rate"),
            "resolution_date": g.get("resolution_date"),
            "origin": "gap",
            "uncertainty_source": g.get("uncertainty_source"),
            "resolution_source": g.get("resolution_source"),
        })

    # Add market-only signals (not matched to any top-down)
    for mo in market_only[:50]:  # Limit to top 50 by similarity
        merged_signals.append({
            "id": mo["id"],
            "source": mo["source"],
            "text": mo.get("question") or mo.get("text"),
            "base_rate": mo.get("base_rate"),
            "url": mo.get("url"),
            "resolution_date": mo.get("resolution_date"),
            "origin": "market_only",
        })

    print(f"  Merged signal set: {len(merged_signals)}")
    print(f"    From matches: {len(matched)}")
    print(f"    From gaps:    {len(gaps)}")
    print(f"    Market-only:  {min(len(market_only), 50)}")

    # ============================================================
    # PHASE 4: Rank and Generate Scenarios
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 4: RANK AND GENERATE SCENARIOS")
    print("=" * 60)

    # Step 4.1: Rank merged signals by VOI
    print("\n[4.1] Ranking signals by VOI (batch API)...")
    ranked_signals = await rank_signals_by_voi(
        signals=merged_signals,
        target=TARGET_QUESTION,
    )

    # Count by origin
    origin_above_floor = {"matched": 0, "gap": 0, "market_only": 0}
    for s in ranked_signals:
        if s.get("voi", 0) >= args.voi_floor:
            origin = s.get("origin", "unknown")
            origin_above_floor[origin] = origin_above_floor.get(origin, 0) + 1

    print(f"\n  Signals above VOI floor ({args.voi_floor}):")
    print(f"    From matches:    {origin_above_floor.get('matched', 0)}")
    print(f"    From gaps:       {origin_above_floor.get('gap', 0)}")
    print(f"    From market-only: {origin_above_floor.get('market_only', 0)}")

    print(f"\n  Top 5 by VOI:")
    for s in ranked_signals[:5]:
        origin = s.get("origin", "?")
        print(f"    VOI={s.get('voi', 0):.2f} [{origin}] {s['text'][:50]}...")

    # Step 4.2: Deduplicate
    print("\n[4.2] Deduplicating...")
    before_dedup = len(ranked_signals)
    deduped_signals = deduplicate_signals(
        [{"text": s["text"], "source": s["source"], **s} for s in ranked_signals[:100]],
        threshold=0.45
    )
    print(f"  {before_dedup} → {len(deduped_signals)} (removed {before_dedup - len(deduped_signals)} duplicates)")

    # Step 4.3: Generate scenarios
    print("\n[4.3] Generating MECE scenarios...")
    result = await generate_mece_scenarios(
        signals=[{"text": s["text"], "source": s["source"], "voi": s.get("voi", 0)} for s in deduped_signals[:50]],
        question=TARGET_QUESTION,
        context=CONTEXT,
        voi_floor=args.voi_floor,
    )

    # ============================================================
    # OUTPUT
    # ============================================================
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

    # ============================================================
    # GAP ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("GAP ANALYSIS: Resolution Sources to Track")
    print("=" * 60)

    # Group gaps by resolution source
    gaps_by_source = {}
    for g in gaps:
        src = g.get("resolution_source", "unspecified")
        if src not in gaps_by_source:
            gaps_by_source[src] = []
        gaps_by_source[src].append(g)

    print(f"\n{len(gaps)} cruxy signals have NO market coverage.")
    print("These require tracking via resolution sources (metadata only, no adapters):\n")

    for src, signals in sorted(gaps_by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {src} ({len(signals)} signals):")
        for sig in signals[:3]:
            print(f"    - {sig['text'][:70]}...")
        if len(signals) > 3:
            print(f"    ... and {len(signals) - 3} more")

    # Save results
    output_file = OUTPUT_DIR / f"dual_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Build Signal instances
    signals_v7 = [
        Signal(
            id=s["id"],
            source=s["source"],
            text=s["text"],
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

    # Build v7 scenario format
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

    # Gap details for JSON output
    gaps_v7 = [
        {
            "text": g["text"],
            "resolution_source": g.get("resolution_source"),
            "uncertainty_source": g.get("uncertainty_source"),
            "base_rate": g.get("base_rate"),
            "best_market_similarity": g.get("best_market_similarity", 0),
        }
        for g in gaps
    ]

    results = {
        "id": f"dual_{args.target}_{uuid4().hex[:8]}",
        "name": f"{args.target} (dual)",
        "target": args.target,
        "approach": "dual_v7",
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
            "match_threshold": args.match_threshold,
            "n_uncertainties": args.n_uncertainties,
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
        # Phase 1: Top-down
        "topdown_signals_generated": len(topdown_signals),
        # Phase 2: Bottom-up
        "market_signals_retrieved": len(market_signals),
        # Phase 3: Match
        "matched_signals": len(matched),
        "gap_signals": len(gaps),
        "market_only_signals": len(market_only),
        # Phase 4: Merged
        "merged_signals_total": len(merged_signals),
        "signals_above_floor": sum(1 for s in ranked_signals if s.get("voi", 0) >= args.voi_floor),
        "signals_above_floor_by_origin": origin_above_floor,
        # Output
        "signals": [s.model_dump(mode="json", exclude_none=True) for s in signals_v7],
        "scenarios": scenarios_v7,
        "mece_reasoning": result.mece_reasoning,
        "coverage_gaps": result.coverage_gaps,
        # Gap analysis
        "gaps": gaps_v7,
        "gaps_by_resolution_source": {
            src: len(sigs) for src, sigs in gaps_by_source.items()
        },
        "created_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
