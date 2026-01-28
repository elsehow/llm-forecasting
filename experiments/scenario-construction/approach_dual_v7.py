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

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_forecasting.semantic_search import SemanticSignalSearcher

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import (
    load_market_signals_semantic,
    rank_signals_by_voi,
    enrich_with_resolution_data,
    filter_by_resolution_date,
    build_signal_models,
)
from shared.generation import generate_signals_for_uncertainty
from shared.uncertainties import identify_and_report_uncertainties
from shared.scenarios import generate_and_print_scenarios
from shared.refresh import refresh_if_stale
from shared.setup import (
    create_base_parser,
    add_uncertainty_args,
    add_matching_args,
    load_config,
    print_header,
    DEFAULT_SOURCES,
)
from shared.output import (
    build_scenario_dicts,
    build_base_results,
    save_results,
)

load_dotenv()

# Parse arguments
parser = create_base_parser("Dual v7 scenario generation")
add_uncertainty_args(parser)
add_matching_args(parser)
args = parser.parse_args()

# Load config
cfg = load_config(args, Path(__file__))
SOURCES = DEFAULT_SOURCES


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
    print_header(
        "DUAL",
        cfg.question_text,
        sources=SOURCES,
        max_horizon_days=cfg.max_horizon_days,
        n_uncertainties=cfg.n_uncertainties,
        voi_floor=cfg.voi_floor,
        match_threshold=cfg.match_threshold,
    )

    # ============================================================
    # PHASE 1: Top-Down (LLM-generated signals)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: TOP-DOWN (LLM-generated signals)")
    print("=" * 60)

    # Step 1.1: Identify key uncertainties
    print("\n[1.1] Identifying key uncertainties...")
    uncertainties = await identify_and_report_uncertainties(
        question=cfg.question_text,
        context=cfg.context,
        n=cfg.n_uncertainties,
    )

    # Step 1.2: Generate signals per uncertainty
    print("\n[1.2] Generating signals per uncertainty...")
    topdown_signals = []

    for u in uncertainties:
        print(f"\n  Generating for: {u.name}")
        signals = await generate_signals_for_uncertainty(
            question=cfg.question_text,
            uncertainty_name=u.name,
            uncertainty_description=u.description,
            n=10,
            max_horizon_days=cfg.max_horizon_days,
            target_resolution_date=cfg.target_resolution_date,
        )
        print(f"    Generated {len(signals)} signals")
        topdown_signals.extend(signals)

    print(f"\n  Total top-down signals: {len(topdown_signals)}")

    # Filter by resolution date (same function used for all signals)
    before_filter = len(topdown_signals)
    topdown_signals = filter_by_resolution_date(topdown_signals, cfg.max_horizon_days)
    print(f"  After date filtering: {len(topdown_signals)} (excluded {before_filter - len(topdown_signals)} outside horizon)")

    # ============================================================
    # PHASE 2: Bottom-Up (Market signals)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: BOTTOM-UP (Market signals)")
    print("=" * 60)

    # Step 2.1: Refresh market data if stale
    print("\n[2.1] Checking data freshness...")
    await refresh_if_stale(str(cfg.db_path), SOURCES)

    # Step 2.2: Semantic search for market signals
    print("\n[2.2] Loading signals via semantic search...")
    market_signals = load_market_signals_semantic(
        db_path=cfg.db_path,
        query=f"What signals affect {cfg.question_text}",
        sources=SOURCES,
        top_k=200,
    )
    print(f"  Retrieved {len(market_signals)} semantically relevant signals")

    # Enrich with resolution data from DB
    enrich_with_resolution_data(market_signals, cfg.db_path, cfg.max_horizon_days)

    # Filter by resolution date (same function used for all signals)
    before_filter = len(market_signals)
    market_signals = filter_by_resolution_date(market_signals, cfg.max_horizon_days)
    print(f"  After date filtering: {len(market_signals)} (excluded {before_filter - len(market_signals)} outside horizon)")

    # ============================================================
    # PHASE 3: Match and Merge
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: MATCH AND MERGE")
    print("=" * 60)

    # Step 3.1: Match top-down signals to market signals
    print(f"\n[3.1] Matching top-down to market (threshold={cfg.match_threshold})...")
    matched, gaps, market_only = match_topdown_to_market(
        topdown_signals,
        market_signals,
        threshold=cfg.match_threshold,
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
    is_continuous = cfg.question_type == "continuous"
    ranked_signals = await rank_signals_by_voi(
        signals=merged_signals,
        target=cfg.question_text,
        is_continuous=is_continuous,
    )

    # Count by origin
    origin_above_floor = {"matched": 0, "gap": 0, "market_only": 0}
    for s in ranked_signals:
        if s.get("voi", 0) >= cfg.voi_floor:
            origin = s.get("origin", "unknown")
            origin_above_floor[origin] = origin_above_floor.get(origin, 0) + 1

    print(f"\n  Signals above VOI floor ({cfg.voi_floor}):")
    print(f"    From matches:    {origin_above_floor.get('matched', 0)}")
    print(f"    From gaps:       {origin_above_floor.get('gap', 0)}")
    print(f"    From market-only: {origin_above_floor.get('market_only', 0)}")

    print(f"\n  Top 5 by VOI:")
    for s in ranked_signals[:5]:
        origin = s.get("origin", "?")
        print(f"    VOI={s.get('voi', 0):.2f} [{origin}] {s['text'][:50]}...")

    # ============================================================
    # CONDITIONAL ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("CONDITIONAL ANALYSIS: How Signals Move the Target")
    print("=" * 60)

    # Show top cruxy signals with conditional probabilities
    cruxy_signals = [s for s in ranked_signals if s.get("voi", 0) >= cfg.voi_floor][:10]
    if cruxy_signals:
        print(f"\nTop {len(cruxy_signals)} signals by VOI (showing P(target|signal resolution)):\n")
        for i, s in enumerate(cruxy_signals, 1):
            p_yes = s.get("p_target_given_yes") or 0.5
            p_no = s.get("p_target_given_no") or 0.5
            spread = s.get("cruxiness_spread") or 0
            base_rate = s.get("base_rate") or 0.5
            rho = s.get("rho") or 0

            print(f"  {i}. {s['text'][:70]}...")
            print(f"     P(signal=YES): {base_rate:.0%} | ρ={rho:+.2f}")
            print(f"     If YES → P(target) = {p_yes:.0%}")
            print(f"     If NO  → P(target) = {p_no:.0%}")
            print(f"     Spread: {spread:.0%} (cruxiness)")
            print()
    else:
        print("\n  No signals above VOI floor for conditional analysis.")

    # Step 4.2: Deduplicate (prefer higher VOI, not observed source)
    print("\n[4.2] Deduplicating (VOI-first)...")
    before_dedup = len(ranked_signals)
    # For dual approach: prefer higher VOI signals regardless of source
    # Sort by VOI descending first, then deduplicate
    signals_for_dedup = sorted(
        [{"text": s["text"], "source": s["source"], **s} for s in ranked_signals[:100]],
        key=lambda x: -x.get("voi", 0)
    )
    searcher = SemanticSignalSearcher()
    deduped_signals = searcher.deduplicate(
        signals_for_dedup,
        threshold=0.65,  # Higher threshold: keep more distinct signals
        prefer_observed=False,  # Don't re-sort by source
    )
    print(f"  {before_dedup} → {len(deduped_signals)} (removed {before_dedup - len(deduped_signals)} duplicates)")

    # Step 4.3: Generate scenarios
    print("\n[4.3] Generating scenarios...")
    result = await generate_and_print_scenarios(
        signals=deduped_signals[:50],
        question=cfg.question_text,
        context=cfg.context,
        question_type=cfg.question_type,
        voi_floor=cfg.voi_floor,
    )

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
    output_file = cfg.output_dir / f"dual_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    signals_v7 = build_signal_models(
        deduped_signals[:50],
        text_key="text",
        include_uncertainty_source=True,
    )
    scenarios_v7 = build_scenario_dicts(result.scenarios)

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

    results = build_base_results(
        approach="dual",
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
    results["config"]["match_threshold"] = cfg.match_threshold
    results["config"]["n_uncertainties"] = cfg.n_uncertainties
    results.update({
        "sources_used": SOURCES,
        "uncertainties": [
            {"name": u.name, "description": u.description, "search_query": u.search_query}
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
        "signals_above_floor": sum(1 for s in ranked_signals if s.get("voi", 0) >= cfg.voi_floor),
        "signals_above_floor_by_origin": origin_above_floor,
        # Gap analysis
        "gaps": gaps_v7,
        "gaps_by_resolution_source": {
            src: len(sigs) for src, sigs in gaps_by_source.items()
        },
    })

    save_results(results, output_file)


if __name__ == "__main__":
    asyncio.run(main())
