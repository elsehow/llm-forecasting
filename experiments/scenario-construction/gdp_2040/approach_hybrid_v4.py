#!/usr/bin/env python3
"""
Hybrid approach v4: Single-call MECE scenario generation.

Combines LLM-brainstormed signals WITH market signals from unified database.

Key features:
- LLM brainstorms cruxy signals (what SHOULD matter)
- Database provides market signals (what IS being tracked) via semantic search
- Deduplication prefers observed signals over LLM-generated
- Combine, cluster, then generate MECE scenarios

Usage:
    uv run python experiments/scenario-construction/gdp_2040/approach_hybrid_v4.py --target gdp_2040
    uv run python experiments/scenario-construction/gdp_2040/approach_hybrid_v4.py --target renewable_2050
"""

import argparse
import json
import asyncio
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import litellm

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.scenario_mece import map_signals_to_all_scenarios
from shared.signals import (
    RESOLVABILITY_REQUIREMENTS,
    load_market_signals_semantic,
    deduplicate_signals,
    DEFAULT_KNOWLEDGE_CUTOFF,
    resolution_proximity_score,
    get_resolution_bucket,
    categorize_signal,
)
from shared.config import get_target, TARGETS

load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description="Hybrid scenario generation approach")
parser.add_argument(
    "--target",
    choices=list(TARGETS.keys()),
    default="gdp_2040",
    help="Target question to generate scenarios for",
)
args = parser.parse_args()

# Load config
config = get_target(args.target)
TARGET_QUESTION = config.question.text
CONTEXT = config.context
UNIT = config.question.unit

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = REPO_ROOT / "data" / "forecastbench.db"
OUTPUT_DIR = Path(__file__).parent / "results" / args.target
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LLM config
MODEL = "claude-sonnet-4-20250514"

# Knowledge cutoff (model's training data end date) and sources
KNOWLEDGE_CUTOFF = DEFAULT_KNOWLEDGE_CUTOFF  # "2025-10-01"
SOURCES = ["polymarket", "metaculus", "fred", "infer", "manifold"]


# ============================================================
# Data classes
# ============================================================

@dataclass
class Signal:
    """A cruxy signal."""
    text: str
    reasoning: str
    source: str = "llm"  # "llm" or market source name
    cluster: int | None = None


# ============================================================
# Pydantic models
# ============================================================

class SignalResponse(BaseModel):
    text: str
    reasoning: str


class SignalsResponse(BaseModel):
    signals: list[SignalResponse]


class CorrelationEstimate(BaseModel):
    pair_id: str
    rho: float


class CorrelationsResponse(BaseModel):
    correlations: list[CorrelationEstimate]


class MECEScenario(BaseModel):
    """A single MECE scenario."""
    name: str = Field(description="2-4 word memorable name")
    description: str = Field(description="2-3 sentences describing this world state")
    gdp_range: str = Field(description="Expected GDP range in 2040, e.g., '$45-55T'")
    key_drivers: list[str] = Field(description="3-5 factors that define this scenario")
    why_exclusive: str = Field(description="Why this scenario CANNOT co-occur with the others")
    signals_from_clusters: list[str] = Field(description="Which input signals map to this scenario")
    indicator_bundle: dict[str, str] = Field(description="3-5 measurable indicators")


class MECEScenariosResponse(BaseModel):
    """Response from single-call MECE scenario generation."""
    scenarios: list[MECEScenario]
    mece_reasoning: str = Field(description="Explanation of why these scenarios are MECE")
    coverage_gaps: list[str] = Field(description="Any outcomes not covered (should be empty or minimal)")


# ============================================================
# Pydantic models for market relevance scoring
# ============================================================

class MarketRelevance(BaseModel):
    id: str
    question: str
    gdp_relevance: str = Field(description="How this relates to GDP")
    relevance_score: int = Field(description="1-5, 5 being most relevant")


class MarketRelevanceResponse(BaseModel):
    markets: list[MarketRelevance]


# ============================================================
# Core functions
# ============================================================

def load_market_signals_from_db(limit: int = 100) -> list[dict]:
    """
    Load GDP-relevant market signals from unified database (keyword-based).

    NOTE: This is the legacy keyword-based approach. Consider using
    `load_market_signals_semantic()` for better recall (~95% vs ~70%).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # GDP-relevant keywords for filtering
    gdp_keywords = [
        "gdp", "economy", "economic", "growth", "recession",
        "inflation", "fed", "federal reserve", "interest rate",
        "trade", "tariff", "china", "ai", "automation",
        "tax", "spending", "deficit", "oil", "energy",
        "labor", "jobs", "wage", "taiwan", "war", "conflict",
        "election", "president", "rate", "yield", "bond",
    ]

    placeholders = ",".join("?" * len(SOURCES))
    query = f"""
        SELECT id, source, text, background, resolution_date
        FROM questions
        WHERE source IN ({placeholders})
        AND (resolution_date > ? OR resolution_date IS NULL)
        AND resolved = 0
    """

    cursor.execute(query, (*SOURCES, KNOWLEDGE_CUTOFF))
    rows = cursor.fetchall()
    conn.close()

    # Filter for GDP relevance
    filtered = []
    for row in rows:
        text_lower = (row["text"] or "").lower()
        bg_lower = (row["background"] or "").lower()
        combined = text_lower + " " + bg_lower

        if any(kw in combined for kw in gdp_keywords):
            filtered.append({
                "id": row["id"],
                "source": row["source"],
                "question": row["text"],
                "background": row["background"] or "",
            })

    return filtered[:limit]


async def score_market_signals(markets: list[dict], batch_size: int = 30) -> list[dict]:
    """LLM scores market signal relevance to GDP."""
    scored = []

    for i in range(0, len(markets), batch_size):
        batch = markets[i : i + batch_size]
        batch_items = [{"id": m["id"], "question": m["question"]} for m in batch]

        prompt = f"""Score these forecasting questions for GDP 2040 relevance (1-5):
- 5: Directly affects GDP (Fed policy, major trade events)
- 4: Strong indirect effect (tech milestones, geopolitics)
- 3: Moderate relevance
- 2: Weak relevance
- 1: Minimal relevance

Questions:
{json.dumps(batch_items, indent=2)}
"""

        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=MarketRelevanceResponse,
        )

        result = MarketRelevanceResponse.model_validate_json(response.choices[0].message.content)
        score_by_id = {m.id: m for m in result.markets}

        for m in batch:
            if m["id"] in score_by_id:
                m["relevance_score"] = score_by_id[m["id"]].relevance_score
                m["gdp_relevance"] = score_by_id[m["id"]].gdp_relevance
            else:
                m["relevance_score"] = 0

        scored.extend(batch)
        print(f"    Scored market batch {i // batch_size + 1}/{(len(markets) + batch_size - 1) // batch_size}")

    return scored


async def brainstorm_signals(question: str, n: int = 20) -> list[Signal]:
    """Step 1: LLM brainstorms cruxy signals that are resolvable as prediction market questions."""
    prompt = f"""You are a superforecaster analyzing: {question}

What are the {n} signals that would MOST change your forecast?

{RESOLVABILITY_REQUIREMENTS}

Focus on signals that resolve in 2-5 years. Explain WHY each signal is cruxy for the forecast.
Cover diverse domains: AI/tech, geopolitics, fiscal policy, demographics, energy.
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SignalsResponse,
    )

    content = response.choices[0].message.content
    try:
        result = SignalsResponse.model_validate_json(content)
    except Exception:
        # Handle case where LLM returns stringified signals array
        data = json.loads(content)
        if isinstance(data.get("signals"), str):
            data["signals"] = json.loads(data["signals"])
        result = SignalsResponse.model_validate(data)

    return [Signal(text=s.text, reasoning=s.reasoning) for s in result.signals]


async def estimate_rho_matrix(signals: list[Signal]) -> np.ndarray:
    """Step 2: Estimate correlation matrix."""
    n = len(signals)
    rho_matrix = np.eye(n)
    pairs = list(combinations(range(n), 2))

    batch_size = 15
    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]

        pair_descriptions = [
            {"pair_id": f"{i}_{j}", "signal_a": signals[i].text, "signal_b": signals[j].text}
            for i, j in batch_pairs
        ]

        prompt = f"""Estimate correlation (ρ) between each signal pair.
ρ: -1 (opposite) to +1 (move together), 0 = independent.

Pairs:
{json.dumps(pair_descriptions, indent=2)}
"""

        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=CorrelationsResponse,
        )

        result = CorrelationsResponse.model_validate_json(response.choices[0].message.content)
        for corr in result.correlations:
            parts = corr.pair_id.split("_")
            if len(parts) == 2:
                i, j = int(parts[0]), int(parts[1])
                rho_matrix[i, j] = corr.rho
                rho_matrix[j, i] = corr.rho

    return rho_matrix


def cluster_signals(signals: list[Signal], rho_matrix: np.ndarray, n_clusters: int = 4) -> list[list[Signal]]:
    """Step 3: Cluster signals by correlation."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(signals)
    if n < n_clusters:
        return [[s] for s in signals]

    distance_matrix = 1 - np.abs(rho_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 1)

    condensed = squareform(distance_matrix)
    linkage_matrix = linkage(condensed, method='ward')
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    clusters = [[] for _ in range(n_clusters)]
    for signal, label in zip(signals, labels):
        signal.cluster = label - 1
        clusters[label - 1].append(signal)

    clusters.sort(key=len, reverse=True)
    return clusters


# ============================================================
# NEW: Single-call MECE scenario generation (Steps 4-6 merged)
# ============================================================

async def generate_mece_scenarios_from_clusters(
    clusters: list[list[Signal]],
    question: str,
    context: str,
) -> MECEScenariosResponse:
    """
    Single LLM call that generates MECE scenarios from signal clusters.

    The MECE constraint is embedded in the prompt, not validated post-hoc.
    """

    # Format clusters for the prompt
    cluster_summaries = []
    for i, cluster in enumerate(clusters):
        cluster_summaries.append({
            "cluster_id": i + 1,
            "size": len(cluster),
            "signals": [s.text for s in cluster],
            "sample_reasoning": cluster[0].reasoning if cluster else "",
        })

    prompt = f"""You are constructing MECE scenarios for forecasting.

TARGET QUESTION: {question}
CONTEXT: {context}

I have clustered {sum(len(c) for c in clusters)} cruxy signals into {len(clusters)} groups based on correlation:

{json.dumps(cluster_summaries, indent=2)}

YOUR TASK: Transform these clusters into 3-5 MECE SCENARIOS.

CRITICAL REQUIREMENTS:

1. MUTUALLY EXCLUSIVE: Each scenario must be a distinct world-state that CANNOT co-occur with the others.
   - Bad: "AI Boom" and "Green Revolution" (these can happen together)
   - Good: "AI-Led Growth" vs "AI Winter" (opposite states of same variable)
   - Good: "High Growth" vs "Stagnation" vs "Crisis" (different outcome ranges)

2. COLLECTIVELY EXHAUSTIVE: Together, the scenarios should cover ALL plausible outcomes.
   - Don't leave gaps in the possibility space
   - Include baseline/moderate outcomes, not just extremes

3. USE THE CLUSTERS AS INPUT, NOT OUTPUT:
   - Clusters show which signals correlate — use this to understand the structure
   - But scenarios should be about OUTCOMES, not about which signals fire
   - Multiple clusters might map to the same scenario
   - A single cluster might split across scenarios

4. For each scenario, explain WHY it cannot co-occur with the others.

5. Acknowledge any coverage gaps honestly (ideally there are none).

Think step by step:
- What are the key axes of uncertainty?
- How can we partition the outcome space so scenarios don't overlap?
- What GDP ranges correspond to each scenario?
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=MECEScenariosResponse,
    )

    return MECEScenariosResponse.model_validate_json(response.choices[0].message.content)


async def main():
    """Run v4 hybrid approach: LLM signals + market signals combined."""
    print("=" * 60)
    print("HYBRID APPROACH v4: LLM + Multi-Source Market Signals")
    print("=" * 60)
    print(f"\nTarget: {TARGET_QUESTION}")
    print(f"Sources: {', '.join(SOURCES)}")
    print(f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}")

    # Step 1a: Brainstorm LLM signals (overgenerate, will dedupe)
    print("\n[1/7] Brainstorming LLM signals...")
    llm_signals = await brainstorm_signals(TARGET_QUESTION, n=30)
    print(f"  Generated {len(llm_signals)} LLM signals")

    # Step 1b: Load market signals via semantic search + enrich with resolution data
    print("\n[2/7] Loading market signals via semantic search...")
    raw_markets = load_market_signals_semantic(
        db_path=DB_PATH,
        query=f"What signals affect {TARGET_QUESTION}",
        sources=SOURCES,
        top_k=150,
    )
    print(f"  Found {len(raw_markets)} semantically relevant market questions")

    # Enrich with resolution data from DB
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    for m in raw_markets:
        cursor.execute(
            "SELECT resolution_date, resolved, resolution_value FROM questions WHERE id = ? AND source = ?",
            (m["id"], m["source"])
        )
        row = cursor.fetchone()
        if row:
            res_date = row["resolution_date"]
            resolved = bool(row["resolved"])
            m["resolution_date"] = str(res_date) if res_date else None
            m["resolved"] = resolved
            m["resolution_value"] = row["resolution_value"]
            m["signal_category"] = categorize_signal(m["resolution_date"], resolved, KNOWLEDGE_CUTOFF)
        else:
            m["resolution_date"] = None
            m["resolved"] = False
            m["resolution_value"] = None
            m["signal_category"] = "unknown"
    conn.close()

    # Filter out signals resolved BEFORE cutoff (model already knows)
    before_filter = len(raw_markets)
    raw_markets = [m for m in raw_markets if m["signal_category"] != "exclude"]
    print(f"  After resolution filtering: {len(raw_markets)} (excluded {before_filter - len(raw_markets)} pre-cutoff resolved)")

    # Show resolution category breakdown
    by_category = {}
    for m in raw_markets:
        cat = m.get("signal_category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    print(f"  Resolution categories: {by_category}")

    print("\n[3/7] Scoring market signal relevance...")
    scored_markets = await score_market_signals(raw_markets)

    # Apply proximity boost to relevance scores
    for m in scored_markets:
        base_score = m.get("relevance_score", 0)
        proximity = resolution_proximity_score(m.get("resolution_date"), KNOWLEDGE_CUTOFF)
        m["relevance_score_raw"] = base_score
        m["resolution_proximity"] = proximity
        m["relevance_score"] = base_score * (0.7 + 0.3 * proximity)

    high_relevance = [m for m in scored_markets if m.get("relevance_score", 0) >= 4 * 0.7]
    print(f"  High relevance (adjusted): {len(high_relevance)}")

    # Show gold signals
    gold_signals = [m for m in high_relevance if m.get("signal_category") == "gold"]
    if gold_signals:
        print(f"  Gold signals (post-cutoff resolved): {len(gold_signals)}")

    # Convert top market signals to Signal objects
    market_signals = [
        Signal(
            text=m["question"],
            reasoning=m.get("gdp_relevance", "Market signal"),
            source=m["source"],
        )
        for m in high_relevance[:50]  # Top 50 market signals (more than before, will dedupe)
    ]

    # Step 2: Deduplicate, preferring observed signals
    print("\n[4/7] Deduplicating signals (preferring observed over LLM)...")
    signals_before = len(llm_signals) + len(market_signals)
    # Combine market (observed) + LLM signals; deduplicate_signals prefers observed
    combined_signals = market_signals + llm_signals
    # Convert to dicts for deduplication
    combined_dicts = [{"text": s.text, "reasoning": s.reasoning, "source": s.source} for s in combined_signals]
    unique_dicts = deduplicate_signals(combined_dicts, threshold=0.45)
    # Convert back to Signal objects
    signals = [Signal(text=d["text"], reasoning=d["reasoning"], source=d["source"]) for d in unique_dicts]
    signals_removed = signals_before - len(signals)

    # Count by source after deduplication
    llm_count = sum(1 for s in signals if s.source == "llm")
    market_count = len(signals) - llm_count
    print(f"  Before: {signals_before} ({len(llm_signals)} LLM + {len(market_signals)} market)")
    print(f"  After:  {len(signals)} ({llm_count} LLM + {market_count} market)")
    print(f"  Removed: {signals_removed} duplicates")

    # Step 3: Estimate correlations
    print("\n[5/7] Estimating correlations...")
    rho_matrix = await estimate_rho_matrix(signals)
    print(f"  Mean |ρ|: {np.mean(np.abs(rho_matrix)):.3f}")

    # Step 4: Cluster signals
    print("\n[6/7] Clustering and generating MECE scenarios...")
    clusters = cluster_signals(signals, rho_matrix, n_clusters=5)
    for i, cluster in enumerate(clusters):
        sources = set(s.source for s in cluster)
        print(f"  Cluster {i+1}: {len(cluster)} signals (sources: {', '.join(sources)})")
        for s in cluster[:2]:
            print(f"    - [{s.source}] {s.text[:45]}...")

    # Step 4: Single-call MECE scenario generation
    print("\n  Generating MECE scenarios (single call)...")
    result = await generate_mece_scenarios_from_clusters(clusters, TARGET_QUESTION, CONTEXT)

    # Step 6: Map signals to ALL scenarios with directions
    print("\n[7/7] Mapping signals to scenarios with directions...")
    signal_dicts = [{"text": s.text, "reasoning": s.reasoning, "source": s.source} for s in signals]
    scenario_dicts = [
        {"name": s.name, "description": s.description, "gdp_range": s.gdp_range}
        for s in result.scenarios
    ]
    signal_scenario_matrix = await map_signals_to_all_scenarios(
        signals=signal_dicts,
        scenarios=scenario_dicts,
        target_question=TARGET_QUESTION,
        batch_size=10,
    )
    print(f"  Mapped {len(signal_scenario_matrix)} signals to {len(result.scenarios)} scenarios")

    # Output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMECE Reasoning: {result.mece_reasoning}")

    if result.coverage_gaps:
        print(f"\nCoverage Gaps:")
        for gap in result.coverage_gaps:
            print(f"  - {gap}")
    else:
        print(f"\nCoverage Gaps: None identified")

    print("\n" + "-" * 40)
    print("SCENARIOS:")

    for s in result.scenarios:
        print(f"\n### {s.name}")
        print(f"  GDP Range: {s.gdp_range}")
        print(f"  {s.description}")
        print(f"\n  Why Exclusive: {s.why_exclusive}")
        print(f"\n  Key Drivers:")
        for d in s.key_drivers[:3]:
            print(f"    - {d}")
        print(f"\n  Indicators:")
        for ind, thresh in list(s.indicator_bundle.items())[:3]:
            print(f"    - {ind}: {thresh}")

    # Quick MECE check
    print("\n" + "-" * 40)
    print("MECE QUICK CHECK:")
    print(f"  Number of scenarios: {len(result.scenarios)}")

    outcome_ranges = [s.gdp_range for s in result.scenarios]
    print(f"  Outcome ranges: {outcome_ranges}")

    # Check for obvious overlaps in stated exclusivity
    exclusivity_statements = [s.why_exclusive for s in result.scenarios]
    print(f"  All scenarios have exclusivity reasoning: {all(exclusivity_statements)}")

    # Save results
    output_file = OUTPUT_DIR / f"hybrid_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Count signals by source
    source_counts = {}
    for s in signals:
        source_counts[s.source] = source_counts.get(s.source, 0) + 1

    results = {
        "target": args.target,
        "approach": "hybrid_v4_semantic_search",
        "question": {
            "id": config.question.id,
            "text": config.question.text,
            "unit": config.question.unit.type if config.question.unit else None,
            "base_rate": config.question.base_rate,
            "value_range": config.question.value_range,
        },
        "config": {
            "context": config.context,
            "cruxiness_normalizer": config.cruxiness_normalizer,
        },
        "knowledge_cutoff": KNOWLEDGE_CUTOFF,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "sources_used": SOURCES,
        "signal_breakdown": source_counts,
        "resolution_distribution": by_category,
        "deduplication": {
            "llm_generated": len(llm_signals),
            "market_retrieved": len(market_signals),
            "before_dedup": signals_before,
            "after_dedup": len(signals),
            "duplicates_removed": signals_removed,
            "threshold": 0.45,
            "method": "semantic_similarity_with_observed_preference",
        },
        "signals": [
            {
                "text": s.text,
                "reasoning": s.reasoning,
                "source": s.source,
                "cluster": int(s.cluster) if s.cluster is not None else None,
                # Resolution metadata (only for market signals)
                "resolution_date": next((m.get("resolution_date") for m in high_relevance if m["question"] == s.text), None),
                "resolution_bucket": next((get_resolution_bucket(m.get("resolution_date"), KNOWLEDGE_CUTOFF) for m in high_relevance if m["question"] == s.text), None) if s.source != "llm" else None,
                "signal_category": next((m.get("signal_category") for m in high_relevance if m["question"] == s.text), None) if s.source != "llm" else "llm_generated",
                "resolved": next((m.get("resolved", False) for m in high_relevance if m["question"] == s.text), False) if s.source != "llm" else False,
                "resolution_value": next((m.get("resolution_value") for m in high_relevance if m["question"] == s.text), None) if s.source != "llm" else None,
            }
            for s in signals
        ],
        "rho_matrix": rho_matrix.tolist(),
        "clusters": [[{"text": s.text, "source": s.source} for s in cluster] for cluster in clusters],
        "scenarios": [
            {
                "name": s.name,
                "description": s.description,
                "gdp_range": s.gdp_range,
                "key_drivers": s.key_drivers,
                "why_exclusive": s.why_exclusive,
                "signals_from_clusters": s.signals_from_clusters,
                "indicator_bundle": s.indicator_bundle,
            }
            for s in result.scenarios
        ],
        "mece_reasoning": result.mece_reasoning,
        "coverage_gaps": result.coverage_gaps,
        "signal_scenario_matrix": signal_scenario_matrix,
        "created_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
