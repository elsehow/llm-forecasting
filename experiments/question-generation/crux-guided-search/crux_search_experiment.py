#!/usr/bin/env python3
"""
Crux-Guided Search Experiment

Use LLM-generated cruxes as search queries to discover relevant Polymarket signals.
Validate that discovered signals show actual information flow to the ultimate.

Pipeline:
1. Select test ultimates from Polymarket (high-volume, good price history)
2. Generate cruxes for each ultimate
3. Search Polymarket via semantic search for each crux
4. Compute VOI for (ultimate, found_signal) pairs
5. Select diverse high-VOI signals with MMR
6. Validate: price correlation between signal and ultimate

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/crux-guided-search/crux_search_experiment.py
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import litellm

from llm_forecasting.market_data.polymarket import PolymarketData
from llm_forecasting.semantic_search.searcher import SemanticSignalSearcher
from llm_forecasting.voi import (
    estimate_rho_market_aware,
    linear_voi_from_rho,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

load_dotenv()

# Paths
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Database path for semantic search
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "forecastbench.db"

# Config
MODEL = "anthropic/claude-sonnet-4-20250514"
MODEL_CHEAP = "anthropic/claude-3-haiku-20240307"
N_ULTIMATES = 5
N_CRUXES_PER_ULTIMATE = 5
SEARCH_TOP_K = 10  # Results per crux search
VOI_THRESHOLD = 0.01
MMR_K = 5  # Signals to select per ultimate
MMR_LAMBDA = 0.7


# =============================================================================
# Step 1: Select Ultimates
# =============================================================================

async def select_ultimates(n: int = N_ULTIMATES) -> list[dict]:
    """Select diverse, high-liquidity ultimates from Polymarket."""
    print("\n[1/6] Selecting ultimates from Polymarket...")

    polymarket = PolymarketData()
    markets = await polymarket.fetch_markets(
        active_only=True,
        min_liquidity=50000,  # $50k minimum liquidity
        limit=200,
    )

    print(f"      Fetched {len(markets)} active markets")

    # Filter for good ultimates (not too specific)
    excluded_patterns = [
        "2025 nba",
        "2025 nfl",
        "2025 nhl",
        "march madness",
        "super bowl",
        "will reach",  # Price targets often lack good signals
        "nomination",  # Too many similar nomination questions
    ]

    def is_good_ultimate(m):
        title = m.title.lower()
        return not any(p in title for p in excluded_patterns)

    filtered = [m for m in markets if is_good_ultimate(m)]
    print(f"      {len(filtered)} markets after filtering")

    # Sort by volume and take top N
    sorted_markets = sorted(
        filtered,
        key=lambda m: (m.volume_total or 0),
        reverse=True
    )[:n]

    # Convert to dicts with fields we need
    ultimates = []
    for m in sorted_markets:
        ultimates.append({
            "id": m.id,
            "title": m.title,
            "url": m.url,
            "current_probability": m.current_probability,
            "liquidity": m.liquidity,
            "volume_total": m.volume_total,
            "clob_token_ids": m.clob_token_ids,
        })

    for i, u in enumerate(ultimates):
        vol = u.get("volume_total", 0)
        print(f"      {i+1}. ${vol:,.0f} - {u['title'][:50]}...")

    return ultimates


# =============================================================================
# Step 2: Generate Cruxes
# =============================================================================

CRUX_GENERATION_PROMPT = """You are a superforecaster. Given an ultimate question (a prediction market question), generate {n} intermediate "crux" questions that would help predict the answer.

A good crux question:
1. Resolves BEFORE the ultimate question
2. Would significantly update your probability if answered
3. Is specific and measurable
4. Could plausibly be another prediction market question

Ultimate Question: {ultimate}

Generate exactly {n} crux questions. Make them concrete and specific.
Format as a JSON array of strings.

JSON array only:"""


async def generate_cruxes(ultimate: str, n: int = N_CRUXES_PER_ULTIMATE) -> list[str]:
    """Generate crux questions for an ultimate."""
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": CRUX_GENERATION_PROMPT.format(ultimate=ultimate, n=n)
            }],
            max_tokens=1000,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        cruxes = json.loads(text)
        return cruxes[:n]
    except Exception as e:
        print(f"      Error generating cruxes: {e}")
        return []


async def generate_all_cruxes(ultimates: list[dict]) -> dict[str, list[str]]:
    """Generate cruxes for all ultimates."""
    print(f"\n[2/6] Generating cruxes for {len(ultimates)} ultimates...")

    cruxes_by_ultimate = {}

    for i, ultimate in enumerate(ultimates):
        title = ultimate["title"]
        print(f"      [{i+1}/{len(ultimates)}] {title[:50]}...")

        cruxes = await generate_cruxes(title)
        cruxes_by_ultimate[ultimate["id"]] = cruxes
        print(f"            Generated {len(cruxes)} cruxes")

        # Rate limiting
        await asyncio.sleep(0.5)

    total_cruxes = sum(len(c) for c in cruxes_by_ultimate.values())
    print(f"      Total: {total_cruxes} cruxes generated")

    return cruxes_by_ultimate


# =============================================================================
# Step 3: Semantic Search for Signals
# =============================================================================

def search_for_signals(
    cruxes_by_ultimate: dict[str, list[str]],
    searcher: SemanticSignalSearcher,
    top_k: int = SEARCH_TOP_K,
) -> dict[str, list[dict]]:
    """Search for Polymarket signals matching each crux."""
    print(f"\n[3/6] Searching for signals via semantic search...")

    signals_by_ultimate = defaultdict(list)
    total_searches = sum(len(c) for c in cruxes_by_ultimate.values())
    search_count = 0
    cruxes_with_results = 0

    for ultimate_id, cruxes in cruxes_by_ultimate.items():
        seen_signal_ids = set()

        for crux in cruxes:
            search_count += 1
            results = searcher.search(crux, top_k=top_k)

            # Filter to Polymarket only and dedupe
            polymarket_results = []
            for r in results:
                if r.source == "polymarket" and r.id not in seen_signal_ids:
                    polymarket_results.append({
                        "id": r.id,
                        "text": r.text,
                        "source": r.source,
                        "similarity": r.similarity,
                        "crux_query": crux,
                    })
                    seen_signal_ids.add(r.id)

            if polymarket_results:
                cruxes_with_results += 1

            signals_by_ultimate[ultimate_id].extend(polymarket_results)

            if search_count % 10 == 0:
                print(f"      {search_count}/{total_searches} searches complete...")

    total_signals = sum(len(s) for s in signals_by_ultimate.values())
    coverage = cruxes_with_results / total_searches if total_searches > 0 else 0

    print(f"      Found {total_signals} unique signals")
    print(f"      Retrieval coverage: {coverage:.1%} ({cruxes_with_results}/{total_searches} cruxes)")

    return dict(signals_by_ultimate)


# =============================================================================
# Step 4: Compute VOI
# =============================================================================

async def compute_voi_for_signals(
    ultimates: list[dict],
    signals_by_ultimate: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Compute VOI for each (ultimate, signal) pair."""
    print(f"\n[4/6] Computing VOI for signal pairs...")

    # Build lookup for ultimates
    ultimate_lookup = {u["id"]: u for u in ultimates}

    scored_signals_by_ultimate = {}
    total_pairs = sum(len(s) for s in signals_by_ultimate.values())
    pair_count = 0
    above_threshold = 0

    for ultimate_id, signals in signals_by_ultimate.items():
        ultimate = ultimate_lookup[ultimate_id]
        p_ultimate = ultimate.get("current_probability", 0.5)

        scored_signals = []

        for signal in signals:
            pair_count += 1

            # Estimate rho using market-aware approach
            rho, reasoning = await estimate_rho_market_aware(
                ultimate["title"],
                signal["text"],
                model=MODEL_CHEAP,
            )

            # For p_signal, we'd need to fetch from Polymarket
            # Use 0.5 as default (maximum uncertainty)
            p_signal = 0.5

            # Compute VOI
            voi = linear_voi_from_rho(rho, p_ultimate, p_signal)

            scored_signal = {
                **signal,
                "rho": rho,
                "rho_reasoning": reasoning,
                "voi": voi,
                "p_ultimate": p_ultimate,
                "p_signal": p_signal,
            }
            scored_signals.append(scored_signal)

            if voi >= VOI_THRESHOLD:
                above_threshold += 1

            if pair_count % 20 == 0:
                print(f"      {pair_count}/{total_pairs} pairs scored...")

            # Rate limiting for API calls
            await asyncio.sleep(0.1)

        scored_signals_by_ultimate[ultimate_id] = scored_signals

    pass_rate = above_threshold / total_pairs if total_pairs > 0 else 0
    print(f"      VOI pass rate: {pass_rate:.1%} ({above_threshold}/{total_pairs} above threshold)")

    return scored_signals_by_ultimate


# =============================================================================
# Step 5: MMR Selection
# =============================================================================

def compute_similarity_matrix(texts: list[str]) -> np.ndarray:
    """Compute pairwise cosine similarity using TF-IDF."""
    if len(texts) < 2:
        return np.ones((len(texts), len(texts)))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    return sklearn_cosine_similarity(tfidf)


def select_mmr(
    signals: list[dict],
    k: int,
    sim_matrix: np.ndarray,
    lambda_param: float = 0.7
) -> list[dict]:
    """MMR selection: balance VOI with diversity."""
    if len(signals) <= k:
        return signals

    n = len(signals)
    selected_indices = []
    remaining_indices = list(range(n))

    # Normalize VOI to [0, 1]
    vois = np.array([s["voi"] for s in signals])
    voi_min, voi_max = vois.min(), vois.max()
    if voi_max > voi_min:
        vois_norm = (vois - voi_min) / (voi_max - voi_min)
    else:
        vois_norm = np.ones(n) * 0.5

    while len(selected_indices) < k and remaining_indices:
        best_score = -float("inf")
        best_idx = None

        for idx in remaining_indices:
            relevance = vois_norm[idx]

            if selected_indices:
                redundancy = max(sim_matrix[idx, s] for s in selected_indices)
            else:
                redundancy = 0

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [signals[i] for i in selected_indices]


def apply_mmr_selection(
    scored_signals_by_ultimate: dict[str, list[dict]],
    k: int = MMR_K,
    lambda_param: float = MMR_LAMBDA,
) -> dict[str, list[dict]]:
    """Apply MMR selection to each ultimate's signals."""
    print(f"\n[5/6] Applying MMR selection (k={k}, λ={lambda_param})...")

    selected_by_ultimate = {}

    for ultimate_id, signals in scored_signals_by_ultimate.items():
        if not signals:
            selected_by_ultimate[ultimate_id] = []
            continue

        # Compute similarity matrix
        texts = [s["text"] for s in signals]
        sim_matrix = compute_similarity_matrix(texts)

        # Apply MMR
        selected = select_mmr(signals, k, sim_matrix, lambda_param)
        selected_by_ultimate[ultimate_id] = selected

        print(f"      {ultimate_id[:8]}...: {len(signals)} → {len(selected)} signals")

    total_selected = sum(len(s) for s in selected_by_ultimate.values())
    print(f"      Total selected: {total_selected} signals")

    return selected_by_ultimate


# =============================================================================
# Step 6: Validation with Price Correlation
# =============================================================================

async def validate_with_price_correlation(
    ultimates: list[dict],
    selected_by_ultimate: dict[str, list[dict]],
) -> list[dict]:
    """Validate selected signals by computing price correlation with ultimates."""
    print(f"\n[6/6] Validating with price correlation...")

    polymarket = PolymarketData()
    ultimate_lookup = {u["id"]: u for u in ultimates}

    validations = []

    for ultimate_id, signals in selected_by_ultimate.items():
        ultimate = ultimate_lookup[ultimate_id]

        if not signals:
            continue

        # Fetch ultimate price history
        try:
            ultimate_prices = await polymarket.fetch_price_history(
                ultimate_id,
                start=datetime.now(timezone.utc) - timedelta(days=30),
                interval="1d",
            )
        except Exception as e:
            print(f"      Error fetching ultimate prices: {e}")
            continue

        if len(ultimate_prices) < 5:
            print(f"      Insufficient price history for {ultimate['title'][:30]}...")
            continue

        # Convert to daily returns
        ultimate_returns = []
        for i in range(1, len(ultimate_prices)):
            ret = ultimate_prices[i].price - ultimate_prices[i-1].price
            ultimate_returns.append((ultimate_prices[i].timestamp, ret))

        ultimate_ts_to_return = {t: r for t, r in ultimate_returns}

        # For each signal, try to compute correlation
        for signal in signals:
            try:
                signal_prices = await polymarket.fetch_price_history(
                    signal["id"],
                    start=datetime.now(timezone.utc) - timedelta(days=30),
                    interval="1d",
                )
            except Exception as e:
                continue

            if len(signal_prices) < 5:
                continue

            # Convert to daily returns
            signal_returns = []
            for i in range(1, len(signal_prices)):
                ret = signal_prices[i].price - signal_prices[i-1].price
                signal_returns.append((signal_prices[i].timestamp, ret))

            # Align timestamps
            aligned_ultimate = []
            aligned_signal = []

            for ts, sig_ret in signal_returns:
                if ts in ultimate_ts_to_return:
                    aligned_ultimate.append(ultimate_ts_to_return[ts])
                    aligned_signal.append(sig_ret)

            if len(aligned_ultimate) >= 5:
                # Compute Pearson correlation
                corr, p_value = stats.pearsonr(aligned_ultimate, aligned_signal)

                validations.append({
                    "ultimate_id": ultimate_id,
                    "ultimate_title": ultimate["title"],
                    "signal_id": signal["id"],
                    "signal_text": signal["text"],
                    "estimated_rho": signal["rho"],
                    "voi": signal["voi"],
                    "observed_correlation": corr,
                    "p_value": p_value,
                    "n_observations": len(aligned_ultimate),
                })

        await asyncio.sleep(0.2)  # Rate limiting

    # Summarize validation results
    if validations:
        corrs = [v["observed_correlation"] for v in validations]
        significant = [v for v in validations if v["p_value"] < 0.1]

        print(f"      Validated {len(validations)} signal-ultimate pairs")
        print(f"      Mean observed correlation: {np.mean(corrs):.3f}")
        print(f"      Pairs with |r| > 0.2: {sum(1 for c in corrs if abs(c) > 0.2)}")
        print(f"      Pairs significant at p<0.1: {len(significant)}")
    else:
        print(f"      No pairs validated (insufficient price data)")

    return validations


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 70)
    print("CRUX-GUIDED SEARCH EXPERIMENT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  N_ULTIMATES: {N_ULTIMATES}")
    print(f"  N_CRUXES_PER_ULTIMATE: {N_CRUXES_PER_ULTIMATE}")
    print(f"  SEARCH_TOP_K: {SEARCH_TOP_K}")
    print(f"  VOI_THRESHOLD: {VOI_THRESHOLD}")
    print(f"  MMR_K: {MMR_K}")

    # Initialize semantic searcher
    print(f"\nInitializing semantic searcher...")
    if not DB_PATH.exists():
        print(f"  WARNING: Database not found at {DB_PATH}")
        print(f"  Run: uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py --db {DB_PATH}")
        return

    searcher = SemanticSignalSearcher(db_path=str(DB_PATH))

    # Check if cache exists
    if not searcher.cache.exists:
        print(f"  Building embedding cache (this may take a while)...")
        searcher.build_cache()
    print(f"  Searcher ready")

    # Step 1: Select ultimates
    ultimates = await select_ultimates(N_ULTIMATES)

    # Step 2: Generate cruxes
    cruxes_by_ultimate = await generate_all_cruxes(ultimates)

    # Step 3: Search for signals
    signals_by_ultimate = search_for_signals(cruxes_by_ultimate, searcher)

    # Step 4: Compute VOI
    scored_signals_by_ultimate = await compute_voi_for_signals(
        ultimates, signals_by_ultimate
    )

    # Step 5: MMR selection
    selected_by_ultimate = apply_mmr_selection(scored_signals_by_ultimate)

    # Step 6: Validation
    validations = await validate_with_price_correlation(ultimates, selected_by_ultimate)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_cruxes = sum(len(c) for c in cruxes_by_ultimate.values())
    total_signals = sum(len(s) for s in signals_by_ultimate.values())
    total_scored = sum(len(s) for s in scored_signals_by_ultimate.values())
    total_selected = sum(len(s) for s in selected_by_ultimate.values())

    # Calculate metrics
    all_vois = [
        s["voi"]
        for signals in scored_signals_by_ultimate.values()
        for s in signals
    ]

    above_threshold = sum(1 for v in all_vois if v >= VOI_THRESHOLD)
    pass_rate = above_threshold / len(all_vois) if all_vois else 0

    # Retrieval coverage
    cruxes_with_results = sum(
        1 for ultimate_id, cruxes in cruxes_by_ultimate.items()
        for crux in cruxes
        if any(s.get("crux_query") == crux for s in signals_by_ultimate.get(ultimate_id, []))
    )
    # Simplified: count cruxes that contributed to signals
    retrieval_coverage = len([
        1 for signals in signals_by_ultimate.values()
        if signals
    ]) / len(ultimates) if ultimates else 0

    print(f"\nPipeline Stats:")
    print(f"  Ultimates: {len(ultimates)}")
    print(f"  Cruxes generated: {total_cruxes}")
    print(f"  Signals found: {total_signals}")
    print(f"  Signals scored: {total_scored}")
    print(f"  Signals selected (MMR): {total_selected}")

    print(f"\nSuccess Criteria:")
    print(f"  Retrieval coverage: {retrieval_coverage:.1%} (target: >50%)")
    print(f"  VOI pass rate: {pass_rate:.1%} (target: >30%)")

    if validations:
        corrs = [v["observed_correlation"] for v in validations]
        high_corr = sum(1 for c in corrs if abs(c) > 0.2)
        print(f"  Info flow validation: {high_corr}/{len(validations)} pairs with |r|>0.2 (target: r>0.2)")

    # Top signals by VOI
    print("\n" + "-" * 70)
    print("TOP 10 SIGNALS BY VOI")
    print("-" * 70)

    all_signals = [
        {**s, "ultimate_id": uid}
        for uid, signals in scored_signals_by_ultimate.items()
        for s in signals
    ]

    top_signals = sorted(all_signals, key=lambda s: -s["voi"])[:10]
    ultimate_lookup = {u["id"]: u for u in ultimates}

    for i, signal in enumerate(top_signals):
        ultimate = ultimate_lookup.get(signal["ultimate_id"], {})
        print(f"\n{i+1}. VOI={signal['voi']:.3f}, ρ={signal['rho']:.2f}")
        print(f"   Ultimate: {ultimate.get('title', 'Unknown')[:50]}...")
        print(f"   Signal: {signal['text'][:60]}...")
        print(f"   Crux query: {signal.get('crux_query', 'N/A')[:50]}...")

    # Save results
    output = {
        "metadata": {
            "experiment": "crux_guided_search",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_ultimates": N_ULTIMATES,
                "n_cruxes_per_ultimate": N_CRUXES_PER_ULTIMATE,
                "search_top_k": SEARCH_TOP_K,
                "voi_threshold": VOI_THRESHOLD,
                "mmr_k": MMR_K,
                "mmr_lambda": MMR_LAMBDA,
            },
        },
        "statistics": {
            "n_ultimates": len(ultimates),
            "n_cruxes": total_cruxes,
            "n_signals_found": total_signals,
            "n_signals_scored": total_scored,
            "n_signals_selected": total_selected,
            "retrieval_coverage": retrieval_coverage,
            "voi_pass_rate": pass_rate,
            "mean_voi": float(np.mean(all_vois)) if all_vois else None,
            "max_voi": float(np.max(all_vois)) if all_vois else None,
        },
        "ultimates": ultimates,
        "cruxes_by_ultimate": cruxes_by_ultimate,
        "scored_signals_by_ultimate": scored_signals_by_ultimate,
        "selected_by_ultimate": selected_by_ultimate,
        "validations": validations,
    }

    output_path = RESULTS_DIR / f"crux_search_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
