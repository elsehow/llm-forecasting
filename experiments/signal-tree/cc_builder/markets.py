"""Phase 2: Market discovery via semantic search.

Discovers related prediction markets for a target question:
- Direct target search
- Entity-based search
- Competing outcome search
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import litellm
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class MarketSignal:
    """A market signal discovered via search."""

    market_id: str
    platform: str
    title: str
    similarity: float
    current_probability: float | None = None
    url: str | None = None
    search_type: str = "direct"  # "direct", "entity", "competing"


class MarketSemanticSearcher:
    """Semantic search over markets using sentence-transformer embeddings.

    Works with the markets table in forecastbench.db (market_data schema).
    """

    def __init__(
        self, db_path: str | Path = "forecastbench.db", cache_dir: Path | None = None
    ):
        self.db_path = Path(db_path)
        if cache_dir is None:
            cache_dir = self.db_path.parent / "embeddings" / "markets"
        self.cache_dir = Path(cache_dir)
        self._embeddings: np.ndarray | None = None
        self._index: dict[str, int] | None = None
        self._markets: list[dict] | None = None

    @property
    def cache_exists(self) -> bool:
        """Check if embedding cache exists."""
        return (self.cache_dir / "embeddings.npy").exists() and (
            self.cache_dir / "index.json"
        ).exists()

    def build_cache(self, batch_size: int = 100) -> int:
        """Build embedding cache from markets table.

        Returns the number of markets embedded.
        """
        from llm_forecasting.semantic_search.embeddings import embed_texts
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT id, platform, title, current_probability, url
            FROM markets
            WHERE title IS NOT NULL AND title != ''
        """
        )

        markets = []
        texts = []
        index = {}

        for row in cursor:
            market_id, platform, title, prob, url = row
            key = f"{platform}|{market_id}"
            index[key] = len(markets)
            markets.append(
                {
                    "id": market_id,
                    "platform": platform,
                    "title": title,
                    "current_probability": prob,
                    "url": url,
                }
            )
            texts.append(title)

        conn.close()

        if not texts:
            logger.warning("No markets found to embed")
            return 0

        logger.info(f"Embedding {len(texts)} markets...")
        embeddings = embed_texts(texts)

        # Save cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / "embeddings.npy", embeddings.astype(np.float32))
        with open(self.cache_dir / "index.json", "w") as f:
            json.dump(index, f)
        with open(self.cache_dir / "markets.json", "w") as f:
            json.dump(markets, f)

        self._embeddings = embeddings
        self._index = index
        self._markets = markets

        logger.info(f"Cache saved to {self.cache_dir}")
        return len(markets)

    def _load_cache(self) -> None:
        """Load embeddings and markets from cache."""
        if self._embeddings is None:
            self._embeddings = np.load(self.cache_dir / "embeddings.npy")
            with open(self.cache_dir / "index.json") as f:
                self._index = json.load(f)
            with open(self.cache_dir / "markets.json") as f:
                self._markets = json.load(f)

    def search(
        self,
        query: str,
        top_k: int = 20,
        platforms: list[str] | None = None,
        search_type: str = "direct",
    ) -> list[MarketSignal]:
        """Find markets most similar to query.

        Args:
            query: Search query text
            top_k: Number of results to return
            platforms: Filter to specific platforms (e.g., ["polymarket"])
            search_type: Type of search (for tagging results)

        Returns:
            List of MarketSignal sorted by similarity (descending)
        """
        if not self.cache_exists:
            raise ValueError("Cache not built. Call build_cache() first.")

        from llm_forecasting.semantic_search.embeddings import embed_single

        self._load_cache()

        query_emb = embed_single(query)

        # Compute similarities
        similarities = np.dot(self._embeddings, query_emb)

        # Get top indices
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break

            market = self._markets[idx]

            # Filter by platform if specified
            if platforms and market["platform"] not in platforms:
                continue

            results.append(
                MarketSignal(
                    market_id=market["id"],
                    platform=market["platform"],
                    title=market["title"],
                    similarity=float(similarities[idx]),
                    current_probability=market.get("current_probability"),
                    url=market.get("url"),
                    search_type=search_type,
                )
            )

        return results


# Singleton instance
_market_searcher: MarketSemanticSearcher | None = None


def get_market_searcher(db_path: str | Path = "forecastbench.db") -> MarketSemanticSearcher:
    """Get or create the market semantic searcher singleton."""
    global _market_searcher
    if _market_searcher is None or str(_market_searcher.db_path) != str(db_path):
        _market_searcher = MarketSemanticSearcher(db_path)
    return _market_searcher


class EntitiesResponse(BaseModel):
    """LLM response for entity extraction."""

    entities: list[str] = Field(description="Named entities relevant to the question")


class CompetingOutcomesResponse(BaseModel):
    """LLM response for competing outcomes."""

    outcomes: list[str] = Field(description="Competing outcomes that preclude target")


async def extract_entities(
    target: str,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> list[str]:
    """Extract named entities from a target question.

    Args:
        target: The target question
        model: LLM model to use

    Returns:
        List of entity names (people, films, companies, etc.)
    """
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Extract the key named entities from this forecasting question:

Question: {target}

List specific named entities (people, films, companies, events, etc.) that are central to this question.
Do NOT include generic terms like "election", "award", or "market".
Focus on proper nouns that could be searched for in prediction markets.""",
            }
        ],
        response_format=EntitiesResponse,
    )

    result = EntitiesResponse.model_validate_json(
        response.choices[0].message.content
    )
    return result.entities


async def generate_competing_outcomes(
    target: str,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> list[str]:
    """Generate competing outcomes that preclude the target.

    Args:
        target: The target question
        model: LLM model to use

    Returns:
        List of competing outcome descriptions
    """
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""For this forecasting question, identify competing outcomes that would preclude it:

Question: {target}

List outcomes that are mutually exclusive with the target. For example:
- If target is "Will X win Best Picture?", competitors are other films that could win
- If target is "Will X be elected?", competitors are other candidates
- If target is "Will company X acquire Y?", competitors are other potential acquirers

Focus on specific named alternatives, not generic "something else".""",
            }
        ],
        response_format=CompetingOutcomesResponse,
    )

    result = CompetingOutcomesResponse.model_validate_json(
        response.choices[0].message.content
    )
    return result.outcomes


def deduplicate_markets(markets: list[MarketSignal]) -> list[MarketSignal]:
    """Remove duplicate markets, keeping highest similarity."""
    seen: dict[str, MarketSignal] = {}
    for market in markets:
        key = f"{market.platform}|{market.market_id}"
        if key not in seen or market.similarity > seen[key].similarity:
            seen[key] = market
    return sorted(seen.values(), key=lambda m: m.similarity, reverse=True)


async def discover_markets(
    target: str,
    db_path: str | Path = "forecastbench.db",
    platforms: list[str] | None = None,
    top_k: int = 10,
) -> list[MarketSignal]:
    """Find semantically related markets for a target question.

    Uses three search strategies:
    1. Direct target search
    2. Entity-based search (extract entities, search for each)
    3. Competing outcome search (generate competitors, search for each)

    Args:
        target: The target question
        db_path: Path to market database
        platforms: Platforms to search (default: polymarket, metaculus)
        top_k: Number of results per search type

    Returns:
        Deduplicated list of MarketSignal sorted by similarity
    """
    platforms = platforms or ["polymarket", "metaculus"]

    searcher = get_market_searcher(db_path)

    # Build cache if needed
    if not searcher.cache_exists:
        logger.info("Building market embedding cache (one-time)...")
        count = searcher.build_cache()
        if count == 0:
            logger.warning("No markets in database")
            return []

    all_markets: list[MarketSignal] = []

    # 1. Direct target search
    logger.info(f"Searching for: {target[:50]}...")
    direct = searcher.search(target, top_k=top_k, platforms=platforms, search_type="direct")
    all_markets.extend(direct)

    # 2. Entity-based search
    try:
        entities = await extract_entities(target)
        logger.info(f"Found entities: {entities}")
        for entity in entities[:5]:  # Limit to top 5 entities
            entity_results = searcher.search(
                entity, top_k=top_k // 2, platforms=platforms, search_type="entity"
            )
            all_markets.extend(entity_results)
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")

    # 3. Competing outcome search
    try:
        competitors = await generate_competing_outcomes(target)
        logger.info(f"Found competitors: {competitors}")
        for competitor in competitors[:5]:  # Limit to top 5 competitors
            comp_results = searcher.search(
                competitor, top_k=top_k // 2, platforms=platforms, search_type="competing"
            )
            all_markets.extend(comp_results)
    except Exception as e:
        logger.warning(f"Competitor generation failed: {e}")

    # Deduplicate and sort
    return deduplicate_markets(all_markets)


async def refresh_market_data(
    platforms: list[str] | None = None,
    min_liquidity: float = 10000,
    limit: int = 1000,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    """Refresh market data from prediction market platforms.

    Fetches fresh market data and saves to the local database cache.
    Call this before tree building to ensure market matching has current data.

    Args:
        platforms: List of platforms to refresh (default: ["polymarket"])
        min_liquidity: Minimum liquidity filter (default: 10000)
        limit: Max markets to fetch per platform (default: 1000)
        db_path: Path to SQLite database (default: forecastbench.db)

    Returns:
        Dict mapping platform name to number of markets saved
    """
    from llm_forecasting.market_data import (
        MarketDataStorage,
        PolymarketData,
        MetaculusData,
    )

    platforms = platforms or ["polymarket"]
    db_path = db_path or "forecastbench.db"

    storage = MarketDataStorage(db_path)
    counts = {}

    try:
        for platform in platforms:
            logger.info(f"Refreshing {platform} markets...")

            if platform == "polymarket":
                provider = PolymarketData()
                markets = await provider.fetch_markets(
                    min_liquidity=min_liquidity,
                    limit=limit,
                )
            elif platform == "metaculus":
                provider = MetaculusData()
                markets = await provider.fetch_markets(limit=limit)
            else:
                logger.warning(f"Unknown platform: {platform}")
                continue

            await storage.save_markets(markets)
            counts[platform] = len(markets)
            logger.info(f"Saved {len(markets)} {platform} markets")

    finally:
        await storage.close()

    return counts


async def get_market_data_stats(db_path: str | Path | None = None) -> dict[str, Any]:
    """Get statistics about cached market data.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dict with stats including counts by platform and data freshness
    """
    from llm_forecasting.market_data import MarketDataStorage

    db_path = db_path or "forecastbench.db"
    storage = MarketDataStorage(db_path)

    try:
        # Get counts by platform
        markets = await storage.get_markets()
        by_platform: dict[str, int] = {}
        latest_update = None

        for m in markets:
            platform = m.platform
            by_platform[platform] = by_platform.get(platform, 0) + 1
            if m.created_at and (latest_update is None or m.created_at > latest_update):
                latest_update = m.created_at

        return {
            "total_markets": len(markets),
            "by_platform": by_platform,
            "latest_update": latest_update,
        }
    finally:
        await storage.close()


async def check_market_price(
    signal_text: str,
    db_path: str | Path | None = None,
    min_similarity: float = 0.4,
    min_confidence: float = 0.5,
    platforms: list[str] | None = None,
) -> dict[str, Any] | None:
    """Check prediction market price for a signal using semantic search.

    Args:
        signal_text: The signal question text
        db_path: Path to market data SQLite database
        min_similarity: Minimum embedding similarity to consider
        min_confidence: Minimum LLM match confidence to accept
        platforms: Platforms to search

    Returns:
        Dict with market_price, platform, url, match_confidence
        or None if no match found
    """
    from llm_forecasting.market_data.matcher import _llm_rerank
    from llm_forecasting.market_data.models import Market, MarketStatus

    db_path = db_path or "forecastbench.db"
    platforms = platforms or ["polymarket", "metaculus"]

    searcher = get_market_searcher(db_path)

    # Build cache if needed
    if not searcher.cache_exists:
        logger.info("Building market embedding cache (one-time)...")
        count = searcher.build_cache()
        if count == 0:
            logger.warning("No markets in database")
            return None

    # Semantic search
    results = searcher.search(signal_text, top_k=10, platforms=platforms)

    if not results:
        logger.debug(f"No semantic matches for: {signal_text[:50]}...")
        return None

    # Filter by minimum similarity
    candidates = [r for r in results if r.similarity >= min_similarity]
    if not candidates:
        logger.debug(f"No matches above similarity threshold {min_similarity}")
        return None

    # Filter to those with probability
    candidates_with_prob = [r for r in candidates if r.current_probability is not None]
    if not candidates_with_prob:
        logger.debug("No candidates with probability")
        return None

    # Create Market objects for LLM reranking
    candidate_markets = [
        Market(
            id=r.market_id,
            platform=r.platform,
            title=r.title,
            current_probability=r.current_probability,
            url=r.url,
            status=MarketStatus.OPEN,
            created_at=datetime.now(),
            fetched_at=datetime.now(),
        )
        for r in candidates_with_prob
    ]

    # LLM rerank to select best match
    if len(candidate_markets) > 1:
        best_idx, confidence, reasoning = await _llm_rerank(signal_text, candidate_markets)

        if best_idx is None:
            logger.debug(f"LLM found no match: {reasoning}")
            return None

        if best_idx >= len(candidate_markets):
            best_idx = 0

        best = candidate_markets[best_idx]
    else:
        best = candidate_markets[0]
        confidence = candidates_with_prob[0].similarity

    if confidence < min_confidence:
        logger.debug(
            f"Match below confidence threshold: {confidence:.0%} < {min_confidence:.0%}"
        )
        return None

    return {
        "market_price": best.current_probability,
        "platform": best.platform,
        "url": best.url,
        "match_confidence": confidence,
        "matched_question": best.title,
        "similarity": candidates_with_prob[0].similarity if candidates_with_prob else None,
    }
