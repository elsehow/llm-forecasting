"""Utility functions for CC-driven tree building."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalNode, SignalTree
from shared.registry import TreeRegistry
from shared.rollup import rollup_tree, analyze_tree

logger = logging.getLogger(__name__)


# ============================================================
# Semantic Market Search
# ============================================================

@dataclass
class MarketSearchResult:
    """A single market search result."""
    market_id: str
    platform: str
    title: str
    similarity: float
    current_probability: Optional[float] = None
    url: Optional[str] = None


class MarketSemanticSearcher:
    """
    Semantic search over markets using sentence-transformer embeddings.

    Works with the markets table in forecastbench.db (market_data schema).
    """

    def __init__(self, db_path: str | Path = "forecastbench.db", cache_dir: Optional[Path] = None):
        self.db_path = Path(db_path)
        if cache_dir is None:
            cache_dir = self.db_path.parent / "embeddings" / "markets"
        self.cache_dir = Path(cache_dir)
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[dict[str, int]] = None
        self._markets: Optional[list[dict]] = None

    @property
    def cache_exists(self) -> bool:
        """Check if embedding cache exists."""
        return (self.cache_dir / "embeddings.npy").exists() and \
               (self.cache_dir / "index.json").exists()

    def build_cache(self, batch_size: int = 100) -> int:
        """
        Build embedding cache from markets table.

        Returns the number of markets embedded.
        """
        from llm_forecasting.semantic_search.embeddings import embed_texts
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT id, platform, title, current_probability, url
            FROM markets
            WHERE title IS NOT NULL AND title != ''
        """)

        markets = []
        texts = []
        index = {}

        for row in cursor:
            market_id, platform, title, prob, url = row
            key = f"{platform}|{market_id}"
            index[key] = len(markets)
            markets.append({
                "id": market_id,
                "platform": platform,
                "title": title,
                "current_probability": prob,
                "url": url,
            })
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

    def search(self, query: str, top_k: int = 20, platforms: list[str] | None = None) -> list[MarketSearchResult]:
        """
        Find markets most similar to query.

        Args:
            query: Search query text
            top_k: Number of results to return
            platforms: Filter to specific platforms (e.g., ["polymarket"])

        Returns:
            List of MarketSearchResult sorted by similarity (descending)
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

            results.append(MarketSearchResult(
                market_id=market["id"],
                platform=market["platform"],
                title=market["title"],
                similarity=float(similarities[idx]),
                current_probability=market.get("current_probability"),
                url=market.get("url"),
            ))

        return results


# Singleton instance
_market_searcher: Optional[MarketSemanticSearcher] = None


def get_market_searcher(db_path: str | Path = "forecastbench.db") -> MarketSemanticSearcher:
    """Get or create the market semantic searcher singleton."""
    global _market_searcher
    if _market_searcher is None or str(_market_searcher.db_path) != str(db_path):
        _market_searcher = MarketSemanticSearcher(db_path)
    return _market_searcher


async def check_market_price_semantic(
    signal_text: str,
    db_path: str | Path | None = None,
    min_similarity: float = 0.4,
    min_confidence: float = 0.5,
    platforms: list[str] | None = None,
) -> dict[str, Any] | None:
    """Check prediction market price using semantic search.

    Uses sentence-transformer embeddings for much better matching than keywords.

    Args:
        signal_text: The signal question text
        db_path: Path to market data SQLite database
        min_similarity: Minimum embedding similarity to consider (default: 0.4)
        min_confidence: Minimum LLM match confidence to accept (default: 0.5)
        platforms: Platforms to search (default: polymarket, metaculus)

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
            logger.warning("No markets in database - run refresh_market_data() first")
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
        confidence = candidates_with_prob[0].similarity  # Use similarity as confidence

    if confidence < min_confidence:
        logger.debug(f"Match below confidence threshold: {confidence:.0%} < {min_confidence:.0%}")
        return None

    return {
        "market_price": best.current_probability,
        "platform": best.platform,
        "url": best.url,
        "match_confidence": confidence,
        "matched_question": best.title,
        "similarity": candidates_with_prob[0].similarity if candidates_with_prob else None,
    }


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

    Example:
        >>> counts = await refresh_market_data()
        >>> print(f"Refreshed {counts['polymarket']} Polymarket markets")
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
        by_platform = {}
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


def create_signal(
    text: str,
    parent_id: str,
    resolution_date: date | str,
    base_rate: float,
    rho: float = 0.0,
    rho_reasoning: str = "",
    relationship_type: str = "correlation",
    depth: int = 1,
    is_leaf: bool = True,
) -> SignalNode:
    """Create a signal node with auto-generated ID.

    Args:
        text: The signal question text
        parent_id: ID of parent node
        resolution_date: When signal resolves (date or YYYY-MM-DD string)
        base_rate: P(signal=YES)
        rho: Correlation coefficient (-1 to +1). Only used for correlation type.
        rho_reasoning: Explanation of the relationship
        relationship_type: One of:
            - "correlation" (default): Statistical relationship, uses rho
            - "necessity": Signal=NO implies Parent=0 (e.g., must be nominated to win)
            - "sufficiency": Signal=YES implies Parent=1 (rare)
        depth: Depth in tree (1 = direct child of target)
        is_leaf: Whether this is a leaf node

    Returns:
        SignalNode with auto-generated ID
    """
    if isinstance(resolution_date, str):
        resolution_date = date.fromisoformat(resolution_date)

    return SignalNode(
        id=f"sig_{uuid.uuid4().hex[:8]}",
        text=text,
        resolution_date=resolution_date,
        base_rate=base_rate,
        probability_source="manual",
        parent_id=parent_id,
        relationship_type=relationship_type,
        rho=rho if relationship_type == "correlation" else None,
        rho_reasoning=rho_reasoning,
        depth=depth,
        is_leaf=is_leaf,
    )


async def create_signal_with_market(
    text: str,
    parent_id: str,
    resolution_date: date | str,
    rho: float = 0.0,
    rho_reasoning: str = "",
    relationship_type: str = "correlation",
    depth: int = 1,
    is_leaf: bool = True,
    fetch_market: bool = True,
    db_path: str | Path | None = None,
) -> SignalNode:
    """Create a signal, optionally fetching market price.

    This is an async version of create_signal that automatically checks
    prediction markets for matching questions and populates market fields.

    Args:
        text: The signal question text
        parent_id: ID of parent node
        resolution_date: When signal resolves (date or YYYY-MM-DD string)
        rho: Correlation coefficient (-1 to +1). Only used for correlation type.
        rho_reasoning: Explanation of the relationship
        relationship_type: "correlation", "necessity", or "sufficiency"
        depth: Depth in tree (1 = direct child of target)
        is_leaf: Whether this is a leaf node
        fetch_market: Whether to check prediction markets (default True)
        db_path: Path to market data SQLite database

    Returns:
        SignalNode with market fields populated if match found

    Example:
        >>> signal = await create_signal_with_market(
        ...     "Will OBAA win Best Picture at 2026 Oscars?",
        ...     parent_id="target",
        ...     resolution_date="2026-03-03",
        ...     rho=0.9,
        ...     rho_reasoning="Same entity, same outcome",
        ... )
        >>> if signal.market_price:
        ...     print(f"Market: {signal.market_platform} @ {signal.market_price:.0%}")
    """
    if isinstance(resolution_date, str):
        resolution_date = date.fromisoformat(resolution_date)

    # Create base signal
    signal = SignalNode(
        id=f"sig_{uuid.uuid4().hex[:8]}",
        text=text,
        resolution_date=resolution_date,
        probability_source="manual",
        parent_id=parent_id,
        relationship_type=relationship_type,
        rho=rho if relationship_type == "correlation" else None,
        rho_reasoning=rho_reasoning,
        depth=depth,
        is_leaf=is_leaf,
    )

    # Fetch market price if requested
    if fetch_market:
        market_data = await check_market_price(text, db_path)
        if market_data:
            signal.market_price = market_data["market_price"]
            signal.market_url = market_data["url"]
            signal.market_platform = market_data["platform"]
            signal.market_match_confidence = market_data["match_confidence"]
            signal.probability_source = "market"
            signal.base_rate = market_data["market_price"]  # Set base_rate too
            logger.info(
                f"Found market match: {market_data['platform']} @ {market_data['market_price']:.0%}"
            )
        else:
            logger.info(f"No market match found for: {text[:50]}...")

    return signal


def create_ref_signal(
    ref: str,
    parent_id: str,
    rho: float,
    rho_reasoning: str,
    registry: TreeRegistry,
    depth: int = 1,
) -> SignalNode:
    """Create a signal that references another tree's node.

    Reference signals pull their base_rate from the referenced node at
    rollup time. This enables cross-tree dependencies.

    Args:
        ref: Reference string like 'house_2026:sig_abc' or 'house_2026' (for root)
        parent_id: ID of parent node in THIS tree
        rho: Correlation coefficient (-1 to +1) with parent
        rho_reasoning: Explanation of the correlation
        registry: TreeRegistry to resolve the reference
        depth: Depth in tree (1 = direct child of target)

    Returns:
        SignalNode with ref field set

    Raises:
        ValueError: If the reference cannot be resolved
    """
    referenced_node = registry.get_node(ref)
    if not referenced_node:
        raise ValueError(f"Reference not found: {ref}")

    return SignalNode(
        id=f"ref_{uuid.uuid4().hex[:8]}",
        text=referenced_node.text,  # Copy text for display
        ref=ref,
        parent_id=parent_id,
        rho=rho,
        rho_reasoning=rho_reasoning,
        depth=depth,
        is_leaf=True,  # Refs are always leaves (not decomposed further)
        # base_rate intentionally None - pulled from ref at rollup time
    )


def create_target(
    question: str,
    target_id: str = "target",
) -> SignalNode:
    """Create a target (root) node."""
    return SignalNode(
        id=target_id,
        text=question,
        depth=0,
        is_leaf=False,
    )


def build_tree(
    target: SignalNode,
    signals: list[SignalNode],
) -> SignalTree:
    """Build a tree from target and flat signal list.

    Automatically:
    - Links children to parents
    - Computes max_depth and leaf_count
    """
    # Build parent -> children mapping
    children_map: dict[str, list[SignalNode]] = {}
    for signal in signals:
        if signal.parent_id not in children_map:
            children_map[signal.parent_id] = []
        children_map[signal.parent_id].append(signal)

    # Attach children to nodes
    target.children = children_map.get(target.id, [])
    for signal in signals:
        signal.children = children_map.get(signal.id, [])
        signal.is_leaf = len(signal.children) == 0

    # Compute metadata
    max_depth = max((s.depth for s in signals), default=0)
    leaf_count = sum(1 for s in signals if s.is_leaf)

    return SignalTree(
        target=target,
        signals=signals,
        max_depth=max_depth,
        leaf_count=leaf_count,
    )


def save_tree(
    tree: SignalTree,
    target_slug: str,
    suffix: str = "cc",
    results_dir: Path | None = None,
    validate_market: bool = True,
    db_path: str | Path | None = None,
) -> Path:
    """Save tree to JSON with consistent naming and optional market validation.

    When validate_market=True and computed_probability exists in tree metadata,
    compares against prediction market prices to catch discrepancies.

    Args:
        tree: The signal tree to save
        target_slug: Slug for the target (used for directory name)
        suffix: Suffix for the filename (default "cc")
        results_dir: Directory to save results (default: ../results)
        validate_market: Whether to validate against prediction markets (default True)
        db_path: Path to market data SQLite database (default: forecastbench.db)

    Returns:
        Path to the saved JSON file
    """
    results_dir = results_dir or Path(__file__).parent.parent / "results"
    output_dir = results_dir / target_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = date.today().strftime("%Y%m%d")
    output_file = output_dir / f"tree_{timestamp}_{suffix}.json"

    # Serialize with analysis
    tree_dict = tree.model_dump(mode="json")

    # Market validation
    if validate_market:
        computed_prob = tree_dict.get("computed_probability")
        if computed_prob is not None:
            try:
                # Handle both sync and async contexts
                try:
                    loop = asyncio.get_running_loop()
                    # Already in async context - skip validation (would cause nested loop)
                    validation = None
                    logger.debug("Skipping market validation in async context")
                except RuntimeError:
                    # No running loop - safe to use asyncio.run
                    validation = asyncio.run(_validate_market_price(
                        question=tree.target.text,
                        computed_probability=computed_prob,
                        db_path=db_path,
                    ))
                if validation:
                    tree_dict["market_validation"] = validation
                    # Print validation result for CC visibility
                    gap = validation["gap_pp"]
                    status = validation["status"]
                    market_price = validation["market_price"]
                    print(f"\n=== Market Validation ===")
                    print(f"Market: {validation['matched_question']}")
                    print(f"Market price: {market_price:.1%}")
                    print(f"Computed: {computed_prob:.1%}")
                    print(f"Gap: {gap:+.1f}pp")
                    print(f"Status: {status}")
                    if validation.get("url"):
                        print(f"URL: {validation['url']}")
                    print("=" * 25 + "\n")
            except Exception as e:
                logger.warning(f"Market validation failed: {e}")

    with open(output_file, "w") as f:
        json.dump(tree_dict, f, indent=2, default=str)

    return output_file


async def _validate_market_price(
    question: str,
    computed_probability: float,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Internal: validate computed probability against market price.

    Returns validation dict or None if no match found.
    """
    from llm_forecasting.market_data import compute_validation_status

    db_path = db_path or "forecastbench.db"

    # Use semantic search (more accurate)
    result = await check_market_price_semantic(
        question,
        db_path=db_path,
        min_confidence=0.5,
    )

    if not result:
        return None

    market_price = result["market_price"]
    gap_pp = (computed_probability - market_price) * 100
    status = compute_validation_status(gap_pp)

    return {
        "platform": result["platform"],
        "matched_question": result["matched_question"],
        "market_price": market_price,
        "gap_pp": round(gap_pp, 1),
        "status": status,
        "url": result["url"],
        "match_confidence": result["match_confidence"],
    }


async def check_market_price(
    signal_text: str,
    db_path: str | Path | None = None,
    min_confidence: float = 0.5,
    use_semantic: bool = True,
) -> dict[str, Any] | None:
    """Check prediction market price for a signal.

    Use this during tree building to verify base rates against market prices.

    IMPORTANT: Run `refresh_market_data()` or `scripts/refresh_markets.py`
    before tree building to ensure market data is current.

    Args:
        signal_text: The signal question text
        db_path: Path to market data SQLite database
        min_confidence: Minimum match confidence to accept (default: 0.5)
        use_semantic: Use semantic search (default True, more accurate)

    Returns:
        Dict with market_price, platform, url, match_confidence
        or None if no match found

    Example:
        >>> result = await check_market_price("Will One Battle win Best Picture at the 2026 Oscars?")
        >>> if result:
        ...     print(f"Market: {result['platform']} @ {result['market_price']:.0%}")
    """
    db_path = db_path or "forecastbench.db"

    # Try semantic search first (more accurate)
    if use_semantic:
        result = await check_market_price_semantic(
            signal_text,
            db_path=db_path,
            min_confidence=min_confidence,
        )
        if result:
            return result
        logger.debug("Semantic search found no match, trying keyword search...")

    # Fallback to keyword search
    from llm_forecasting.market_data import (
        MarketDataStorage,
        find_matching_market,
        extract_keywords,
    )

    storage = MarketDataStorage(db_path)
    try:
        match = await find_matching_market(
            question=signal_text,
            storage=storage,
            platforms=["polymarket", "metaculus"],
            use_llm_rerank=True,
        )

        if not match:
            keywords = extract_keywords(signal_text)
            logger.debug(f"No market match for '{signal_text[:50]}...' (keywords: {keywords[:5]})")
            return None

        if match.market.current_probability is None:
            logger.debug(f"Market match has no probability: {match.market.title}")
            return None

        if match.match_confidence < min_confidence:
            logger.debug(
                f"Market match below confidence threshold: {match.market.title} "
                f"(confidence: {match.match_confidence:.0%} < {min_confidence:.0%})"
            )
            return None

        return {
            "market_price": match.market.current_probability,
            "platform": match.market.platform,
            "url": match.market.url,
            "match_confidence": match.match_confidence,
            "matched_question": match.market.title,
        }
    finally:
        await storage.close()


def print_tree_summary(
    tree: SignalTree,
    target_prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> dict:
    """Analyze and print tree summary.

    Args:
        tree: The signal tree to analyze
        target_prior: Prior probability for target
        registry: Optional TreeRegistry for resolving cross-tree refs
    """
    analysis = analyze_tree(tree, target_prior, registry)

    print(f"\nTarget: {tree.target.text}")
    print(f"Total signals: {len(tree.signals)}")
    print(f"Leaves: {tree.leaf_count}")
    print(f"Max depth: {tree.max_depth}")

    # Count refs
    ref_count = sum(1 for s in tree.signals if s.ref)
    if ref_count:
        print(f"Cross-tree refs: {ref_count}")

    print(f"\nComputed probability: {analysis['computed_probability']:.1%}")
    print(f"(Prior was {target_prior:.0%})")

    print(f"\nEvidence breakdown:")
    print(f"  Positive: {analysis['evidence_breakdown']['positive']:+.4f}")
    print(f"  Negative: {analysis['evidence_breakdown']['negative']:+.4f}")
    print(f"  Net: {analysis['evidence_breakdown']['net']:+.4f}")

    print(f"\nTop contributors:")
    for c in analysis["top_contributors"][:5]:
        sign = "+" if c["evidence"] >= 0 else ""
        ref_marker = f" [ref:{c['ref']}]" if c.get("ref") else ""
        print(f"  {sign}{c['evidence']:.4f} | rho={c['rho']:+.2f} | p={c['base_rate']:.0%} | {c['text'][:40]}...{ref_marker}")

    return analysis
