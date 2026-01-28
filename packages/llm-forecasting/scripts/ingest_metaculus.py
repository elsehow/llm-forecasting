#!/usr/bin/env python3
"""Ingest Metaculus questions into the database.

Fetches all resolved binary questions with pagination and saves them
with topic and tournament categories extracted from the projects field.

Usage:
    uv run python packages/llm-forecasting/scripts/ingest_metaculus.py
    uv run python packages/llm-forecasting/scripts/ingest_metaculus.py --resolved-only
    uv run python packages/llm-forecasting/scripts/ingest_metaculus.py --limit 1000
"""

import argparse
import asyncio
import logging
from pathlib import Path

import httpx

from llm_forecasting.market_data.metaculus import MetaculusData
from llm_forecasting.market_data.models import Market
from llm_forecasting.market_data.storage import MarketDataStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.metaculus.com/api"


async def fetch_all_questions(
    *,
    resolved_only: bool = False,
    min_forecasters: int = 0,
    limit: int | None = None,
) -> list[Market]:
    """Fetch all binary questions from Metaculus with pagination.

    Args:
        resolved_only: Only fetch resolved questions
        min_forecasters: Minimum number of forecasters (default 0 for all)
        limit: Maximum total questions to fetch (None for all)

    Returns:
        List of Market objects
    """
    metaculus = MetaculusData()
    all_markets: list[Market] = []
    offset = 0
    page_size = 100

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {
                "statuses": "resolved" if resolved_only else "open,resolved",
                "forecast_type": "binary",
                "limit": page_size,
                "offset": offset,
            }

            try:
                response = await client.get(f"{BASE_URL}/posts/", params=params)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                # Parse each result using the MetaculusData parser
                for raw in results:
                    if min_forecasters > 0:
                        if raw.get("nr_forecasters", 0) < min_forecasters:
                            continue

                    market = metaculus._parse_market(raw)
                    if market:
                        all_markets.append(market)

                logger.info(
                    f"Fetched {len(results)} at offset {offset}, "
                    f"total parsed: {len(all_markets)}"
                )

                if not data.get("next"):
                    break

                offset += page_size

                if limit and len(all_markets) >= limit:
                    all_markets = all_markets[:limit]
                    break

            except httpx.HTTPError as e:
                logger.error(f"HTTP error at offset {offset}: {e}")
                break

    return all_markets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Metaculus questions")
    parser.add_argument(
        "--resolved-only",
        action="store_true",
        help="Only fetch resolved questions",
    )
    parser.add_argument(
        "--min-forecasters",
        type=int,
        default=0,
        help="Minimum number of forecasters (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum questions to fetch (default: all)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/forecastbench.db",
        help="Database path (default: data/forecastbench.db)",
    )
    args = parser.parse_args()

    logger.info(
        f"Fetching Metaculus questions "
        f"(resolved_only={args.resolved_only}, "
        f"min_forecasters={args.min_forecasters}, "
        f"limit={args.limit})"
    )

    markets = await fetch_all_questions(
        resolved_only=args.resolved_only,
        min_forecasters=args.min_forecasters,
        limit=args.limit,
    )

    logger.info(f"Fetched {len(markets)} markets total")

    # Count by status and categories
    resolved = sum(1 for m in markets if m.resolved_value is not None)
    with_topics = sum(1 for m in markets if m.topic_categories)
    with_tournaments = sum(1 for m in markets if m.tournament_categories)

    logger.info(f"  Resolved: {resolved}")
    logger.info(f"  With topic categories: {with_topics}")
    logger.info(f"  With tournament categories: {with_tournaments}")

    # Save to database
    db_path = Path(args.db)
    storage = MarketDataStorage(db_path)

    logger.info(f"Saving to {db_path}...")
    await storage.save_markets(markets)
    await storage.close()

    logger.info("Done!")

    # Show category breakdown
    from collections import Counter

    topic_counts: Counter[str] = Counter()
    for m in markets:
        if m.topic_categories:
            for cat in m.topic_categories:
                topic_counts[cat] += 1

    logger.info("\nTop topic categories:")
    for cat, count in topic_counts.most_common(15):
        logger.info(f"  {count:4d}  {cat}")


if __name__ == "__main__":
    asyncio.run(main())
