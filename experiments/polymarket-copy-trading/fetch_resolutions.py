"""Fetch market resolution data for backtesting."""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import select, distinct, update

from llm_forecasting.market_data.polymarket import PolymarketData
from llm_forecasting.market_data.storage import MarketDataStorage, TraderActivityRow, MarketRow
from llm_forecasting.market_data.models import MarketStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"

MAX_CONCURRENT = 20  # Concurrent API requests


async def get_unique_market_ids(storage: MarketDataStorage) -> set[str]:
    """Get all unique market (condition) IDs from trader activity."""
    async with await storage._get_session() as session:
        result = await session.execute(
            select(distinct(TraderActivityRow.condition_id)).where(
                TraderActivityRow.condition_id.isnot(None)
            )
        )
        return {row[0] for row in result.fetchall()}


async def fetch_and_save_market(
    provider: PolymarketData,
    storage: MarketDataStorage,
    condition_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, float | None]:
    """Fetch market data and return (condition_id, status, resolved_value)."""
    async with semaphore:
        try:
            market = await asyncio.wait_for(
                provider.fetch_market(condition_id),
                timeout=30.0
            )

            if not market:
                return (condition_id, "not_found", None)

            # Save to storage
            await storage.save_markets([market])

            status = market.status.value
            resolved_value = market.resolved_value

            return (condition_id, status, resolved_value)

        except asyncio.TimeoutError:
            return (condition_id, "timeout", None)
        except Exception as e:
            return (condition_id, f"error:{str(e)[:30]}", None)


async def main():
    """Fetch market resolution data."""
    provider = PolymarketData()
    storage = MarketDataStorage(DB_PATH)

    try:
        # Get unique market IDs
        logger.info("Getting unique market IDs from trader activity...")
        market_ids = await get_unique_market_ids(storage)
        logger.info(f"Found {len(market_ids)} unique markets")

        # Check how many we already have
        async with await storage._get_session() as session:
            result = await session.execute(
                select(MarketRow.id).where(MarketRow.platform == "polymarket")
            )
            existing = {row[0] for row in result.fetchall()}

        to_fetch = market_ids - existing
        logger.info(f"Already have {len(existing)}, need to fetch {len(to_fetch)}")

        if not to_fetch:
            logger.info("All markets already fetched!")
            return

        # Fetch in parallel
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        market_list = list(to_fetch)

        resolved_count = 0
        open_count = 0
        error_count = 0

        # Process in batches
        batch_size = 500
        for batch_start in range(0, len(market_list), batch_size):
            batch_end = min(batch_start + batch_size, len(market_list))
            batch = market_list[batch_start:batch_end]

            tasks = [
                fetch_and_save_market(provider, storage, mid, semaphore)
                for mid in batch
            ]

            results = await asyncio.gather(*tasks)

            for cid, status, resolved_value in results:
                if status == "resolved":
                    resolved_count += 1
                elif status == "open":
                    open_count += 1
                else:
                    error_count += 1

            logger.info(
                f"Progress: {batch_end}/{len(market_list)} "
                f"(resolved={resolved_count}, open={open_count}, errors={error_count})"
            )

        logger.info(f"Done! Resolved: {resolved_count}, Open: {open_count}, Errors: {error_count}")

        # Print resolution summary
        async with await storage._get_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(MarketRow.status, func.count())
                .where(MarketRow.platform == "polymarket")
                .group_by(MarketRow.status)
            )
            logger.info("Resolution status summary:")
            for status, count in result.fetchall():
                logger.info(f"  {status}: {count}")

    finally:
        await provider.close()
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
