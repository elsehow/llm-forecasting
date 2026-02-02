"""Fetch price histories in parallel.

Resume script - uses already collected trader data to fetch price histories.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_forecasting.market_data.polymarket import PolymarketData
from llm_forecasting.market_data.storage import MarketDataStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"

MAX_CONCURRENT = 10  # Max concurrent price history fetches


async def get_all_market_ids(storage: MarketDataStorage) -> set[str]:
    """Get all market IDs from stored trader activities."""
    from sqlalchemy import select, distinct
    from llm_forecasting.market_data.storage import TraderActivityRow

    async with await storage._get_session() as session:
        result = await session.execute(
            select(distinct(TraderActivityRow.condition_id)).where(
                TraderActivityRow.condition_id.isnot(None)
            )
        )
        return {row[0] for row in result.fetchall()}


async def fetch_single_market_price_history(
    provider: PolymarketData,
    storage: MarketDataStorage,
    market_id: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """Fetch price history for a single market. Returns (market_id, status)."""
    async with semaphore:
        try:
            # Check if we already have price history
            if await storage.has_price_history("polymarket", market_id):
                return (market_id, "skipped")

            # Fetch market to get token ID (with timeout)
            try:
                market = await asyncio.wait_for(
                    provider.fetch_market(market_id),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                return (market_id, "timeout_market")

            if not market or not market.clob_token_ids:
                return (market_id, "no_token")

            # Fetch price history (use 1h interval for reasonable granularity)
            # Go back 1 year for comprehensive history
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=365)

            try:
                prices = await asyncio.wait_for(
                    provider.fetch_price_history_by_token(
                        market.clob_token_ids[0],
                        start=start,
                        end=end,
                        interval="1h",
                    ),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                return (market_id, "timeout_prices")

            if prices:
                await storage.save_price_history(market_id, "polymarket", prices)
                return (market_id, "fetched")
            return (market_id, "empty")

        except Exception as e:
            # Only log first few errors to avoid spam
            return (market_id, f"error:{str(e)[:50]}")


async def main():
    """Fetch price histories in parallel."""
    provider = PolymarketData()
    storage = MarketDataStorage(DB_PATH)

    try:
        # Get all market IDs from stored trader data
        logger.info("Getting market IDs from trader activity...")
        market_ids = await get_all_market_ids(storage)
        logger.info(f"Found {len(market_ids)} unique markets")

        # Check how many we already have
        existing = 0
        for market_id in list(market_ids)[:100]:  # Sample check
            if await storage.has_price_history("polymarket", market_id):
                existing += 1
        logger.info(f"Sample check: ~{existing}% already have price history")

        # Fetch remaining
        logger.info(f"Fetching price history (parallel with {MAX_CONCURRENT} concurrent)...")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        market_list = list(market_ids)

        fetched = 0
        skipped = 0
        errors = 0
        error_sample = []

        # Process in batches for progress reporting
        batch_size = 500
        for batch_start in range(0, len(market_list), batch_size):
            batch_end = min(batch_start + batch_size, len(market_list))
            batch = market_list[batch_start:batch_end]

            # Create tasks for this batch
            tasks = [
                fetch_single_market_price_history(provider, storage, market_id, semaphore)
                for market_id in batch
            ]

            # Run batch concurrently
            results = await asyncio.gather(*tasks)

            # Count results
            for market_id, status in results:
                if status == "fetched":
                    fetched += 1
                elif status == "skipped" or status == "empty" or status == "no_token":
                    skipped += 1
                else:
                    errors += 1
                    if len(error_sample) < 5:
                        error_sample.append(f"{market_id[:10]}: {status}")

            logger.info(f"Progress: {batch_end}/{len(market_list)} markets (fetched={fetched}, skipped={skipped}, errors={errors})")

        logger.info(f"Price history complete: {fetched} fetched, {skipped} skipped, {errors} errors")
        if error_sample:
            logger.info(f"Sample errors: {error_sample}")

    finally:
        await provider.close()
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
