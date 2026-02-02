"""Fetch data for Polymarket copy trading experiment.

Phase 1: Data collection
1. Fetch top 500 traders (ALL time) from leaderboard
2. For each trader, fetch complete trade history
3. Fetch price history for all markets they traded
4. Store in SQLite via storage layer
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

# Rate limiting: 200 requests/10 seconds = 20 req/sec
# Be conservative but parallel: 10 concurrent requests with throttling
RATE_LIMIT_DELAY = 0.1  # 100ms between requests
MAX_CONCURRENT = 10  # Max concurrent price history fetches


async def fetch_leaderboard(
    provider: PolymarketData,
    storage: MarketDataStorage,
    n_traders: int = 500,
) -> list[str]:
    """Fetch top N traders from leaderboard.

    Returns list of user addresses.
    """
    logger.info(f"Fetching top {n_traders} traders from leaderboard...")

    entries = await provider.fetch_leaderboard(
        time_period="ALL",
        category="OVERALL",
        order_by="PNL",
        limit=n_traders,
    )

    # Save snapshot
    await storage.save_leaderboard_snapshot(entries)

    user_addresses = [e.user_address for e in entries]
    logger.info(f"Fetched {len(entries)} leaderboard entries")

    # Summary stats
    if entries:
        total_pnl = sum(e.pnl for e in entries)
        total_volume = sum(e.volume for e in entries)
        logger.info(f"Total PNL: ${total_pnl:,.2f}, Total Volume: ${total_volume:,.2f}")

    return user_addresses


async def fetch_trader_histories(
    provider: PolymarketData,
    storage: MarketDataStorage,
    user_addresses: list[str],
) -> set[str]:
    """Fetch complete trade history for all traders.

    Returns set of all market condition IDs traded.
    """
    all_market_ids = set()
    total_trades = 0

    for i, user_address in enumerate(user_addresses):
        logger.info(f"Fetching trades for trader {i+1}/{len(user_addresses)}: {user_address[:10]}...")

        try:
            # Fetch all activity (trades, splits, merges, redeems)
            activities = await provider.fetch_user_activity(
                user_address,
                limit=10000,  # Get complete history
            )

            if activities:
                await storage.save_trader_activity(activities)
                trades = [a for a in activities if a.activity_type == "TRADE"]
                total_trades += len(trades)

                # Collect market IDs
                market_ids = {a.condition_id for a in activities if a.condition_id}
                all_market_ids.update(market_ids)

                logger.info(f"  {len(trades)} trades across {len(market_ids)} markets")

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.warning(f"  Error fetching trader {user_address[:10]}: {e}")
            continue

    logger.info(f"Total: {total_trades} trades across {len(all_market_ids)} unique markets")
    return all_market_ids


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

            # Fetch market to get token ID
            market = await provider.fetch_market(market_id)
            if not market or not market.clob_token_ids:
                return (market_id, "skipped")

            # Fetch price history (use 1h interval for reasonable granularity)
            # Go back 1 year for comprehensive history
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=365)

            prices = await provider.fetch_price_history_by_token(
                market.clob_token_ids[0],
                start=start,
                end=end,
                interval="1h",
            )

            if prices:
                await storage.save_price_history(market_id, "polymarket", prices)
                return (market_id, "fetched")
            return (market_id, "empty")

        except Exception as e:
            logger.warning(f"Error fetching price history for {market_id}: {e}")
            return (market_id, "error")


async def fetch_market_price_histories(
    provider: PolymarketData,
    storage: MarketDataStorage,
    market_ids: set[str],
) -> None:
    """Fetch price history for all markets traded by top traders (parallel)."""
    logger.info(f"Fetching price history for {len(market_ids)} markets (parallel with {MAX_CONCURRENT} concurrent)...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    market_list = list(market_ids)

    fetched = 0
    skipped = 0
    errors = 0

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
            elif status == "skipped" or status == "empty":
                skipped += 1
            else:
                errors += 1

        logger.info(f"Progress: {batch_end}/{len(market_list)} markets (fetched={fetched}, skipped={skipped}, errors={errors})")

    logger.info(f"Price history complete: {fetched} fetched, {skipped} skipped, {errors} errors")


async def main():
    """Run full data collection pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    provider = PolymarketData()
    storage = MarketDataStorage(DB_PATH)

    try:
        # Phase 1: Fetch leaderboard
        logger.info("=" * 60)
        logger.info("PHASE 1: Fetching leaderboard")
        logger.info("=" * 60)
        user_addresses = await fetch_leaderboard(provider, storage, n_traders=500)

        # Phase 2: Fetch trader histories
        logger.info("=" * 60)
        logger.info("PHASE 2: Fetching trader histories")
        logger.info("=" * 60)
        market_ids = await fetch_trader_histories(provider, storage, user_addresses)

        # Phase 3: Fetch price histories
        logger.info("=" * 60)
        logger.info("PHASE 3: Fetching market price histories")
        logger.info("=" * 60)
        await fetch_market_price_histories(provider, storage, market_ids)

        # Summary
        logger.info("=" * 60)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Database: {DB_PATH}")

        # Print some stats
        tracked_traders = await storage.get_tracked_traders()
        logger.info(f"Tracked traders: {len(tracked_traders)}")

    finally:
        await provider.close()
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
