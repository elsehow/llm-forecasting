"""Fetch market resolution data using slugs."""

import asyncio
import logging
from pathlib import Path
import httpx
from sqlalchemy import select, distinct, update

from llm_forecasting.market_data.storage import MarketDataStorage, TraderActivityRow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"
GAMMA_API_URL = "https://gamma-api.polymarket.com"

MAX_CONCURRENT = 20


async def get_unique_slugs(storage: MarketDataStorage) -> dict[str, str]:
    """Get unique market slugs with their condition IDs."""
    async with await storage._get_session() as session:
        result = await session.execute(
            select(
                TraderActivityRow.market_slug,
                TraderActivityRow.condition_id
            )
            .where(TraderActivityRow.market_slug.isnot(None))
            .distinct(TraderActivityRow.market_slug)
        )
        # Return dict: slug -> condition_id
        return {row.market_slug: row.condition_id for row in result.fetchall() if row.market_slug}


async def fetch_market_by_slug(
    client: httpx.AsyncClient,
    slug: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None, list | None]:
    """Fetch market data by slug. Returns (slug, status, outcome_prices list)."""
    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.get(f"{GAMMA_API_URL}/markets", params={"slug": slug}),
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return (slug, None, None)

            market = data[0] if isinstance(data, list) else data

            status = market.get("umaResolutionStatus")
            outcome_prices = None

            if status == "resolved":
                try:
                    prices = market.get("outcomePrices", [])
                    if isinstance(prices, str):
                        import json
                        prices = json.loads(prices)
                    # Return all outcome prices as floats
                    outcome_prices = [float(p) for p in prices]
                except (ValueError, TypeError):
                    pass

            return (slug, status, outcome_prices)

        except asyncio.TimeoutError:
            return (slug, "timeout", None)
        except Exception as e:
            return (slug, f"error", None)


async def main():
    """Fetch resolution data by slug and create a mapping file."""
    storage = MarketDataStorage(DB_PATH)

    try:
        logger.info("Getting unique market slugs from trader activity...")
        slug_to_condition = await get_unique_slugs(storage)
        logger.info(f"Found {len(slug_to_condition)} unique slugs")

        # Fetch in parallel
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        slugs = list(slug_to_condition.keys())

        # Results: condition_id -> resolved_value
        resolutions = {}
        resolved_count = 0
        open_count = 0
        error_count = 0

        async with httpx.AsyncClient() as client:
            batch_size = 500
            for batch_start in range(0, len(slugs), batch_size):
                batch_end = min(batch_start + batch_size, len(slugs))
                batch = slugs[batch_start:batch_end]

                tasks = [
                    fetch_market_by_slug(client, slug, semaphore)
                    for slug in batch
                ]

                results = await asyncio.gather(*tasks)

                for slug, status, outcome_prices in results:
                    condition_id = slug_to_condition[slug]
                    if status == "resolved" and outcome_prices is not None:
                        # Store all outcome prices
                        resolutions[condition_id] = outcome_prices
                        resolved_count += 1
                    elif status == "open" or status is None:
                        open_count += 1
                    else:
                        error_count += 1

                logger.info(
                    f"Progress: {batch_end}/{len(slugs)} "
                    f"(resolved={resolved_count}, open={open_count}, errors={error_count})"
                )

        logger.info(f"Done! Resolved: {resolved_count}, Open: {open_count}, Errors: {error_count}")

        # Save to a simple JSON file for the backtest to use
        import json
        output_file = DATA_DIR / "market_resolutions.json"
        with open(output_file, "w") as f:
            json.dump(resolutions, f)
        logger.info(f"Saved {len(resolutions)} resolutions to {output_file}")

        # Print some examples
        logger.info("Sample resolutions:")
        for i, (cid, val) in enumerate(list(resolutions.items())[:5]):
            logger.info(f"  {cid[:30]}... -> {val}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
