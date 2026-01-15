#!/usr/bin/env python3
"""
Fetch historical price data using the llm_forecasting market_data layer.
Saves to SQLite for shared access across experiments.
"""

import asyncio
import json
from pathlib import Path

from llm_forecasting.market_data import MarketDataStorage, PolymarketData

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_DIR = DATA_DIR / "price_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_and_cache_history(
    days: int = 60,
    db_path: Path | None = None,
) -> dict:
    """
    Fetch price history for all cached markets.

    Args:
        days: Number of days of history to fetch
        db_path: Optional custom DB path (default: forecastbench.db)

    Returns:
        Stats dict with success/failure counts
    """
    storage = MarketDataStorage(db_path or "forecastbench.db")
    provider = PolymarketData()

    stats = {"successful": 0, "skipped": 0, "failed": 0, "insufficient": 0}

    try:
        # Get markets from SQLite cache
        markets = await storage.get_markets(platform="polymarket")

        if not markets:
            # Fallback: load from JSON if SQLite is empty
            markets_json = DATA_DIR / "markets.json"
            if markets_json.exists():
                print("Loading markets from JSON (SQLite cache empty)...")
                with open(markets_json) as f:
                    market_data = json.load(f)

                # Fetch fresh from API and save to SQLite
                print("Fetching markets to populate SQLite cache...")
                provider_markets = await provider.fetch_markets(limit=500)
                await storage.save_markets(provider_markets)
                markets = provider_markets
            else:
                print("No markets found. Run fetch_markets.py first.")
                return stats

        print(f"Processing {len(markets)} markets...")

        for i, market in enumerate(markets):
            if not market.clob_token_ids:
                stats["failed"] += 1
                continue

            # Check if already cached in SQLite
            if await storage.has_price_history("polymarket", market.id):
                stats["skipped"] += 1
                continue

            # Fetch price history
            token_id = market.clob_token_ids[0]
            title = market.title[:40] if market.title else "Unknown"
            print(f"[{i+1}/{len(markets)}] {title}...", flush=True)

            try:
                points = await provider.fetch_price_history_by_token(
                    token_id, interval="1d"
                )

                if points and len(points) >= 7:
                    await storage.save_price_history(market.id, "polymarket", points)
                    stats["successful"] += 1
                    print(f"  -> {len(points)} points")

                    # Also write JSON for backward compatibility
                    json_path = HISTORY_DIR / f"{market.id[:40]}.json"
                    if not json_path.exists():
                        candles = [
                            {
                                "timestamp": int(p.timestamp.timestamp()),
                                "close": p.price,
                                "open": p.price,
                                "high": p.price,
                                "low": p.price,
                            }
                            for p in points
                        ]
                        with open(json_path, "w") as f:
                            json.dump(
                                {
                                    "condition_id": market.id,
                                    "token_id": token_id,
                                    "question": market.title,
                                    "candles": candles,
                                },
                                f,
                                indent=2,
                            )
                elif points:
                    stats["insufficient"] += 1
                    print(f"  -> only {len(points)} points (need 7+)")
                else:
                    stats["failed"] += 1
                    print("  -> no data")
            except Exception as e:
                stats["failed"] += 1
                print(f"  -> error: {e}")

            await asyncio.sleep(0.3)  # Rate limit

        return stats

    finally:
        await provider.close()
        await storage.close()


def main():
    stats = asyncio.run(fetch_and_cache_history())

    print("\nDone:")
    print(f"  {stats['successful']} markets with sufficient history")
    print(f"  {stats['insufficient']} markets with insufficient history (<7 days)")
    print(f"  {stats['skipped']} already cached")
    print(f"  {stats['failed']} failed/no data")


if __name__ == "__main__":
    main()
