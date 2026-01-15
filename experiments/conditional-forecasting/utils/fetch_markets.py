#!/usr/bin/env python3
"""
Fetch active Polymarket markets using the llm_forecasting market_data layer.
Saves to both SQLite (for shared access) and JSON (for backward compatibility).
"""

import asyncio
import json
from pathlib import Path

from llm_forecasting.market_data import MarketDataStorage, PolymarketData

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


async def fetch_and_cache_markets(
    min_volume_usd: float = 10_000,
    limit: int = 500,
    db_path: Path | None = None,
) -> list[dict]:
    """
    Fetch Polymarket markets and cache to SQLite.

    Args:
        min_volume_usd: Minimum 24h volume
        limit: Max markets
        db_path: Optional custom DB path (default: forecastbench.db)

    Returns:
        List of market dicts (for backward compatibility)
    """
    provider = PolymarketData()
    storage = MarketDataStorage(db_path or "forecastbench.db")

    try:
        print(f"Fetching markets with min volume ${min_volume_usd:,.0f}...")

        # Fetch from API
        markets = await provider.fetch_markets(
            active_only=True,
            min_volume=min_volume_usd,
            limit=limit,
        )

        print(f"Fetched {len(markets)} markets from Polymarket")

        # Save to SQLite cache
        await storage.save_markets(markets)
        print(f"Saved to SQLite cache")

        # Convert to dicts for backward compatibility
        market_dicts = []
        for m in markets:
            market_dicts.append(
                {
                    "condition_id": m.id,
                    "question": m.title,
                    "slug": m.url.split("/")[-1] if m.url else None,
                    "volume_24h": m.volume_24h,
                    "volume_total": m.volume_total,
                    "liquidity": m.liquidity,
                    "clob_token_ids": m.clob_token_ids,
                    "outcomes": ["Yes", "No"],  # Only binary markets
                    "end_date": m.close_date.isoformat() if m.close_date else None,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
            )

        return market_dicts

    finally:
        await provider.close()
        await storage.close()


def main():
    output_path = DATA_DIR / "markets.json"

    # Check JSON cache for backward compatibility
    if output_path.exists():
        print(f"Using cached markets from {output_path}")
        print("(Delete data/markets.json to refetch)")
        with open(output_path) as f:
            markets = json.load(f)
        print(f"Loaded {len(markets)} cached markets")
    else:
        markets = asyncio.run(fetch_and_cache_markets())

        # Write JSON for backward compatibility with other scripts
        with open(output_path, "w") as f:
            json.dump(markets, f, indent=2)
        print(f"Saved {len(markets)} markets to {output_path}")

    # Show sample
    if markets:
        print("\nTop markets by volume:")
        for m in markets[:5]:
            q = m.get("question", "Unknown")[:50]
            vol = m.get("volume_24h", 0)
            print(f"  ${vol:>10,.0f}  {q}...")


if __name__ == "__main__":
    main()
