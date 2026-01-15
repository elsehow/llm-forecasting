#!/usr/bin/env python3
"""Ingest Kalshi markets into the database.

Fetches non-sports prediction markets from Kalshi's public API
and saves them to the forecastbench database.

Usage:
    uv run python packages/llm-forecasting/scripts/ingest_kalshi.py
    uv run python packages/llm-forecasting/scripts/ingest_kalshi.py --db data/forecastbench.db
    uv run python packages/llm-forecasting/scripts/ingest_kalshi.py --limit 100
"""

import argparse
import asyncio
from pathlib import Path

from llm_forecasting.sources.kalshi import KalshiSource
from llm_forecasting.storage.sqlite import SQLiteStorage


async def main():
    parser = argparse.ArgumentParser(description="Ingest Kalshi markets")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/forecastbench.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of markets to fetch",
    )
    parser.add_argument(
        "--include-sports",
        action="store_true",
        help="Include sports betting markets (excluded by default)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("KALSHI MARKET INGESTION")
    print("=" * 60)
    print(f"\nDatabase: {args.db}")
    print(f"Limit: {args.limit or 'None'}")
    print(f"Include sports: {args.include_sports}")

    # Initialize storage
    storage = SQLiteStorage(db_path=args.db)

    # Fetch markets
    print("\n[1/2] Fetching Kalshi markets...")
    source = KalshiSource()

    # Override the data provider's settings if needed
    source._data_provider._exclude_sports = not args.include_sports

    markets = await source._data_provider.fetch_markets(
        active_only=False,  # Include closed markets for historical data
        exclude_sports=not args.include_sports,
        limit=args.limit,
    )
    print(f"  Fetched {len(markets)} markets")

    # Convert to questions
    questions = []
    for market in markets:
        q = source._market_to_question(market)
        if q:
            questions.append(q)

    print(f"  Converted {len(questions)} to questions")

    # Save to database
    print("\n[2/2] Saving to database...")
    existing_count = 0
    new_count = 0

    for q in questions:
        try:
            await storage.save_question(q)
            new_count += 1
        except Exception as e:
            if "UNIQUE constraint" in str(e):
                existing_count += 1
            else:
                print(f"  Error saving {q.id}: {e}")

    print(f"  New questions: {new_count}")
    print(f"  Already existed: {existing_count}")

    # Show sample
    print("\n" + "-" * 40)
    print("Sample questions:")
    for q in questions[:5]:
        print(f"\n  {q.id}")
        print(f"    {q.text[:60]}...")
        print(f"    URL: {q.url}")
        print(f"    Resolution: {q.resolution_date}")
        print(f"    Prob: {q.base_rate}")

    await source.close()
    print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
