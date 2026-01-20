#!/usr/bin/env python
"""Refresh prediction market data cache.

Usage:
    python scripts/refresh_markets.py
    python scripts/refresh_markets.py --platforms polymarket metaculus
    python scripts/refresh_markets.py --min-liquidity 5000 --limit 2000
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cc_builder.utils import refresh_market_data, get_market_data_stats


async def main():
    parser = argparse.ArgumentParser(description="Refresh prediction market data cache")
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["polymarket"],
        help="Platforms to refresh (default: polymarket)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=5000,
        help="Minimum liquidity filter (default: 5000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max markets per platform (default: 2000)",
    )
    parser.add_argument(
        "--db-path",
        default="forecastbench.db",
        help="Database path (default: forecastbench.db)",
    )
    args = parser.parse_args()

    print("Refreshing market data...")
    print(f"  Platforms: {args.platforms}")
    print(f"  Min liquidity: ${args.min_liquidity:,.0f}")
    print(f"  Limit: {args.limit}")
    print()

    counts = await refresh_market_data(
        platforms=args.platforms,
        min_liquidity=args.min_liquidity,
        limit=args.limit,
        db_path=args.db_path,
    )

    print("\nResults:")
    for platform, count in counts.items():
        print(f"  {platform}: {count} markets")

    print("\nDatabase stats:")
    stats = await get_market_data_stats(args.db_path)
    print(f"  Total markets: {stats['total_markets']}")
    for platform, count in stats["by_platform"].items():
        print(f"  - {platform}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
