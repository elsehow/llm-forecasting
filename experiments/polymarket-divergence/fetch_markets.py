"""Fetch Polymarket markets for divergence experiment."""
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from llm_forecasting.market_data.polymarket import PolymarketData

RESULTS_DIR = Path(__file__).parent / "results"
MARKETS_FILE = RESULTS_DIR / "markets.jsonl"
MIN_LIQUIDITY = 10_000


async def main(limit: int = 10, max_days: int | None = None):
    """Fetch markets from Polymarket.

    Args:
        limit: Max number of markets to fetch
        max_days: If set, only include markets closing within this many days
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    provider = PolymarketData()
    try:
        # Fetch more than limit to account for filtering
        fetch_limit = limit * 3 if max_days else limit
        markets = await provider.fetch_markets(
            active_only=True,
            min_liquidity=MIN_LIQUIDITY,
            limit=fetch_limit,
        )

        # Filter by close date if specified
        if max_days:
            cutoff = datetime.now().astimezone() + timedelta(days=max_days)
            markets = [
                m for m in markets
                if m.close_date and m.close_date <= cutoff
            ][:limit]

        timestamp = datetime.now().isoformat()
        records = []

        with open(MARKETS_FILE, "a") as f:
            for m in markets:
                record = {
                    "fetch_timestamp": timestamp,
                    "market_id": m.id,
                    "question": m.title,
                    "resolution_rules": m.description or "(No resolution rules provided)",
                    "market_price": m.current_probability,
                    "close_date": m.close_date.isoformat() if m.close_date else None,
                    "liquidity": m.liquidity,
                    "url": m.url,
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)

        print(f"Appended {len(records)} markets to {MARKETS_FILE}")

        # Print summary for quick review
        for i, m in enumerate(records[:10]):
            print(f"\n[{i+1}] {m['question'][:70]}...")
            print(f"    Market: {m['market_price']:.1%} | Close: {m['close_date'][:10] if m['close_date'] else 'N/A'} | Liquidity: ${m['liquidity']:,.0f}")

    finally:
        await provider.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-days", type=int, help="Only markets closing within N days")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.max_days))
