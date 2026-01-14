#!/usr/bin/env python3
"""
Fetch active Polymarket markets via public gamma API.
Sorted by volume to get established markets with price history.
"""

import json
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Public API - no key needed, but be respectful
RATE_LIMIT_DELAY = 0.5
API_BASE = "https://gamma-api.polymarket.com"


def fetch_markets(min_volume_usd: float = 10_000, limit: int = 500) -> list[dict]:
    """
    Fetch Polymarket markets sorted by 24h volume.

    Args:
        min_volume_usd: Minimum 24h volume to include
        limit: Max markets to fetch

    Returns:
        List of market dicts
    """
    all_markets = []
    offset = 0
    batch_size = 100  # API limit per request

    while len(all_markets) < limit:
        print(f"Fetching markets offset={offset}...")

        params = {
            "active": "true",
            "closed": "false",
            "limit": batch_size,
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false",
        }

        try:
            response = httpx.get(f"{API_BASE}/markets", params=params, timeout=30)
            response.raise_for_status()
            batch = response.json()
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

        if not batch:
            break

        # First batch: show sample
        if offset == 0 and batch:
            m = batch[0]
            print(f"\nFirst market keys: {list(m.keys())[:15]}")
            print(f"  question: {m.get('question', 'N/A')[:60]}")
            print(f"  volume24hr: ${float(m.get('volume24hr', 0)):,.0f}")
            print(f"  conditionId: {m.get('conditionId', 'N/A')[:30]}...")
            print()

        all_markets.extend(batch)
        offset += batch_size
        time.sleep(RATE_LIMIT_DELAY)

        # Stop if we got a partial batch
        if len(batch) < batch_size:
            break

    print(f"\nFetched {len(all_markets)} total markets")

    # Filter and normalize
    filtered = []
    for m in all_markets:
        condition_id = m.get("conditionId")
        if not condition_id:
            continue

        volume_24h = float(m.get("volume24hr", 0) or 0)
        if volume_24h < min_volume_usd:
            continue

        # Get token IDs for price history
        # clobTokenIds is a JSON string like "[\"token1\", \"token2\"]"
        clob_token_ids = m.get("clobTokenIds", "[]")
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except:
                clob_token_ids = []

        market_dict = {
            "condition_id": condition_id,
            "question": m.get("question", "Unknown"),
            "slug": m.get("slug"),
            "volume_24h": volume_24h,
            "volume_total": float(m.get("volume", 0) or 0),
            "liquidity": float(m.get("liquidity", 0) or 0),
            "clob_token_ids": clob_token_ids,
            "outcomes": m.get("outcomes"),  # e.g., "Yes, No"
            "end_date": m.get("endDate"),
            "created_at": m.get("createdAt"),
        }

        filtered.append(market_dict)

    print(f"Found {len(filtered)} markets with volume >= ${min_volume_usd:,.0f}")
    return filtered


def main():
    output_path = DATA_DIR / "markets.json"

    # Always refetch to get current high-volume markets
    # (Delete cache manually if you want to refresh)
    if output_path.exists():
        print(f"Using cached markets from {output_path}")
        print("(Delete data/markets.json to refetch)")
        with open(output_path) as f:
            markets = json.load(f)
        print(f"Loaded {len(markets)} cached markets")
    else:
        markets = fetch_markets(min_volume_usd=10_000, limit=500)
        with open(output_path, "w") as f:
            json.dump(markets, f, indent=2)
        print(f"Saved {len(markets)} markets to {output_path}")

    # Show sample
    if markets:
        print("\nTop markets by volume:")
        for m in markets[:5]:
            q = m.get('question', 'Unknown')[:50]
            vol = m.get('volume_24h', 0)
            print(f"  ${vol:>10,.0f}  {q}...")


if __name__ == "__main__":
    main()
