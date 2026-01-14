#!/usr/bin/env python3
"""
Fetch historical price data for markets using Polymarket CLOB API.
Saves to data/price_history/{condition_id}.json
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_DIR = DATA_DIR / "price_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Public API - be respectful
RATE_LIMIT_DELAY = 0.3
CLOB_API = "https://clob.polymarket.com"


def fetch_price_history(
    token_id: str,
    days: int = 60,
    interval: str = "1d",  # 1m, 5m, 1h, 4h, 1d
) -> list[dict]:
    """
    Fetch price history for a token (YES outcome).
    Fetches in 7-day chunks to avoid API limits.

    Args:
        token_id: CLOB token ID
        days: Number of days of history
        interval: Candle interval (1d = daily)

    Returns:
        List of {timestamp, open, high, low, close} dicts
    """
    # Convert interval to fidelity parameter
    fidelity_map = {"1m": 1, "5m": 5, "1h": 60, "4h": 240, "1d": 1440}
    fidelity = fidelity_map.get(interval, 1440)

    all_candles = []
    chunk_days = 7  # Fetch in 7-day chunks to avoid "interval too long" error

    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())

    # Fetch in chunks from oldest to newest
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + (chunk_days * 86400), end_ts)

        try:
            url = f"{CLOB_API}/prices-history"
            params = {
                "market": token_id,
                "interval": interval,
                "fidelity": fidelity,
                "startTs": chunk_start,
                "endTs": chunk_end,
            }

            response = httpx.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Response is {"history": [{"t": timestamp, "p": price}, ...]}
            history = data.get("history", [])

            for point in history:
                ts = point.get("t")
                price = point.get("p")
                if ts and price:
                    all_candles.append({
                        "timestamp": ts,
                        "close": float(price),
                        "open": float(price),
                        "high": float(price),
                        "low": float(price),
                    })

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []  # Token not found
            # Continue to next chunk on other errors
            pass
        except Exception:
            pass

        chunk_start = chunk_end
        time.sleep(0.1)  # Small delay between chunks

    # Sort by timestamp and deduplicate
    seen = set()
    unique = []
    for c in sorted(all_candles, key=lambda x: x["timestamp"]):
        if c["timestamp"] not in seen:
            seen.add(c["timestamp"])
            unique.append(c)

    return unique


def main():
    # Load markets
    markets_path = DATA_DIR / "markets.json"
    if not markets_path.exists():
        print("Run fetch_markets.py first")
        return

    with open(markets_path) as f:
        markets = json.load(f)

    import sys
    print(f"Fetching history for {len(markets)} markets...", flush=True)
    successful = 0
    skipped = 0
    failed = 0
    insufficient = 0

    for i, market in enumerate(markets):
        condition_id = market.get("condition_id")
        clob_token_ids = market.get("clob_token_ids", [])

        if not condition_id or not clob_token_ids:
            failed += 1
            continue

        # Use first token (YES outcome typically)
        token_id = clob_token_ids[0] if clob_token_ids else None
        if not token_id:
            failed += 1
            continue

        # Check cache (use condition_id for filename)
        output_path = HISTORY_DIR / f"{condition_id[:40]}.json"
        if output_path.exists():
            skipped += 1
            continue

        # Fetch price history
        q = market.get('question', 'Unknown')[:40]
        print(f"[{i+1}/{len(markets)}] {q}...", flush=True)

        candles = fetch_price_history(token_id, days=60)

        if candles and len(candles) >= 7:  # Need at least a week of data
            with open(output_path, "w") as f:
                json.dump({
                    "condition_id": condition_id,
                    "token_id": token_id,
                    "question": market.get("question"),
                    "candles": candles,
                }, f, indent=2)
            successful += 1
            print(f"  -> {len(candles)} candles", flush=True)
        elif candles:
            insufficient += 1
            print(f"  -> only {len(candles)} candles (need 7+)", flush=True)
        else:
            failed += 1
            print(f"  -> no data", flush=True)

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone:", flush=True)
    print(f"  {successful} markets with sufficient history", flush=True)
    print(f"  {insufficient} markets with insufficient history (<7 days)", flush=True)
    print(f"  {skipped} already cached", flush=True)
    print(f"  {failed} failed/no data", flush=True)


if __name__ == "__main__":
    main()
