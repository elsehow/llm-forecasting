#!/usr/bin/env python3
"""Update market probabilities in database from scenario result files.

Reads monkeypatched probabilities from scenario JSON files and updates
the MarketDataStorage database with these values.
"""

import json
import asyncio
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

from llm_forecasting.market_data import MarketDataStorage
from llm_forecasting.market_data.models import Market


async def main():
    # Collect all market signals with probabilities from all result files
    results_dir = Path(__file__).parent.parent / "experiments/scenario-construction/results"
    market_sources = ["polymarket", "metaculus", "manifold", "kalshi"]

    # Use dict to dedupe by (source, id), keeping latest
    signals_to_update = {}

    for result_file in results_dir.glob("*/*.json"):
        data = json.loads(result_file.read_text())
        for s in data.get("signals", []):
            source = s.get("source", "")
            signal_id = s.get("id")
            base_rate = s.get("base_rate")

            if source in market_sources and signal_id and base_rate is not None:
                key = (source, signal_id)
                signals_to_update[key] = {
                    "id": signal_id,
                    "source": source,
                    "text": s.get("text", ""),
                    "base_rate": base_rate,
                }

    print(f"Found {len(signals_to_update)} unique market signals with probabilities")

    # Group by source
    by_source = defaultdict(list)
    for (source, _), signal in signals_to_update.items():
        by_source[source].append(signal)

    for source, signals in sorted(by_source.items()):
        print(f"  {source}: {len(signals)} signals")

    # Initialize storage (lazy init)
    storage = MarketDataStorage()

    # Update each market
    updated = 0
    created = 0
    for (source, signal_id), signal in signals_to_update.items():
        # Check if market exists
        existing = await storage.get_market(source, signal_id)

        if existing:
            # Update probability by creating updated Market
            updated_market = Market(
                id=existing.id,
                platform=existing.platform,
                title=existing.title,
                description=existing.description,
                url=existing.url,
                created_at=existing.created_at,
                close_date=existing.close_date,
                resolution_date=existing.resolution_date,
                status=existing.status,
                resolved_value=existing.resolved_value,
                current_probability=signal["base_rate"],  # Update this
                liquidity=existing.liquidity,
                volume_24h=existing.volume_24h,
                volume_total=existing.volume_total,
                num_forecasters=existing.num_forecasters,
                clob_token_ids=existing.clob_token_ids,
                fetched_at=datetime.now(timezone.utc),
            )
            await storage.save_markets([updated_market])
            updated += 1
        else:
            # Create minimal market entry
            new_market = Market(
                id=signal_id,
                platform=source,
                title=signal["text"],
                current_probability=signal["base_rate"],
                created_at=datetime.now(timezone.utc),
                fetched_at=datetime.now(timezone.utc),
            )
            await storage.save_markets([new_market])
            created += 1

    print(f"\nUpdated {updated} existing markets, created {created} new entries")


if __name__ == "__main__":
    asyncio.run(main())
