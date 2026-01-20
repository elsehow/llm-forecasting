"""Entity market lookup: find markets for entities mentioned in a tree.

Simple, general-purpose tool that:
1. Extracts entities (quoted strings, proper nouns) from signals
2. Searches for markets mentioning each entity
3. Reports findings

No diagnosis, no weight calculations - just data for the user to interpret.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tree import SignalTree


@dataclass
class EntityMarket:
    """An entity mentioned in the tree and its market data (if found)."""

    name: str
    signal_ids: list[str] = field(default_factory=list)
    market_title: str | None = None
    market_probability: float | None = None
    market_url: str | None = None


def extract_entities(tree: SignalTree) -> dict[str, list[str]]:
    """Extract entities from signal texts.

    Returns:
        Dict mapping entity name -> list of signal IDs mentioning it
    """
    entities: dict[str, list[str]] = {}

    # Also check target
    all_texts = [(tree.target.id, tree.target.text)]
    all_texts.extend((s.id, s.text) for s in tree.signals)

    for signal_id, text in all_texts:
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        for name in quoted:
            if len(name) > 2:  # Skip very short strings
                entities.setdefault(name, []).append(signal_id)

        # Extract capitalized sequences (2+ words)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        for name in proper_nouns:
            entities.setdefault(name, []).append(signal_id)

    # Deduplicate signal IDs
    return {name: list(set(ids)) for name, ids in entities.items()}


async def find_entity_markets(
    tree: SignalTree,
    db_path: str = "forecastbench.db",
) -> list[EntityMarket]:
    """Find markets for entities mentioned in the tree.

    Args:
        tree: The signal tree
        db_path: Path to market database

    Returns:
        List of EntityMarket, sorted by market probability (highest first)
    """
    from llm_forecasting.market_data import MarketDataStorage

    entities = extract_entities(tree)
    if not entities:
        return []

    # Filter out the target entity itself
    target_entities = set(re.findall(r'"([^"]+)"', tree.target.text))
    for name in target_entities:
        entities.pop(name, None)

    storage = MarketDataStorage(db_path)
    results = []

    try:
        for name, signal_ids in entities.items():
            # Search by entity name only
            candidates = await storage.search_by_title(
                keywords=[name],
                platform="polymarket",
                limit=10,
            )

            # Find first candidate containing the entity name
            market = None
            name_lower = name.lower()
            for c in candidates:
                if name_lower in c.title.lower() and c.current_probability is not None:
                    market = c
                    break

            results.append(EntityMarket(
                name=name,
                signal_ids=signal_ids,
                market_title=market.title if market else None,
                market_probability=market.current_probability if market else None,
                market_url=market.url if market else None,
            ))

    finally:
        await storage.close()

    # Sort: markets found first (by probability), then no-market entities
    results.sort(key=lambda e: (e.market_probability is None, -(e.market_probability or 0)))

    return results


def print_entity_report(entities: list[EntityMarket], tree: SignalTree) -> None:
    """Print entity market findings."""
    print("\n" + "=" * 70)
    print("ENTITY MARKETS")
    print("=" * 70)

    with_markets = [e for e in entities if e.market_probability is not None]
    without_markets = [e for e in entities if e.market_probability is None]

    if with_markets:
        print("\nEntities with markets:")
        for e in with_markets:
            print(f"\n  {e.name}")
            print(f"    Market: {e.market_probability:.1%} - {e.market_title[:50]}...")
            print(f"    Mentioned in {len(e.signal_ids)} signal(s)")

    if without_markets:
        print(f"\nEntities without markets ({len(without_markets)}):")
        for e in without_markets[:10]:  # Limit to avoid noise
            print(f"  - {e.name} ({len(e.signal_ids)} signals)")
        if len(without_markets) > 10:
            print(f"  ... and {len(without_markets) - 10} more")
