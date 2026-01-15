"""Refresh market data if stale.

Ensures questions table has fresh base_rate values before scenario generation.
"""

from datetime import datetime, timedelta, timezone

from llm_forecasting.sources import PolymarketSource, MetaculusSource, ManifoldSource
from llm_forecasting.storage import SQLiteStorage

# Maximum age before refresh is triggered
MAX_AGE = timedelta(hours=24)

# Map source names to source classes
SOURCE_CLASSES = {
    "polymarket": PolymarketSource,
    "metaculus": MetaculusSource,
    "manifold": ManifoldSource,
}


async def refresh_if_stale(
    db_path: str,
    sources: list[str],
    max_age: timedelta = MAX_AGE,
    force: bool = False,
) -> dict[str, int]:
    """Refresh questions from source APIs if data is stale.

    Args:
        db_path: Path to SQLite database
        sources: List of source names to check (e.g., ["polymarket", "metaculus"])
        max_age: Maximum age before data is considered stale (default: 24 hours)
        force: If True, refresh regardless of age

    Returns:
        Dict mapping source name to number of questions refreshed (0 if skipped)
    """
    storage = SQLiteStorage(db_path)
    results = {}

    for source_name in sources:
        if source_name not in SOURCE_CLASSES:
            print(f"  {source_name}: unknown source, skipping")
            results[source_name] = 0
            continue

        # Check freshness via most recent created_at
        if not force:
            questions = await storage.get_questions(source=source_name, limit=1)
            if questions:
                # Handle timezone-aware vs naive datetimes
                created_at = questions[0].created_at
                now = datetime.now(timezone.utc) if created_at.tzinfo else datetime.now()
                age = now - created_at

                if age < max_age:
                    hours = int(age.total_seconds() // 3600)
                    print(f"  {source_name}: fresh ({hours}h old), skipping")
                    results[source_name] = 0
                    continue

        # Stale or empty or forced - refresh from API
        print(f"  {source_name}: fetching fresh data...")
        source_cls = SOURCE_CLASSES[source_name]
        source = source_cls()

        try:
            fresh = await source.fetch_questions()
            await storage.save_questions(fresh)
            print(f"  {source_name}: refreshed {len(fresh)} questions")
            results[source_name] = len(fresh)
        except Exception as e:
            print(f"  {source_name}: error fetching - {e}")
            results[source_name] = 0

    return results
