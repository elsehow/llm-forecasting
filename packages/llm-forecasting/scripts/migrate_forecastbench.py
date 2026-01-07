#!/usr/bin/env python3
"""Migrate ForecastBench datasets to local SQLite database.

Downloads question sets and resolution sets from the ForecastBench public datasets
repository and imports them into the llm-forecasting schema.

Usage:
    uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py
    uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py --db forecasts.db
    uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py --question-sets 2025-12-21-llm
"""

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path

import httpx

from llm_forecasting.models import Forecast, Question, QuestionType, SourceType
from llm_forecasting.storage.sqlite import Resolution, SQLiteStorage

# ForecastBench dataset URLs
BASE_URL = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"

# Available question sets (as of 2026-01-06)
QUESTION_SETS = [
    "2024-07-21-human",
    "2024-07-21-llm",
    "2025-03-02-llm",
    "2025-03-16-llm",
    "2025-03-30-llm",
    "2025-04-13-llm",
    "2025-04-27-llm",
    "2025-05-11-llm",
    "2025-05-25-llm",
    "2025-06-08-llm",
    "2025-06-22-llm",
    "2025-08-03-llm",
    "2025-08-17-llm",
    "2025-08-31-llm",
    "2025-10-26-llm",
    "2025-11-09-llm",
    "2025-11-23-llm",
    "2025-12-07-llm",
    "2025-12-21-llm",
    "2026-01-04-llm",
]

# Resolution sets match question sets by date prefix
RESOLUTION_SETS = [
    "2024-07-21",
    "2025-03-02",
    "2025-03-16",
    "2025-03-30",
    "2025-04-13",
    "2025-04-27",
    "2025-05-11",
    "2025-05-25",
    "2025-06-08",
    "2025-06-22",
    "2025-08-03",
    "2025-08-17",
    "2025-08-31",
    "2025-10-26",
    "2025-11-09",
    "2025-11-23",
    "2025-12-07",
    "2025-12-21",
]

# Map ForecastBench sources to our source types
SOURCE_TYPE_MAP = {
    "manifold": SourceType.MARKET,
    "metaculus": SourceType.MARKET,
    "polymarket": SourceType.MARKET,
    "infer": SourceType.MARKET,
    "fred": SourceType.DATA,
    "acled": SourceType.DATA,
    "wikipedia": SourceType.DATA,
    "dbnomics": SourceType.DATA,
}


def parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO 8601 datetime string."""
    if not dt_str or dt_str == "N/A":
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_date(dt_str: str | None) -> date | None:
    """Parse date from datetime string or date string."""
    if not dt_str or dt_str == "N/A":
        return None
    try:
        if "T" in dt_str:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).date()
        return date.fromisoformat(dt_str)
    except ValueError:
        return None


def fb_question_to_question(fb_q: dict) -> Question | None:
    """Convert ForecastBench question to our Question model."""
    # Skip questions with composite IDs (lists)
    question_id = fb_q.get("id")
    if isinstance(question_id, list):
        return None

    source = fb_q.get("source", "unknown")

    # Combine background with resolution_criteria if present
    background_parts = []
    if fb_q.get("background"):
        background_parts.append(fb_q["background"])
    if fb_q.get("resolution_criteria") and fb_q["resolution_criteria"] != "N/A":
        background_parts.append(f"\n\nResolution criteria: {fb_q['resolution_criteria']}")

    # Determine question type from freeze value
    freeze_value = fb_q.get("freeze_datetime_value")
    question_type = QuestionType.BINARY
    if freeze_value is not None:
        try:
            fv = float(freeze_value)
            if fv < 0 or fv > 1:
                question_type = QuestionType.CONTINUOUS
        except (ValueError, TypeError):
            pass

    return Question(
        id=fb_q["id"],
        source=source,
        source_type=SOURCE_TYPE_MAP.get(source, SourceType.MARKET),
        text=fb_q.get("question", ""),
        background="\n".join(background_parts) if background_parts else None,
        url=fb_q.get("url"),
        question_type=question_type,
        created_at=parse_datetime(fb_q.get("market_info_open_datetime")) or datetime.now(),
        resolution_date=parse_date(fb_q.get("market_info_close_datetime")),
        resolved=False,  # Will be updated from resolution set
        resolution_value=None,
    )


def fb_question_to_market_forecast(fb_q: dict) -> Forecast | None:
    """Extract market consensus forecast from ForecastBench question."""
    freeze_value = fb_q.get("freeze_datetime_value")
    if freeze_value is None:
        return None

    # Convert to float if string
    try:
        freeze_value = float(freeze_value)
    except (ValueError, TypeError):
        return None

    freeze_dt = parse_datetime(fb_q.get("freeze_datetime"))

    # Determine if this is a probability (0-1) or a point estimate (continuous)
    is_probability = 0 <= freeze_value <= 1

    return Forecast(
        question_id=fb_q["id"],
        source=fb_q.get("source", "unknown"),
        forecaster="market",
        probability=freeze_value if is_probability else None,
        point_estimate=freeze_value if not is_probability else None,
        created_at=freeze_dt or datetime.now(),
        reasoning=fb_q.get("freeze_datetime_value_explanation"),
    )


def fb_resolution_to_resolution(fb_r: dict) -> Resolution | None:
    """Convert ForecastBench resolution to our Resolution model."""
    # Skip composite IDs
    question_id = fb_r.get("id")
    if isinstance(question_id, list):
        return None

    if not fb_r.get("resolved"):
        return None

    resolution_date = parse_date(fb_r.get("resolution_date"))
    if not resolution_date:
        return None

    return Resolution(
        question_id=question_id,
        source=fb_r.get("source", "unknown"),
        date=resolution_date,
        value=fb_r.get("resolved_to", 0.0),
    )


async def fetch_json(client: httpx.AsyncClient, url: str) -> list[dict] | dict | None:
    """Fetch JSON from URL."""
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        print(f"  Error fetching {url}: {e}")
        return None


async def migrate_question_set(
    client: httpx.AsyncClient,
    storage: SQLiteStorage,
    qs_name: str,
) -> tuple[int, int]:
    """Migrate a single question set. Returns (questions_count, forecasts_count)."""
    url = f"{BASE_URL}/question_sets/{qs_name}.json"
    print(f"  Fetching {qs_name}...")

    data = await fetch_json(client, url)
    if not data:
        return 0, 0

    # Handle both list format and dict with questions key
    questions_data = data if isinstance(data, list) else data.get("questions", data)
    if not isinstance(questions_data, list):
        questions_data = [questions_data]

    questions = []
    forecasts = []

    for fb_q in questions_data:
        q = fb_question_to_question(fb_q)
        if q is None:
            continue  # Skip composite/invalid questions
        questions.append(q)

        f = fb_question_to_market_forecast(fb_q)
        if f:
            forecasts.append(f)

    if questions:
        await storage.save_questions(questions)

    if forecasts:
        await storage.save_forecasts(forecasts)

    return len(questions), len(forecasts)


async def migrate_resolution_set(
    client: httpx.AsyncClient,
    storage: SQLiteStorage,
    rs_date: str,
) -> int:
    """Migrate a single resolution set. Returns count of resolutions."""
    url = f"{BASE_URL}/resolution_sets/{rs_date}_resolution_set.json"
    print(f"  Fetching {rs_date} resolutions...")

    data = await fetch_json(client, url)
    if not data:
        return 0

    # Handle dict with resolutions key
    resolutions_data = data.get("resolutions", data) if isinstance(data, dict) else data
    if not isinstance(resolutions_data, list):
        resolutions_data = [resolutions_data]

    count = 0
    for fb_r in resolutions_data:
        r = fb_resolution_to_resolution(fb_r)
        if r:
            await storage.save_resolution(r)
            count += 1

    return count


async def main():
    parser = argparse.ArgumentParser(description="Migrate ForecastBench datasets to SQLite")
    parser.add_argument(
        "--db",
        type=str,
        default="forecastbench.db",
        help="Path to SQLite database (default: forecastbench.db)",
    )
    parser.add_argument(
        "--question-sets",
        type=str,
        nargs="*",
        help="Specific question sets to migrate (default: all)",
    )
    parser.add_argument(
        "--resolution-sets",
        type=str,
        nargs="*",
        help="Specific resolution sets to migrate (default: all)",
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Skip question set migration",
    )
    parser.add_argument(
        "--skip-resolutions",
        action="store_true",
        help="Skip resolution set migration",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    storage = SQLiteStorage(db_path)

    question_sets = args.question_sets or QUESTION_SETS
    resolution_sets = args.resolution_sets or RESOLUTION_SETS

    print(f"Migrating ForecastBench data to {db_path}")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        total_questions = 0
        total_forecasts = 0
        total_resolutions = 0

        if not args.skip_questions:
            print("Migrating question sets...")
            for qs_name in question_sets:
                q_count, f_count = await migrate_question_set(client, storage, qs_name)
                total_questions += q_count
                total_forecasts += f_count
                print(f"    {qs_name}: {q_count} questions, {f_count} market forecasts")
            print()

        if not args.skip_resolutions:
            print("Migrating resolution sets...")
            for rs_date in resolution_sets:
                r_count = await migrate_resolution_set(client, storage, rs_date)
                total_resolutions += r_count
                print(f"    {rs_date}: {r_count} resolutions")
            print()

        print("Migration complete!")
        print(f"  Questions: {total_questions}")
        print(f"  Market forecasts: {total_forecasts}")
        print(f"  Resolutions: {total_resolutions}")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
