#!/usr/bin/env python3
"""
Estimate how many Metaculus question pairs have overlapping history data
for VOI validation.

Goal: Find pairs (Q, X) where:
1. Q resolved
2. X had probability history during Q's active period
3. Both have enough overlapping data points to compute co-movement

This tells us if we have enough data for Metaculus VOI validation.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import httpx

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data"
MIN_OVERLAP_DAYS = 7  # Minimum overlapping period
MIN_HISTORY_ENTRIES = 5  # Minimum data points in overlap
MAX_QUESTIONS = 200  # Limit for initial scan (API calls)
CATEGORIES_OF_INTEREST = ["politics", "technology", "ai", "science", "geopolitics"]


@dataclass
class QuestionSummary:
    id: int
    title: str
    open_time: datetime | None
    close_time: datetime | None
    resolve_time: datetime | None
    resolved: bool
    history_entries: int
    history_start: datetime | None
    history_end: datetime | None
    forecasts_count: int
    category: str | None


async def fetch_resolved_questions(client: httpx.AsyncClient, limit: int = 100) -> list[dict]:
    """Fetch resolved binary questions with high forecast counts."""
    url = "https://www.metaculus.com/api/posts/"
    params = {
        "limit": limit,
        "forecast_type": "binary",
        "statuses": "resolved",
        "order_by": "-forecasts_count",
    }

    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("results", [])


async def fetch_question_detail(client: httpx.AsyncClient, question_id: int) -> dict | None:
    """Fetch individual question with full history."""
    url = f"https://www.metaculus.com/api/posts/{question_id}/"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error fetching {question_id}: {e}")
        return None


def parse_question(data: dict) -> QuestionSummary | None:
    """Parse question data into summary."""
    try:
        q = data.get("question", {})
        hist = q.get("aggregations", {}).get("recency_weighted", {}).get("history", [])

        # Parse times
        open_time = None
        if data.get("open_time"):
            open_time = datetime.fromisoformat(data["open_time"].replace("Z", "+00:00"))

        close_time = None
        if data.get("actual_close_time"):
            close_time = datetime.fromisoformat(data["actual_close_time"].replace("Z", "+00:00"))

        resolve_time = None
        if data.get("actual_resolve_time"):
            resolve_time = datetime.fromisoformat(data["actual_resolve_time"].replace("Z", "+00:00"))

        # History times
        history_start = None
        history_end = None
        if hist:
            history_start = datetime.fromtimestamp(hist[0]["start_time"])
            history_end = datetime.fromtimestamp(hist[-1]["start_time"])

        # Category from projects
        category = None
        projects = data.get("projects", {})
        if isinstance(projects, dict):
            # Look for category in projects
            for proj_type, projs in projects.items():
                if isinstance(projs, list):
                    for p in projs:
                        if p.get("type") == "category":
                            category = p.get("name", "").lower()
                            break

        return QuestionSummary(
            id=data["id"],
            title=data.get("title", "")[:80],
            open_time=open_time,
            close_time=close_time,
            resolve_time=resolve_time,
            resolved=data.get("resolved", False),
            history_entries=len(hist),
            history_start=history_start,
            history_end=history_end,
            forecasts_count=data.get("forecasts_count", 0),
            category=category,
        )
    except Exception as e:
        print(f"  Error parsing question: {e}")
        return None


def compute_overlap(q1: QuestionSummary, q2: QuestionSummary) -> tuple[int, int]:
    """
    Compute overlapping period between two questions.
    Returns (overlap_days, estimated_shared_entries).
    """
    if not (q1.history_start and q1.history_end and q2.history_start and q2.history_end):
        return 0, 0

    # Find overlap period
    overlap_start = max(q1.history_start, q2.history_start)
    overlap_end = min(q1.history_end, q2.history_end)

    if overlap_start >= overlap_end:
        return 0, 0

    overlap_days = (overlap_end - overlap_start).days

    # Estimate shared entries (rough: take min density)
    q1_span = (q1.history_end - q1.history_start).days or 1
    q2_span = (q2.history_end - q2.history_start).days or 1

    q1_density = q1.history_entries / q1_span  # entries per day
    q2_density = q2.history_entries / q2_span

    min_density = min(q1_density, q2_density)
    estimated_entries = int(overlap_days * min_density)

    return overlap_days, estimated_entries


async def main():
    print("=" * 70)
    print("METACULUS PAIR ESTIMATION")
    print("=" * 70)
    print(f"\nFetching up to {MAX_QUESTIONS} resolved questions...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch list of resolved questions
        questions_list = await fetch_resolved_questions(client, limit=MAX_QUESTIONS)
        print(f"Got {len(questions_list)} questions from list endpoint")

        # Fetch full details for each (to get history)
        questions: list[QuestionSummary] = []

        print(f"\nFetching individual question details...")
        for i, q in enumerate(questions_list):
            if i % 20 == 0:
                print(f"  [{i+1}/{len(questions_list)}]...")

            detail = await fetch_question_detail(client, q["id"])
            if detail:
                summary = parse_question(detail)
                if summary and summary.history_entries > 0:
                    questions.append(summary)

            # Small delay to be nice to API
            await asyncio.sleep(0.1)

        print(f"\nQuestions with history data: {len(questions)}")

    if len(questions) < 2:
        print("Not enough questions with history!")
        return

    # Show distribution
    print("\n" + "-" * 70)
    print("QUESTION STATISTICS")
    print("-" * 70)

    history_counts = [q.history_entries for q in questions]
    print(f"History entries: min={min(history_counts)}, max={max(history_counts)}, median={sorted(history_counts)[len(history_counts)//2]}")

    forecast_counts = [q.forecasts_count for q in questions]
    print(f"Forecast counts: min={min(forecast_counts)}, max={max(forecast_counts)}")

    # Compute all pairs
    print("\n" + "-" * 70)
    print("PAIR ANALYSIS")
    print("-" * 70)

    total_pairs = 0
    usable_pairs = 0
    good_pairs = 0
    excellent_pairs = 0

    pair_details = []

    for i, q1 in enumerate(questions):
        for q2 in questions[i+1:]:
            total_pairs += 1

            overlap_days, shared_entries = compute_overlap(q1, q2)

            if overlap_days >= MIN_OVERLAP_DAYS and shared_entries >= MIN_HISTORY_ENTRIES:
                usable_pairs += 1

                if shared_entries >= 20:
                    good_pairs += 1

                if shared_entries >= 50:
                    excellent_pairs += 1
                    pair_details.append({
                        "q1_id": q1.id,
                        "q1_title": q1.title,
                        "q2_id": q2.id,
                        "q2_title": q2.title,
                        "overlap_days": overlap_days,
                        "shared_entries": shared_entries,
                    })

    print(f"Total pairs: {total_pairs}")
    print(f"Usable pairs (≥{MIN_OVERLAP_DAYS} days, ≥{MIN_HISTORY_ENTRIES} entries): {usable_pairs}")
    print(f"Good pairs (≥20 shared entries): {good_pairs}")
    print(f"Excellent pairs (≥50 shared entries): {excellent_pairs}")

    # Show some excellent pairs
    if pair_details:
        print("\n" + "-" * 70)
        print("SAMPLE EXCELLENT PAIRS")
        print("-" * 70)
        for p in sorted(pair_details, key=lambda x: -x["shared_entries"])[:10]:
            print(f"\n{p['overlap_days']} days, ~{p['shared_entries']} entries:")
            print(f"  Q1: {p['q1_title']}")
            print(f"  Q2: {p['q2_title']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"From {len(questions)} resolved questions with history:")
    print(f"  - {usable_pairs} pairs have sufficient overlap for co-movement analysis")
    print(f"  - {good_pairs} pairs have good data (≥20 shared entries)")
    print(f"  - {excellent_pairs} pairs have excellent data (≥50 shared entries)")

    if usable_pairs > 100:
        print("\n✓ SUFFICIENT DATA for Metaculus VOI validation")
    elif usable_pairs > 30:
        print("\n~ MARGINAL: May have enough for a pilot study")
    else:
        print("\n✗ INSUFFICIENT: Need more questions or different approach")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_questions": MAX_QUESTIONS,
            "min_overlap_days": MIN_OVERLAP_DAYS,
            "min_history_entries": MIN_HISTORY_ENTRIES,
        },
        "summary": {
            "questions_with_history": len(questions),
            "total_pairs": total_pairs,
            "usable_pairs": usable_pairs,
            "good_pairs": good_pairs,
            "excellent_pairs": excellent_pairs,
        },
        "questions": [
            {
                "id": q.id,
                "title": q.title,
                "history_entries": q.history_entries,
                "forecasts_count": q.forecasts_count,
                "category": q.category,
            }
            for q in questions
        ],
        "excellent_pairs": pair_details[:50],  # Save top 50
    }

    output_path = OUTPUT_DIR / "metaculus_pair_estimate.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
