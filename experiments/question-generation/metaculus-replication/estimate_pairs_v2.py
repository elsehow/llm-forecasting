#!/usr/bin/env python3
"""
Estimate Metaculus pairs for LLM conditional validation.

Goal: Find pairs (Q, X) where:
1. Q resolved at time T
2. X had probability data around T (before and after)
3. We can compute Δp on X when Q resolved

This is simpler than co-movement analysis. We just need to measure
whether X's probability shifted when Q resolved.

Rate limited to avoid 429 errors.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import httpx

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data"
RATE_LIMIT_DELAY = 1.5  # seconds between requests
MAX_RESOLVED_QUESTIONS = 100  # Q's to check
MAX_CANDIDATE_X = 500  # X's to scan for each time window
MIN_DP_WINDOW_DAYS = 3  # Need data within this many days of resolution


@dataclass
class ResolvedQuestion:
    """A question that resolved (potential Q)."""
    id: int
    title: str
    resolve_time: datetime
    resolution: float  # 0 or 1


@dataclass
class CandidateX:
    """A question that could be X (had data around Q's resolution)."""
    id: int
    title: str
    prob_before: float | None  # P(X) before Q resolved
    prob_after: float | None   # P(X) after Q resolved
    delta_p: float | None      # Observed shift
    time_before: datetime | None
    time_after: datetime | None


@dataclass
class ValidPair:
    """A usable (Q, X) pair."""
    q_id: int
    q_title: str
    q_resolve_time: str
    q_resolution: float
    x_id: int
    x_title: str
    x_prob_before: float
    x_prob_after: float
    x_delta_p: float
    days_before: float  # How many days before Q resolved
    days_after: float   # How many days after Q resolved


async def fetch_with_rate_limit(
    client: httpx.AsyncClient,
    url: str,
    params: dict = None,
    delay: float = RATE_LIMIT_DELAY
) -> dict | None:
    """Fetch with rate limiting and error handling."""
    await asyncio.sleep(delay)
    try:
        resp = await client.get(url, params=params)
        if resp.status_code == 429:
            print(f"  Rate limited, waiting 30s...")
            await asyncio.sleep(30)
            resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error: {e}")
        return None


async def fetch_resolved_questions(
    client: httpx.AsyncClient,
    limit: int
) -> list[ResolvedQuestion]:
    """Fetch resolved binary questions ordered by forecast count."""
    print(f"Fetching {limit} resolved questions...")

    url = "https://www.metaculus.com/api/posts/"
    params = {
        "limit": limit,
        "forecast_type": "binary",
        "statuses": "resolved",
        "order_by": "-forecasts_count",
    }

    data = await fetch_with_rate_limit(client, url, params, delay=0.5)
    if not data:
        return []

    questions = []
    for q in data.get("results", []):
        resolve_time_str = q.get("actual_resolve_time")
        if not resolve_time_str:
            continue

        resolve_time = datetime.fromisoformat(resolve_time_str.replace("Z", "+00:00"))

        # Get resolution value
        resolution = q.get("question", {}).get("resolution")
        if resolution is None:
            continue

        # Parse resolution to float
        if resolution in [0, 1, 0.0, 1.0]:
            res_float = float(resolution)
        elif resolution in ["yes", "Yes", "YES"]:
            res_float = 1.0
        elif resolution in ["no", "No", "NO"]:
            res_float = 0.0
        else:
            continue  # Skip non-binary resolutions

        questions.append(ResolvedQuestion(
            id=q["id"],
            title=q.get("title", "")[:80],
            resolve_time=resolve_time,
            resolution=res_float,
        ))

    print(f"  Got {len(questions)} resolved questions")
    return questions


async def fetch_questions_active_around(
    client: httpx.AsyncClient,
    target_time: datetime,
    limit: int = 100,
) -> list[dict]:
    """Fetch questions that were active around a target time."""
    # Fetch questions that were open before target_time
    # The API doesn't have great filtering for this, so we fetch recent
    # and filter client-side

    url = "https://www.metaculus.com/api/posts/"
    params = {
        "limit": limit,
        "forecast_type": "binary",
        "order_by": "-forecasts_count",
    }

    data = await fetch_with_rate_limit(client, url, params)
    if not data:
        return []

    return data.get("results", [])


async def get_question_history(
    client: httpx.AsyncClient,
    question_id: int,
) -> list[dict]:
    """Fetch full history for a single question."""
    url = f"https://www.metaculus.com/api/posts/{question_id}/"
    data = await fetch_with_rate_limit(client, url)

    if not data:
        return []

    return (data
            .get("question", {})
            .get("aggregations", {})
            .get("recency_weighted", {})
            .get("history", []))


def find_prob_around_time(
    history: list[dict],
    target_time: datetime,
    window_days: int = MIN_DP_WINDOW_DAYS,
) -> tuple[float | None, float | None, datetime | None, datetime | None]:
    """
    Find probability before and after target_time.

    Returns (prob_before, prob_after, time_before, time_after)
    """
    if not history:
        return None, None, None, None

    # Make target_time naive for comparison
    if target_time.tzinfo is not None:
        target_time = target_time.replace(tzinfo=None)

    target_ts = target_time.timestamp()
    window_seconds = window_days * 24 * 3600

    # Find closest entry before target
    prob_before = None
    time_before = None
    best_before_diff = float('inf')

    # Find closest entry after target
    prob_after = None
    time_after = None
    best_after_diff = float('inf')

    for entry in history:
        entry_ts = entry.get("start_time", 0)
        entry_time = datetime.fromtimestamp(entry_ts)
        centers = entry.get("centers", [])
        if not centers:
            continue
        prob = centers[0]

        diff = entry_ts - target_ts

        if diff < 0 and abs(diff) < window_seconds:
            # Before target
            if abs(diff) < best_before_diff:
                best_before_diff = abs(diff)
                prob_before = prob
                time_before = entry_time

        elif diff > 0 and diff < window_seconds:
            # After target
            if diff < best_after_diff:
                best_after_diff = diff
                prob_after = prob
                time_after = entry_time

    return prob_before, prob_after, time_before, time_after


async def main():
    print("=" * 70)
    print("METACULUS PAIR ESTIMATION v2")
    print("=" * 70)
    print(f"Rate limit: {RATE_LIMIT_DELAY}s between requests")
    print(f"Looking for pairs where X had data within {MIN_DP_WINDOW_DAYS} days of Q's resolution")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get resolved questions (these are our Q's)
        resolved_qs = await fetch_resolved_questions(client, MAX_RESOLVED_QUESTIONS)

        if not resolved_qs:
            print("No resolved questions found!")
            return

        print(f"\n{len(resolved_qs)} resolved questions to check as Q")

        # Step 2: Get a pool of candidate X questions
        print(f"\nFetching candidate X questions...")
        candidate_pool = await fetch_questions_active_around(
            client,
            datetime.now(),
            limit=MAX_CANDIDATE_X,
        )
        print(f"  Got {len(candidate_pool)} candidate X questions")

        # Step 3: For each Q, find X's that had data around Q's resolution
        print(f"\nChecking for valid pairs...")
        valid_pairs: list[ValidPair] = []
        questions_checked = 0

        # Get unique X ids from pool
        x_ids = [q["id"] for q in candidate_pool]

        # Cache X histories as we fetch them
        x_histories: dict[int, list[dict]] = {}

        for i, q in enumerate(resolved_qs[:50]):  # Limit Q's for initial test
            if i % 10 == 0:
                print(f"  [{i+1}/{min(50, len(resolved_qs))}] Q: {q.title[:40]}...")

            questions_checked += 1

            # Check each candidate X
            for x_meta in candidate_pool:
                x_id = x_meta["id"]

                # Skip if X is the same as Q
                if x_id == q.id:
                    continue

                # Fetch X's history if not cached
                if x_id not in x_histories:
                    history = await get_question_history(client, x_id)
                    x_histories[x_id] = history

                history = x_histories[x_id]

                if not history:
                    continue

                # Check if X had data around Q's resolution time
                prob_before, prob_after, time_before, time_after = find_prob_around_time(
                    history,
                    q.resolve_time,
                    window_days=MIN_DP_WINDOW_DAYS,
                )

                if prob_before is not None and prob_after is not None:
                    delta_p = prob_after - prob_before

                    # Make resolve_time naive for comparison
                    resolve_naive = q.resolve_time.replace(tzinfo=None) if q.resolve_time.tzinfo else q.resolve_time
                    days_before = (resolve_naive - time_before).total_seconds() / 86400
                    days_after = (time_after - resolve_naive).total_seconds() / 86400

                    valid_pairs.append(ValidPair(
                        q_id=q.id,
                        q_title=q.title,
                        q_resolve_time=q.resolve_time.isoformat(),
                        q_resolution=q.resolution,
                        x_id=x_id,
                        x_title=x_meta.get("title", "")[:80],
                        x_prob_before=prob_before,
                        x_prob_after=prob_after,
                        x_delta_p=delta_p,
                        days_before=days_before,
                        days_after=days_after,
                    ))

        # Summary
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nQ's checked: {questions_checked}")
        print(f"X's with cached history: {len(x_histories)}")
        print(f"Valid pairs found: {len(valid_pairs)}")

        if valid_pairs:
            # Stats on delta_p
            delta_ps = [abs(p.x_delta_p) for p in valid_pairs]
            print(f"\n|Δp| distribution:")
            print(f"  min: {min(delta_ps):.3f}")
            print(f"  max: {max(delta_ps):.3f}")
            print(f"  mean: {sum(delta_ps)/len(delta_ps):.3f}")
            print(f"  median: {sorted(delta_ps)[len(delta_ps)//2]:.3f}")

            # Pairs with notable movement
            notable = [p for p in valid_pairs if abs(p.x_delta_p) > 0.05]
            print(f"\nPairs with |Δp| > 5%: {len(notable)}")

            # Sample pairs
            print("\nSample pairs with largest |Δp|:")
            for p in sorted(valid_pairs, key=lambda x: -abs(x.x_delta_p))[:10]:
                print(f"\n  Q: {p.q_title}")
                print(f"  X: {p.x_title}")
                print(f"  Δp: {p.x_delta_p:+.1%} ({p.x_prob_before:.1%} → {p.x_prob_after:.1%})")

        # Assessment
        print("\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)

        if len(valid_pairs) >= 200:
            print("✓ SUFFICIENT pairs for LLM validation experiment")
        elif len(valid_pairs) >= 50:
            print("~ MARGINAL: Could run pilot study")
        else:
            print("✗ Need more data - try larger candidate pool or longer time window")

        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_resolved_questions": MAX_RESOLVED_QUESTIONS,
                "max_candidate_x": MAX_CANDIDATE_X,
                "min_dp_window_days": MIN_DP_WINDOW_DAYS,
                "rate_limit_delay": RATE_LIMIT_DELAY,
            },
            "summary": {
                "qs_checked": questions_checked,
                "xs_with_history": len(x_histories),
                "valid_pairs": len(valid_pairs),
                "pairs_with_notable_movement": len([p for p in valid_pairs if abs(p.x_delta_p) > 0.05]),
            },
            "pairs": [
                {
                    "q_id": p.q_id,
                    "q_title": p.q_title,
                    "q_resolve_time": p.q_resolve_time,
                    "q_resolution": p.q_resolution,
                    "x_id": p.x_id,
                    "x_title": p.x_title,
                    "x_prob_before": p.x_prob_before,
                    "x_prob_after": p.x_prob_after,
                    "x_delta_p": p.x_delta_p,
                    "days_before": p.days_before,
                    "days_after": p.days_after,
                }
                for p in sorted(valid_pairs, key=lambda x: -abs(x.x_delta_p))
            ],
        }

        output_path = OUTPUT_DIR / "metaculus_pairs_v2.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
