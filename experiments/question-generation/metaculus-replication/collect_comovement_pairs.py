#!/usr/bin/env python3
"""
Collect Metaculus pairs with historical co-movement (ρ) for VOI validation.

Goal: Find pairs (Q, X) where:
1. Q resolved at time T
2. X had overlapping probability history with Q (before Q resolved)
3. We can compute:
   - Co-movement ρ from probability changes during overlap
   - ΔP on X around Q's resolution time

This differs from estimate_pairs_v2.py which only measured ΔP.
Here we compute actual correlation from overlapping histories.

Rate limited to avoid 429 errors.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import httpx
import numpy as np
from scipy import stats

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data"
RATE_LIMIT_DELAY = 1.5  # seconds between requests
MAX_RESOLVED_QUESTIONS = 200  # Q's to fetch
MIN_OVERLAP_DAYS = 14  # Need at least this much overlapping history
MIN_SHARED_POINTS = 10  # Need at least this many shared data points
MAX_RETRIES = 3


@dataclass
class QuestionHistory:
    """A question with its probability history."""
    id: int
    title: str
    resolve_time: datetime | None  # None if still open
    resolution: float | None  # 0, 1, or None
    history: list[tuple[datetime, float]]  # (timestamp, probability)


@dataclass
class ComovementPair:
    """A (Q, X) pair with co-movement data."""
    q_id: int
    q_title: str
    q_resolve_time: str
    q_resolution: float
    x_id: int
    x_title: str
    x_prob_before: float
    x_prob_after: float
    x_delta_p: float
    rho: float  # Co-movement correlation
    overlap_days: float
    shared_points: int
    days_before: float
    days_after: float


async def fetch_with_rate_limit(
    client: httpx.AsyncClient,
    url: str,
    params: dict = None,
    delay: float = RATE_LIMIT_DELAY,
    headers: dict = None,
) -> dict | None:
    """Fetch with rate limiting and exponential backoff."""
    await asyncio.sleep(delay)

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 429:
                wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s
                print(f"  Rate limited (429), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  Error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(5)

    print(f"  Failed after {MAX_RETRIES} attempts")
    return None


def get_auth_headers() -> dict:
    """Get authorization headers if API key is available."""
    api_key = os.environ.get("METACULUS_API_KEY")
    if api_key:
        return {"Authorization": f"Token {api_key}"}
    return {}


async def fetch_resolved_questions(
    client: httpx.AsyncClient,
    limit: int,
    headers: dict,
) -> list[dict]:
    """Fetch resolved binary questions ordered by forecast count."""
    print(f"Fetching {limit} resolved questions...")

    url = "https://www.metaculus.com/api/posts/"
    params = {
        "limit": limit,
        "forecast_type": "binary",
        "statuses": "resolved",
        "order_by": "-forecasts_count",
    }

    data = await fetch_with_rate_limit(client, url, params, delay=0.5, headers=headers)
    if not data:
        return []

    print(f"  Got {len(data.get('results', []))} resolved questions")
    return data.get("results", [])


async def fetch_question_detail(
    client: httpx.AsyncClient,
    question_id: int,
    headers: dict,
) -> QuestionHistory | None:
    """Fetch full detail for a question including history."""
    url = f"https://www.metaculus.com/api/posts/{question_id}/"
    data = await fetch_with_rate_limit(client, url, headers=headers)

    if not data:
        return None

    # Extract history
    history_data = (data
        .get("question", {})
        .get("aggregations", {})
        .get("recency_weighted", {})
        .get("history", []))

    if not history_data:
        return None

    # Convert history to (datetime, probability) tuples
    history = []
    for entry in history_data:
        ts = entry.get("start_time", 0)
        centers = entry.get("centers", [])
        if ts and centers:
            dt = datetime.fromtimestamp(ts)
            prob = centers[0]
            history.append((dt, prob))

    if not history:
        return None

    # Sort by time
    history.sort(key=lambda x: x[0])

    # Get resolution info
    resolve_time_str = data.get("actual_resolve_time")
    resolve_time = None
    if resolve_time_str:
        resolve_time = datetime.fromisoformat(resolve_time_str.replace("Z", "+00:00"))
        # Make naive for comparison
        if resolve_time.tzinfo:
            resolve_time = resolve_time.replace(tzinfo=None)

    resolution = data.get("question", {}).get("resolution")
    res_float = None
    if resolution in [0, 1, 0.0, 1.0]:
        res_float = float(resolution)
    elif resolution in ["yes", "Yes", "YES"]:
        res_float = 1.0
    elif resolution in ["no", "No", "NO"]:
        res_float = 0.0

    return QuestionHistory(
        id=question_id,
        title=data.get("title", "")[:80],
        resolve_time=resolve_time,
        resolution=res_float,
        history=history,
    )


def interpolate_to_daily(
    history: list[tuple[datetime, float]],
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float]]:
    """Interpolate probability history to daily values."""
    if not history:
        return []

    # Create daily timestamps
    current = start
    daily_points = []

    while current <= end:
        # Find closest point at or before current
        prob = None
        for dt, p in history:
            if dt <= current:
                prob = p
            else:
                break

        if prob is not None:
            daily_points.append((current, prob))

        current += timedelta(days=1)

    return daily_points


def compute_comovement(
    q_history: list[tuple[datetime, float]],
    x_history: list[tuple[datetime, float]],
    q_resolve_time: datetime,
) -> tuple[float | None, int, float]:
    """
    Compute co-movement correlation from overlapping histories.

    Uses probability CHANGES (diff) to compute correlation, not raw levels.
    Only uses data from BEFORE Q resolves to avoid information leakage.

    Returns: (rho, shared_points, overlap_days)
    """
    if not q_history or not x_history:
        return None, 0, 0.0

    # Find overlap period (before Q resolves)
    q_start = q_history[0][0]
    q_end = min(q_history[-1][0], q_resolve_time - timedelta(days=1))
    x_start = x_history[0][0]
    x_end = x_history[-1][0]

    overlap_start = max(q_start, x_start)
    overlap_end = min(q_end, x_end)

    if overlap_start >= overlap_end:
        return None, 0, 0.0

    overlap_days = (overlap_end - overlap_start).days

    # Interpolate both to daily
    q_daily = interpolate_to_daily(q_history, overlap_start, overlap_end)
    x_daily = interpolate_to_daily(x_history, overlap_start, overlap_end)

    if len(q_daily) < 2 or len(x_daily) < 2:
        return None, 0, overlap_days

    # Align by date
    q_dict = {dt.date(): prob for dt, prob in q_daily}
    x_dict = {dt.date(): prob for dt, prob in x_daily}

    shared_dates = sorted(set(q_dict.keys()) & set(x_dict.keys()))
    if len(shared_dates) < MIN_SHARED_POINTS:
        return None, len(shared_dates), overlap_days

    # Get aligned series
    q_probs = [q_dict[d] for d in shared_dates]
    x_probs = [x_dict[d] for d in shared_dates]

    # Compute changes (differences)
    q_changes = np.diff(q_probs)
    x_changes = np.diff(x_probs)

    if len(q_changes) < MIN_SHARED_POINTS - 1:
        return None, len(shared_dates), overlap_days

    # Compute Pearson correlation on changes
    # Handle constant series (no variance)
    if np.std(q_changes) < 1e-10 or np.std(x_changes) < 1e-10:
        return None, len(shared_dates), overlap_days

    rho, _ = stats.pearsonr(q_changes, x_changes)

    if np.isnan(rho):
        return None, len(shared_dates), overlap_days

    return rho, len(shared_dates), overlap_days


def find_prob_around_time(
    history: list[tuple[datetime, float]],
    target_time: datetime,
    window_days: int = 3,
) -> tuple[float | None, float | None, datetime | None, datetime | None]:
    """
    Find probability before and after target_time.

    Returns (prob_before, prob_after, time_before, time_after)
    """
    if not history:
        return None, None, None, None

    window = timedelta(days=window_days)

    # Find closest entry before target
    prob_before = None
    time_before = None
    best_before_diff = timedelta.max

    # Find closest entry after target
    prob_after = None
    time_after = None
    best_after_diff = timedelta.max

    for dt, prob in history:
        diff = dt - target_time

        if diff < timedelta(0) and abs(diff) < window:
            # Before target
            if abs(diff) < best_before_diff:
                best_before_diff = abs(diff)
                prob_before = prob
                time_before = dt

        elif diff > timedelta(0) and diff < window:
            # After target
            if diff < best_after_diff:
                best_after_diff = diff
                prob_after = prob
                time_after = dt

    return prob_before, prob_after, time_before, time_after


async def main():
    print("=" * 70)
    print("METACULUS CO-MOVEMENT PAIR COLLECTION")
    print("=" * 70)
    print(f"Rate limit: {RATE_LIMIT_DELAY}s between requests")
    print(f"Min overlap: {MIN_OVERLAP_DAYS} days, {MIN_SHARED_POINTS} shared points")

    headers = get_auth_headers()
    if headers:
        print("Using API key authentication")
    else:
        print("No API key - using unauthenticated access (stricter rate limits)")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get resolved questions (these are our Q's)
        resolved_list = await fetch_resolved_questions(
            client, MAX_RESOLVED_QUESTIONS, headers
        )

        if not resolved_list:
            print("No resolved questions found!")
            return

        print(f"\n{len(resolved_list)} resolved questions to process")

        # Step 2: Fetch full details for resolved questions
        print("\nFetching question details...")
        resolved_qs: list[QuestionHistory] = []

        for i, q_meta in enumerate(resolved_list):
            if i % 20 == 0:
                print(f"  [{i+1}/{len(resolved_list)}] Fetching details...")

            q_detail = await fetch_question_detail(client, q_meta["id"], headers)
            if q_detail and q_detail.resolve_time and q_detail.resolution is not None:
                if len(q_detail.history) >= MIN_SHARED_POINTS:
                    resolved_qs.append(q_detail)

        print(f"  Got {len(resolved_qs)} resolved questions with history")

        # Step 3: For each Q, find X's with overlapping history
        print("\nBuilding candidate X pool...")

        # Fetch open/recently resolved questions as candidates
        url = "https://www.metaculus.com/api/posts/"
        params = {
            "limit": 300,
            "forecast_type": "binary",
            "order_by": "-forecasts_count",
        }

        data = await fetch_with_rate_limit(client, url, params, headers=headers)
        candidate_metas = data.get("results", []) if data else []
        print(f"  Got {len(candidate_metas)} candidate X questions")

        # Fetch details for candidates
        x_details: dict[int, QuestionHistory] = {}
        for i, x_meta in enumerate(candidate_metas):
            if i % 20 == 0:
                print(f"  [{i+1}/{len(candidate_metas)}] Fetching X details...")

            x_id = x_meta["id"]
            if x_id not in x_details:
                x_detail = await fetch_question_detail(client, x_id, headers)
                if x_detail and len(x_detail.history) >= MIN_SHARED_POINTS:
                    x_details[x_id] = x_detail

        print(f"  Got {len(x_details)} X questions with history")

        # Step 4: Compute co-movement for all (Q, X) pairs
        print("\nComputing co-movement pairs...")
        valid_pairs: list[ComovementPair] = []

        for i, q in enumerate(resolved_qs):
            if i % 10 == 0:
                print(f"  [{i+1}/{len(resolved_qs)}] Q: {q.title[:40]}...")

            for x_id, x in x_details.items():
                # Skip if same question
                if x_id == q.id:
                    continue

                # Compute co-movement
                rho, shared_points, overlap_days = compute_comovement(
                    q.history, x.history, q.resolve_time
                )

                if rho is None or overlap_days < MIN_OVERLAP_DAYS:
                    continue

                # Compute ΔP around resolution
                prob_before, prob_after, time_before, time_after = find_prob_around_time(
                    x.history, q.resolve_time, window_days=3
                )

                if prob_before is None or prob_after is None:
                    continue

                delta_p = prob_after - prob_before
                days_before = (q.resolve_time - time_before).total_seconds() / 86400
                days_after = (time_after - q.resolve_time).total_seconds() / 86400

                valid_pairs.append(ComovementPair(
                    q_id=q.id,
                    q_title=q.title,
                    q_resolve_time=q.resolve_time.isoformat(),
                    q_resolution=q.resolution,
                    x_id=x_id,
                    x_title=x.title,
                    x_prob_before=prob_before,
                    x_prob_after=prob_after,
                    x_delta_p=delta_p,
                    rho=rho,
                    overlap_days=overlap_days,
                    shared_points=shared_points,
                    days_before=days_before,
                    days_after=days_after,
                ))

        # Summary
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nQ's processed: {len(resolved_qs)}")
        print(f"X's with history: {len(x_details)}")
        print(f"Valid pairs found: {len(valid_pairs)}")

        if valid_pairs:
            # Stats on rho
            rhos = [p.rho for p in valid_pairs]
            print(f"\nρ distribution:")
            print(f"  min: {min(rhos):.3f}")
            print(f"  max: {max(rhos):.3f}")
            print(f"  mean: {np.mean(rhos):.3f}")
            print(f"  median: {np.median(rhos):.3f}")

            # Stats on delta_p
            delta_ps = [abs(p.x_delta_p) for p in valid_pairs]
            print(f"\n|ΔP| distribution:")
            print(f"  min: {min(delta_ps):.3f}")
            print(f"  max: {max(delta_ps):.3f}")
            print(f"  mean: {np.mean(delta_ps):.3f}")
            print(f"  median: {np.median(delta_ps):.3f}")

            # Pairs with notable movement
            notable = [p for p in valid_pairs if abs(p.x_delta_p) > 0.05]
            print(f"\nPairs with |ΔP| > 5%: {len(notable)}")

            # High correlation pairs
            high_rho = [p for p in valid_pairs if abs(p.rho) > 0.3]
            print(f"Pairs with |ρ| > 0.3: {len(high_rho)}")

            # Sample pairs
            print("\nSample pairs with highest |ρ|:")
            for p in sorted(valid_pairs, key=lambda x: -abs(x.rho))[:5]:
                print(f"\n  Q: {p.q_title}")
                print(f"  X: {p.x_title}")
                print(f"  ρ: {p.rho:+.3f}, ΔP: {p.x_delta_p:+.1%}")

        # Assessment
        print("\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)

        if len(valid_pairs) >= 100:
            print("✓ SUFFICIENT pairs for VOI validation experiment")
        elif len(valid_pairs) >= 50:
            print("~ MARGINAL: Could run pilot study")
        else:
            print("✗ Need more data - try expanding question pool")

        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_resolved_questions": MAX_RESOLVED_QUESTIONS,
                "min_overlap_days": MIN_OVERLAP_DAYS,
                "min_shared_points": MIN_SHARED_POINTS,
                "rate_limit_delay": RATE_LIMIT_DELAY,
            },
            "summary": {
                "qs_processed": len(resolved_qs),
                "xs_with_history": len(x_details),
                "valid_pairs": len(valid_pairs),
                "pairs_with_notable_movement": len([p for p in valid_pairs if abs(p.x_delta_p) > 0.05]),
                "pairs_with_high_rho": len([p for p in valid_pairs if abs(p.rho) > 0.3]),
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
                    "rho": p.rho,
                    "overlap_days": p.overlap_days,
                    "shared_points": p.shared_points,
                    "days_before": p.days_before,
                    "days_after": p.days_after,
                }
                for p in sorted(valid_pairs, key=lambda x: -abs(x.rho))
            ],
        }

        output_path = OUTPUT_DIR / "metaculus_comovement_pairs.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
