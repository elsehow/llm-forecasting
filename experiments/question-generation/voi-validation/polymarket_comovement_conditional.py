"""Polymarket Comovement Conditional Experiment.

Replicate the Metaculus closed-question conditional experiment on Polymarket data
using the H1 hybrid prompt with confidence gating.

Design:
1. Fetch all resolved binary markets from Polymarket
2. Get price histories for each resolved market
3. Compute comovement (Pearson correlation of daily price CHANGES) between pairs
4. Build pairs with ground truth (both markets resolved)
5. Run H1+confidence prompt for conditional estimation
6. Compute Brier scores vs market baseline

This tests if LLM conditional forecasting (H1 prompt) beats market baseline
on Polymarket pairs, matching the Metaculus result (+0.064 overall improvement).

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/polymarket_comovement_conditional.py
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env from monorepo root before importing llm_forecasting
_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import numpy as np
from scipy import stats

from llm_forecasting.market_data.polymarket import PolymarketData
from llm_forecasting.market_data.models import Market, MarketStatus, PricePoint
from llm_forecasting.market_data.storage import MarketDataStorage

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "results"
PAIRS_FILE = OUTPUT_DIR / "polymarket_comovement_pairs.json"
RESULTS_FILE = OUTPUT_DIR / "polymarket_comovement_conditional_results.json"
DB_PATH = _monorepo_root / "forecastbench.db"

# Model
MODEL = "claude-opus-4-5-20251101"

# Minimum requirements for valid pairs
MIN_OVERLAP_DAYS = 14
MIN_SHARED_POINTS = 10

# H1 + Confidence Gating Prompt (same as run_h1_confidence_full.py)
PROMPT_H1_CONFIDENCE = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

PHASE 1 - INDEPENDENCE TEST (be skeptical):
Most prediction market question pairs are UNRELATED. Default assumption: INDEPENDENT.

To claim a relationship, you need a SPECIFIC mechanism - not just topical similarity.
Ask: Would a rational bettor change their X position by >5% after learning Q's outcome?

PHASE 2 - CONFIDENCE CHECK:
Rate your confidence (0-100) that a genuine causal/logical relationship exists:
- 0-50: Probably unrelated, any connection is speculative
- 51-79: Possible connection but uncertain
- 80-100: Strong evidence of direct relationship

RULE: Only claim "related" if confidence >= 80.

PHASE 3 - IF RELATED (confidence >= 80), estimate magnitude:
- How much would knowing Q shift your belief about X? (0.05 to 0.5)
- Does Q=YES make X more likely (positive) or less likely (negative)?

Respond with JSON only:
{{"confidence": <int 0-100>, "is_related": true | false, "mechanism": "<specific mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float, 0 if confidence < 80>, "direction": "positive" | "negative" | "none"}}"""


def parse_response(text: str) -> tuple[float, float, float, dict]:
    """Parse H1+confidence response. Returns (p_yes, p_no, shift, raw_data)."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        confidence = int(data.get("confidence", 0))
        base = float(data.get("base_p_x", 0.5))
        shift = float(data.get("shift_magnitude", 0))
        direction = data.get("direction", "none")
        is_related = data.get("is_related", False)

        # Enforce confidence threshold
        if confidence < 80 or not is_related:
            return base, base, 0, data

        if direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


async def fetch_resolved_markets(storage: MarketDataStorage | None = None) -> list[Market]:
    """Fetch all resolved binary markets from Polymarket.

    First checks database cache, then fetches from API and saves to cache.
    """
    print("\n[Phase 1] Fetching resolved markets from Polymarket...")

    # Check database cache first
    if storage:
        print("  Checking database cache...")
        cached = await storage.get_markets(
            platform="polymarket",
            status=MarketStatus.RESOLVED,
        )
        if cached:
            # Filter to those with resolved_value
            cached = [m for m in cached if m.resolved_value is not None]
            if len(cached) >= 100:  # Use cache if we have substantial data
                print(f"  Found {len(cached)} resolved markets in cache")
                return cached
            print(f"  Cache has only {len(cached)} markets, fetching fresh data...")

    provider = PolymarketData()

    # Fetch markets (including closed/resolved)
    # We need to fetch in multiple batches since API may have limits
    all_markets = []

    # Fetch active=false to get closed/resolved markets
    print("  Fetching closed/resolved markets from API...")

    # Direct API call for resolved markets since the provider defaults to active_only
    import httpx
    from llm_forecasting.market_data.polymarket import GAMMA_API_URL

    async with httpx.AsyncClient(timeout=60.0) as client:
        offset = 0
        batch_size = 100

        while True:
            params = {
                "limit": batch_size,
                "offset": offset,
                "closed": "true",  # Get closed markets
                "order": "volume24hr",
                "ascending": "false",
            }

            try:
                response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
                response.raise_for_status()
                batch = response.json()

                if not batch:
                    break

                for raw in batch:
                    market = provider._parse_market(raw)
                    if market and market.status == MarketStatus.RESOLVED:
                        if market.resolved_value is not None:
                            all_markets.append(market)

                print(f"    Fetched {offset + len(batch)} markets, {len(all_markets)} resolved so far...")

                if len(batch) < batch_size:
                    break

                offset += batch_size

                # Don't fetch too many
                if offset >= 2000:
                    break

            except Exception as e:
                print(f"    Error at offset {offset}: {e}")
                break

    print(f"  Found {len(all_markets)} resolved binary markets")

    # Save to database cache
    if storage and all_markets:
        print(f"  Saving {len(all_markets)} markets to database cache...")
        await storage.save_markets(all_markets)

    return all_markets


async def fetch_all_histories(
    markets: list[Market],
    provider: PolymarketData,
    storage: MarketDataStorage | None = None,
) -> dict[str, list[PricePoint]]:
    """Fetch price history for each resolved market.

    First checks database cache, then fetches from API and saves to cache.
    """
    print(f"\n[Phase 2] Fetching price histories for {len(markets)} markets...")

    histories = {}
    skipped = 0
    from_cache = 0
    from_api = 0

    for i, market in enumerate(markets):
        if i > 0 and i % 20 == 0:
            print(f"  Processed {i}/{len(markets)} markets, {len(histories)} with valid history (cache: {from_cache}, api: {from_api})...")

        if not market.clob_token_ids:
            skipped += 1
            continue

        # Check database cache first
        if storage:
            try:
                cached_candles = await storage.get_price_history(
                    platform="polymarket",
                    market_id=market.id,
                )
                if cached_candles and len(cached_candles) >= MIN_OVERLAP_DAYS:
                    # Convert Candle to PricePoint for compatibility
                    histories[market.id] = [
                        PricePoint(
                            market_id=c.market_id,
                            platform=c.platform,
                            timestamp=c.timestamp,
                            price=c.close,  # Use close price
                        )
                        for c in cached_candles
                    ]
                    from_cache += 1
                    continue
            except Exception:
                pass  # Fall through to API fetch

        try:
            # Fetch 1 year of history to maximize overlap
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=365)

            history = await provider.fetch_price_history(
                market.id,
                start=start,
                end=end,
                interval="1d",
            )

            if len(history) >= MIN_OVERLAP_DAYS:
                histories[market.id] = history
                from_api += 1

                # Save to database cache
                if storage:
                    try:
                        await storage.save_price_history(
                            market_id=market.id,
                            platform="polymarket",
                            candles=history,
                        )
                    except Exception:
                        pass  # Don't fail on cache write errors

        except Exception as e:
            # Silently skip failures
            pass

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.05)

    print(f"  Got price history for {len(histories)} markets (cache: {from_cache}, api: {from_api}, skipped: {skipped})")
    return histories


def compute_comovement(
    q_history: list[PricePoint],
    x_history: list[PricePoint],
    q_resolve_time: datetime | None,
) -> tuple[float | None, int, float]:
    """Compute Pearson correlation of daily price CHANGES.

    Only uses data BEFORE Q resolves.

    Returns:
        Tuple of (rho, shared_points, overlap_days)
        rho is None if insufficient data
    """
    if not q_history or not x_history:
        return None, 0, 0

    # Convert to daily prices dict
    def to_daily_dict(history: list[PricePoint]) -> dict[str, float]:
        daily = {}
        for p in history:
            day = p.timestamp.strftime("%Y-%m-%d")
            daily[day] = p.price
        return daily

    q_daily = to_daily_dict(q_history)
    x_daily = to_daily_dict(x_history)

    # Find overlapping days (before Q resolves)
    if q_resolve_time:
        cutoff = q_resolve_time.strftime("%Y-%m-%d")
    else:
        cutoff = "9999-12-31"

    common_days = sorted(
        d for d in q_daily.keys()
        if d in x_daily and d < cutoff
    )

    if len(common_days) < MIN_OVERLAP_DAYS:
        return None, len(common_days), len(common_days)

    # Compute daily changes
    q_changes = []
    x_changes = []

    for i in range(1, len(common_days)):
        prev_day = common_days[i - 1]
        curr_day = common_days[i]

        q_change = q_daily[curr_day] - q_daily[prev_day]
        x_change = x_daily[curr_day] - x_daily[prev_day]

        q_changes.append(q_change)
        x_changes.append(x_change)

    if len(q_changes) < MIN_SHARED_POINTS:
        return None, len(q_changes), len(common_days)

    # Compute Pearson correlation
    try:
        rho, _ = stats.pearsonr(q_changes, x_changes)
        if np.isnan(rho):
            return None, len(q_changes), len(common_days)
        return float(rho), len(q_changes), len(common_days)
    except Exception:
        return None, len(q_changes), len(common_days)


def get_price_at_time(
    history: list[PricePoint],
    target_time: datetime,
) -> float | None:
    """Get the price closest to (but before) the target time."""
    if not history:
        return None

    # Ensure target_time is timezone-aware
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    valid_points = [p for p in history if p.timestamp <= target_time]
    if not valid_points:
        # Fall back to earliest available
        return history[0].price if history else None

    # Return most recent before target
    return max(valid_points, key=lambda p: p.timestamp).price


def build_pairs(
    markets: list[Market],
    histories: dict[str, list[PricePoint]],
) -> list[dict]:
    """Build all valid pairs with comovement stats."""
    print(f"\n[Phase 3] Building comovement pairs from {len(markets)} markets...")

    # Filter to markets with history
    markets_with_history = [m for m in markets if m.id in histories]
    print(f"  Markets with history: {len(markets_with_history)}")

    pairs = []
    checked = 0

    for i, q in enumerate(markets_with_history):
        for x in markets_with_history:
            if q.id == x.id:
                continue

            checked += 1

            # Get resolution time for Q
            q_resolve_time = None
            if q.resolution_date:
                q_resolve_time = datetime.combine(
                    q.resolution_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )

            rho, points, days = compute_comovement(
                histories[q.id],
                histories[x.id],
                q_resolve_time,
            )

            if rho is not None:
                # Get x_prob_before (X's price when Q resolved)
                x_prob_before = get_price_at_time(
                    histories[x.id],
                    q_resolve_time or datetime.now(timezone.utc),
                )

                if x_prob_before is None:
                    continue

                pairs.append({
                    "q_id": q.id,
                    "q_title": q.title,
                    "q_resolution": q.resolved_value,  # 1.0 or 0.0
                    "q_resolve_date": q.resolution_date.isoformat() if q.resolution_date else None,
                    "x_id": x.id,
                    "x_title": x.title,
                    "x_resolution": x.resolved_value,  # 1.0 or 0.0
                    "x_prob_before": x_prob_before,
                    "rho": rho,
                    "overlap_days": days,
                    "shared_points": points,
                })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(markets_with_history)} Q markets, {len(pairs)} valid pairs...")

    print(f"  Built {len(pairs)} valid pairs from {checked} comparisons")
    return pairs


def sample_pairs(pairs: list[dict], n: int = 50) -> list[dict]:
    """Sample balanced high-rho and low-rho pairs."""
    high_rho = [p for p in pairs if abs(p["rho"]) > 0.5]
    low_rho = [p for p in pairs if abs(p["rho"]) <= 0.5]

    print(f"\n[Phase 4] Sampling pairs...")
    print(f"  High |rho| > 0.5: {len(high_rho)} pairs")
    print(f"  Low |rho| <= 0.5: {len(low_rho)} pairs")

    # Sample ~half from each category
    n_high = min(n // 2, len(high_rho))
    n_low = min(n - n_high, len(low_rho))

    random.seed(42)  # Reproducibility
    sampled = random.sample(high_rho, n_high) if n_high > 0 else []
    sampled += random.sample(low_rho, n_low) if n_low > 0 else []

    print(f"  Sampled {len(sampled)} pairs ({n_high} high-rho, {n_low} low-rho)")
    return sampled


async def run_h1_on_pair(
    semaphore: asyncio.Semaphore,
    pair: dict,
    pair_idx: int,
) -> dict:
    """Run H1+confidence prompt on a single pair."""
    import litellm

    async with semaphore:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": PROMPT_H1_CONFIDENCE.format(
                        question_q=pair["q_title"],
                        question_x=pair["x_title"],
                    )
                }],
                max_tokens=600,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no, shift, raw = parse_response(text)
        except Exception as e:
            text = str(e)
            p_yes, p_no, shift, raw = 0.5, 0.5, 0, {"error": str(e)}

        # Determine Q outcome and select appropriate conditional
        q_outcome = pair["q_resolution"] > 0.5
        p_conditional = p_yes if q_outcome else p_no

        # X outcome
        x_actual = pair["x_resolution"]

        # Brier scores
        brier = (p_conditional - x_actual) ** 2
        baseline = pair["x_prob_before"]
        brier_baseline = (baseline - x_actual) ** 2

        # Confidence check
        confidence = raw.get("confidence", 0)
        is_related = raw.get("is_related", False) and confidence >= 80

        return {
            "q_id": pair["q_id"],
            "x_id": pair["x_id"],
            "question_q": pair["q_title"][:80],
            "question_x": pair["x_title"][:80],
            "q_outcome": q_outcome,
            "x_outcome": x_actual,
            "rho": pair["rho"],
            "x_prob_before": baseline,
            # H1 outputs
            "p_x_given_q_yes": p_yes,
            "p_x_given_q_no": p_no,
            "p_conditional": p_conditional,
            "spread": abs(p_yes - p_no),
            "confidence": confidence,
            "is_related": is_related,
            "mechanism": raw.get("mechanism", ""),
            # Brier scores
            "brier_h1": brier,
            "brier_baseline": brier_baseline,
            "improvement": brier_baseline - brier,
        }


async def run_experiment(pairs: list[dict]) -> list[dict]:
    """Run H1+confidence experiment on all pairs."""
    print(f"\n[Phase 5] Running H1+confidence on {len(pairs)} pairs...")
    print(f"  Model: {MODEL}")

    semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls

    tasks = [
        run_h1_on_pair(semaphore, pair, i)
        for i, pair in enumerate(pairs)
    ]

    results = []
    batch_size = 10

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"  Processed {min(i + batch_size, len(tasks))}/{len(tasks)}...")

    return results


def evaluate(results: list[dict]) -> dict:
    """Compute summary statistics and compare to baseline."""
    if not results:
        return {"error": "No results"}

    briers_h1 = [r["brier_h1"] for r in results]
    briers_baseline = [r["brier_baseline"] for r in results]
    improvements = [r["improvement"] for r in results]
    spreads = [r["spread"] for r in results]
    confidences = [r["confidence"] for r in results]

    # Overall
    overall = {
        "n": len(results),
        "mean_brier_h1": float(np.mean(briers_h1)),
        "mean_brier_baseline": float(np.mean(briers_baseline)),
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements),
        "mean_spread": float(np.mean(spreads)),
        "mean_confidence": float(np.mean(confidences)),
        "pct_related": sum(1 for r in results if r["is_related"]) / len(results),
    }

    # t-test
    if len(improvements) >= 3:
        t, p = stats.ttest_1samp(improvements, 0)
        overall["t_stat"] = float(t)
        overall["p_value"] = float(p)

    # By rho category
    high_rho = [r for r in results if abs(r.get("rho", 0) or 0) > 0.5]
    low_rho = [r for r in results if abs(r.get("rho", 0) or 0) <= 0.5]

    def summarize_subset(subset: list[dict], name: str) -> dict:
        if not subset:
            return {"n": 0}
        impr = [r["improvement"] for r in subset]
        return {
            "n": len(subset),
            "mean_improvement": float(np.mean(impr)),
            "pct_improved": sum(1 for i in impr if i > 0) / len(impr),
            "pct_related": sum(1 for r in subset if r["is_related"]) / len(subset),
            "mean_confidence": float(np.mean([r["confidence"] for r in subset])),
        }

    return {
        "overall": overall,
        "high_rho": summarize_subset(high_rho, "high_rho"),
        "low_rho": summarize_subset(low_rho, "low_rho"),
    }


def make_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


async def main():
    """Run the Polymarket comovement conditional experiment."""
    print("=" * 80)
    print("POLYMARKET COMOVEMENT CONDITIONAL EXPERIMENT")
    print("=" * 80)
    print("Goal: Replicate Metaculus H1+confidence results on Polymarket data")

    # Initialize database storage for caching
    storage = MarketDataStorage(db_path=DB_PATH)
    print(f"\nUsing database cache: {DB_PATH}")

    # Phase 1: Fetch resolved markets
    markets = await fetch_resolved_markets(storage=storage)

    if len(markets) < 10:
        print("\nERROR: Not enough resolved markets found. Exiting.")
        return

    # Phase 2: Get price histories
    provider = PolymarketData()
    histories = await fetch_all_histories(markets, provider, storage=storage)

    if len(histories) < 10:
        print("\nERROR: Not enough price histories available. Exiting.")
        return

    # Phase 3: Build pairs
    all_pairs = build_pairs(markets, histories)

    if len(all_pairs) < 10:
        print("\nERROR: Not enough valid pairs found. Exiting.")
        return

    # Save all pairs for reference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_FILE, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "n_markets": len(markets),
            "n_with_history": len(histories),
            "n_pairs": len(all_pairs),
            "pairs": make_json_serializable(all_pairs),
        }, f, indent=2)
    print(f"\n  Saved {len(all_pairs)} pairs to {PAIRS_FILE}")

    # Phase 4: Sample for experiment
    sampled = sample_pairs(all_pairs, n=50)

    if len(sampled) < 10:
        print("\nWARNING: Sample size is small. Results may not be statistically meaningful.")

    # Phase 5: Run H1+confidence
    results = await run_experiment(sampled)

    # Phase 6: Evaluate
    summary = evaluate(results)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    overall = summary["overall"]
    print(f"\n--- Overall (n={overall['n']}) ---")
    print(f"  Mean Brier (H1+Conf):    {overall['mean_brier_h1']:.4f}")
    print(f"  Mean Brier (baseline):   {overall['mean_brier_baseline']:.4f}")
    print(f"  Mean improvement:        {overall['mean_improvement']:+.4f}")
    print(f"  % improved:              {100*overall['pct_improved']:.1f}%")
    print(f"  Mean spread:             {overall['mean_spread']:.3f}")
    print(f"  Mean confidence:         {overall['mean_confidence']:.1f}")
    print(f"  % detected related:      {100*overall['pct_related']:.1f}%")

    if "p_value" in overall:
        sig = "***" if overall["p_value"] < 0.001 else "**" if overall["p_value"] < 0.01 else "*" if overall["p_value"] < 0.05 else ""
        print(f"  t-test (improvement>0):  t={overall['t_stat']:.2f}, p={overall['p_value']:.4f} {sig}")

    high_rho = summary["high_rho"]
    low_rho = summary["low_rho"]

    print(f"\n--- High |rho| > 0.5 (n={high_rho['n']}) ---")
    if high_rho["n"] > 0:
        print(f"  Mean improvement:        {high_rho['mean_improvement']:+.4f}")
        print(f"  % improved:              {100*high_rho['pct_improved']:.1f}%")
        print(f"  % detected related:      {100*high_rho['pct_related']:.1f}%")

    print(f"\n--- Low |rho| <= 0.5 (n={low_rho['n']}) ---")
    if low_rho["n"] > 0:
        print(f"  Mean improvement:        {low_rho['mean_improvement']:+.4f}")
        print(f"  % improved:              {100*low_rho['pct_improved']:.1f}%")
        print(f"  % independent (not rel): {100*(1-low_rho['pct_related']):.1f}%")

    # Comparison to Metaculus
    print("\n--- Comparison to Metaculus H1+Confidence (n=598) ---")
    print("  | Metric              | Metaculus | Polymarket |")
    print("  |---------------------|-----------|------------|")
    print(f"  | High-rho improvement| +0.078    | {high_rho.get('mean_improvement', 0):+.3f}      |")
    print(f"  | Low-rho improvement | -0.019    | {low_rho.get('mean_improvement', 0):+.3f}      |")
    print(f"  | Overall improvement | -0.015    | {overall['mean_improvement']:+.3f}      |")
    fp_rate = 1 - low_rho.get("pct_related", 1) if low_rho["n"] > 0 else 0
    # Actually false positive = detected related when low rho
    fp_rate = low_rho.get("pct_related", 0) if low_rho["n"] > 0 else 0
    print(f"  | False positive rate | 4.9%      | {100*fp_rate:.1f}%       |")

    # Save results
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "prompt": "H1_confidence_gating",
            "confidence_threshold": 80,
            "n_markets": len(markets),
            "n_with_history": len(histories),
            "n_total_pairs": len(all_pairs),
            "n_sampled": len(sampled),
        },
        "summary": summary,
        "results": results,
    }

    output = make_json_serializable(output)

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
