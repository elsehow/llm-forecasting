#!/usr/bin/env python3
"""
Step 1: Select ultimate markets for LLM crux validation.

Filters markets.json to find ultimates that:
- Have end_date > Feb 2026 (time to observe crux resolution)
- Have volume_total > $100k (real interest)
- Are at moderate probability (20-80%) — room to move
- Span diverse topics

Usage:
    uv run python experiments/question-generation/llm-crux-validation/select_ultimates.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import random

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting" / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selection criteria
MIN_VOLUME = 100_000  # $100k minimum volume
MIN_PROB = 0.20  # 20% minimum probability
MAX_PROB = 0.80  # 80% maximum probability
MIN_END_DATE = datetime(2026, 2, 1, tzinfo=timezone.utc)  # End date > Feb 2026
TARGET_COUNT = 50  # Target number of ultimates

# Topic keywords for diversity tracking
TOPIC_KEYWORDS = {
    "politics_us": ["trump", "biden", "republican", "democrat", "congress", "senate", "election", "president"],
    "politics_intl": ["iran", "russia", "ukraine", "china", "israel", "gaza", "nato", "war"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "xrp"],
    "finance": ["fed", "interest rate", "s&p", "gdp", "inflation", "recession"],
    "sports": ["nba", "nfl", "super bowl", "world cup", "champions league", "premier league"],
    "tech": ["ai", "openai", "gpt", "google", "apple", "tesla", "meta"],
    "entertainment": ["oscar", "grammy", "emmy", "movie", "film"],
}


def detect_topic(question: str) -> str:
    """Detect primary topic from question text."""
    q_lower = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return topic
    return "other"


def get_current_price(condition_id: str) -> float | None:
    """Get current price from price history (last candle close)."""
    price_file = PRICE_HISTORY_DIR / f"{condition_id[:40]}.json"
    if not price_file.exists():
        return None

    try:
        with open(price_file) as f:
            data = json.load(f)
        candles = data.get("candles", [])
        if candles:
            return candles[-1]["close"]
    except Exception:
        pass
    return None


def main():
    print("=" * 70)
    print("SELECT ULTIMATE MARKETS FOR LLM CRUX VALIDATION")
    print("=" * 70)
    print(f"\nCriteria:")
    print(f"  Volume: >${MIN_VOLUME:,}")
    print(f"  Probability: {MIN_PROB*100:.0f}% - {MAX_PROB*100:.0f}%")
    print(f"  End date: >{MIN_END_DATE.date()}")
    print(f"  Target count: {TARGET_COUNT}")

    # Load markets
    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)
    print(f"\nLoaded {len(markets)} markets")

    # Filter markets
    candidates = []
    filter_stats = {
        "total": len(markets),
        "volume_fail": 0,
        "end_date_fail": 0,
        "probability_fail": 0,
        "no_price": 0,
        "passed": 0,
    }

    for market in markets:
        # Volume filter
        if market.get("volume_total", 0) < MIN_VOLUME:
            filter_stats["volume_fail"] += 1
            continue

        # End date filter
        end_date_str = market.get("end_date")
        if not end_date_str:
            filter_stats["end_date_fail"] += 1
            continue

        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            if end_date <= MIN_END_DATE:
                filter_stats["end_date_fail"] += 1
                continue
        except (ValueError, TypeError):
            filter_stats["end_date_fail"] += 1
            continue

        # Get current price
        condition_id = market["condition_id"]
        current_price = get_current_price(condition_id)

        if current_price is None:
            filter_stats["no_price"] += 1
            continue

        # Probability filter
        if not (MIN_PROB <= current_price <= MAX_PROB):
            filter_stats["probability_fail"] += 1
            continue

        filter_stats["passed"] += 1

        # Detect topic
        topic = detect_topic(market["question"])

        candidates.append({
            "condition_id": condition_id,
            "question": market["question"],
            "slug": market.get("slug"),
            "end_date": end_date_str,
            "volume_total": market["volume_total"],
            "current_price": current_price,
            "topic": topic,
        })

    print(f"\nFilter results:")
    print(f"  Volume < ${MIN_VOLUME:,}: {filter_stats['volume_fail']}")
    print(f"  End date <= {MIN_END_DATE.date()}: {filter_stats['end_date_fail']}")
    print(f"  No price history: {filter_stats['no_price']}")
    print(f"  Probability outside {MIN_PROB*100:.0f}%-{MAX_PROB*100:.0f}%: {filter_stats['probability_fail']}")
    print(f"  ✅ Passed all filters: {filter_stats['passed']}")

    if not candidates:
        print("\n❌ No candidates found!")
        return

    # Topic distribution
    topic_counts = Counter(c["topic"] for c in candidates)
    print(f"\nTopic distribution (all candidates):")
    for topic, count in topic_counts.most_common():
        print(f"  {topic}: {count}")

    # Select diverse sample
    # Stratified sampling: try to get at least some from each topic
    selected = []
    topic_pools = {topic: [] for topic in TOPIC_KEYWORDS.keys()}
    topic_pools["other"] = []

    for c in candidates:
        topic_pools[c["topic"]].append(c)

    # First pass: take from each topic proportionally
    target_per_topic = max(1, TARGET_COUNT // len([t for t, pool in topic_pools.items() if pool]))

    for topic, pool in topic_pools.items():
        if pool:
            # Sort by volume (higher is better) then sample
            pool.sort(key=lambda x: -x["volume_total"])
            take = min(target_per_topic, len(pool))
            selected.extend(pool[:take])

    # Second pass: if we need more, add from largest pools
    if len(selected) < TARGET_COUNT:
        remaining = [c for c in candidates if c not in selected]
        remaining.sort(key=lambda x: -x["volume_total"])
        need = TARGET_COUNT - len(selected)
        selected.extend(remaining[:need])

    # If we have too many, trim by volume
    if len(selected) > TARGET_COUNT:
        selected.sort(key=lambda x: -x["volume_total"])
        selected = selected[:TARGET_COUNT]

    print(f"\n✅ Selected {len(selected)} ultimates")

    # Final topic distribution
    final_topics = Counter(c["topic"] for c in selected)
    print(f"\nTopic distribution (selected):")
    for topic, count in final_topics.most_common():
        print(f"  {topic}: {count}")

    # Show sample
    print(f"\nSample of selected ultimates:")
    for i, ult in enumerate(selected[:10]):
        print(f"\n{i+1}. [{ult['topic']}] p={ult['current_price']:.2f}, vol=${ult['volume_total']/1e6:.1f}M")
        print(f"   {ult['question'][:75]}...")

    # Save
    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_volume": MIN_VOLUME,
            "min_prob": MIN_PROB,
            "max_prob": MAX_PROB,
            "min_end_date": MIN_END_DATE.isoformat(),
            "n_candidates": len(candidates),
            "n_selected": len(selected),
            "topic_distribution": dict(final_topics),
        },
        "ultimates": selected,
    }

    output_path = OUTPUT_DIR / "ultimates.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
