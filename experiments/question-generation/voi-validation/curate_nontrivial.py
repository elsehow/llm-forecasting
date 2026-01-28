#!/usr/bin/env python3
"""
Curate pairs where the OTHER market has non-trivial probability.

The previous approach classified pairs blind to the other market's probability.
Most resolved pairs had the other market at <5%, limiting observable shift.

This script:
1. Filters to pairs where other market is at 10-90% probability
2. THEN classifies with LLM
3. Should yield many more usable pairs for validation

Usage:
    uv run python experiments/question-generation/voi-validation/curate_nontrivial.py
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
from dotenv import load_dotenv
import litellm

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent

# Thresholds
MIN_VOLUME = 100_000
RESOLUTION_THRESHOLD_HIGH = 0.90
RESOLUTION_THRESHOLD_LOW = 0.10
MIN_OTHER_PROB = 0.10  # Other market must be above this
MAX_OTHER_PROB = 0.90  # Other market must be below this

MODEL = "claude-sonnet-4-20250514"

# Topic patterns
TOPIC_PATTERNS = {
    "iran": r"\biran\b|\bkhamenei\b|\btehran\b",
    "russia_ukraine": r"\brussia\b|\bukraine\b|\bputin\b|\bzelensky\b|\bceasefire\b",
    "fed": r"\bfed\b|\bfederal reserve\b|\binterest rate\b|\bfed chair\b",
    "trump": r"\btrump\b",
    "bitcoin": r"\bbitcoin\b|\bbtc\b",
    "crypto": r"\bcrypto\b|\bethereum\b|\bxrp\b|\bsolana\b",
    "nba": r"\bnba\b|\bceltics\b|\blakers\b|\bwarriors\b|\bnba finals\b",
    "nfl": r"\bnfl\b|\bsuper bowl\b|\bnfc\b|\bafc\b",
    "soccer": r"\bchampions league\b|\bla liga\b|\bpremier league\b|\bworld cup\b|\bfifa\b",
    "election_2028": r"\b2028\b.*(election|nomination|president)",
    "market_cap": r"\bmarket cap\b|\blargest company\b",
    "ai": r"\bopenai\b|\bgpt\b|\bclaude\b|\bai model\b",
}


def extract_topics(question: str) -> set[str]:
    question_lower = question.lower()
    topics = set()
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, question_lower):
            topics.add(topic)
    return topics


def load_data():
    with open(DATA_DIR / "markets.json") as f:
        markets = {m["condition_id"]: m for m in json.load(f)}

    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = data

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)

    return markets, histories, pairs


def detect_resolution(history: dict) -> tuple[str, datetime, int] | None:
    """Returns (outcome, date, resolution_idx)."""
    candles = history.get("candles", [])
    if len(candles) < 3:
        return None

    final_price = candles[-1]["close"]
    if final_price > RESOLUTION_THRESHOLD_HIGH:
        outcome = "YES"
    elif final_price < RESOLUTION_THRESHOLD_LOW:
        outcome = "NO"
    else:
        return None

    threshold = RESOLUTION_THRESHOLD_HIGH if outcome == "YES" else RESOLUTION_THRESHOLD_LOW
    for i, candle in enumerate(candles):
        if outcome == "YES" and candle["close"] >= threshold:
            ts = candle["timestamp"]
            resolution_date = datetime.fromtimestamp(ts, tz=timezone.utc)
            if resolution_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                return outcome, resolution_date, i
            return None
        elif outcome == "NO" and candle["close"] <= threshold:
            ts = candle["timestamp"]
            resolution_date = datetime.fromtimestamp(ts, tz=timezone.utc)
            if resolution_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                return outcome, resolution_date, i
            return None
    return None


def get_price_before_resolution(candles: list[dict], resolution_idx: int) -> float | None:
    """Get price of other market before resolution."""
    if resolution_idx < 3:
        return candles[0]["close"] if candles else None
    # Average of 3 candles before resolution
    prices = [c["close"] for c in candles[max(0, resolution_idx-3):resolution_idx]]
    return np.mean(prices) if prices else None


CLASSIFICATION_PROMPT = """Classify this prediction market pair as having a REAL relationship or SPURIOUS correlation.

Question A: {question_a}
Question B: {question_b}
Observed ρ: {rho:.2f}
Shared topics: {topics}

Categories:
1. same_event_different_timeline - Same event, different time horizons
2. sequential_prerequisite - One is prerequisite for other
3. mutually_exclusive - Same race/competition, only one can win
4. causal_relationship - Clear causal mechanism
5. shared_driver - Both driven by same factor
6. spurious - No real relationship
7. weak_or_unclear - Might have relationship but indirect

JSON only: {{"category": "...", "confidence": "high/medium/low", "rationale": "..."}}
"""


async def classify_pair(q_a: str, q_b: str, rho: float, topics: set[str]) -> dict:
    prompt = CLASSIFICATION_PROMPT.format(
        question_a=q_a, question_b=q_b, rho=rho,
        topics=", ".join(topics) if topics else "none"
    )
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content
        if "```" in text:
            text = text.split("```")[1].split("```")[0]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        return {"category": "error", "confidence": "none", "rationale": str(e)}


async def main():
    print("=" * 70)
    print("CURATE NON-TRIVIAL PROBABILITY PAIRS")
    print("=" * 70)
    print(f"\nFilters:")
    print(f"  Min volume: ${MIN_VOLUME:,}")
    print(f"  Other market probability: {MIN_OTHER_PROB} - {MAX_OTHER_PROB}")
    print(f"  Resolution after: {MODEL_KNOWLEDGE_CUTOFF}")

    markets, histories, pairs = load_data()
    print(f"\nLoaded {len(markets)} markets, {len(histories)} histories, {len(pairs)} pairs")

    # Find pairs meeting all criteria
    candidates = []

    for pair in pairs:
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        rho = pair["rho"]

        if np.isnan(rho):
            continue

        # Volume filter
        market_a = markets.get(cond_a)
        market_b = markets.get(cond_b)
        if not market_a or not market_b:
            continue
        if market_a.get("volume_total", 0) < MIN_VOLUME or market_b.get("volume_total", 0) < MIN_VOLUME:
            continue

        # Get histories
        hist_a = histories.get(cond_a)
        hist_b = histories.get(cond_b)
        if not hist_a or not hist_b:
            continue

        # Check for resolution
        res_a = detect_resolution(hist_a)
        res_b = detect_resolution(hist_b)

        if not res_a and not res_b:
            continue

        # Determine which resolved and check other market's probability
        if res_a:
            resolved = "A"
            resolution_outcome, resolution_date, res_idx = res_a
            other_candles = hist_b.get("candles", [])
            # Find other market's price at resolution time
            res_ts = hist_a["candles"][res_idx]["timestamp"]
            other_price = None
            for c in reversed(other_candles):
                if c["timestamp"] <= res_ts:
                    other_price = c["close"]
                    break
            if other_price is None and other_candles:
                other_price = other_candles[0]["close"]
        else:
            resolved = "B"
            resolution_outcome, resolution_date, res_idx = res_b
            other_candles = hist_a.get("candles", [])
            res_ts = hist_b["candles"][res_idx]["timestamp"]
            other_price = None
            for c in reversed(other_candles):
                if c["timestamp"] <= res_ts:
                    other_price = c["close"]
                    break
            if other_price is None and other_candles:
                other_price = other_candles[0]["close"]

        if other_price is None:
            continue

        # KEY FILTER: other market must have non-trivial probability
        if not (MIN_OTHER_PROB <= other_price <= MAX_OTHER_PROB):
            continue

        # Topic filter
        topics_a = extract_topics(q_a)
        topics_b = extract_topics(q_b)
        shared_topics = topics_a & topics_b

        if not shared_topics:
            continue

        candidates.append({
            "question_a": q_a,
            "question_b": q_b,
            "condition_id_a": cond_a,
            "condition_id_b": cond_b,
            "rho": rho,
            "shared_topics": list(shared_topics),
            "resolved": resolved,
            "resolution_outcome": resolution_outcome,
            "resolution_date": resolution_date.isoformat(),
            "other_price_at_resolution": other_price,
        })

    print(f"\nFound {len(candidates)} candidates with non-trivial other-market probability")

    if not candidates:
        print("No candidates found!")
        return

    # Sort by |ρ|
    candidates.sort(key=lambda p: -abs(p["rho"]))

    # Show distribution
    topic_counts = defaultdict(int)
    for c in candidates:
        for t in c["shared_topics"]:
            topic_counts[t] += 1

    print("\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")

    print("\nTop 10 candidates:")
    for i, c in enumerate(candidates[:10]):
        print(f"\n{i+1}. ρ={c['rho']:+.2f}, other_p={c['other_price_at_resolution']:.2f}, topics={c['shared_topics']}")
        print(f"   A: {c['question_a'][:65]}...")
        print(f"   B: {c['question_b'][:65]}...")

    # Classify all
    print("\n" + "=" * 70)
    print("LLM CLASSIFICATION")
    print("=" * 70)
    print(f"\nClassifying {len(candidates)} candidates...")

    batch_size = 10
    classifications = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            classify_pair(c["question_a"], c["question_b"], c["rho"], set(c["shared_topics"]))
            for c in batch
        ])
        classifications.extend(batch_results)
        print(f"  Processed {min(i+batch_size, len(candidates))}/{len(candidates)}")

    # Stats
    categories = [c.get("category", "unknown") for c in classifications]
    print("\nClassification results:")
    for cat in sorted(set(categories)):
        count = categories.count(cat)
        pct = count / len(categories) * 100
        print(f"  {cat}: {count} ({pct:.0f}%)")

    # Filter to real
    real_categories = {
        "same_event_different_timeline",
        "sequential_prerequisite",
        "mutually_exclusive",
        "causal_relationship",
        "shared_driver",
    }

    real_pairs = []
    for i, (cand, classif) in enumerate(zip(candidates, classifications)):
        if classif.get("category") in real_categories:
            if classif.get("confidence") in ("high", "medium"):
                cand["classification"] = classif
                real_pairs.append(cand)

    print(f"\n✅ Real relationships: {len(real_pairs)}")
    print(f"   All have other market at {MIN_OTHER_PROB}-{MAX_OTHER_PROB} probability!")

    # Save
    output = {
        "metadata": {
            "n_candidates": len(candidates),
            "n_real": len(real_pairs),
            "min_other_prob": MIN_OTHER_PROB,
            "max_other_prob": MAX_OTHER_PROB,
            "generated_at": datetime.now().isoformat(),
        },
        "curated_pairs": real_pairs,
    }

    output_path = OUTPUT_DIR / "curated_pairs_nontrivial.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Show curated pairs
    if real_pairs:
        print("\n" + "-" * 70)
        print("CURATED PAIRS")
        print("-" * 70)
        for i, p in enumerate(real_pairs[:20]):
            c = p["classification"]
            print(f"\n{i+1}. [{c['category']}] ρ={p['rho']:+.2f}, other_p={p['other_price_at_resolution']:.2f}")
            print(f"   A: {p['question_a'][:65]}...")
            print(f"   B: {p['question_b'][:65]}...")


if __name__ == "__main__":
    asyncio.run(main())
