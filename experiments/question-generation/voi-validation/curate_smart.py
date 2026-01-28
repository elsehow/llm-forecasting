#!/usr/bin/env python3
"""
Smart pair curation: filter by topic similarity BEFORE LLM classification.

The problem: 90% of high-|ρ| pairs are spurious noise.
The solution: Pre-filter pairs where questions share keywords/topics.

Strategy:
1. Extract keywords from each question
2. Find pairs with keyword overlap (likely real relationships)
3. Then classify with LLM

Usage:
    uv run python experiments/question-generation/voi-validation/curate_smart.py
"""

import json
import re
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
import litellm

load_dotenv()

# Import config from conditional-forecasting (shared)
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

# LLM config
MODEL = "claude-sonnet-4-20250514"

# Topic keywords to look for
TOPIC_PATTERNS = {
    "iran": r"\biran\b|\bkhamenei\b|\btehran\b|\bpersia\b",
    "russia_ukraine": r"\brussia\b|\bukraine\b|\bputin\b|\bzelensky\b|\bceasefire\b",
    "fed": r"\bfed\b|\bfederal reserve\b|\binterest rate\b|\bfomc\b|\bfed chair\b",
    "trump": r"\btrump\b",
    "bitcoin": r"\bbitcoin\b|\bbtc\b",
    "crypto": r"\bcrypto\b|\bethereumr\b|\bxrp\b|\bsolana\b|\bdogecoin\b",
    "nba": r"\bnba\b|\bceltics\b|\blakers\b|\bwarriors\b|\brockets\b|\bnba finals\b",
    "nfl": r"\bnfl\b|\bsuper bowl\b|\bnfc\b|\bafc\b|\brams\b|\bchiefs\b|\beagles\b",
    "soccer": r"\bchampions league\b|\bla liga\b|\bpremier league\b|\bworld cup\b|\bfifa\b",
    "election_2028": r"\b2028\b.*(election|nomination|president)",
    "election_2026": r"\b2026\b.*(election|midterm)",
    "china": r"\bchina\b|\bxi jinping\b|\bbeijing\b|\btaiwan\b",
    "ai": r"\bopenai\b|\bgpt\b|\bclaude\b|\banthropicb|\bai model\b",
    "nvidia": r"\bnvidia\b",
    "apple": r"\bapple\b",
    "market_cap": r"\bmarket cap\b|\blargest company\b",
}


def extract_topics(question: str) -> set[str]:
    """Extract topic tags from a question."""
    question_lower = question.lower()
    topics = set()
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, question_lower):
            topics.add(topic)
    return topics


def load_data():
    """Load markets, histories, and pairs."""
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


def detect_resolution(history: dict) -> tuple[str, datetime] | None:
    """Detect if market resolved after cutoff."""
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
    for candle in candles:
        if outcome == "YES" and candle["close"] >= threshold:
            ts = candle["timestamp"]
            break
        elif outcome == "NO" and candle["close"] <= threshold:
            ts = candle["timestamp"]
            break
    else:
        return None

    resolution_date = datetime.fromtimestamp(ts, tz=timezone.utc)
    if resolution_date.date() < MODEL_KNOWLEDGE_CUTOFF:
        return None

    return outcome, resolution_date


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

JSON response only:
{{"category": "...", "confidence": "high/medium/low", "rationale": "..."}}
"""


async def classify_pair(question_a: str, question_b: str, rho: float, topics: set[str]) -> dict:
    """Classify a single pair."""
    prompt = CLASSIFICATION_PROMPT.format(
        question_a=question_a,
        question_b=question_b,
        rho=rho,
        topics=", ".join(topics) if topics else "none detected"
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
    print("SMART PAIR CURATION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    markets, histories, pairs = load_data()
    print(f"Loaded {len(markets)} markets, {len(histories)} histories, {len(pairs)} pairs")

    # Find pairs with topic overlap
    print("\nFinding pairs with topic overlap...")
    topic_pairs = []

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

        # Resolution filter
        hist_a = histories.get(cond_a)
        hist_b = histories.get(cond_b)
        resolved = None
        resolution_outcome = None
        resolution_date = None

        if hist_a:
            res = detect_resolution(hist_a)
            if res:
                resolved = "A"
                resolution_outcome, resolution_date = res
        if hist_b and not resolved:
            res = detect_resolution(hist_b)
            if res:
                resolved = "B"
                resolution_outcome, resolution_date = res

        if not resolved:
            continue

        # Topic extraction
        topics_a = extract_topics(q_a)
        topics_b = extract_topics(q_b)
        shared_topics = topics_a & topics_b

        if shared_topics:
            topic_pairs.append({
                "question_a": q_a,
                "question_b": q_b,
                "condition_id_a": cond_a,
                "condition_id_b": cond_b,
                "rho": rho,
                "topics_a": list(topics_a),
                "topics_b": list(topics_b),
                "shared_topics": list(shared_topics),
                "resolved": resolved,
                "resolution_outcome": resolution_outcome,
                "resolution_date": resolution_date.isoformat() if resolution_date else None,
            })

    print(f"Found {len(topic_pairs)} pairs with shared topics")

    # Sort by number of shared topics, then |ρ|
    topic_pairs.sort(key=lambda p: (-len(p["shared_topics"]), -abs(p["rho"])))

    # Show topic distribution
    topic_counts = defaultdict(int)
    for p in topic_pairs:
        for t in p["shared_topics"]:
            topic_counts[t] += 1

    print("\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")

    # Show top candidates
    print("\nTop 20 topic-filtered candidates:")
    for i, p in enumerate(topic_pairs[:20]):
        print(f"\n{i+1}. ρ={p['rho']:+.2f} | topics: {', '.join(p['shared_topics'])}")
        print(f"   A: {p['question_a'][:70]}...")
        print(f"   B: {p['question_b'][:70]}...")

    # Classify
    print("\n" + "=" * 70)
    print("LLM CLASSIFICATION")
    print("=" * 70)

    limit = min(100, len(topic_pairs))
    print(f"\nClassifying {limit} topic-filtered pairs...")

    classifications = []
    batch_size = 10
    for i in range(0, limit, batch_size):
        batch = topic_pairs[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            classify_pair(p["question_a"], p["question_b"], p["rho"], set(p["shared_topics"]))
            for p in batch
        ])
        classifications.extend(batch_results)
        print(f"  Processed {min(i+batch_size, limit)}/{limit}")

    # Stats
    categories = [c.get("category", "unknown") for c in classifications]
    print("\nClassification results:")
    for cat in sorted(set(categories)):
        count = categories.count(cat)
        pct = count / len(categories) * 100
        print(f"  {cat}: {count} ({pct:.0f}%)")

    # Filter to real relationships
    real_categories = {
        "same_event_different_timeline",
        "sequential_prerequisite",
        "mutually_exclusive",
        "causal_relationship",
        "shared_driver",
    }
    real_pairs = []
    for i, (pair, classification) in enumerate(zip(topic_pairs[:limit], classifications)):
        if classification.get("category") in real_categories:
            if classification.get("confidence") in ("high", "medium"):
                pair["classification"] = classification
                real_pairs.append(pair)

    print(f"\n✅ Real relationships (high/medium confidence): {len(real_pairs)}")
    print(f"   Hit rate: {len(real_pairs)/limit*100:.0f}% (vs 8% without topic filter)")

    # Save
    output = {
        "metadata": {
            "n_topic_pairs": len(topic_pairs),
            "n_classified": limit,
            "n_real": len(real_pairs),
            "hit_rate": len(real_pairs) / limit if limit > 0 else 0,
            "generated_at": datetime.now().isoformat(),
        },
        "curated_pairs": real_pairs
    }

    output_path = OUTPUT_DIR / "curated_pairs_smart.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(real_pairs)} curated pairs to {output_path}")

    # Details
    if real_pairs:
        print("\n" + "-" * 70)
        print("CURATED PAIRS")
        print("-" * 70)
        for i, p in enumerate(real_pairs):
            c = p["classification"]
            print(f"\n{i+1}. [{c['category']}] ρ={p['rho']:+.2f}")
            print(f"   A: {p['question_a'][:70]}...")
            print(f"   B: {p['question_b'][:70]}...")
            print(f"   → {c['rationale']}")


if __name__ == "__main__":
    asyncio.run(main())
