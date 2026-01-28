#!/usr/bin/env python3
"""
Step 3: Match LLM-generated cruxes to existing Polymarket markets.

For each crux:
1. Find candidate markets via text similarity (fast filter)
2. Use LLM to judge semantic match quality (accurate filter)
3. Track match rate and match quality

Usage:
    uv run python experiments/question-generation/llm-crux-validation/match_cruxes.py
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import litellm
import numpy as np

load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting" / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
RESULTS_DIR = Path(__file__).parent / "results"

# Model config
MODEL = "claude-sonnet-4-20250514"

# Matching config
TOP_K_CANDIDATES = 5  # Number of candidate markets to consider per crux
MIN_MATCH_CONFIDENCE = "high"  # Only accept high-confidence matches

MATCH_JUDGMENT_PROMPT = """Does market A ask essentially the same question as crux B?

Market A: "{market_question}"
Crux B: "{crux_question}"

We need markets that would RESOLVE in the same way as the crux.
They don't need identical wording, but must ask about the same event/outcome.

Examples of MATCHES:
- "Will Bitcoin reach $100k by March?" ↔ "Does BTC price exceed $100,000 before April?"
- "Fed cuts rates at Jan meeting?" ↔ "Will the Federal Reserve lower rates in January?"

Examples of NON-MATCHES:
- "Will Trump win 2028 nomination?" ↔ "Trump runs in 2028?" (running ≠ winning)
- "Bitcoin above $100k?" ↔ "Crypto market cap exceeds $5T?" (different metrics)
- "Will X happen by June?" ↔ "Will X happen by December?" (different timelines)

JSON only: {{"match": true/false, "confidence": "high/medium/low", "reasoning": "<brief explanation>"}}
"""


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    import re
    text = text.lower()
    # Remove common question prefixes
    text = re.sub(r"^(will |does |is |are |has |have |can |could |would |should )", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def get_word_overlap_score(text1: str, text2: str) -> float:
    """Simple word overlap score for fast filtering."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())

    # Remove stop words
    stop_words = {"the", "a", "an", "by", "in", "on", "at", "to", "for", "of", "and", "or", "is", "be", "will", "would"}
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def get_current_price(condition_id: str) -> float | None:
    """Get current price from price history."""
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


async def judge_match(market_question: str, crux_question: str) -> dict:
    """Use LLM to judge if market and crux are semantically equivalent."""
    prompt = MATCH_JUDGMENT_PROMPT.format(
        market_question=market_question,
        crux_question=crux_question,
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)
    except Exception as e:
        return {"match": False, "confidence": "none", "reasoning": f"Error: {e}"}


async def find_matches_for_crux(
    crux: dict,
    ultimate_id: str,
    markets: list[dict],
    market_prices: dict[str, float],
) -> list[dict]:
    """Find matching markets for a single crux."""
    crux_text = crux["crux"]

    # Fast filter: word overlap
    candidates = []
    for market in markets:
        # Skip the ultimate itself
        if market["condition_id"] == ultimate_id:
            continue

        # Must have price data
        price = market_prices.get(market["condition_id"])
        if price is None:
            continue

        # Must not be resolved (price not near 0 or 1)
        if price < 0.05 or price > 0.95:
            continue

        overlap = get_word_overlap_score(crux_text, market["question"])
        if overlap > 0.1:  # Very loose threshold
            candidates.append({
                "market": market,
                "overlap": overlap,
                "price": price,
            })

    # Sort by overlap and take top K
    candidates.sort(key=lambda x: -x["overlap"])
    candidates = candidates[:TOP_K_CANDIDATES]

    if not candidates:
        return []

    # LLM judge: check each candidate
    matches = []
    for cand in candidates:
        judgment = await judge_match(cand["market"]["question"], crux_text)

        if judgment.get("match") and judgment.get("confidence") in ("high", "medium"):
            matches.append({
                "market_id": cand["market"]["condition_id"],
                "market_question": cand["market"]["question"],
                "market_slug": cand["market"].get("slug"),
                "market_price": cand["price"],
                "overlap_score": cand["overlap"],
                "match_confidence": judgment["confidence"],
                "match_reasoning": judgment.get("reasoning", ""),
            })

    return matches


async def main():
    print("=" * 70)
    print("MATCH CRUXES TO POLYMARKET MARKETS")
    print("=" * 70)

    # Load cruxes
    cruxes_path = RESULTS_DIR / "cruxes.json"
    if not cruxes_path.exists():
        print(f"\n❌ {cruxes_path} not found. Run generate_cruxes.py first.")
        return

    with open(cruxes_path) as f:
        cruxes_data = json.load(f)
    crux_results = cruxes_data["cruxes"]

    total_cruxes = sum(len(r["cruxes"]) for r in crux_results)
    print(f"\nLoaded {total_cruxes} cruxes from {len(crux_results)} ultimates")

    # Load all markets
    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)
    print(f"Loaded {len(markets)} markets")

    # Load prices
    print("Loading prices...")
    market_prices = {}
    for market in markets:
        price = get_current_price(market["condition_id"])
        if price is not None:
            market_prices[market["condition_id"]] = price
    print(f"  Loaded prices for {len(market_prices)} markets")

    # Match cruxes
    print(f"\nMatching cruxes to markets using {MODEL}...")
    matched_pairs = []
    crux_match_count = 0
    crux_total = 0

    for i, ult_result in enumerate(crux_results):
        ultimate_id = ult_result["ultimate_id"]
        ultimate_question = ult_result["ultimate_question"]

        for crux in ult_result["cruxes"]:
            crux_total += 1
            matches = await find_matches_for_crux(crux, ultimate_id, markets, market_prices)

            if matches:
                crux_match_count += 1
                # Take best match (highest confidence, then highest overlap)
                best_match = sorted(
                    matches,
                    key=lambda x: (x["match_confidence"] == "high", x["overlap_score"]),
                    reverse=True
                )[0]

                matched_pairs.append({
                    "ultimate_id": ultimate_id,
                    "ultimate_question": ultimate_question,
                    "crux": crux,
                    "match": best_match,
                })

        if (i + 1) % 10 == 0 or i == len(crux_results) - 1:
            print(f"  Processed {i+1}/{len(crux_results)} ultimates ({crux_match_count}/{crux_total} cruxes matched)")

    # Stats
    match_rate = crux_match_count / crux_total if crux_total > 0 else 0
    print(f"\n{'='*70}")
    print("MATCHING RESULTS")
    print("=" * 70)
    print(f"\nTotal cruxes: {crux_total}")
    print(f"Matched cruxes: {crux_match_count}")
    print(f"Match rate: {match_rate:.1%}")

    # Unique ultimates with matches
    ultimates_with_matches = len(set(p["ultimate_id"] for p in matched_pairs))
    print(f"\nUltimates with at least one match: {ultimates_with_matches}/{len(crux_results)}")

    # Confidence distribution
    confidences = [p["match"]["match_confidence"] for p in matched_pairs]
    conf_counts = Counter(confidences)
    print(f"\nMatch confidence distribution:")
    for conf, count in conf_counts.most_common():
        print(f"  {conf}: {count}")

    # Magnitude distribution of matched cruxes
    magnitudes = [p["crux"].get("magnitude", "unknown") for p in matched_pairs]
    mag_counts = Counter(magnitudes)
    print(f"\nMagnitude distribution (matched cruxes):")
    for mag, count in mag_counts.most_common():
        print(f"  {mag}: {count}")

    # Show samples
    print(f"\nSample matched pairs:")
    for i, pair in enumerate(matched_pairs[:5]):
        print(f"\n{i+1}. Ultimate: {pair['ultimate_question'][:50]}...")
        print(f"   Crux: {pair['crux']['crux'][:50]}...")
        print(f"   Match: {pair['match']['market_question'][:50]}...")
        print(f"   [{pair['match']['match_confidence']}] {pair['match']['match_reasoning'][:60]}...")

    # Save
    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "total_cruxes": crux_total,
            "matched_cruxes": crux_match_count,
            "match_rate": match_rate,
            "ultimates_with_matches": ultimates_with_matches,
            "confidence_distribution": dict(conf_counts),
            "magnitude_distribution": dict(mag_counts),
        },
        "matched_pairs": matched_pairs,
    }

    output_path = RESULTS_DIR / "matched_pairs.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved {len(matched_pairs)} matched pairs to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
