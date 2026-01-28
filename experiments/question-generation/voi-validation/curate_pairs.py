#!/usr/bin/env python3
"""
Curate pairs for VOI validation.

Strategy:
1. Filter pairs by volume (both markets > $100k)
2. Filter pairs where at least one resolved (price → 0 or 1)
3. Use LLM to classify if relationship is real (not spurious)
4. Output curated pairs for validation

Usage:
    uv run python experiments/question-generation/voi-validation/curate_pairs.py
    uv run python experiments/question-generation/voi-validation/curate_pairs.py --classify  # Run LLM classification
"""

import json
import asyncio
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
import litellm

load_dotenv()

# Import config from conditional-forecasting (shared)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths - data lives in conditional-forecasting
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent

# Thresholds
MIN_VOLUME = 100_000  # $100k minimum volume
MIN_LIQUIDITY = 10_000  # $10k minimum liquidity
RESOLUTION_THRESHOLD_HIGH = 0.90
RESOLUTION_THRESHOLD_LOW = 0.10

# LLM config
MODEL = "claude-sonnet-4-20250514"


@dataclass
class CandidatePair:
    """A candidate pair for curation."""
    question_a: str
    question_b: str
    condition_id_a: str
    condition_id_b: str
    rho: float
    volume_a: float
    volume_b: float
    resolved: str | None  # "A", "B", or None
    resolution_outcome: str | None
    resolution_date: datetime | None


def load_markets() -> dict[str, dict]:
    """Load markets with volume data."""
    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)
    return {m["condition_id"]: m for m in markets}


def load_price_histories() -> dict[str, dict]:
    """Load price histories."""
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = data
    return histories


def detect_resolution(history: dict) -> tuple[str, datetime] | None:
    """Detect if market resolved."""
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

    # Find resolution timestamp
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

    # Filter by cutoff
    if resolution_date.date() < MODEL_KNOWLEDGE_CUTOFF:
        return None

    return outcome, resolution_date


def find_candidate_pairs() -> list[CandidatePair]:
    """Find candidate pairs meeting volume and resolution criteria."""
    print("Loading data...")
    markets = load_markets()
    histories = load_price_histories()

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)

    print(f"Loaded {len(markets)} markets, {len(histories)} histories, {len(pairs)} pairs")

    # Build condition_id to market mapping
    cond_to_market = markets

    # Build question to condition_id mapping from histories
    question_to_cond = {}
    for cond_id, hist in histories.items():
        question_to_cond[hist["question"]] = cond_id

    candidates = []

    for pair in pairs:
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        rho = pair["rho"]

        # Skip NaN correlations
        if np.isnan(rho):
            continue

        # Get market data
        market_a = cond_to_market.get(cond_a)
        market_b = cond_to_market.get(cond_b)

        if not market_a or not market_b:
            continue

        volume_a = market_a.get("volume_total", 0)
        volume_b = market_b.get("volume_total", 0)

        # Filter by volume
        if volume_a < MIN_VOLUME or volume_b < MIN_VOLUME:
            continue

        # Check for resolution
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

        # Only keep pairs where at least one resolved
        if not resolved:
            continue

        candidates.append(CandidatePair(
            question_a=q_a,
            question_b=q_b,
            condition_id_a=cond_a,
            condition_id_b=cond_b,
            rho=rho,
            volume_a=volume_a,
            volume_b=volume_b,
            resolved=resolved,
            resolution_outcome=resolution_outcome,
            resolution_date=resolution_date,
        ))

    return candidates


CLASSIFICATION_PROMPT = """You are evaluating whether two prediction market questions have a REAL causal or logical relationship, or if their price correlation is SPURIOUS (just noise from trading patterns).

Question A: {question_a}
Question B: {question_b}
Observed correlation (ρ): {rho:.2f}

Classify this pair into ONE of these categories:

1. **same_event_different_timeline** - Same underlying event with different time horizons
   Example: "Will Iran regime fall by March?" and "Will Iran regime fall by 2027?"

2. **sequential_prerequisite** - One event is a prerequisite for the other
   Example: "Will Rams win NFC Championship?" and "Will Rams win Super Bowl?"

3. **mutually_exclusive** - Only one can happen (same race/competition)
   Example: "Will Celtics win NBA Finals?" and "Will Rockets win NBA Finals?"

4. **causal_relationship** - Clear causal mechanism connecting them
   Example: "US strikes Iran?" and "Khamenei out as Supreme Leader?"

5. **shared_driver** - Both driven by same underlying factor
   Example: "Bitcoin reaches $110k?" and "XRP reaches $3.20?" (both driven by crypto sentiment)

6. **spurious** - No real relationship, correlation is noise
   Example: "Club Puebla wins soccer match?" and "NBA MVP winner?"

7. **weak_or_unclear** - Might have relationship but unclear/indirect

Respond with ONLY a JSON object:
{{
  "category": "one of the 7 categories above",
  "confidence": "high" or "medium" or "low",
  "rationale": "brief explanation (1-2 sentences)"
}}
"""


async def classify_pair(pair: CandidatePair) -> dict:
    """Use LLM to classify if pair has real relationship."""
    prompt = CLASSIFICATION_PROMPT.format(
        question_a=pair.question_a,
        question_b=pair.question_b,
        rho=pair.rho,
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        text = response.choices[0].message.content

        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())
        return result
    except Exception as e:
        return {"category": "error", "confidence": "none", "rationale": str(e)}


async def classify_pairs(pairs: list[CandidatePair], limit: int = None) -> list[dict]:
    """Classify multiple pairs in parallel."""
    if limit:
        pairs = pairs[:limit]

    print(f"Classifying {len(pairs)} pairs...")

    # Process in batches to avoid rate limits
    batch_size = 10
    results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        batch_results = await asyncio.gather(*[classify_pair(p) for p in batch])
        results.extend(batch_results)
        print(f"  Processed {min(i+batch_size, len(pairs))}/{len(pairs)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classify", action="store_true", help="Run LLM classification")
    parser.add_argument("--limit", type=int, default=100, help="Max pairs to classify")
    args = parser.parse_args()

    print("=" * 70)
    print("PAIR CURATION FOR VOI VALIDATION")
    print("=" * 70)
    print(f"\nFilters:")
    print(f"  Min volume: ${MIN_VOLUME:,}")
    print(f"  Resolution after: {MODEL_KNOWLEDGE_CUTOFF}")
    print(f"  Resolution threshold: <{RESOLUTION_THRESHOLD_LOW} or >{RESOLUTION_THRESHOLD_HIGH}")

    # Find candidates
    candidates = find_candidate_pairs()
    print(f"\nFound {len(candidates)} candidate pairs (volume + resolution filters)")

    # Sort by |ρ| to prioritize strong relationships
    candidates.sort(key=lambda p: abs(p.rho), reverse=True)

    # Show distribution
    print("\nCorrelation distribution:")
    rhos = [abs(p.rho) for p in candidates]
    print(f"  Strong (|ρ|>0.6): {sum(1 for r in rhos if r > 0.6)}")
    print(f"  Moderate (0.3-0.6): {sum(1 for r in rhos if 0.3 <= r <= 0.6)}")
    print(f"  Weak (0.1-0.3): {sum(1 for r in rhos if 0.1 <= r < 0.3)}")
    print(f"  Independent (<0.1): {sum(1 for r in rhos if r < 0.1)}")

    # Show top candidates
    print("\nTop 20 candidates by |ρ|:")
    for i, p in enumerate(candidates[:20]):
        print(f"\n{i+1}. ρ={p.rho:+.2f} | {p.resolved} resolved {p.resolution_outcome}")
        print(f"   A: {p.question_a[:70]}...")
        print(f"   B: {p.question_b[:70]}...")

    # Save candidates
    output = {
        "metadata": {
            "min_volume": MIN_VOLUME,
            "resolution_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "n_candidates": len(candidates),
            "generated_at": datetime.now().isoformat(),
        },
        "candidates": [
            {
                "question_a": p.question_a,
                "question_b": p.question_b,
                "condition_id_a": p.condition_id_a,
                "condition_id_b": p.condition_id_b,
                "rho": p.rho,
                "volume_a": p.volume_a,
                "volume_b": p.volume_b,
                "resolved": p.resolved,
                "resolution_outcome": p.resolution_outcome,
                "resolution_date": p.resolution_date.isoformat() if p.resolution_date else None,
            }
            for p in candidates
        ]
    }

    candidates_path = OUTPUT_DIR / "candidate_pairs.json"
    with open(candidates_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved {len(candidates)} candidates to {candidates_path}")

    # Run classification if requested
    if args.classify:
        print("\n" + "=" * 70)
        print("LLM CLASSIFICATION")
        print("=" * 70)

        classifications = asyncio.run(classify_pairs(candidates, limit=args.limit))

        # Merge results
        for i, (pair, classification) in enumerate(zip(candidates[:args.limit], classifications)):
            output["candidates"][i]["classification"] = classification

        # Stats
        categories = [c.get("category", "unknown") for c in classifications]
        print("\nClassification results:")
        for cat in set(categories):
            count = categories.count(cat)
            print(f"  {cat}: {count}")

        # Filter to real relationships
        real_categories = {
            "same_event_different_timeline",
            "sequential_prerequisite",
            "mutually_exclusive",
            "causal_relationship",
            "shared_driver",
        }
        real_pairs = [
            (candidates[i], classifications[i])
            for i in range(len(classifications))
            if classifications[i].get("category") in real_categories
            and classifications[i].get("confidence") in ("high", "medium")
        ]

        print(f"\nReal relationships (high/medium confidence): {len(real_pairs)}")

        # Save classified results
        classified_path = OUTPUT_DIR / "classified_pairs.json"
        with open(classified_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved classified pairs to {classified_path}")

        # Save just the real pairs for validation
        real_output = {
            "metadata": {
                "source": "curate_pairs.py --classify",
                "n_classified": len(classifications),
                "n_real": len(real_pairs),
                "generated_at": datetime.now().isoformat(),
            },
            "curated_pairs": [
                {
                    "question_a": p.question_a,
                    "question_b": p.question_b,
                    "condition_id_a": p.condition_id_a,
                    "condition_id_b": p.condition_id_b,
                    "rho": p.rho,
                    "category": c["category"],
                    "rationale": c["rationale"],
                    "resolved": p.resolved,
                    "resolution_outcome": p.resolution_outcome,
                    "resolution_date": p.resolution_date.isoformat() if p.resolution_date else None,
                }
                for p, c in real_pairs
            ]
        }

        curated_path = OUTPUT_DIR / "curated_pairs_auto.json"
        with open(curated_path, "w") as f:
            json.dump(real_output, f, indent=2)
        print(f"Saved {len(real_pairs)} curated pairs to {curated_path}")


if __name__ == "__main__":
    main()
