#!/usr/bin/env python3
"""
Find resolved pairs within categories.

For each within-category pair, identify cases where:
1. One market has resolved (price < 0.05 or > 0.95)
2. The other market has price history around resolution time

This allows us to measure actual belief shifts when information arrives.
"""

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sys

# Import config from conditional-forecasting
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
INPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Resolution thresholds
RESOLUTION_THRESHOLD_HIGH = 0.95
RESOLUTION_THRESHOLD_LOW = 0.05
MIN_PRICE_CHANGE = 0.10

# Windows
WINDOW_BEFORE_DAYS = 3
WINDOW_AFTER_DAYS = 3
SECONDS_PER_DAY = 86400


@dataclass
class PriceHistory:
    """Price history for a single market."""
    condition_id: str
    question: str
    candles: list[dict]

    @property
    def first_timestamp(self) -> int:
        return self.candles[0]["timestamp"] if self.candles else 0

    @property
    def last_timestamp(self) -> int:
        return self.candles[-1]["timestamp"] if self.candles else 0

    @property
    def first_price(self) -> float:
        return self.candles[0]["close"] if self.candles else 0.5

    @property
    def last_price(self) -> float:
        return self.candles[-1]["close"] if self.candles else 0.5

    def price_at(self, timestamp: int) -> float | None:
        """Get price at or before given timestamp."""
        for candle in reversed(self.candles):
            if candle["timestamp"] <= timestamp:
                return candle["close"]
        return None

    def price_after(self, timestamp: int, window_seconds: int) -> float | None:
        """Get average price in window after timestamp."""
        prices = []
        for candle in self.candles:
            if timestamp < candle["timestamp"] <= timestamp + window_seconds:
                prices.append(candle["close"])
        return np.mean(prices) if prices else None


@dataclass
class Resolution:
    """A market resolution event."""
    condition_id: str
    question: str
    resolution_timestamp: int
    resolution_outcome: str
    price_before: float
    price_after: float

    @property
    def resolution_date(self) -> datetime:
        return datetime.fromtimestamp(self.resolution_timestamp, tz=timezone.utc)


def load_price_histories() -> dict[str, PriceHistory]:
    """Load all price histories from disk."""
    histories = {}

    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)

        condition_id = data["condition_id"]
        histories[condition_id] = PriceHistory(
            condition_id=condition_id,
            question=data["question"],
            candles=data["candles"]
        )

    return histories


def detect_resolution(history: PriceHistory) -> Resolution | None:
    """Detect if a market resolved during its price history."""
    if len(history.candles) < 5:
        return None

    final_price = history.last_price

    # Check if final price is extreme
    if not (final_price < RESOLUTION_THRESHOLD_LOW or final_price > RESOLUTION_THRESHOLD_HIGH):
        return None

    # Find when the resolution happened
    resolution_outcome = "YES" if final_price > RESOLUTION_THRESHOLD_HIGH else "NO"
    threshold = RESOLUTION_THRESHOLD_HIGH if resolution_outcome == "YES" else RESOLUTION_THRESHOLD_LOW

    resolution_idx = None
    for i, candle in enumerate(history.candles):
        if resolution_outcome == "YES" and candle["close"] >= threshold:
            resolution_idx = i
            break
        elif resolution_outcome == "NO" and candle["close"] <= threshold:
            resolution_idx = i
            break

    if resolution_idx is None or resolution_idx < 3:
        return None

    resolution_timestamp = history.candles[resolution_idx]["timestamp"]

    # Get price before resolution
    before_prices = [c["close"] for c in history.candles[max(0, resolution_idx-3):resolution_idx]]
    price_before = np.mean(before_prices) if before_prices else history.first_price

    # Check if there was meaningful movement
    if abs(final_price - price_before) < MIN_PRICE_CHANGE:
        return None

    return Resolution(
        condition_id=history.condition_id,
        question=history.question,
        resolution_timestamp=resolution_timestamp,
        resolution_outcome=resolution_outcome,
        price_before=price_before,
        price_after=final_price
    )


def main():
    print("=" * 70)
    print("FIND RESOLVED WITHIN-CATEGORY PAIRS")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")

    # Load price histories
    print("\n[1/4] Loading price histories...")
    histories = load_price_histories()
    print(f"      Loaded {len(histories)} market histories")

    # Detect resolutions
    print("\n[2/4] Detecting resolutions...")
    resolutions = {}
    for cond_id, history in histories.items():
        res = detect_resolution(history)
        if res:
            if res.resolution_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                resolutions[cond_id] = res

    print(f"      Found {len(resolutions)} resolved markets (after {MODEL_KNOWLEDGE_CUTOFF})")

    # Process each category
    print("\n[3/4] Finding resolved pairs in each category...")

    categories_to_process = ["fed_monetary", "politics"]
    results = {}

    for cat_name in categories_to_process:
        input_path = INPUT_DIR / f"{cat_name}_pairs.json"
        if not input_path.exists():
            print(f"      {cat_name}: No pairs file found, skipping")
            continue

        with open(input_path) as f:
            pairs = json.load(f)

        resolved_pairs = []
        for pair in pairs:
            cond_a = pair["market_a"]["condition_id"]
            cond_b = pair["market_b"]["condition_id"]
            rho = pair["rho"]

            if math.isnan(rho):
                continue

            res_a = resolutions.get(cond_a)
            res_b = resolutions.get(cond_b)

            if not res_a and not res_b:
                continue

            # Determine which market resolved
            if res_a:
                resolved_market = "A"
                resolution = res_a
                other_cond = cond_b
                other_question = pair["market_b"]["question"]
            else:
                resolved_market = "B"
                resolution = res_b
                other_cond = cond_a
                other_question = pair["market_a"]["question"]

            # Check if we have price history for the other market
            other_history = histories.get(other_cond)
            if not other_history:
                continue

            resolution_ts = resolution.resolution_timestamp

            # Get price before resolution
            price_before = other_history.price_at(resolution_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY)
            if price_before is None:
                price_before = other_history.price_at(resolution_ts)

            # Get price after resolution
            price_after = other_history.price_after(resolution_ts, WINDOW_AFTER_DAYS * SECONDS_PER_DAY)
            if price_after is None:
                final_ts = other_history.last_timestamp
                if final_ts > resolution_ts:
                    price_after = other_history.last_price
                else:
                    continue

            if price_before is None:
                continue

            resolved_pairs.append({
                "pair": pair,
                "resolved_market": resolved_market,
                "resolution": {
                    "condition_id": resolution.condition_id,
                    "question": resolution.question,
                    "outcome": resolution.resolution_outcome,
                    "timestamp": resolution.resolution_timestamp,
                    "date": resolution.resolution_date.isoformat(),
                    "price_before": resolution.price_before,
                    "price_after": resolution.price_after,
                },
                "other_market": {
                    "condition_id": other_cond,
                    "question": other_question,
                    "price_before": price_before,
                    "price_after": price_after,
                }
            })

        results[cat_name] = resolved_pairs
        print(f"      {cat_name}: {len(resolved_pairs)} resolved pairs out of {len(pairs)}")

    # Save results
    print("\n[4/4] Saving results...")
    for cat_name, resolved_pairs in results.items():
        output_path = OUTPUT_DIR / f"resolved_{cat_name}_pairs.json"
        with open(output_path, "w") as f:
            json.dump(resolved_pairs, f, indent=2)
        print(f"      Saved {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for cat_name, resolved_pairs in results.items():
        print(f"\n{cat_name}:")
        print(f"  Resolved pairs: {len(resolved_pairs)}")

        if resolved_pairs:
            # Show resolution date range
            dates = [datetime.fromisoformat(p["resolution"]["date"]) for p in resolved_pairs]
            print(f"  Resolution dates: {min(dates).date()} to {max(dates).date()}")

            # Show examples
            print(f"\n  Examples:")
            for i, rp in enumerate(resolved_pairs[:2]):
                print(f"    {i+1}. Resolved: {rp['resolution']['question'][:50]}...")
                print(f"       -> {rp['resolution']['outcome']} on {rp['resolution']['date'][:10]}")
                print(f"       Other: {rp['other_market']['question'][:50]}...")
                print(f"       rho={rp['pair']['rho']:.2f}")


if __name__ == "__main__":
    main()
