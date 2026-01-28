#!/usr/bin/env python3
"""
Validate VOI within-category vs cross-category.

Key question: Does VOI discriminate better within categories or between categories?

Approach:
1. Find all resolved pairs (one market resolved, other has price history)
2. Tag each pair as within-category or cross-category
3. Compute VOI vs actual shift correlation for each group
4. Compare to overall Polymarket (r=0.65) and Russell earnings (r=-0.15)

This tests whether within-category discrimination is fundamentally hard,
or whether the Russell 2000 failure was domain-specific.

Modes:
- Default: Use resolution-based ground truth (original)
- --delta-p: Use Δp (price change) as ground truth for expanded sample
- --league: Add league-level stratification for sports
"""

import json
import re
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from scipy import stats
import numpy as np
import sys

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho

# Import config from conditional-forecasting
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent / "data"

# Resolution thresholds
RESOLUTION_THRESHOLD_HIGH = 0.95
RESOLUTION_THRESHOLD_LOW = 0.05
MIN_PRICE_CHANGE = 0.10

# Windows
WINDOW_BEFORE_DAYS = 3
WINDOW_AFTER_DAYS = 3
SECONDS_PER_DAY = 86400

# Category patterns
CATEGORIES = {
    "crypto": [
        r"\bBitcoin\b", r"\bBTC\b", r"\bEthereum\b", r"\bETH\b",
        r"\bSolana\b", r"\bSOL\b", r"\bcrypto",
    ],
    "fed_monetary": [
        r"\bFed\b.*\b(rate|interest|bps|meeting|decrease|increase)",
        r"\bFOMC\b", r"\binterest rate",
    ],
    "politics": [
        r"\belection\b", r"\bpresident", r"\bnominate",
        r"\bTrump\b", r"\bBiden\b", r"\bRepublican", r"\bDemocrat",
    ],
    "sports": [
        r"\bwin on 2026\b", r"\bvs\.?\b", r"\bNFL\b", r"\bNBA\b",
        r"\bMLB\b", r"\bNHL\b", r"\bFC\b",
    ],
}


def categorize_market(question: str) -> str:
    """Categorize a market. Returns 'other' if no match."""
    for cat_name, patterns in CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return cat_name
    return "other"


class PriceHistory:
    def __init__(self, condition_id: str, question: str, candles: list):
        self.condition_id = condition_id
        self.question = question
        self.candles = candles

    @property
    def last_price(self) -> float:
        return self.candles[-1]["close"] if self.candles else 0.5

    @property
    def last_timestamp(self) -> int:
        return self.candles[-1]["timestamp"] if self.candles else 0

    def price_at(self, timestamp: int) -> float | None:
        for candle in reversed(self.candles):
            if candle["timestamp"] <= timestamp:
                return candle["close"]
        return None

    def price_after(self, timestamp: int, window_seconds: int) -> float | None:
        prices = []
        for candle in self.candles:
            if timestamp < candle["timestamp"] <= timestamp + window_seconds:
                prices.append(candle["close"])
        return np.mean(prices) if prices else None


def load_price_histories() -> dict[str, PriceHistory]:
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = PriceHistory(
            data["condition_id"], data["question"], data["candles"]
        )
    return histories


def detect_resolution(history: PriceHistory):
    """Detect if and when a market resolved."""
    if len(history.candles) < 5:
        return None

    final_price = history.last_price
    if not (final_price < RESOLUTION_THRESHOLD_LOW or final_price > RESOLUTION_THRESHOLD_HIGH):
        return None

    outcome = "YES" if final_price > RESOLUTION_THRESHOLD_HIGH else "NO"
    threshold = RESOLUTION_THRESHOLD_HIGH if outcome == "YES" else RESOLUTION_THRESHOLD_LOW

    for i, candle in enumerate(history.candles):
        if (outcome == "YES" and candle["close"] >= threshold) or \
           (outcome == "NO" and candle["close"] <= threshold):
            if i < 3:
                return None
            before_prices = [c["close"] for c in history.candles[max(0, i-3):i]]
            price_before = np.mean(before_prices) if before_prices else 0.5
            if abs(final_price - price_before) < MIN_PRICE_CHANGE:
                return None
            return {
                "timestamp": candle["timestamp"],
                "outcome": outcome,
                "price_before": price_before,
            }
    return None


def main():
    print("=" * 70)
    print("WITHIN-CATEGORY VS CROSS-CATEGORY VOI VALIDATION")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")

    # Load data
    print("\n[1/4] Loading data...")
    histories = load_price_histories()
    print(f"      Price histories: {len(histories)}")

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Pairs: {len(pairs)}")

    # Detect resolutions
    print("\n[2/4] Detecting resolutions...")
    resolutions = {}
    for cond_id, history in histories.items():
        res = detect_resolution(history)
        if res:
            res_date = datetime.fromtimestamp(res["timestamp"], tz=timezone.utc)
            if res_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                resolutions[cond_id] = res
    print(f"      Resolved markets: {len(resolutions)}")

    # Categorize all markets
    print("\n[3/4] Categorizing markets...")
    market_categories = {}
    for cond_id, history in histories.items():
        market_categories[cond_id] = categorize_market(history.question)

    cat_counts = defaultdict(int)
    for cat in market_categories.values():
        cat_counts[cat] += 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"      {cat}: {count}")

    # Validate pairs
    print("\n[4/4] Validating pairs...")

    results = []
    for pair in pairs:
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        rho = pair["rho"]

        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            continue

        res_a = resolutions.get(cond_a)
        res_b = resolutions.get(cond_b)

        if not res_a and not res_b:
            continue

        # Use whichever resolved
        if res_a:
            resolution = res_a
            resolved_cond = cond_a
            resolved_q = pair["market_a"]["question"]
            other_cond = cond_b
            other_q = pair["market_b"]["question"]
        else:
            resolution = res_b
            resolved_cond = cond_b
            resolved_q = pair["market_b"]["question"]
            other_cond = cond_a
            other_q = pair["market_a"]["question"]

        other_history = histories.get(other_cond)
        if not other_history:
            continue

        resolution_ts = resolution["timestamp"]

        # Get prices for other market
        price_before = other_history.price_at(resolution_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY)
        if price_before is None:
            price_before = other_history.price_at(resolution_ts)
        if price_before is None:
            continue

        price_after = other_history.price_after(resolution_ts, WINDOW_AFTER_DAYS * SECONDS_PER_DAY)
        if price_after is None:
            if other_history.last_timestamp > resolution_ts:
                price_after = other_history.last_price
            else:
                continue

        # Compute VOI and actual shift
        linear_voi = linear_voi_from_rho(rho, price_before, resolution["price_before"])
        actual_shift = abs(price_after - price_before)

        # Categorize
        cat_resolved = market_categories.get(resolved_cond, "other")
        cat_other = market_categories.get(other_cond, "other")
        is_within_category = (cat_resolved == cat_other) and (cat_resolved != "other")

        results.append({
            "resolved_question": resolved_q,
            "other_question": other_q,
            "rho": rho,
            "linear_voi": float(linear_voi),
            "actual_shift": actual_shift,
            "cat_resolved": cat_resolved,
            "cat_other": cat_other,
            "is_within_category": is_within_category,
            "category_pair": f"{cat_resolved}-{cat_other}",
        })

    print(f"      Validated pairs: {len(results)}")

    # Split by within/cross category
    within = [r for r in results if r["is_within_category"]]
    cross = [r for r in results if not r["is_within_category"]]

    print(f"      Within-category: {len(within)}")
    print(f"      Cross-category: {len(cross)}")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def analyze_group(name: str, group: list):
        if len(group) < 3:
            print(f"\n{name}: Too few pairs ({len(group)})")
            return None

        vois = np.array([r["linear_voi"] for r in group])
        shifts = np.array([r["actual_shift"] for r in group])
        rhos = np.array([abs(r["rho"]) for r in group])

        r_voi, p_voi = stats.pearsonr(vois, shifts)
        r_rho, p_rho = stats.pearsonr(rhos, shifts)

        print(f"\n{name} (n={len(group)}):")
        print(f"  VOI mean: {np.mean(vois):.3f}, shift mean: {np.mean(shifts):.3f}")
        print(f"  Correlation (VOI vs shift):  r={r_voi:.3f}, p={p_voi:.4f}")
        print(f"  Correlation (|rho| vs shift): r={r_rho:.3f}, p={p_rho:.4f}")

        return {"r": r_voi, "p": p_voi, "n": len(group)}

    all_result = analyze_group("ALL PAIRS", results)
    within_result = analyze_group("WITHIN-CATEGORY", within)
    cross_result = analyze_group("CROSS-CATEGORY", cross)

    # Category breakdown for within-category
    if within:
        print("\n" + "-" * 70)
        print("WITHIN-CATEGORY BREAKDOWN")
        print("-" * 70)

        cat_groups = defaultdict(list)
        for r in within:
            cat_groups[r["cat_resolved"]].append(r)

        for cat, group in sorted(cat_groups.items(), key=lambda x: -len(x[1])):
            analyze_group(f"  {cat}", group)

    # Comparison with baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)

    print("\nBaselines:")
    print("  Overall Polymarket (previous): r=0.65")
    print("  Russell 2000 earnings:         r=-0.15")

    print("\nCurrent results:")
    if all_result:
        print(f"  All resolved pairs:   r={all_result['r']:.2f} (n={all_result['n']})")
    if within_result:
        print(f"  Within-category:      r={within_result['r']:.2f} (n={within_result['n']})")
    if cross_result:
        print(f"  Cross-category:       r={cross_result['r']:.2f} (n={cross_result['n']})")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if within_result and within_result["n"] >= 10:
        r_within = within_result["r"]
        if r_within > 0.15:
            print("\n POSITIVE: Within-category VOI shows positive correlation")
            print("  -> Russell 2000 failure is likely domain-specific (equities confounds)")
        elif r_within < -0.10:
            print("\n NEGATIVE: Within-category VOI shows negative correlation")
            print("  -> Consistent with Russell 2000; within-category may be hard")
        else:
            print("\n NULL: Within-category VOI shows near-zero correlation")
            print("  -> VOI may be fundamentally a category detector, not ranker")
    else:
        print(f"\n LOW N: Only {within_result['n'] if within_result else 0} within-category pairs")
        print("  -> Cannot draw strong conclusions about within-category discrimination")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "experiment": "within_vs_cross_category_voi",
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "n_total": len(results),
            "n_within": len(within),
            "n_cross": len(cross),
            "run_at": datetime.now().isoformat(),
        },
        "results": {
            "all": all_result,
            "within_category": within_result,
            "cross_category": cross_result,
        },
        "pairs": results,
    }

    output_path = OUTPUT_DIR / "within_vs_cross_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
