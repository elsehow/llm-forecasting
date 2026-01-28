#!/usr/bin/env python3
"""
Validate VOI using Δp (price change) as ground truth for unresolved pairs.

Problem: Only 14 resolved sports pairs (11 within-category in original analysis).
Solution: Use price changes over 1-week window as proxy for "ground truth shift".

Methodology:
- For each pair (A, B), compute Δp = max(|ΔA|, |ΔB|) over 1-week window
- Compare VOI predictions to Δp
- If r(VOI, Δp) > 0.30: Δp method is viable for expansion
- If r(VOI, Δp) < 0.10: Δp is noise; stick with resolved-only

Key insight: We're testing whether VOI predicts which market *will move more*,
not just which resolution *caused* movement. This is a weaker but still useful signal.

Modes:
- --sports-only: Original behavior, only sports pairs
- (default): All pairs, with category breakdown
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

# Time windows
SECONDS_PER_DAY = 86400
DELTA_P_WINDOW_DAYS = 7  # 1-week window for price changes

# Resolution thresholds (for comparison with resolved-only)
RESOLUTION_THRESHOLD_HIGH = 0.95
RESOLUTION_THRESHOLD_LOW = 0.05
MIN_PRICE_CHANGE = 0.10

# Category patterns (matches validate_within_vs_cross.py)
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
        r"\bwin on 20\d{2}\b", r"\bvs\.?\b", r"\bNFL\b", r"\bNBA\b",
        r"\bMLB\b", r"\bNHL\b", r"\bFC\b", r"Super Bowl", r"World Cup",
        r"Championship", r"Finals", r"Lakers", r"Warriors", r"Chiefs",
        r"Eagles", r"MVP", r"Playoffs", r"Stanley Cup",
    ],
}

# League detection patterns (for sports subcategories)
LEAGUE_PATTERNS = {
    "NBA": [
        r"\bNBA\b", r"NBA Finals", r"\bCeltics\b", r"\bLakers\b",
        r"\bBucks\b", r"\bRaptors\b", r"\bKings\b", r"\bRockets\b",
        r"\bWarriors\b", r"\bKnicks\b", r"\b76ers\b", r"\bNuggets\b",
    ],
    "NFL": [
        r"\bNFL\b", r"Super Bowl", r"\bChiefs\b", r"\bEagles\b",
        r"\bBears\b", r"\bTexans\b", r"\bRavens\b", r"\bBills\b",
    ],
    "NHL": [
        r"\bNHL\b", r"Stanley Cup", r"\bvs\.\s*\w+", r"\bIslanders\b",
        r"\bJets\b", r"\bPenguins\b", r"\bBruins\b",
    ],
    "FIFA": [
        r"FIFA World Cup", r"\bWorld Cup\b",
    ],
    "Soccer_Club": [
        r"win on 20\d{2}-\d{2}-\d{2}", r"\bFC\b", r"Manchester",
        r"Barcelona", r"Real Madrid", r"Bayern", r"Dortmund",
    ],
}


def categorize_market(question: str) -> str:
    """Categorize a market. Returns 'other' if no match."""
    for cat_name, patterns in CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return cat_name
    return "other"


def detect_league(question: str) -> str:
    """Detect league from market question."""
    for league, patterns in LEAGUE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return league
    return "Other"


def is_sports(question: str) -> bool:
    """Check if market is sports-related."""
    sports_patterns = [
        r"win on 20\d{2}", r"vs\.", r"\bNFL\b", r"\bNBA\b",
        r"\bMLB\b", r"\bNHL\b", r"\bFC\b", r"Super Bowl", r"World Cup",
        r"Championship", r"Finals", r"Lakers", r"Warriors", r"Chiefs",
        r"Eagles", r"MVP", r"Playoffs", r"Stanley Cup",
    ]
    for pattern in sports_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False


class PriceHistory:
    def __init__(self, condition_id: str, question: str, candles: list):
        self.condition_id = condition_id
        self.question = question
        self.candles = candles

    @property
    def last_price(self) -> float:
        return self.candles[-1]["close"] if self.candles else 0.5

    @property
    def first_timestamp(self) -> int:
        return self.candles[0]["timestamp"] if self.candles else 0

    @property
    def last_timestamp(self) -> int:
        return self.candles[-1]["timestamp"] if self.candles else 0

    def price_at(self, timestamp: int) -> float | None:
        """Get price at or before timestamp."""
        for candle in reversed(self.candles):
            if candle["timestamp"] <= timestamp:
                return candle["close"]
        return None

    def price_at_or_after(self, timestamp: int) -> tuple[float, int] | None:
        """Get price at or after timestamp, return (price, actual_timestamp)."""
        for candle in self.candles:
            if candle["timestamp"] >= timestamp:
                return candle["close"], candle["timestamp"]
        return None

    def delta_p(self, start_ts: int, window_seconds: int) -> float | None:
        """Compute absolute price change over window."""
        start_price = self.price_at(start_ts)
        if start_price is None:
            return None

        end_ts = start_ts + window_seconds
        end_price = self.price_at(end_ts)
        if end_price is None:
            # Use last available price if window extends beyond data
            if self.last_timestamp > start_ts:
                end_price = self.last_price
            else:
                return None

        return abs(end_price - start_price)

    def is_resolved(self) -> bool:
        """Check if market appears resolved."""
        if not self.candles:
            return False
        final_price = self.last_price
        return final_price < RESOLUTION_THRESHOLD_LOW or final_price > RESOLUTION_THRESHOLD_HIGH


def load_price_histories() -> dict[str, PriceHistory]:
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = PriceHistory(
            data["condition_id"], data["question"], data["candles"]
        )
    return histories


def analyze_group(name: str, group: list) -> dict:
    """Compute correlation stats for a group of pairs."""
    if len(group) < 3:
        return {
            "name": name,
            "n": len(group),
            "r": None,
            "p": None,
            "msg": "Too few pairs",
        }

    vois = np.array([r["voi"] for r in group])
    ground_truth = np.array([r["ground_truth"] for r in group])

    # Filter out zero-variance cases
    if np.std(vois) == 0 or np.std(ground_truth) == 0:
        return {
            "name": name,
            "n": len(group),
            "r": None,
            "p": None,
            "msg": "Zero variance in VOI or ground truth",
        }

    r, p = stats.pearsonr(vois, ground_truth)

    return {
        "name": name,
        "n": len(group),
        "r": float(r),
        "p": float(p),
        "voi_mean": float(np.mean(vois)),
        "ground_truth_mean": float(np.mean(ground_truth)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sports-only", action="store_true", help="Only analyze sports pairs (original behavior)")
    args = parser.parse_args()

    mode = "SPORTS ONLY" if args.sports_only else "ALL CATEGORIES"
    print("=" * 70)
    print(f"Δp VALIDATION ({mode})")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")
    print(f"Δp window: {DELTA_P_WINDOW_DAYS} days")

    # Load data
    print("\n[1/5] Loading data...")
    histories = load_price_histories()
    print(f"      Price histories: {len(histories)}")

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Total pairs: {len(pairs)}")

    # Filter pairs if sports-only mode
    if args.sports_only:
        print("\n[2/5] Filtering to sports pairs...")
        filtered_pairs = []
        for p in pairs:
            q_a = p["market_a"]["question"]
            q_b = p["market_b"]["question"]
            if is_sports(q_a) and is_sports(q_b):
                filtered_pairs.append(p)
        pairs = filtered_pairs
        print(f"      Sports-sports pairs: {len(pairs)}")
    else:
        print("\n[2/5] Processing all pairs...")

    # Find common time window across all pairs
    print("\n[3/5] Finding common evaluation window...")

    # Use the model knowledge cutoff as start point
    cutoff_ts = int(datetime.combine(
        MODEL_KNOWLEDGE_CUTOFF,
        datetime.min.time(),
        tzinfo=timezone.utc
    ).timestamp())

    print(f"      Evaluation starts: {MODEL_KNOWLEDGE_CUTOFF}")

    # Compute Δp for each pair
    print("\n[4/5] Computing Δp ground truth...")

    results = []
    skipped_no_history = 0
    skipped_no_overlap = 0
    skipped_nan_rho = 0

    for pair in pairs:
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        rho = pair["rho"]

        # Skip invalid rho
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            skipped_nan_rho += 1
            continue

        # Get histories
        hist_a = histories.get(cond_a)
        hist_b = histories.get(cond_b)

        if not hist_a or not hist_b:
            skipped_no_history += 1
            continue

        # Find overlapping time window after cutoff
        # Start from cutoff or first available data, whichever is later
        start_ts = max(cutoff_ts, hist_a.first_timestamp, hist_b.first_timestamp)

        # Check if we have enough data after start
        window_seconds = DELTA_P_WINDOW_DAYS * SECONDS_PER_DAY
        end_ts = start_ts + window_seconds

        if hist_a.last_timestamp < end_ts or hist_b.last_timestamp < end_ts:
            skipped_no_overlap += 1
            continue

        # Get prices at start of window
        price_a_start = hist_a.price_at(start_ts)
        price_b_start = hist_b.price_at(start_ts)

        if price_a_start is None or price_b_start is None:
            skipped_no_overlap += 1
            continue

        # Compute Δp for each market
        delta_p_a = hist_a.delta_p(start_ts, window_seconds)
        delta_p_b = hist_b.delta_p(start_ts, window_seconds)

        if delta_p_a is None or delta_p_b is None:
            skipped_no_overlap += 1
            continue

        # Ground truth: max movement
        ground_truth = max(delta_p_a, delta_p_b)

        # VOI prediction (using start-of-window prices)
        voi = linear_voi_from_rho(rho, price_a_start, price_b_start)

        # Detect categories
        cat_a = categorize_market(q_a)
        cat_b = categorize_market(q_b)
        is_within_category = (cat_a == cat_b) and (cat_a != "other")

        # Detect leagues (for sports)
        league_a = detect_league(q_a) if cat_a == "sports" else None
        league_b = detect_league(q_b) if cat_b == "sports" else None
        is_within_league = (league_a == league_b) and (league_a is not None) and (league_a != "Other")

        # Resolution status
        resolved_a = hist_a.is_resolved()
        resolved_b = hist_b.is_resolved()

        results.append({
            "question_a": q_a,
            "question_b": q_b,
            "rho": rho,
            "voi": float(voi),
            "delta_p_a": delta_p_a,
            "delta_p_b": delta_p_b,
            "ground_truth": ground_truth,
            "category_a": cat_a,
            "category_b": cat_b,
            "is_within_category": is_within_category,
            "league_a": league_a,
            "league_b": league_b,
            "is_within_league": is_within_league,
            "resolved_a": resolved_a,
            "resolved_b": resolved_b,
            "both_resolved": resolved_a and resolved_b,
            "one_resolved": resolved_a != resolved_b,
            "neither_resolved": not resolved_a and not resolved_b,
        })

    print(f"      Valid pairs for Δp: {len(results)}")
    print(f"      Skipped (no history): {skipped_no_history}")
    print(f"      Skipped (no overlap): {skipped_no_overlap}")
    print(f"      Skipped (NaN rho): {skipped_nan_rho}")

    # Split by resolution status
    both_resolved = [r for r in results if r["both_resolved"]]
    one_resolved = [r for r in results if r["one_resolved"]]
    neither_resolved = [r for r in results if r["neither_resolved"]]

    # Split by within/cross category
    within_category = [r for r in results if r["is_within_category"]]
    cross_category = [r for r in results if not r["is_within_category"]]

    # Sports-specific splits
    sports_pairs = [r for r in results if r["category_a"] == "sports" and r["category_b"] == "sports"]
    within_league = [r for r in sports_pairs if r["is_within_league"]]
    cross_league = [r for r in sports_pairs if not r["is_within_league"]]

    print(f"\n      By resolution status:")
    print(f"        Both resolved: {len(both_resolved)}")
    print(f"        One resolved: {len(one_resolved)}")
    print(f"        Neither resolved: {len(neither_resolved)}")
    print(f"\n      By category:")
    print(f"        Within-category: {len(within_category)}")
    print(f"        Cross-category: {len(cross_category)}")
    if sports_pairs:
        print(f"\n      Sports breakdown:")
        print(f"        Within-league: {len(within_league)}")
        print(f"        Cross-league: {len(cross_league)}")

    # Analysis
    print("\n[5/5] Computing correlations...")
    print("\n" + "=" * 70)
    print("RESULTS: VOI vs Δp CORRELATION")
    print("=" * 70)

    def print_result(r):
        if r["r"] is None:
            print(f"\n{r['name']}: {r.get('msg', 'N/A')} (n={r['n']})")
        else:
            print(f"\n{r['name']} (n={r['n']}):")
            print(f"  VOI mean: {r['voi_mean']:.4f}")
            print(f"  Δp mean: {r['ground_truth_mean']:.4f}")
            print(f"  r(VOI, Δp): {r['r']:.3f} (p={r['p']:.4f})")

    # Overall
    all_result = analyze_group("ALL PAIRS", results)
    print_result(all_result)

    # By resolution status
    print("\n" + "-" * 70)
    print("BY RESOLUTION STATUS")
    print("-" * 70)

    both_result = analyze_group("Both resolved", both_resolved)
    one_result = analyze_group("One resolved", one_resolved)
    neither_result = analyze_group("Neither resolved", neither_resolved)

    print_result(both_result)
    print_result(one_result)
    print_result(neither_result)

    # By within/cross category
    print("\n" + "-" * 70)
    print("BY CATEGORY (WITHIN vs CROSS)")
    print("-" * 70)

    within_cat_result = analyze_group("Within-category (all)", within_category)
    cross_cat_result = analyze_group("Cross-category (all)", cross_category)

    print_result(within_cat_result)
    print_result(cross_cat_result)

    # Category breakdown
    print("\n" + "-" * 70)
    print("WITHIN-CATEGORY BREAKDOWN")
    print("-" * 70)

    category_results = {}
    for cat in ["sports", "fed_monetary", "crypto", "politics"]:
        cat_pairs = [r for r in within_category if r["category_a"] == cat]
        if cat_pairs:
            result = analyze_group(f"  {cat}", cat_pairs)
            category_results[cat] = result
            print_result(result)

    # Sports league breakdown (if applicable)
    if sports_pairs:
        print("\n" + "-" * 70)
        print("SPORTS: WITHIN-LEAGUE vs CROSS-LEAGUE")
        print("-" * 70)

        within_league_result = analyze_group("Within-league", within_league)
        cross_league_result = analyze_group("Cross-league", cross_league)

        print_result(within_league_result)
        print_result(cross_league_result)
    else:
        within_league_result = None
        cross_league_result = None

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if all_result["r"] is not None:
        r_all = all_result["r"]
        baseline_r = 0.65  # Overall Polymarket resolved pairs

        print(f"\nOverall Δp correlation: r = {r_all:.3f}")
        print(f"Baseline (resolved pairs): r = {baseline_r:.3f}")

        if r_all > 0.30:
            print("\n✓ VIABLE: Δp shows meaningful correlation with VOI")
        elif r_all > 0.10:
            print("\n~ MARGINAL: Δp shows weak correlation with VOI")
        else:
            print("\n✗ NOT VIABLE: Δp shows near-zero correlation")

        # Within vs Cross comparison
        if within_cat_result["r"] is not None and cross_cat_result["r"] is not None:
            r_within = within_cat_result["r"]
            r_cross = cross_cat_result["r"]
            print(f"\nWithin-category r = {r_within:.3f} (n={within_cat_result['n']})")
            print(f"Cross-category r = {r_cross:.3f} (n={cross_cat_result['n']})")

            if r_within > 0.20:
                print("→ Within-category VOI shows positive correlation!")
                print("  Original r=-0.02 (n=11) was likely underpowered")
            elif r_within < 0.05:
                print("→ Within-category still shows near-zero correlation")
                print("  Confirms original finding with larger sample")

        # Compare resolved to unresolved
        if neither_result["r"] is not None and both_result["r"] is not None:
            r_unres = neither_result["r"]
            r_res = both_result["r"]
            print(f"\nResolved pairs r = {r_res:.3f}")
            print(f"Unresolved pairs r = {r_unres:.3f}")

    else:
        print("\nInsufficient data for Δp validation")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "experiment": "delta_p_validation_all_categories",
            "mode": "sports_only" if args.sports_only else "all_categories",
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "delta_p_window_days": DELTA_P_WINDOW_DAYS,
            "n_total": len(results),
            "n_both_resolved": len(both_resolved),
            "n_one_resolved": len(one_resolved),
            "n_neither_resolved": len(neither_resolved),
            "n_within_category": len(within_category),
            "n_cross_category": len(cross_category),
            "n_sports": len(sports_pairs),
            "n_within_league": len(within_league),
            "run_at": datetime.now().isoformat(),
        },
        "results": {
            "all": all_result,
            "both_resolved": both_result,
            "one_resolved": one_result,
            "neither_resolved": neither_result,
            "within_category": within_cat_result,
            "cross_category": cross_cat_result,
            "by_category": category_results,
            "within_league": within_league_result,
            "cross_league": cross_league_result,
        },
        "pairs": results,
    }

    output_path = OUTPUT_DIR / "delta_p_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
