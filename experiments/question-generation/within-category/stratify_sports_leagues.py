#!/usr/bin/env python3
"""
Stratify sports pairs by league to understand within-league vs cross-league VOI.

Goal: Understand if mixing NBA/NFL/soccer adds noise to the r=-0.02 finding.

Outputs:
- Within-league correlations for each sport
- Cross-league correlations
- Comparison with overall sports r=-0.02
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

# League detection patterns
LEAGUE_PATTERNS = {
    "NBA": [
        r"\bNBA\b", r"NBA Finals", r"\bCeltics\b", r"\bLakers\b",
        r"\bBucks\b", r"\bRaptors\b", r"\bKings\b", r"\bRockets\b",
        r"\bWarriors\b", r"\bKnicks\b", r"\b76ers\b", r"\bNuggets\b",
        r"\bHeat\b", r"\bSuns\b", r"\bMavericks\b", r"\bClippers\b",
        r"\bThunder\b", r"\bGrizzlies\b", r"\bTimberwolves\b",
    ],
    "NFL": [
        r"\bNFL\b", r"Super Bowl", r"\bChiefs\b", r"\bEagles\b",
        r"\bBears\b", r"\bTexans\b", r"\bRavens\b", r"\bBills\b",
        r"\bLions\b", r"\b49ers\b", r"\bPackers\b", r"\bCowboys\b",
        r"\bSteelers\b", r"\bBroncos\b", r"\bDolphins\b",
    ],
    "NHL": [
        r"\bNHL\b", r"Stanley Cup", r"\bvs\.\s*\w+", r"\bIslanders\b",
        r"\bJets\b", r"\bPenguins\b", r"\bBruins\b", r"\bOilers\b",
        r"\bMaple Leafs\b", r"\bFlames\b", r"\bCanucks\b",
    ],
    "FIFA": [
        r"FIFA World Cup", r"\bWorld Cup\b",
    ],
    "Soccer_Club": [
        r"win on 20\d{2}-\d{2}-\d{2}", r"\bFC\b", r"Manchester",
        r"Barcelona", r"Real Madrid", r"Bayern", r"Dortmund",
        r"Liverpool", r"Chelsea", r"Arsenal", r"Juventus",
        r"Inter Milan", r"AC Milan", r"PSG", r"Marseille",
    ],
}


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


def analyze_group(name: str, group: list) -> dict:
    """Compute correlation stats for a group of pairs."""
    if len(group) < 3:
        return {
            "name": name,
            "n": len(group),
            "r_voi": None,
            "p_voi": None,
            "r_rho": None,
            "p_rho": None,
            "voi_mean": None,
            "shift_mean": None,
            "msg": "Too few pairs",
        }

    vois = np.array([r["linear_voi"] for r in group])
    shifts = np.array([r["actual_shift"] for r in group])
    rhos = np.array([abs(r["rho"]) for r in group])

    r_voi, p_voi = stats.pearsonr(vois, shifts)
    r_rho, p_rho = stats.pearsonr(rhos, shifts)

    return {
        "name": name,
        "n": len(group),
        "r_voi": float(r_voi),
        "p_voi": float(p_voi),
        "r_rho": float(r_rho),
        "p_rho": float(p_rho),
        "voi_mean": float(np.mean(vois)),
        "shift_mean": float(np.mean(shifts)),
    }


def main():
    print("=" * 70)
    print("SPORTS LEAGUE STRATIFICATION ANALYSIS")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")

    # Load data
    print("\n[1/5] Loading data...")
    histories = load_price_histories()
    print(f"      Price histories: {len(histories)}")

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Total pairs: {len(pairs)}")

    # Filter to sports-sports pairs
    print("\n[2/5] Filtering to sports pairs...")
    sports_pairs = []
    for p in pairs:
        q_a = p["market_a"]["question"]
        q_b = p["market_b"]["question"]
        if is_sports(q_a) and is_sports(q_b):
            sports_pairs.append(p)
    print(f"      Sports-sports pairs: {len(sports_pairs)}")

    # Detect resolutions
    print("\n[3/5] Detecting resolutions...")
    resolutions = {}
    for cond_id, history in histories.items():
        res = detect_resolution(history)
        if res:
            res_date = datetime.fromtimestamp(res["timestamp"], tz=timezone.utc)
            if res_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                resolutions[cond_id] = res
    print(f"      Resolved markets: {len(resolutions)}")

    # Validate sports pairs
    print("\n[4/5] Validating sports pairs...")
    results = []

    for pair in sports_pairs:
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

        # Detect leagues
        league_resolved = detect_league(resolved_q)
        league_other = detect_league(other_q)
        is_within_league = (league_resolved == league_other) and (league_resolved != "Other")

        results.append({
            "resolved_question": resolved_q,
            "other_question": other_q,
            "rho": rho,
            "linear_voi": float(linear_voi),
            "actual_shift": actual_shift,
            "league_resolved": league_resolved,
            "league_other": league_other,
            "is_within_league": is_within_league,
            "league_pair": f"{league_resolved}-{league_other}",
        })

    print(f"      Validated sports pairs: {len(results)}")

    # Split by within/cross league
    within_league = [r for r in results if r["is_within_league"]]
    cross_league = [r for r in results if not r["is_within_league"]]

    print(f"      Within-league: {len(within_league)}")
    print(f"      Cross-league: {len(cross_league)}")

    # Analysis
    print("\n[5/5] Computing correlations...")
    print("\n" + "=" * 70)
    print("RESULTS: ALL SPORTS")
    print("=" * 70)

    all_result = analyze_group("ALL SPORTS", results)
    within_result = analyze_group("WITHIN-LEAGUE", within_league)
    cross_result = analyze_group("CROSS-LEAGUE", cross_league)

    def print_result(r):
        if r["r_voi"] is None:
            print(f"\n{r['name']}: Too few pairs (n={r['n']})")
        else:
            print(f"\n{r['name']} (n={r['n']}):")
            print(f"  VOI mean: {r['voi_mean']:.3f}, shift mean: {r['shift_mean']:.3f}")
            print(f"  r(VOI, shift):  {r['r_voi']:.3f} (p={r['p_voi']:.4f})")
            print(f"  r(|ρ|, shift):  {r['r_rho']:.3f} (p={r['p_rho']:.4f})")

    print_result(all_result)
    print_result(within_result)
    print_result(cross_result)

    # Within-league breakdown by sport
    print("\n" + "-" * 70)
    print("WITHIN-LEAGUE BREAKDOWN BY SPORT")
    print("-" * 70)

    league_groups = defaultdict(list)
    for r in within_league:
        league_groups[r["league_resolved"]].append(r)

    league_results = {}
    for league, group in sorted(league_groups.items(), key=lambda x: -len(x[1])):
        result = analyze_group(f"  {league}", group)
        league_results[league] = result
        print_result(result)

    # Cross-league breakdown
    print("\n" + "-" * 70)
    print("CROSS-LEAGUE PAIR BREAKDOWN")
    print("-" * 70)

    cross_pair_groups = defaultdict(list)
    for r in cross_league:
        # Normalize pair name (alphabetical)
        leagues = sorted([r["league_resolved"], r["league_other"]])
        pair_name = f"{leagues[0]}-{leagues[1]}"
        cross_pair_groups[pair_name].append(r)

    cross_pair_results = {}
    for pair_name, group in sorted(cross_pair_groups.items(), key=lambda x: -len(x[1])):
        result = analyze_group(f"  {pair_name}", group)
        cross_pair_results[pair_name] = result
        if len(group) >= 3:
            print_result(result)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if within_result and within_result["r_voi"] is not None:
        r_within = within_result["r_voi"]
        r_cross = cross_result["r_voi"] if cross_result and cross_result["r_voi"] else 0

        print(f"\nWithin-league r = {r_within:.3f}")
        print(f"Cross-league r = {r_cross:.3f}")

        if abs(r_within - r_cross) < 0.10:
            print("\n→ NO DIFFERENCE: Within and cross-league correlations are similar")
            print("  Mixing leagues is NOT the source of the null finding")
        elif r_within > r_cross + 0.10:
            print("\n→ WITHIN-LEAGUE BETTER: Within-league shows stronger correlation")
            print("  Mixing leagues may be adding noise")
        else:
            print("\n→ CROSS-LEAGUE BETTER: Cross-league shows stronger correlation")
            print("  Within-league may be fundamentally hard")
    else:
        print(f"\nLOW N: Only {len(within_league)} within-league pairs")
        print("Cannot draw strong conclusions about league stratification")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "experiment": "sports_league_stratification",
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "n_total": len(results),
            "n_within_league": len(within_league),
            "n_cross_league": len(cross_league),
            "run_at": datetime.now().isoformat(),
        },
        "results": {
            "all_sports": all_result,
            "within_league": within_result,
            "cross_league": cross_result,
            "by_league": league_results,
            "by_cross_pair": cross_pair_results,
        },
        "pairs": results,
    }

    output_path = OUTPUT_DIR / "sports_league_stratification.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
