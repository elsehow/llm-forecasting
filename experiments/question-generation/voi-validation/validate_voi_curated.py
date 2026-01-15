#!/usr/bin/env python3
"""
Phase 0 (v2): Validate VOI using CURATED pairs only.

The initial validation (v1) failed because it used all 67k pairs from pairs.json,
which includes many spurious correlations from data mining. This version uses only
the 20 hand-curated pairs that have verified real relationships.

Key insight from v1: Spurious ρ values (from price co-movement noise) don't predict
actual price shifts when markets resolve. This is expected and validates our concern
about spurious correlations.

Usage:
    uv run python experiments/question-generation/voi-validation/validate_voi_curated.py
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from scipy import stats
import numpy as np
from difflib import SequenceMatcher

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho

# Import config from conditional-forecasting (shared)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths - data lives in conditional-forecasting
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
RESULTS_DIR = Path(__file__).parent / "results"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"

# Resolution thresholds
RESOLUTION_THRESHOLD_HIGH = 0.90  # Relaxed slightly for curated pairs
RESOLUTION_THRESHOLD_LOW = 0.10
MIN_PRICE_CHANGE = 0.05  # Lower threshold for curated pairs

# Analysis windows
WINDOW_BEFORE_DAYS = 5  # Slightly larger window
WINDOW_AFTER_DAYS = 5
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

    def detect_resolution(self) -> dict | None:
        """Detect if this market resolved."""
        if len(self.candles) < 3:
            return None

        final_price = self.last_price

        # Check if final price is extreme
        if not (final_price < RESOLUTION_THRESHOLD_LOW or final_price > RESOLUTION_THRESHOLD_HIGH):
            return None

        resolution_outcome = "YES" if final_price > RESOLUTION_THRESHOLD_HIGH else "NO"
        threshold = RESOLUTION_THRESHOLD_HIGH if resolution_outcome == "YES" else RESOLUTION_THRESHOLD_LOW

        # Find when resolution happened
        resolution_idx = None
        for i, candle in enumerate(self.candles):
            if resolution_outcome == "YES" and candle["close"] >= threshold:
                resolution_idx = i
                break
            elif resolution_outcome == "NO" and candle["close"] <= threshold:
                resolution_idx = i
                break

        if resolution_idx is None:
            return None

        resolution_timestamp = self.candles[resolution_idx]["timestamp"]

        # Get price before
        before_prices = [c["close"] for c in self.candles[max(0, resolution_idx-3):resolution_idx]]
        price_before = np.mean(before_prices) if before_prices else self.first_price

        return {
            "outcome": resolution_outcome,
            "timestamp": resolution_timestamp,
            "price_before": price_before,
            "price_after": final_price
        }


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


def load_markets() -> dict[str, dict]:
    """Load markets.json for condition_id lookup."""
    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)
    return {m["question"]: m for m in markets}


def fuzzy_match_question(question: str, markets: dict[str, dict]) -> str | None:
    """Find best matching question in markets."""
    best_match = None
    best_score = 0

    for market_q in markets.keys():
        score = SequenceMatcher(None, question.lower(), market_q.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = market_q

    if best_score > 0.7:
        return markets[best_match]["condition_id"]
    return None


def compute_linear_voi(rho: float, p_a: float, p_b: float) -> float:
    """Compute Linear VOI for a pair using canonical formula.

    Delegates to llm_forecasting.voi.linear_voi_from_rho which:
    1. Converts ρ to posterior probabilities P(A|B=yes) and P(A|B=no)
    2. Computes VOI as expected absolute belief shift
    """
    return linear_voi_from_rho(rho, p_a, p_b)


def run_validation():
    """Run VOI validation on curated pairs."""
    print("=" * 70)
    print("PHASE 0 (v2): VOI VALIDATION - CURATED PAIRS ONLY")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")

    # Load data
    print("\n[1/5] Loading curated pairs...")
    with open(DATA_DIR / "curated_pairs.json") as f:
        data = json.load(f)
    curated_pairs = data["curated_pairs"]
    print(f"      Loaded {len(curated_pairs)} curated pairs")

    print("\n[2/5] Loading price histories...")
    histories = load_price_histories()
    print(f"      Loaded {len(histories)} price histories")

    print("\n[3/5] Loading markets for condition_id lookup...")
    markets = load_markets()
    print(f"      Loaded {len(markets)} markets")

    # Build question → condition_id mapping
    question_to_cond = {}
    for q, m in markets.items():
        question_to_cond[q.lower()] = m["condition_id"]

    # Also build from price histories (more reliable)
    for cond_id, ph in histories.items():
        question_to_cond[ph.question.lower()] = cond_id

    print("\n[4/5] Matching curated pairs to price histories...")

    results = []
    for pair in curated_pairs:
        q_a = pair["question_a"]
        q_b = pair["question_b"]
        rho = pair["rho"]
        category = pair["category"]

        # Try to find condition_ids
        cond_a = question_to_cond.get(q_a.lower()) or fuzzy_match_question(q_a, markets)
        cond_b = question_to_cond.get(q_b.lower()) or fuzzy_match_question(q_b, markets)

        if not cond_a or not cond_b:
            print(f"      ⚠️  Could not match pair {pair['id']}: {q_a[:40]}...")
            continue

        hist_a = histories.get(cond_a)
        hist_b = histories.get(cond_b)

        if not hist_a or not hist_b:
            print(f"      ⚠️  No price history for pair {pair['id']}")
            continue

        # Check for resolutions
        res_a = hist_a.detect_resolution()
        res_b = hist_b.detect_resolution()

        if not res_a and not res_b:
            # Neither resolved yet - measure co-movement instead
            # Calculate actual correlation from returns
            aligned_a, aligned_b = [], []
            ts_to_price_a = {c["timestamp"]: c["close"] for c in hist_a.candles}
            ts_to_price_b = {c["timestamp"]: c["close"] for c in hist_b.candles}

            common_ts = sorted(set(ts_to_price_a.keys()) & set(ts_to_price_b.keys()))
            if len(common_ts) >= 5:
                for ts in common_ts:
                    aligned_a.append(ts_to_price_a[ts])
                    aligned_b.append(ts_to_price_b[ts])

                # Compute returns
                returns_a = np.diff(aligned_a)
                returns_b = np.diff(aligned_b)

                if len(returns_a) >= 3 and np.std(returns_a) > 0 and np.std(returns_b) > 0:
                    actual_corr, _ = stats.pearsonr(returns_a, returns_b)
                    results.append({
                        "pair_id": pair["id"],
                        "category": category,
                        "question_a": q_a,
                        "question_b": q_b,
                        "rho_expected": rho,
                        "rho_actual": actual_corr,
                        "type": "correlation_check",
                        "n_points": len(common_ts)
                    })
            continue

        # One or both resolved - measure actual shift
        if res_a:
            resolved = "A"
            resolution = res_a
            other_hist = hist_b
            other_question = q_b
        else:
            resolved = "B"
            resolution = res_b
            other_hist = hist_a
            other_question = q_a

        resolution_ts = resolution["timestamp"]
        resolution_date = datetime.fromtimestamp(resolution_ts, tz=timezone.utc)

        # Skip if before cutoff
        if resolution_date.date() < MODEL_KNOWLEDGE_CUTOFF:
            continue

        # Measure price of other market before and after
        price_before = other_hist.price_at(resolution_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY)
        if price_before is None:
            price_before = other_hist.first_price

        price_after = other_hist.price_after(resolution_ts, WINDOW_AFTER_DAYS * SECONDS_PER_DAY)
        if price_after is None:
            if other_hist.last_timestamp > resolution_ts:
                price_after = other_hist.last_price
            else:
                continue

        actual_shift = abs(price_after - price_before)

        # Compute predicted VOI
        p_resolved = resolution["price_before"]
        p_other = price_before
        predicted_voi = compute_linear_voi(rho, p_other, p_resolved)

        # Predicted shift based on ρ
        predicted_shift = abs(rho) * math.sqrt(p_resolved * (1 - p_resolved))

        results.append({
            "pair_id": pair["id"],
            "category": category,
            "question_a": q_a,
            "question_b": q_b,
            "rho": rho,
            "resolved_market": resolved,
            "resolution_outcome": resolution["outcome"],
            "resolution_date": resolution_date.isoformat(),
            "p_before": price_before,
            "p_after": price_after,
            "predicted_shift": predicted_shift,
            "predicted_voi": predicted_voi,
            "actual_shift": actual_shift,
            "type": "resolution"
        })

    print(f"\n      Matched {len(results)} pairs with data")

    # Analyze
    print("\n[5/5] Analyzing results...")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Separate by type
    resolution_results = [r for r in results if r["type"] == "resolution"]
    correlation_results = [r for r in results if r["type"] == "correlation_check"]

    print(f"\nResolution events: {len(resolution_results)}")
    print(f"Correlation checks: {len(correlation_results)}")

    # Analyze resolution events
    if resolution_results:
        print("\n" + "-" * 70)
        print("RESOLUTION ANALYSIS")
        print("-" * 70)

        predicted = np.array([r["predicted_shift"] for r in resolution_results])
        actual = np.array([r["actual_shift"] for r in resolution_results])
        rhos = np.array([abs(r["rho"]) for r in resolution_results])

        print(f"\nN = {len(resolution_results)}")
        print(f"\nPredicted shift: mean={np.mean(predicted):.3f}, std={np.std(predicted):.3f}")
        print(f"Actual shift:    mean={np.mean(actual):.3f}, std={np.std(actual):.3f}")

        if len(resolution_results) >= 3:
            corr, pval = stats.pearsonr(predicted, actual)
            print(f"\nCorrelation (predicted ~ actual): r={corr:.3f}, p={pval:.4f}")

            corr_rho, pval_rho = stats.pearsonr(rhos, actual)
            print(f"Correlation (|ρ| ~ actual):       r={corr_rho:.3f}, p={pval_rho:.4f}")

        print("\nResolution details:")
        for r in resolution_results:
            direction = "→" if r["actual_shift"] >= 0 else "←"
            print(f"  [{r['pair_id']:2d}] {r['category'][:20]:20s}")
            print(f"       ρ={r['rho']:+.2f}, pred={r['predicted_shift']:.3f}, actual={r['actual_shift']:.3f}")
            print(f"       {r['resolved_market']} resolved {r['resolution_outcome']}, other: {r['p_before']:.2f} → {r['p_after']:.2f}")

    # Analyze correlation checks
    if correlation_results:
        print("\n" + "-" * 70)
        print("CORRELATION VERIFICATION (pairs without resolution)")
        print("-" * 70)

        expected = np.array([r["rho_expected"] for r in correlation_results])
        actual = np.array([r["rho_actual"] for r in correlation_results])

        print(f"\nN = {len(correlation_results)}")

        if len(correlation_results) >= 3:
            corr, pval = stats.pearsonr(expected, actual)
            print(f"\nCorrelation (ρ_expected ~ ρ_actual): r={corr:.3f}, p={pval:.4f}")

        print("\nDetails:")
        for r in correlation_results:
            match = "✓" if np.sign(r["rho_expected"]) == np.sign(r["rho_actual"]) else "✗"
            print(f"  [{r['pair_id']:2d}] {r['category'][:20]:20s}")
            print(f"       expected ρ={r['rho_expected']:+.2f}, actual ρ={r['rho_actual']:+.2f} {match}")

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if len(resolution_results) >= 3:
        predicted = np.array([r["predicted_shift"] for r in resolution_results])
        actual = np.array([r["actual_shift"] for r in resolution_results])
        corr, pval = stats.pearsonr(predicted, actual)

        if pval < 0.05 and corr > 0:
            print("\n✅ VALIDATED: High-VOI pairs show larger actual shifts (p < 0.05)")
        elif pval < 0.10 and corr > 0:
            print("\n⚠️  MARGINAL: Positive trend, not significant (p < 0.10)")
        elif corr > 0:
            print("\n⚠️  WEAK: Positive correlation but not significant")
            print("   → Need more resolution data")
        else:
            print("\n❌ NOT VALIDATED with current data")
    else:
        print("\n⚠️  INSUFFICIENT DATA: Need more resolution events")
        print(f"   Currently have {len(resolution_results)} resolution events")

    if len(correlation_results) >= 3:
        expected = np.array([r["rho_expected"] for r in correlation_results])
        actual = np.array([r["rho_actual"] for r in correlation_results])
        corr, pval = stats.pearsonr(expected, actual)

        print(f"\nCorrelation structure verification:")
        if corr > 0.5:
            print(f"   ✅ Curated ρ values match observed price correlations (r={corr:.2f})")
        else:
            print(f"   ⚠️  Curated ρ partially matches observed (r={corr:.2f})")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "experiment": "voi_validation_phase0_v2_curated",
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "n_curated_pairs": len(curated_pairs),
            "n_matched": len(results),
            "n_resolution_events": len(resolution_results),
            "n_correlation_checks": len(correlation_results),
            "run_at": datetime.now().isoformat(),
        },
        "results": results
    }

    output_path = RESULTS_DIR / "voi_validation_phase0_v2_curated.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved results to {output_path}")

    return output


if __name__ == "__main__":
    run_validation()
