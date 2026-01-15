#!/usr/bin/env python3
"""
Phase 0: Validate VOI as a metric for question evaluation.

Key insight: Before using VOI to evaluate question generation, we need to validate
that VOI actually measures something real. This experiment tests:

    Do high-VOI pairs show larger actual price shifts when one market resolves?

If high-VOI pairs predict larger shifts and we observe larger shifts, VOI rankings
are meaningful and we can use them to evaluate generated questions.

Approach:
1. Find Polymarket pairs where one market resolved (price → 0 or 1)
2. For each resolved pair:
   - Compute predicted shift using ρ and Linear VOI
   - Measure actual shift in the other market around resolution time
3. Test correlation: predicted_shift ~ actual_shift

Usage:
    uv run python experiments/question-generation/voi-validation/validate_voi.py
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from scipy import stats
import numpy as np

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
RESOLUTION_THRESHOLD_HIGH = 0.95  # Price above this = resolved YES
RESOLUTION_THRESHOLD_LOW = 0.05   # Price below this = resolved NO
MIN_PRICE_CHANGE = 0.10           # Minimum price change to count as "resolved"

# Analysis windows
WINDOW_BEFORE_DAYS = 3  # Days before resolution to measure "before" price
WINDOW_AFTER_DAYS = 3   # Days after resolution to measure "after" price
SECONDS_PER_DAY = 86400


@dataclass
class PriceHistory:
    """Price history for a single market."""
    condition_id: str
    question: str
    candles: list[dict]  # [{timestamp, open, high, low, close}, ...]

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
    resolution_outcome: str  # "YES" or "NO"
    price_before: float
    price_after: float

    @property
    def resolution_date(self) -> datetime:
        return datetime.fromtimestamp(self.resolution_timestamp, tz=timezone.utc)


@dataclass
class ValidationResult:
    """Result of validating one pair."""
    pair_id: str
    question_a: str
    question_b: str
    rho: float

    # Which market resolved
    resolved_market: str  # "A" or "B"
    resolution_outcome: str
    resolution_date: datetime

    # Predicted shift (from VOI)
    p_a_before: float
    predicted_shift: float  # Expected |P(A|B) - P(A)|

    # Actual shift
    actual_shift: float  # Observed |P(A after) - P(A before)|

    # Metadata
    n_observations: int


def load_price_histories() -> dict[str, PriceHistory]:
    """Load all price histories from disk."""
    histories = {}

    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)

        # Handle truncated condition_id in filename
        condition_id = data["condition_id"]
        histories[condition_id] = PriceHistory(
            condition_id=condition_id,
            question=data["question"],
            candles=data["candles"]
        )

    return histories


def detect_resolution(history: PriceHistory) -> Resolution | None:
    """
    Detect if a market resolved during its price history.

    A market is considered "resolved" if:
    1. Its final price is extreme (< 0.05 or > 0.95)
    2. It moved significantly from its earlier price

    Returns the resolution event, or None if no resolution detected.
    """
    if len(history.candles) < 5:
        return None

    final_price = history.last_price

    # Check if final price is extreme
    if not (final_price < RESOLUTION_THRESHOLD_LOW or final_price > RESOLUTION_THRESHOLD_HIGH):
        return None

    # Find when the resolution happened (first time price went extreme)
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
        return None  # Need some history before resolution

    resolution_timestamp = history.candles[resolution_idx]["timestamp"]

    # Get price before resolution (average of 3 candles before)
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


def compute_predicted_shift(rho: float, p_b: float, p_a: float = 0.5) -> float:
    """Compute expected belief shift using canonical Linear VOI.

    Given ρ between A and B, estimate how much P(A) should shift when B resolves.
    Uses the canonical formula which converts ρ to posteriors and computes
    expected absolute belief shift.

    Args:
        rho: Correlation coefficient between A and B
        p_b: P(B) - probability of the signal that resolves
        p_a: P(A) - prior probability of A (defaults to 0.5 if unknown)
    """
    if math.isnan(rho):
        return 0.0

    return linear_voi_from_rho(rho, p_a, p_b)


def validate_pair(
    pair: dict,
    histories: dict[str, PriceHistory],
    resolutions: dict[str, Resolution]
) -> ValidationResult | None:
    """
    Validate VOI prediction for a single pair.

    If one market in the pair resolved, measure:
    1. Predicted shift (from ρ)
    2. Actual shift (from price data)
    """
    cond_a = pair["market_a"]["condition_id"]
    cond_b = pair["market_b"]["condition_id"]
    rho = pair["rho"]

    if math.isnan(rho):
        return None

    # Check if either market resolved
    res_a = resolutions.get(cond_a)
    res_b = resolutions.get(cond_b)

    if not res_a and not res_b:
        return None  # Neither resolved

    # Prefer the one that resolved (if both, use A)
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

    # Get price history for the other market
    other_history = histories.get(other_cond)
    if not other_history:
        return None

    # Measure actual shift in other market around resolution time
    resolution_ts = resolution.resolution_timestamp

    # Price before resolution
    price_before = other_history.price_at(resolution_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY)
    if price_before is None:
        price_before = other_history.price_at(resolution_ts)

    # Price after resolution
    price_after = other_history.price_after(resolution_ts, WINDOW_AFTER_DAYS * SECONDS_PER_DAY)
    if price_after is None:
        # Use final price if no data in window
        final_ts = other_history.last_timestamp
        if final_ts > resolution_ts:
            price_after = other_history.last_price
        else:
            return None  # No post-resolution data

    if price_before is None:
        return None

    actual_shift = abs(price_after - price_before)

    # Compute predicted shift
    predicted_shift = compute_predicted_shift(rho, resolution.price_before)

    return ValidationResult(
        pair_id=f"{cond_a[:8]}_{cond_b[:8]}",
        question_a=pair["market_a"]["question"],
        question_b=pair["market_b"]["question"],
        rho=rho,
        resolved_market=resolved_market,
        resolution_outcome=resolution.resolution_outcome,
        resolution_date=resolution.resolution_date,
        p_a_before=price_before,
        predicted_shift=predicted_shift,
        actual_shift=actual_shift,
        n_observations=pair["n_observations"]
    )


def run_validation():
    """Run the full VOI validation experiment."""
    print("=" * 70)
    print("PHASE 0: VOI VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")
    print(f"Resolution thresholds: < {RESOLUTION_THRESHOLD_LOW} or > {RESOLUTION_THRESHOLD_HIGH}")
    print(f"Window: {WINDOW_BEFORE_DAYS} days before / {WINDOW_AFTER_DAYS} days after")

    # Step 1: Load price histories
    print("\n[1/4] Loading price histories...")
    histories = load_price_histories()
    print(f"      Loaded {len(histories)} market histories")

    # Step 2: Detect resolutions
    print("\n[2/4] Detecting market resolutions...")
    resolutions = {}
    for cond_id, history in histories.items():
        res = detect_resolution(history)
        if res:
            # Filter by cutoff date
            if res.resolution_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                resolutions[cond_id] = res

    print(f"      Found {len(resolutions)} resolved markets (after {MODEL_KNOWLEDGE_CUTOFF})")

    if not resolutions:
        print("\n⚠️  No resolved markets found after cutoff date!")
        print("    This may mean:")
        print("    - Markets haven't resolved yet (data is too recent)")
        print("    - Resolution detection thresholds are too strict")
        print("    - Need to refresh price history data")
        return None

    # Show some examples
    print("\n      Example resolutions:")
    for i, (cond_id, res) in enumerate(list(resolutions.items())[:5]):
        print(f"      - {res.question[:60]}...")
        print(f"        → {res.resolution_outcome} on {res.resolution_date.date()}")

    # Step 3: Load pairs and validate
    print("\n[3/4] Loading pairs and computing validation metrics...")
    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Loaded {len(pairs)} pairs")

    results = []
    for pair in pairs:
        result = validate_pair(pair, histories, resolutions)
        if result:
            results.append(result)

    print(f"      Validated {len(results)} pairs with resolution data")

    if len(results) < 5:
        print("\n⚠️  Too few validated pairs for meaningful analysis!")
        print("    Need at least 5 pairs with resolution data.")
        return None

    # Step 4: Analyze results
    print("\n[4/4] Analyzing results...")

    predicted = np.array([r.predicted_shift for r in results])
    actual = np.array([r.actual_shift for r in results])
    rhos = np.array([abs(r.rho) for r in results])

    # Correlation between predicted and actual shifts
    if len(results) >= 3:
        corr_pred_actual, pval_pred_actual = stats.pearsonr(predicted, actual)
        corr_rho_actual, pval_rho_actual = stats.pearsonr(rhos, actual)
    else:
        corr_pred_actual, pval_pred_actual = float('nan'), float('nan')
        corr_rho_actual, pval_rho_actual = float('nan'), float('nan')

    # Summary statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nN validated pairs: {len(results)}")
    print(f"\nPredicted shift (from ρ):")
    print(f"  Mean: {np.mean(predicted):.3f}")
    print(f"  Std:  {np.std(predicted):.3f}")
    print(f"  Min:  {np.min(predicted):.3f}")
    print(f"  Max:  {np.max(predicted):.3f}")

    print(f"\nActual shift (observed):")
    print(f"  Mean: {np.mean(actual):.3f}")
    print(f"  Std:  {np.std(actual):.3f}")
    print(f"  Min:  {np.min(actual):.3f}")
    print(f"  Max:  {np.max(actual):.3f}")

    print(f"\n|ρ| (absolute correlation):")
    print(f"  Mean: {np.mean(rhos):.3f}")
    print(f"  Std:  {np.std(rhos):.3f}")

    print("\n" + "-" * 70)
    print("KEY TEST: Do high-VOI pairs show larger actual shifts?")
    print("-" * 70)

    print(f"\nCorrelation (predicted_shift ~ actual_shift):")
    print(f"  r = {corr_pred_actual:.3f}, p = {pval_pred_actual:.4f}")

    print(f"\nCorrelation (|ρ| ~ actual_shift):")
    print(f"  r = {corr_rho_actual:.3f}, p = {pval_rho_actual:.4f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if pval_pred_actual < 0.05 and corr_pred_actual > 0:
        print("\n✅ VALIDATED: High-VOI pairs show larger actual shifts (p < 0.05)")
        print("   → VOI rankings are meaningful for question evaluation")
    elif pval_pred_actual < 0.10 and corr_pred_actual > 0:
        print("\n⚠️  MARGINAL: Positive correlation but not significant (p < 0.10)")
        print("   → More data needed, but trend is encouraging")
    elif corr_pred_actual > 0:
        print("\n⚠️  WEAK: Positive correlation but not significant")
        print("   → Direction is right, but signal is noisy")
    else:
        print("\n❌ NOT VALIDATED: No positive correlation between predicted and actual shifts")
        print("   → VOI may not capture real information flow in markets")

    # Bucket analysis
    print("\n" + "-" * 70)
    print("BUCKET ANALYSIS")
    print("-" * 70)

    buckets = {"strong": [], "moderate": [], "weak": [], "independent": []}
    for r in results:
        rho_abs = abs(r.rho)
        if rho_abs >= 0.6:
            buckets["strong"].append(r.actual_shift)
        elif rho_abs >= 0.3:
            buckets["moderate"].append(r.actual_shift)
        elif rho_abs >= 0.1:
            buckets["weak"].append(r.actual_shift)
        else:
            buckets["independent"].append(r.actual_shift)

    print("\nMean actual shift by |ρ| bucket:")
    for bucket, shifts in buckets.items():
        if shifts:
            print(f"  {bucket:12s}: {np.mean(shifts):.3f} (n={len(shifts)})")
        else:
            print(f"  {bucket:12s}: no data")

    # Top examples
    print("\n" + "-" * 70)
    print("TOP 5 PAIRS BY PREDICTED SHIFT")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda r: r.predicted_shift, reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. ρ = {r.rho:.2f}, predicted = {r.predicted_shift:.3f}, actual = {r.actual_shift:.3f}")
        if r.resolved_market == "A":
            print(f"   Resolved: {r.question_a[:60]}...")
            print(f"   Other:    {r.question_b[:60]}...")
        else:
            print(f"   Resolved: {r.question_b[:60]}...")
            print(f"   Other:    {r.question_a[:60]}...")
        print(f"   → {r.resolution_outcome} on {r.resolution_date.date()}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "experiment": "voi_validation_phase0",
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "n_markets": len(histories),
            "n_resolutions": len(resolutions),
            "n_validated_pairs": len(results),
            "run_at": datetime.now().isoformat(),
        },
        "summary": {
            "predicted_shift_mean": float(np.mean(predicted)),
            "predicted_shift_std": float(np.std(predicted)),
            "actual_shift_mean": float(np.mean(actual)),
            "actual_shift_std": float(np.std(actual)),
            "correlation_pred_actual": float(corr_pred_actual) if not math.isnan(corr_pred_actual) else None,
            "pvalue_pred_actual": float(pval_pred_actual) if not math.isnan(pval_pred_actual) else None,
            "correlation_rho_actual": float(corr_rho_actual) if not math.isnan(corr_rho_actual) else None,
            "pvalue_rho_actual": float(pval_rho_actual) if not math.isnan(pval_rho_actual) else None,
        },
        "bucket_analysis": {
            bucket: {
                "n": len(shifts),
                "mean_actual_shift": float(np.mean(shifts)) if shifts else None
            }
            for bucket, shifts in buckets.items()
        },
        "pairs": [
            {
                "pair_id": r.pair_id,
                "question_a": r.question_a,
                "question_b": r.question_b,
                "rho": r.rho,
                "resolved_market": r.resolved_market,
                "resolution_outcome": r.resolution_outcome,
                "resolution_date": r.resolution_date.isoformat(),
                "p_a_before": r.p_a_before,
                "predicted_shift": r.predicted_shift,
                "actual_shift": r.actual_shift,
            }
            for r in results
        ]
    }

    output_path = RESULTS_DIR / "voi_validation_phase0.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved results to {output_path}")

    return output


if __name__ == "__main__":
    run_validation()
