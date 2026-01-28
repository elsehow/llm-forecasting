#!/usr/bin/env python3
"""
Step 5: Validate VOI predictions against actual price shifts.

When crux markets resolve, this script:
1. Detects which crux markets have resolved
2. Measures actual shift in ultimate price (7-day window)
3. Correlates predicted shift (VOI) with actual shift
4. Compares r to human-curated baseline (r=0.65)

Usage:
    uv run python experiments/question-generation/llm-crux-validation/validate.py

Note: Run this AFTER crux markets have resolved (2+ weeks from experiment start).
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting" / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
RESULTS_DIR = Path(__file__).parent / "results"

# Resolution detection thresholds
RESOLUTION_THRESHOLD_HIGH = 0.90
RESOLUTION_THRESHOLD_LOW = 0.10

# Shift measurement window
SHIFT_WINDOW_DAYS = 7


def load_price_history(condition_id: str) -> list[dict] | None:
    """Load price history for a market."""
    price_file = PRICE_HISTORY_DIR / f"{condition_id[:40]}.json"
    if not price_file.exists():
        return None

    try:
        with open(price_file) as f:
            data = json.load(f)
        return data.get("candles", [])
    except Exception:
        pass
    return None


def detect_resolution(candles: list[dict]) -> tuple[str, int, datetime] | None:
    """Detect if market has resolved.

    Returns (outcome, resolution_idx, resolution_date) or None.
    """
    if len(candles) < 3:
        return None

    final_price = candles[-1]["close"]
    if final_price > RESOLUTION_THRESHOLD_HIGH:
        outcome = "YES"
    elif final_price < RESOLUTION_THRESHOLD_LOW:
        outcome = "NO"
    else:
        return None

    # Find when it first crossed threshold
    threshold = RESOLUTION_THRESHOLD_HIGH if outcome == "YES" else RESOLUTION_THRESHOLD_LOW
    for i, candle in enumerate(candles):
        if outcome == "YES" and candle["close"] >= threshold:
            resolution_date = datetime.fromtimestamp(candle["timestamp"], tz=timezone.utc)
            return outcome, i, resolution_date
        elif outcome == "NO" and candle["close"] <= threshold:
            resolution_date = datetime.fromtimestamp(candle["timestamp"], tz=timezone.utc)
            return outcome, i, resolution_date

    return None


def measure_shift(
    ultimate_candles: list[dict],
    resolution_ts: float,
) -> tuple[float, float, float] | None:
    """Measure ultimate price shift around crux resolution.

    Returns (price_before, price_after, shift) or None.
    """
    if not ultimate_candles:
        return None

    # Find candles before and after resolution
    before_candles = [c for c in ultimate_candles if c["timestamp"] < resolution_ts]
    after_candles = [c for c in ultimate_candles if c["timestamp"] > resolution_ts]

    if not before_candles or not after_candles:
        return None

    # Price before: average of last 3 candles before resolution
    n_before = min(3, len(before_candles))
    price_before = np.mean([c["close"] for c in before_candles[-n_before:]])

    # Price after: use 7-day window (or all available if less)
    window_end = resolution_ts + SHIFT_WINDOW_DAYS * 86400
    window_candles = [c for c in after_candles if c["timestamp"] <= window_end]

    if not window_candles:
        # Use last available candle if no window candles
        price_after = after_candles[-1]["close"]
    else:
        # Average of candles in window
        price_after = np.mean([c["close"] for c in window_candles])

    shift = price_after - price_before
    return price_before, price_after, shift


def main():
    print("=" * 70)
    print("VALIDATE VOI PREDICTIONS AGAINST ACTUAL SHIFTS")
    print("=" * 70)

    # Load pairs with VOI
    voi_path = RESULTS_DIR / "pairs_with_voi.json"
    if not voi_path.exists():
        print(f"\n❌ {voi_path} not found. Run compute_voi.py first.")
        return

    with open(voi_path) as f:
        voi_data = json.load(f)
    pairs = [p for p in voi_data["pairs_with_voi"] if p.get("voi_computed")]
    print(f"\nLoaded {len(pairs)} pairs with VOI")

    # Check for resolutions
    print("\nChecking for crux market resolutions...")
    resolved_pairs = []
    unresolved_count = 0
    no_history_count = 0

    for pair in pairs:
        crux_market_id = pair["match"]["market_id"]
        ultimate_id = pair["ultimate_id"]

        # Load crux market history
        crux_candles = load_price_history(crux_market_id)
        if not crux_candles:
            no_history_count += 1
            continue

        # Check if resolved
        resolution = detect_resolution(crux_candles)
        if not resolution:
            unresolved_count += 1
            continue

        outcome, res_idx, res_date = resolution
        resolution_ts = crux_candles[res_idx]["timestamp"]

        # Load ultimate history
        ultimate_candles = load_price_history(ultimate_id)
        if not ultimate_candles:
            continue

        # Measure shift
        shift_result = measure_shift(ultimate_candles, resolution_ts)
        if shift_result is None:
            continue

        price_before, price_after, actual_shift = shift_result

        resolved_pairs.append({
            **pair,
            "crux_resolved": True,
            "crux_outcome": outcome,
            "resolution_date": res_date.isoformat(),
            "ultimate_price_before": price_before,
            "ultimate_price_after": price_after,
            "actual_shift": actual_shift,
            "actual_abs_shift": abs(actual_shift),
        })

    print(f"\nResolution status:")
    print(f"  Resolved with measurable shift: {len(resolved_pairs)}")
    print(f"  Not yet resolved: {unresolved_count}")
    print(f"  Missing price history: {no_history_count}")

    if len(resolved_pairs) < 5:
        print(f"\n⏳ Not enough resolved pairs yet ({len(resolved_pairs)} < 5).")
        print("   Wait for more crux markets to resolve, then re-run this script.")

        # Save partial results
        output = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_total_pairs": len(pairs),
                "n_resolved": len(resolved_pairs),
                "n_unresolved": unresolved_count,
                "status": "insufficient_data",
            },
            "resolved_pairs": resolved_pairs,
        }
        output_path = RESULTS_DIR / "validation.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved partial results to {output_path}")
        return

    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print("=" * 70)

    # Compute correlations
    predicted_voi = [p["linear_voi"] for p in resolved_pairs]
    actual_abs_shifts = [p["actual_abs_shift"] for p in resolved_pairs]
    actual_shifts = [p["actual_shift"] for p in resolved_pairs]
    rhos = [p["rho"] for p in resolved_pairs]

    # Main result: VOI vs actual absolute shift
    r_voi_shift, p_voi = stats.pearsonr(predicted_voi, actual_abs_shifts)
    tau_voi, p_tau = stats.kendalltau(predicted_voi, actual_abs_shifts)

    print(f"\n📊 VOI vs Actual |Shift| (N={len(resolved_pairs)}):")
    print(f"   Pearson r = {r_voi_shift:.3f} (p={p_voi:.4f})")
    print(f"   Kendall τ = {tau_voi:.3f} (p={p_tau:.4f})")

    # Direction accuracy: does ρ sign predict shift sign?
    direction_correct = 0
    direction_total = 0
    for p in resolved_pairs:
        if abs(p["rho"]) > 0.1 and abs(p["actual_shift"]) > 0.02:
            direction_total += 1
            # Positive ρ → crux YES should increase ultimate → positive shift if YES
            expected_direction = np.sign(p["rho"])
            # If crux resolved NO, flip the expected direction
            if p["crux_outcome"] == "NO":
                expected_direction = -expected_direction
            if np.sign(p["actual_shift"]) == expected_direction:
                direction_correct += 1

    if direction_total > 0:
        direction_accuracy = direction_correct / direction_total
        print(f"\n📊 Direction accuracy (ρ>0.1, |shift|>0.02):")
        print(f"   {direction_correct}/{direction_total} = {direction_accuracy:.1%}")

    # Compare to baseline
    print(f"\n{'='*70}")
    print("COMPARISON TO HUMAN-CURATED BASELINE")
    print("=" * 70)
    print(f"\n  Human-curated pairs: r = 0.65")
    print(f"  LLM-generated cruxes: r = {r_voi_shift:.3f}")

    if r_voi_shift > 0.5:
        print(f"\n  ✅ MATCH BASELINE: LLM cruxes ≈ human cruxes")
        print(f"     Interpretation: Crux generation is not the bottleneck")
    elif r_voi_shift > 0.2:
        print(f"\n  ⚠️ PARTIAL: LLM cruxes work but not as well")
        print(f"     Interpretation: Room for improvement in crux generation")
    else:
        print(f"\n  ❌ FAILURE: LLM cruxes don't predict information flow")
        print(f"     Interpretation: Crux generation is the bottleneck")

    # Breakdown by crux magnitude
    print(f"\n📊 Breakdown by crux magnitude:")
    for mag in ["high", "medium", "low"]:
        mag_pairs = [p for p in resolved_pairs if p["crux"].get("magnitude") == mag]
        if len(mag_pairs) >= 3:
            mag_voi = [p["linear_voi"] for p in mag_pairs]
            mag_shift = [p["actual_abs_shift"] for p in mag_pairs]
            r_mag, _ = stats.pearsonr(mag_voi, mag_shift)
            print(f"   {mag}: r={r_mag:.3f} (N={len(mag_pairs)})")

    # Show best/worst predictions
    by_prediction_quality = []
    for p in resolved_pairs:
        # Prediction error: difference between predicted VOI rank and actual shift rank
        pred_rank = sorted(predicted_voi).index(p["linear_voi"]) / len(predicted_voi)
        actual_rank = sorted(actual_abs_shifts).index(p["actual_abs_shift"]) / len(actual_abs_shifts)
        error = abs(pred_rank - actual_rank)
        by_prediction_quality.append((p, error))

    by_prediction_quality.sort(key=lambda x: x[1])

    print(f"\n📊 Best predictions (low error):")
    for p, err in by_prediction_quality[:3]:
        print(f"   VOI={p['linear_voi']:.4f} → shift={p['actual_shift']:+.3f}")
        print(f"   {p['crux']['crux'][:50]}...")

    print(f"\n📊 Worst predictions (high error):")
    for p, err in by_prediction_quality[-3:]:
        print(f"   VOI={p['linear_voi']:.4f} → shift={p['actual_shift']:+.3f}")
        print(f"   {p['crux']['crux'][:50]}...")

    # Save
    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_total_pairs": len(pairs),
            "n_resolved": len(resolved_pairs),
            "shift_window_days": SHIFT_WINDOW_DAYS,
            "status": "complete",
        },
        "results": {
            "pearson_r": r_voi_shift,
            "pearson_p": p_voi,
            "kendall_tau": tau_voi,
            "kendall_p": p_tau,
            "direction_accuracy": direction_accuracy if direction_total > 0 else None,
            "direction_n": direction_total,
            "baseline_r": 0.65,
            "interpretation": (
                "match_baseline" if r_voi_shift > 0.5 else
                "partial" if r_voi_shift > 0.2 else
                "failure"
            ),
        },
        "resolved_pairs": resolved_pairs,
    }

    output_path = RESULTS_DIR / "validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
