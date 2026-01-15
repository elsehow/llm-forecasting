#!/usr/bin/env python3
"""
Validate VOI using smart-curated pairs (topic-filtered + LLM classified).

Usage:
    uv run python experiments/question-generation/voi-validation/validate_voi_smart.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from scipy import stats
import numpy as np

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = OUTPUT_DIR / "results"

# Windows
WINDOW_BEFORE_DAYS = 5
WINDOW_AFTER_DAYS = 5
SECONDS_PER_DAY = 86400


def load_price_histories() -> dict[str, dict]:
    """Load price histories."""
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = data
    return histories


def price_at(candles: list[dict], timestamp: int) -> float | None:
    """Get price at or before timestamp."""
    for candle in reversed(candles):
        if candle["timestamp"] <= timestamp:
            return candle["close"]
    return None


def price_after(candles: list[dict], timestamp: int, window_seconds: int) -> float | None:
    """Get average price in window after timestamp."""
    prices = []
    for candle in candles:
        if timestamp < candle["timestamp"] <= timestamp + window_seconds:
            prices.append(candle["close"])
    return np.mean(prices) if prices else None


def validate_pair(pair: dict, histories: dict[str, dict]) -> dict | None:
    """Validate a single pair."""
    cond_a = pair["condition_id_a"]
    cond_b = pair["condition_id_b"]
    rho = pair["rho"]
    category = pair.get("classification", {}).get("category", "unknown")

    # Determine which resolved
    resolved = pair.get("resolved")
    if not resolved:
        return None

    if resolved == "A":
        resolved_cond = cond_a
        other_cond = cond_b
        other_question = pair["question_b"]
    else:
        resolved_cond = cond_b
        other_cond = cond_a
        other_question = pair["question_a"]

    # Get histories
    resolved_hist = histories.get(resolved_cond)
    other_hist = histories.get(other_cond)

    if not resolved_hist or not other_hist:
        return None

    # Parse resolution date
    res_date_str = pair.get("resolution_date")
    if not res_date_str:
        return None

    resolution_ts = int(datetime.fromisoformat(res_date_str.replace("Z", "+00:00")).timestamp())

    # Get other market's price before and after
    other_candles = other_hist.get("candles", [])

    price_before = price_at(other_candles, resolution_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY)
    if price_before is None:
        price_before = other_candles[0]["close"] if other_candles else None

    price_after_val = price_after(other_candles, resolution_ts, WINDOW_AFTER_DAYS * SECONDS_PER_DAY)
    if price_after_val is None:
        # Use final price if after resolution
        if other_candles and other_candles[-1]["timestamp"] > resolution_ts:
            price_after_val = other_candles[-1]["close"]

    if price_before is None or price_after_val is None:
        return None

    actual_shift = abs(price_after_val - price_before)

    # Predicted shift using canonical Linear VOI
    predicted_shift = linear_voi_from_rho(rho, price_before, 0.5)

    return {
        "question_a": pair["question_a"],
        "question_b": pair["question_b"],
        "category": category,
        "rho": rho,
        "resolved": resolved,
        "resolution_outcome": pair.get("resolution_outcome"),
        "p_before": price_before,
        "p_after": price_after_val,
        "predicted_shift": predicted_shift,
        "actual_shift": actual_shift,
    }


def main():
    print("=" * 70)
    print("VOI VALIDATION - SMART CURATED PAIRS")
    print("=" * 70)

    # Load curated pairs
    curated_path = OUTPUT_DIR / "curated_pairs_smart.json"
    with open(curated_path) as f:
        data = json.load(f)

    pairs = data["curated_pairs"]
    print(f"\nLoaded {len(pairs)} smart-curated pairs")

    # Load histories
    print("Loading price histories...")
    histories = load_price_histories()
    print(f"Loaded {len(histories)} histories")

    # Validate each pair
    print("\nValidating pairs...")
    results = []
    for pair in pairs:
        result = validate_pair(pair, histories)
        if result:
            results.append(result)

    print(f"Validated {len(results)} pairs")

    if len(results) < 5:
        print("\n⚠️  Too few validated pairs!")
        return

    # Analysis
    predicted = np.array([r["predicted_shift"] for r in results])
    actual = np.array([r["actual_shift"] for r in results])
    rhos = np.array([abs(r["rho"]) for r in results])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nN validated pairs: {len(results)}")

    print(f"\nPredicted shift: mean={np.mean(predicted):.3f}, std={np.std(predicted):.3f}")
    print(f"Actual shift:    mean={np.mean(actual):.3f}, std={np.std(actual):.3f}")
    print(f"|ρ|:             mean={np.mean(rhos):.3f}, std={np.std(rhos):.3f}")

    # Correlations
    if len(results) >= 3:
        corr_pred, pval_pred = stats.pearsonr(predicted, actual)
        corr_rho, pval_rho = stats.pearsonr(rhos, actual)
        spearman_rho, spearman_p = stats.spearmanr(rhos, actual)

        print("\n" + "-" * 70)
        print("KEY TESTS")
        print("-" * 70)

        print(f"\nPearson (predicted ~ actual):   r={corr_pred:.3f}, p={pval_pred:.4f}")
        print(f"Pearson (|ρ| ~ actual):         r={corr_rho:.3f}, p={pval_rho:.4f}")
        print(f"Spearman (|ρ| ~ actual):        ρ={spearman_rho:.3f}, p={spearman_p:.4f}")

        if pval_rho < 0.05 and corr_rho > 0:
            print("\n✅ VALIDATED: High |ρ| predicts larger actual shifts (p < 0.05)")
        elif pval_rho < 0.10 and corr_rho > 0:
            print("\n⚠️  MARGINAL: Positive trend (p < 0.10)")
        else:
            print("\n❌ NOT VALIDATED with current data")

    # By category
    print("\n" + "-" * 70)
    print("BY CATEGORY")
    print("-" * 70)

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cat_results in sorted(categories.items(), key=lambda x: -len(x[1])):
        if len(cat_results) >= 2:
            cat_rhos = [abs(r["rho"]) for r in cat_results]
            cat_actual = [r["actual_shift"] for r in cat_results]
            print(f"\n{cat} (n={len(cat_results)}):")
            print(f"  Mean |ρ|: {np.mean(cat_rhos):.3f}")
            print(f"  Mean actual shift: {np.mean(cat_actual):.3f}")
            if len(cat_results) >= 5:
                r, p = stats.pearsonr(cat_rhos, cat_actual)
                print(f"  Correlation: r={r:.3f}, p={p:.3f}")

    # Examples
    print("\n" + "-" * 70)
    print("TOP 10 BY ACTUAL SHIFT")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda r: r["actual_shift"], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"\n{i+1}. actual={r['actual_shift']:.3f}, |ρ|={abs(r['rho']):.2f}, [{r['category']}]")
        print(f"   {r['question_a'][:60]}...")
        print(f"   {r['question_b'][:60]}...")
        print(f"   {r['p_before']:.2f} → {r['p_after']:.2f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "n_curated": len(pairs),
            "n_validated": len(results),
            "run_at": datetime.now().isoformat(),
        },
        "summary": {
            "predicted_shift_mean": float(np.mean(predicted)),
            "actual_shift_mean": float(np.mean(actual)),
            "correlation_rho_actual": float(corr_rho) if len(results) >= 3 else None,
            "pvalue_rho_actual": float(pval_rho) if len(results) >= 3 else None,
        },
        "results": results,
    }

    output_path = RESULTS_DIR / "voi_validation_smart.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
