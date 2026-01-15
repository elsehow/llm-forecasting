#!/usr/bin/env python3
"""
Validate VOI on non-trivial probability pairs.

These pairs were pre-filtered so the OTHER market has 10-90% probability
at resolution time, giving room for observable shifts.

Usage:
    uv run python experiments/question-generation/voi-validation/validate_voi_nontrivial.py
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho

# Paths
VOI_DIR = Path(__file__).parent
CONDITIONAL_DIR = VOI_DIR.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
RESULTS_DIR = VOI_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load curated pairs and price histories."""
    with open(VOI_DIR / "curated_pairs_nontrivial.json") as f:
        data = json.load(f)

    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            h = json.load(f)
        histories[h["condition_id"]] = h

    return data, histories


def get_price_at_timestamp(candles: list[dict], ts: int) -> float | None:
    """Get price at or before a given timestamp."""
    for c in reversed(candles):
        if c["timestamp"] <= ts:
            return c["close"]
    return candles[0]["close"] if candles else None


def get_price_after_timestamp(candles: list[dict], ts: int, days_after: int = 7) -> float | None:
    """Get price some days after a timestamp."""
    target_ts = ts + (days_after * 24 * 60 * 60)
    for c in candles:
        if c["timestamp"] >= target_ts:
            return c["close"]
    # If no candle found, use last available
    return candles[-1]["close"] if candles else None


def compute_actual_shift(pair: dict, histories: dict) -> dict | None:
    """Compute the actual price shift in the other market after resolution."""
    cond_a = pair["condition_id_a"]
    cond_b = pair["condition_id_b"]
    resolved = pair["resolved"]  # "A" or "B"

    if resolved == "A":
        resolved_cond = cond_a
        other_cond = cond_b
    else:
        resolved_cond = cond_b
        other_cond = cond_a

    resolved_hist = histories.get(resolved_cond)
    other_hist = histories.get(other_cond)

    if not resolved_hist or not other_hist:
        return None

    resolved_candles = resolved_hist.get("candles", [])
    other_candles = other_hist.get("candles", [])

    if not resolved_candles or not other_candles:
        return None

    # Find resolution timestamp
    resolution_date = datetime.fromisoformat(pair["resolution_date"])
    resolution_ts = int(resolution_date.timestamp())

    # Get prices before and after
    p_before = pair["other_price_at_resolution"]  # Already computed in curation
    p_after = get_price_after_timestamp(other_candles, resolution_ts, days_after=7)

    if p_before is None or p_after is None:
        return None

    actual_shift = abs(p_after - p_before)

    return {
        "p_before": p_before,
        "p_after": p_after,
        "actual_shift": actual_shift,
        "direction": "up" if p_after > p_before else "down",
    }


def main():
    print("=" * 70)
    print("VOI VALIDATION ON NON-TRIVIAL PROBABILITY PAIRS")
    print("=" * 70)

    data, histories = load_data()
    pairs = data["curated_pairs"]

    print(f"\nLoaded {len(pairs)} curated pairs")
    print(f"All have other market at 10-90% probability")

    # Compute actual shifts
    results = []
    for pair in pairs:
        shift_data = compute_actual_shift(pair, histories)
        if shift_data:
            results.append({
                "question_a": pair["question_a"],
                "question_b": pair["question_b"],
                "rho": pair["rho"],
                "abs_rho": abs(pair["rho"]),
                "classification": pair["classification"]["category"],
                "resolved": pair["resolved"],
                "outcome": pair["resolution_outcome"],
                **shift_data,
            })

    print(f"Computed shifts for {len(results)} pairs")

    if len(results) < 5:
        print("Too few pairs with shift data!")
        return

    # Extract arrays
    abs_rhos = np.array([r["abs_rho"] for r in results])
    actual_shifts = np.array([r["actual_shift"] for r in results])
    p_befores = np.array([r["p_before"] for r in results])

    # Compute linear VOI using canonical formula
    rhos = np.array([r["rho"] for r in results])
    vois = np.array([
        linear_voi_from_rho(rho, p_before, 0.5)
        for rho, p_before in zip(rhos, p_befores)
    ])

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Test 1: |ρ| vs actual shift
    r_rho, p_rho = stats.pearsonr(abs_rhos, actual_shifts)
    print(f"\n|ρ| vs actual_shift:")
    print(f"  r = {r_rho:.3f}, p = {p_rho:.3f}")

    # Test 2: VOI vs actual shift
    r_voi, p_voi = stats.pearsonr(vois, actual_shifts)
    print(f"\nVOI vs actual_shift:")
    print(f"  r = {r_voi:.3f}, p = {p_voi:.3f}")

    # Test 3: Spearman (rank correlation)
    rho_spearman, p_spearman = stats.spearmanr(abs_rhos, actual_shifts)
    print(f"\nSpearman |ρ| vs actual_shift:")
    print(f"  ρ = {rho_spearman:.3f}, p = {p_spearman:.3f}")

    # Breakdown by classification
    print("\n" + "-" * 70)
    print("BY CLASSIFICATION TYPE")
    print("-" * 70)

    by_category = {}
    for r in results:
        cat = r["classification"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    for cat, cat_results in sorted(by_category.items(), key=lambda x: -len(x[1])):
        n = len(cat_results)
        avg_rho = np.mean([r["abs_rho"] for r in cat_results])
        avg_shift = np.mean([r["actual_shift"] for r in cat_results])
        print(f"\n{cat} (n={n}):")
        print(f"  avg |ρ|: {avg_rho:.2f}, avg shift: {avg_shift:.3f}")

    # Top pairs by actual shift
    print("\n" + "-" * 70)
    print("TOP 10 PAIRS BY ACTUAL SHIFT")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda r: -r["actual_shift"])
    for i, r in enumerate(sorted_results[:10]):
        print(f"\n{i+1}. shift={r['actual_shift']:.3f}, |ρ|={r['abs_rho']:.2f}, p_before={r['p_before']:.2f}")
        print(f"   [{r['classification']}] {r['direction']}")
        print(f"   A: {r['question_a'][:60]}...")
        print(f"   B: {r['question_b'][:60]}...")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nN pairs: {len(results)}")
    print(f"Mean |ρ|: {np.mean(abs_rhos):.3f}")
    print(f"Mean actual shift: {np.mean(actual_shifts):.3f}")
    print(f"Mean p_before: {np.mean(p_befores):.3f}")

    # Power analysis note
    print("\n" + "-" * 70)
    print("STATISTICAL POWER")
    print("-" * 70)

    # For r=0.3, we need ~85 pairs for 80% power at α=0.05
    # For r=0.4, we need ~47 pairs
    # For r=0.5, we need ~30 pairs
    if r_rho > 0:
        required_n = int(np.ceil((1.96 + 0.84) ** 2 / (0.5 * np.log((1 + r_rho) / (1 - r_rho))) ** 2))
        print(f"For observed r={r_rho:.2f}, need ~{required_n} pairs for 80% power")
        print(f"Currently have {len(results)} pairs")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if p_rho < 0.05:
        print(f"\n✅ STATISTICALLY SIGNIFICANT")
        print(f"   |ρ| predicts actual shift (r={r_rho:.2f}, p={p_rho:.3f})")
        print(f"   VOI is validated!")
    elif p_rho < 0.10:
        print(f"\n⚠️ MARGINALLY SIGNIFICANT")
        print(f"   |ρ| shows directional relationship (r={r_rho:.2f}, p={p_rho:.3f})")
        print(f"   VOI is directionally validated, need more data")
    elif r_rho > 0.2:
        print(f"\n📊 DIRECTIONALLY CORRECT BUT UNDERPOWERED")
        print(f"   |ρ| positively correlates with shift (r={r_rho:.2f}, p={p_rho:.3f})")
        print(f"   Effect exists but need more pairs for significance")
    else:
        print(f"\n❌ NO CLEAR RELATIONSHIP")
        print(f"   r={r_rho:.2f}, p={p_rho:.3f}")

    # Save results
    output = {
        "metadata": {
            "n_pairs": len(results),
            "mean_abs_rho": float(np.mean(abs_rhos)),
            "mean_actual_shift": float(np.mean(actual_shifts)),
            "mean_p_before": float(np.mean(p_befores)),
        },
        "correlations": {
            "rho_vs_shift": {"r": float(r_rho), "p": float(p_rho)},
            "voi_vs_shift": {"r": float(r_voi), "p": float(p_voi)},
            "spearman_rho_vs_shift": {"rho": float(rho_spearman), "p": float(p_spearman)},
        },
        "by_category": {
            cat: {
                "n": len(cat_results),
                "avg_abs_rho": float(np.mean([r["abs_rho"] for r in cat_results])),
                "avg_shift": float(np.mean([r["actual_shift"] for r in cat_results])),
            }
            for cat, cat_results in by_category.items()
        },
        "pairs": results,
    }

    output_path = RESULTS_DIR / "voi_validation_nontrivial.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
