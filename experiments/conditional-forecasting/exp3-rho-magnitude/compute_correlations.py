#!/usr/bin/env python3
"""
Compute pairwise correlations between markets based on price returns.
Outputs pairs labeled by correlation strength bucket.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_DIR = DATA_DIR / "price_history"

# Correlation buckets
BUCKETS = {
    "independent": (0.0, 0.1),
    "weak": (0.1, 0.3),
    "moderate": (0.3, 0.6),
    "strong": (0.6, 1.0),
}


def load_price_series(condition_id: str) -> tuple[list[int], list[float]] | None:
    """Load price history and return (timestamps, closes)."""
    # Filename is truncated to 40 chars (matching fetch_history.py)
    path = HISTORY_DIR / f"{condition_id[:40]}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    candles = data.get("candles", [])
    if len(candles) < 10:  # Need sufficient data
        return None

    timestamps = [c["timestamp"] for c in candles if c.get("close") is not None]
    closes = [c["close"] for c in candles if c.get("close") is not None]

    return timestamps, closes


def compute_returns(prices: list[float]) -> np.ndarray:
    """Compute daily returns from price series."""
    prices = np.array(prices)
    # Avoid division by zero
    prices = np.where(prices == 0, 1e-6, prices)
    returns = np.diff(prices) / prices[:-1]
    return returns


def align_series(
    ts1: list[int], vals1: list[float],
    ts2: list[int], vals2: list[float],
) -> tuple[list[float], list[float]]:
    """Align two time series by timestamp, returning only overlapping periods."""
    set1 = set(ts1)
    set2 = set(ts2)
    common = sorted(set1 & set2)

    if len(common) < 10:
        return [], []

    idx1 = {t: i for i, t in enumerate(ts1)}
    idx2 = {t: i for i, t in enumerate(ts2)}

    aligned1 = [vals1[idx1[t]] for t in common]
    aligned2 = [vals2[idx2[t]] for t in common]

    return aligned1, aligned2


def compute_correlation(vals1: list[float], vals2: list[float]) -> tuple[float, float]:
    """Compute Pearson correlation on returns."""
    if len(vals1) < 10:
        return 0.0, 1.0

    returns1 = compute_returns(vals1)
    returns2 = compute_returns(vals2)

    # Remove any NaN/inf
    mask = np.isfinite(returns1) & np.isfinite(returns2)
    returns1 = returns1[mask]
    returns2 = returns2[mask]

    if len(returns1) < 5:
        return 0.0, 1.0

    rho, pval = pearsonr(returns1, returns2)
    return float(rho), float(pval)


def bucket_correlation(rho: float) -> str:
    """Assign correlation to bucket based on absolute value."""
    abs_rho = abs(rho)
    for name, (low, high) in BUCKETS.items():
        if low <= abs_rho < high:
            return name
    return "strong"  # >= 0.6


def main():
    # Load markets metadata
    markets_path = DATA_DIR / "markets.json"
    if not markets_path.exists():
        print("Run fetch_markets.py first")
        return

    with open(markets_path) as f:
        markets = json.load(f)

    # Create lookup
    market_lookup = {m.get("condition_id"): m for m in markets if m.get("condition_id")}

    # Load all price histories
    print("Loading price histories...")
    price_data = {}
    for market in markets:
        cid = market.get("condition_id")
        if not cid:
            continue
        result = load_price_series(cid)
        if result:
            price_data[cid] = result

    print(f"Loaded {len(price_data)} markets with price history")

    # Compute all pairwise correlations
    print(f"Computing correlations for {len(price_data) * (len(price_data) - 1) // 2} pairs...")

    pairs = []
    bucket_counts = {b: 0 for b in BUCKETS}

    condition_ids = list(price_data.keys())
    for i, (cid1, cid2) in enumerate(combinations(condition_ids, 2)):
        ts1, vals1 = price_data[cid1]
        ts2, vals2 = price_data[cid2]

        # Align series
        aligned1, aligned2 = align_series(ts1, vals1, ts2, vals2)
        if len(aligned1) < 10:
            continue

        # Compute correlation
        rho, pval = compute_correlation(aligned1, aligned2)
        bucket = bucket_correlation(rho)
        bucket_counts[bucket] += 1

        m1 = market_lookup.get(cid1, {})
        m2 = market_lookup.get(cid2, {})

        pairs.append({
            "market_a": {
                "condition_id": cid1,
                "question": m1.get("question"),
            },
            "market_b": {
                "condition_id": cid2,
                "question": m2.get("question"),
            },
            "rho": round(rho, 4),
            "pval": round(pval, 4),
            "bucket": bucket,
            "n_observations": len(aligned1),
        })

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} pairs...")

    # Sort by absolute correlation (most interesting first)
    pairs.sort(key=lambda p: abs(p["rho"]), reverse=True)

    # Save results
    output_path = DATA_DIR / "pairs.json"
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nSaved {len(pairs)} pairs to {output_path}")
    print("\nBucket distribution:")
    for bucket, count in bucket_counts.items():
        pct = 100 * count / len(pairs) if pairs else 0
        print(f"  {bucket:12} {count:5} ({pct:.1f}%)")

    # Show samples from each bucket
    print("\nSample pairs by bucket:")
    for bucket in BUCKETS:
        samples = [p for p in pairs if p["bucket"] == bucket][:2]
        print(f"\n  {bucket.upper()}:")
        for s in samples:
            q1 = s["market_a"]["question"][:35] if s["market_a"]["question"] else "?"
            q2 = s["market_b"]["question"][:35] if s["market_b"]["question"] else "?"
            print(f"    ρ={s['rho']:+.3f}: {q1}... ↔ {q2}...")


if __name__ == "__main__":
    main()
