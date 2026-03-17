#!/usr/bin/env python3
"""
Analyze options-implied-vol experiment results.

Computes:
- Width ratios (LLM vs market) at p10-p90, p5-p95, p1-p99
- Per-quantile calibration curves
- Coverage rates
- CRPS (quantile-weighted pinball loss)
- Breakdowns by asset, horizon, model

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/analyze.py
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Quantile pairs for width comparison
QUANTILE_PAIRS = [
    ("p10", "p90", 0.10, 0.90),
    ("p5", "p95", 0.05, 0.95),
    ("p1", "p99", 0.01, 0.99),
]

ALL_QUANTILES = {
    "p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
    "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99,
}


def market_interval(price, vol_pct, days, lo_quantile, hi_quantile):
    """Compute market p_lo and p_hi from vol index using lognormal model."""
    sigma = vol_pct / 100
    t = days / 365
    z_lo = norm.ppf(lo_quantile)
    z_hi = norm.ppf(hi_quantile)
    p_lo = price * math.exp(z_lo * sigma * math.sqrt(t))
    p_hi = price * math.exp(z_hi * sigma * math.sqrt(t))
    return p_lo, p_hi


def compute_crps(percentiles: dict, true_value: float) -> float:
    """Approximate CRPS from quantile-weighted pinball loss."""
    total = 0
    n = 0
    for key, tau in ALL_QUANTILES.items():
        if key not in percentiles:
            continue
        q = percentiles[key]
        if true_value >= q:
            total += tau * (true_value - q)
        else:
            total += (1 - tau) * (q - true_value)
        n += 1
    return (2 / n) * total if n > 0 else None


def analyze():
    forecasts_path = RESULTS_DIR / "forecasts.json"
    with open(forecasts_path) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} forecast results")

    # Filter to results with valid percentiles and ground truth
    valid = [r for r in results
             if r.get("percentiles") and r.get("ground_truth") is not None]
    print(f"Valid (with percentiles + ground truth): {len(valid)}")

    models = sorted(set(r["model"] for r in valid))
    assets = sorted(set(r["asset_id"] for r in valid))
    horizons = ["H1", "H2", "H3"]

    print(f"Models: {[m.split('/')[-1] for m in models]}")
    print(f"Assets: {assets}")

    # ===== WIDTH RATIOS =====
    print(f"\n{'='*80}")
    print("  WIDTH RATIOS: LLM interval / Market interval")
    print(f"{'='*80}")

    # Per horizon, averaged across all models and assets
    for pair_name, (lo_key, hi_key, lo_q, hi_q) in [
        ("p10-p90", QUANTILE_PAIRS[0]),
        ("p5-p95", QUANTILE_PAIRS[1]),
        ("p1-p99", QUANTILE_PAIRS[2]),
    ]:
        print(f"\n  {pair_name}:")
        print(f"  {'Horizon':<8} {'Mean ratio':>12} {'Median':>10} {'Min':>8} {'Max':>8} {'N':>6}")
        print(f"  {'-'*55}")
        for hz in horizons:
            ratios = []
            for r in valid:
                if r["horizon"] != hz:
                    continue
                pcts = r["percentiles"]
                if lo_key not in pcts or hi_key not in pcts:
                    continue
                llm_width = pcts[hi_key] - pcts[lo_key]
                mkt_lo, mkt_hi = market_interval(
                    r["cutoff_price"], r["vol_index_at_cutoff"],
                    r["days_from_cutoff"], lo_q, hi_q
                )
                mkt_width = mkt_hi - mkt_lo
                if mkt_width > 0:
                    ratios.append(llm_width / mkt_width)
            if ratios:
                print(f"  {hz:<8} {np.mean(ratios):>11.2f}x {np.median(ratios):>9.2f}x "
                      f"{min(ratios):>7.2f}x {max(ratios):>7.2f}x {len(ratios):>5}")

    # Per asset at H3 (p10-p90)
    print(f"\n  Per-asset width ratio (p10-p90) at H3:")
    print(f"  {'Asset':<15} {'Mean ratio':>12} {'N':>6}")
    print(f"  {'-'*35}")
    for asset in assets:
        ratios = []
        lo_key, hi_key, lo_q, hi_q = QUANTILE_PAIRS[0]
        for r in valid:
            if r["asset_id"] != asset or r["horizon"] != "H3":
                continue
            pcts = r["percentiles"]
            if lo_key not in pcts or hi_key not in pcts:
                continue
            llm_width = pcts[hi_key] - pcts[lo_key]
            mkt_lo, mkt_hi = market_interval(
                r["cutoff_price"], r["vol_index_at_cutoff"],
                r["days_from_cutoff"], lo_q, hi_q
            )
            mkt_width = mkt_hi - mkt_lo
            if mkt_width > 0:
                ratios.append(llm_width / mkt_width)
        if ratios:
            print(f"  {asset:<15} {np.mean(ratios):>11.2f}x {len(ratios):>5}")

    # Per model at H3 (p10-p90)
    print(f"\n  Per-model width ratio (p10-p90) at H3:")
    print(f"  {'Model':<35} {'Mean ratio':>12} {'N':>6}")
    print(f"  {'-'*55}")
    for model in models:
        ratios = []
        lo_key, hi_key, lo_q, hi_q = QUANTILE_PAIRS[0]
        for r in valid:
            if r["model"] != model or r["horizon"] != "H3":
                continue
            pcts = r["percentiles"]
            if lo_key not in pcts or hi_key not in pcts:
                continue
            llm_width = pcts[hi_key] - pcts[lo_key]
            mkt_lo, mkt_hi = market_interval(
                r["cutoff_price"], r["vol_index_at_cutoff"],
                r["days_from_cutoff"], lo_q, hi_q
            )
            mkt_width = mkt_hi - mkt_lo
            if mkt_width > 0:
                ratios.append(llm_width / mkt_width)
        if ratios:
            print(f"  {model.split('/')[-1]:<35} {np.mean(ratios):>11.2f}x {len(ratios):>5}")

    # ===== CALIBRATION =====
    print(f"\n{'='*80}")
    print("  PER-QUANTILE CALIBRATION")
    print(f"{'='*80}")

    print(f"\n  {'Quantile':<10} {'Nominal':>8} {'Observed':>10} {'Gap':>8} {'N':>6}")
    print(f"  {'-'*45}")
    for key, tau in ALL_QUANTILES.items():
        below = 0
        total = 0
        for r in valid:
            pcts = r["percentiles"]
            if key not in pcts:
                continue
            if r["ground_truth"] < pcts[key]:
                below += 1
            total += 1
        if total > 0:
            observed = below / total
            print(f"  {key:<10} {tau:>7.0%} {observed:>9.1%} {observed - tau:>+7.1%} {total:>5}")

    # Per model calibration
    print(f"\n  Per-model calibration (p10 and p90):")
    print(f"  {'Model':<35} {'p10 (nom 10%)':>15} {'p90 (nom 90%)':>15}")
    print(f"  {'-'*68}")
    for model in models:
        model_results = [r for r in valid if r["model"] == model]
        for key, nom in [("p10", 0.10), ("p90", 0.90)]:
            below = sum(1 for r in model_results
                       if key in r["percentiles"] and r["ground_truth"] < r["percentiles"][key])
            total = sum(1 for r in model_results if key in r["percentiles"])
            if key == "p10":
                p10_str = f"{below/total:.1%}" if total else "?"
            else:
                p90_str = f"{below/total:.1%}" if total else "?"
        print(f"  {model.split('/')[-1]:<35} {p10_str:>15} {p90_str:>15}")

    # ===== COVERAGE =====
    print(f"\n{'='*80}")
    print("  COVERAGE (% of actuals inside interval)")
    print(f"{'='*80}")

    for pair_name, (lo_key, hi_key, lo_q, hi_q) in [
        ("p10-p90 (nom 80%)", QUANTILE_PAIRS[0]),
        ("p5-p95 (nom 90%)", QUANTILE_PAIRS[1]),
        ("p1-p99 (nom 98%)", QUANTILE_PAIRS[2]),
    ]:
        inside = 0
        total = 0
        for r in valid:
            pcts = r["percentiles"]
            if lo_key not in pcts or hi_key not in pcts:
                continue
            if pcts[lo_key] <= r["ground_truth"] <= pcts[hi_key]:
                inside += 1
            total += 1
        if total:
            print(f"  {pair_name}: {inside/total:.1%} ({inside}/{total})")

    # ===== CRPS =====
    print(f"\n{'='*80}")
    print("  CRPS (lower is better)")
    print(f"{'='*80}")

    print(f"\n  {'Model':<35} {'Mean CRPS':>12} {'Relative CRPS':>15} {'N':>6}")
    print(f"  {'-'*70}")
    for model in models:
        crps_vals = []
        rel_crps_vals = []
        for r in valid:
            if r["model"] != model:
                continue
            c = compute_crps(r["percentiles"], r["ground_truth"])
            if c is not None:
                crps_vals.append(c)
                rel_crps_vals.append(c / abs(r["ground_truth"]) if r["ground_truth"] != 0 else None)
        rel_crps_vals = [v for v in rel_crps_vals if v is not None]
        if crps_vals:
            print(f"  {model.split('/')[-1]:<35} {np.mean(crps_vals):>11.2f} "
                  f"{np.mean(rel_crps_vals):>14.4f} {len(crps_vals):>5}")

    # ===== SAVE STRUCTURED OUTPUT =====
    output = {
        "generated": str(np.datetime64("now")),
        "n_results": len(valid),
        "models": [m.split("/")[-1] for m in models],
        "assets": assets,
    }

    out_path = RESULTS_DIR / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis summary to {out_path}")


if __name__ == "__main__":
    analyze()
