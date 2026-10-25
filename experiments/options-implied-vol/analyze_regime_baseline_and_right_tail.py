#!/usr/bin/env python3
"""
Two analyses for tailriskbench paper:

1. Hist-vol baseline regime-conditional miss rates
   (fills TODO in the regime decomposition table)

2. ECI × right-tail metrics
   (fills TODO for symmetric right-tail correlation)

Uses the same data as analyze_baselines.py and analyze_left_tail.py.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

BASE_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/options-implied-vol")
QUESTIONS_FILE = BASE_DIR / "data" / "questions.json"
FORECASTS_FILE = BASE_DIR / "results" / "forecasts.json"
PRICES_DIR = BASE_DIR / "data" / "prices"
OUTPUT_FILE = BASE_DIR / "results" / "regime_baseline_and_right_tail.json"

DRAWDOWN_THRESHOLD = -0.03

ECI_SCORES = {
    "anthropic/claude-opus-4-5-20251101": 150.1,
    "anthropic/claude-opus-4-6": 152.0,
    "anthropic/claude-sonnet-4-5-20250929": 146.6,
    "anthropic/claude-sonnet-4-20250514": 145.0,
    "anthropic/claude-haiku-4-5-20251001": 140.0,
    "anthropic/claude-3-haiku-20240307": 118.0,
    "openai/gpt-5.1-2025-11-13": 149.7,
    "openai/gpt-5-2025-08-07": 150.0,
    "openai/gpt-5-mini-2025-08-07": 144.3,
    "openai/gpt-5-nano-2025-08-07": 138.0,
    "openai/gpt-4.1-2025-04-14": 137.4,
    "openai/gpt-4o": 132.0,
    "openai/gpt-3.5-turbo-0125": 110.0,
    "openai/o3-2025-04-16": 146.6,
    "openai/o3-mini-2025-01-31": 140.0,
    "openai/o4-mini-2025-04-16": 142.0,
    "google/gemini-3-pro-preview": 154.2,
    "google/gemini-2.5-pro": 147.0,
    "google/gemini-2.5-flash": 145.0,
    "google/gemini-2.0-flash-lite-001": 125.0,
    "mistral/mistral-large-2411": 130.0,
    "mistral/mistral-large-2407": 128.0,
    "mistral/mistral-large-latest": 130.0,
    "together_ai/deepseek-ai/DeepSeek-V3": 135.0,
    "together_ai/deepseek-ai/DeepSeek-V3.1": 137.0,
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": 125.0,
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": 110.0,
    "xai/grok-4-0709": 150.0,
    "xai/grok-4-fast-reasoning": 150.0,
    "xai/grok-4-fast-non-reasoning": 145.0,
    "xai/grok-4-1-fast-reasoning": 151.0,
    "xai/grok-4-1-fast-non-reasoning": 146.0,
}

# Quantile levels for the 9-quantile format
QUANTILE_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
QUANTILE_KEYS = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
Z_SCORES = {q: stats.norm.ppf(q) for q in QUANTILE_LEVELS}


def load_prices(asset_id):
    """Load price history for an asset."""
    # Try different file naming conventions
    for suffix in [".csv"]:
        fpath = PRICES_DIR / f"{asset_id}{suffix}"
        if fpath.exists():
            import csv
            prices = {}
            with open(fpath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date = row.get("Date", "")
                    close = row.get("Close") or row.get("Adj Close")
                    if date and close:
                        try:
                            prices[date] = float(close)
                        except (ValueError, TypeError):
                            pass
            return prices
    return {}


def compute_realized_vol(prices_dict, cutoff_date, lookback=60):
    """Compute trailing realized vol from daily log-returns."""
    dates = sorted(prices_dict.keys())
    try:
        idx = dates.index(cutoff_date)
    except ValueError:
        # Find closest date before cutoff
        idx = None
        for i, d in enumerate(dates):
            if d <= cutoff_date:
                idx = i
        if idx is None:
            return None

    start = max(0, idx - lookback)
    window_dates = dates[start:idx + 1]
    if len(window_dates) < 20:
        return None

    prices = [prices_dict[d] for d in window_dates]
    log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1] > 0]
    if len(log_returns) < 10:
        return None

    daily_vol = np.std(log_returns)
    annualized_vol = daily_vol * math.sqrt(252)
    return annualized_vol


def generate_hist_vol_quantiles(price, sigma_ann, days):
    """Generate 9-quantile log-normal forecast from realized vol."""
    t = days / 365.0
    sigma_t = sigma_ann * math.sqrt(t)
    drift = -0.5 * sigma_ann ** 2 * t

    quantiles = {}
    for q, key in zip(QUANTILE_LEVELS, QUANTILE_KEYS):
        z = Z_SCORES[q]
        quantiles[key] = price * math.exp(drift + z * sigma_t)
    return quantiles


def miss_rate(records, quantile_key, direction="below"):
    """Compute miss rate for a quantile."""
    if not records:
        return float("nan")
    if direction == "below":
        misses = sum(1 for r in records if r["ground_truth"] < r["quantiles"][quantile_key])
    else:  # "above"
        misses = sum(1 for r in records if r["ground_truth"] > r["quantiles"][quantile_key])
    return misses / len(records)


def main():
    # Load questions and forecasts
    with open(QUESTIONS_FILE) as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]  # nested under "questions" key
    with open(FORECASTS_FILE) as f:
        forecasts = json.load(f)

    print(f"Loaded {len(questions)} questions, {len(forecasts)} forecasts")

    # Build question lookup
    q_lookup = {}
    for q in questions:
        key = (q["asset_id"], q["cutoff_date"], q["horizon"])
        q_lookup[key] = q

    # Load all price data
    asset_ids = list(set(q["asset_id"] for q in questions))
    price_cache = {}
    for aid in asset_ids:
        price_cache[aid] = load_prices(aid)
    print(f"Loaded prices for {len(price_cache)} assets")

    # =========================================================================
    # PART 1: Hist-vol baseline regime-conditional miss rates
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Hist-vol baseline regime-conditional miss rates")
    print("=" * 70)

    # Generate hist-vol forecasts for each question
    hist_vol_records = []
    for q in questions:
        aid = q["asset_id"]
        cutoff = q["cutoff_date"]
        horizon = q["horizon"]
        cutoff_price = q["cutoff_price"]
        gt = q["ground_truth"]

        if cutoff_price is None or gt is None:
            continue

        vol = compute_realized_vol(price_cache.get(aid, {}), cutoff, lookback=60)
        if vol is None:
            continue

        days = q.get("days_from_cutoff", {"H1": 7, "H2": 14, "H3": 21}.get(horizon, 7))
        quantiles = generate_hist_vol_quantiles(cutoff_price, vol, days)
        realized_return = (gt - cutoff_price) / cutoff_price

        hist_vol_records.append({
            "asset_id": aid,
            "cutoff_date": cutoff,
            "horizon": horizon,
            "ground_truth": gt,
            "cutoff_price": cutoff_price,
            "quantiles": quantiles,
            "realized_return": realized_return,
        })

    print(f"Generated {len(hist_vol_records)} hist-vol forecasts")

    drawdown = [r for r in hist_vol_records if r["realized_return"] < DRAWDOWN_THRESHOLD]
    calm = [r for r in hist_vol_records if r["realized_return"] >= DRAWDOWN_THRESHOLD]

    print(f"  Drawdown: {len(drawdown)}, Calm: {len(calm)}")

    # Regime-conditional miss rates for hist-vol
    print(f"\n  Hist-vol regime-conditional miss rates:")
    print(f"  {'Regime':<25} {'p10 miss':<12} {'p5 miss':<12} {'p1 miss':<12} {'N':<8}")
    print(f"  {'-'*69}")

    for label, recs in [("Drawdown (ret < -3%)", drawdown), ("Calm (ret >= -3%)", calm), ("All (pooled)", hist_vol_records)]:
        p10 = miss_rate(recs, "p10")
        p5 = miss_rate(recs, "p5")
        p1 = miss_rate(recs, "p1")
        print(f"  {label:<25} {p10:<12.1%} {p5:<12.1%} {p1:<12.1%} {len(recs):<8}")

    print(f"  {'Nominal':<25} {'10.0%':<12} {'5.0%':<12} {'1.0%':<12}")

    # Severity gradient for hist-vol
    thresholds = [
        ("Return >= 0%", lambda r: r["realized_return"] >= 0),
        ("-3% to 0%", lambda r: -0.03 <= r["realized_return"] < 0),
        ("-5% to -3%", lambda r: -0.05 <= r["realized_return"] < -0.03),
        ("-10% to -5%", lambda r: -0.10 <= r["realized_return"] < -0.05),
        ("< -10%", lambda r: r["realized_return"] < -0.10),
    ]

    print(f"\n  Hist-vol severity gradient:")
    print(f"  {'Realized return':<25} {'p10 miss':<12} {'p5 miss':<12} {'N':<8}")
    print(f"  {'-'*57}")
    for label, filt in thresholds:
        recs = [r for r in hist_vol_records if filt(r)]
        p10 = miss_rate(recs, "p10")
        p5 = miss_rate(recs, "p5")
        print(f"  {label:<25} {p10:<12.1%} {p5:<12.1%} {len(recs):<8}")

    # =========================================================================
    # PART 2: ECI × right-tail metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: ECI × right-tail metrics (p90 miss = fraction above p90)")
    print("=" * 70)

    # Compute per-model p90 miss rate (fraction of actuals ABOVE model's p90)
    # and right-tail width (distance from p50 to p90)
    model_stats = defaultdict(lambda: {"p90_above": 0, "p10_below": 0, "total": 0,
                                        "right_widths": [], "left_widths": []})

    for r in forecasts:
        model = r.get("model")
        if model not in ECI_SCORES:
            continue
        pcts = r.get("percentiles", {})
        gt = r.get("ground_truth")
        if not pcts or gt is None:
            continue
        if pcts.get("p90") is None or pcts.get("p10") is None:
            continue

        ms = model_stats[model]
        ms["total"] += 1
        if gt > pcts["p90"]:
            ms["p90_above"] += 1
        if gt < pcts["p10"]:
            ms["p10_below"] += 1
        if pcts.get("p50") is not None:
            ms["right_widths"].append(pcts["p90"] - pcts["p50"])
            ms["left_widths"].append(pcts["p50"] - pcts["p10"])

    # Filter to models with enough predictions
    valid_models = {m: s for m, s in model_stats.items() if s["total"] >= 700}
    print(f"\n{len(valid_models)} models with >= 700 valid predictions\n")

    print(f"  {'Model':<45} {'ECI':<8} {'p90 miss':<10} {'p10 miss':<10} {'R-width':<10} {'L-width':<10}")
    print(f"  {'-'*93}")

    ecis = []
    p90_misses = []
    p10_misses = []
    right_widths = []
    left_widths = []
    width_ratios = []

    for model in sorted(valid_models, key=lambda m: ECI_SCORES[m]):
        s = valid_models[model]
        eci = ECI_SCORES[model]
        p90_miss = s["p90_above"] / s["total"]
        p10_miss = s["p10_below"] / s["total"]
        rw = np.mean(s["right_widths"])
        lw = np.mean(s["left_widths"])
        ratio = rw / lw if lw > 0 else float("nan")

        ecis.append(eci)
        p90_misses.append(p90_miss)
        p10_misses.append(p10_miss)
        right_widths.append(rw)
        left_widths.append(lw)
        width_ratios.append(ratio)

        short = model.split("/")[-1][:40]
        print(f"  {short:<45} {eci:<8.1f} {p90_miss:<10.1%} {p10_miss:<10.1%} {rw:<10.1f} {lw:<10.1f}")

    # Correlations
    print(f"\n  Spearman correlations with ECI (N={len(ecis)}):")
    metrics = [
        ("p90 miss (above p90)", p90_misses),
        ("p10 miss (below p10)", p10_misses),
        ("Right-tail width (p90-p50)", right_widths),
        ("Left-tail width (p50-p10)", left_widths),
        ("Width ratio (right/left)", width_ratios),
    ]

    for name, vals in metrics:
        rho, p = stats.spearmanr(ecis, vals)
        sig = "*" if p < 0.05 else ("**" if p < 0.01 else "")
        if p < 0.01:
            sig = "**"
        if p < 0.001:
            sig = "***"
        direction = "pro-g" if rho < 0 else "anti-g"
        print(f"  ECI × {name:<35} ρ={rho:+.3f}  p={p:.4f}  {direction} {sig}")

    # Save results
    results = {
        "hist_vol_regime": {
            "drawdown_p10": miss_rate(drawdown, "p10"),
            "drawdown_p5": miss_rate(drawdown, "p5"),
            "drawdown_p1": miss_rate(drawdown, "p1"),
            "calm_p10": miss_rate(calm, "p10"),
            "calm_p5": miss_rate(calm, "p5"),
            "calm_p1": miss_rate(calm, "p1"),
            "n_drawdown": len(drawdown),
            "n_calm": len(calm),
        },
        "right_tail_correlations": {
            name: {"rho": float(stats.spearmanr(ecis, vals)[0]),
                   "p": float(stats.spearmanr(ecis, vals)[1]),
                   "n": len(ecis)}
            for name, vals in metrics
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
