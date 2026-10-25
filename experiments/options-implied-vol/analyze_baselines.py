#!/usr/bin/env python3
"""
Compare LLM distributional forecasts against two simple baselines:

1. Historical volatility: trailing 60-day realized vol + log-normal model
2. Options-implied volatility: CBOE vol index at cutoff + log-normal model

Computes PIT histograms, per-quantile calibration, and CRPS for all three.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/analyze_baselines.py
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static")

TRADING_DAYS_PER_YEAR = 252

ALL_QUANTILES = {
    "p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
    "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99,
}

ASSETS = {
    "sp500":     {"price_ticker": "^GSPC", "vol_ticker": "VIX"},
    "nasdaq100": {"price_ticker": "^NDX",  "vol_ticker": "VXN"},
    "russell":   {"price_ticker": "^RUT",  "vol_ticker": "RVX"},
    "em":        {"price_ticker": "EEM",   "vol_ticker": "VXEEM"},
    "oil":       {"price_ticker": "CL=F",  "vol_ticker": "OVX"},
    "gold":      {"price_ticker": "GC=F",  "vol_ticker": "GVZ"},
    "treasury":  {"price_ticker": "TLT",   "vol_ticker": "VXTLT"},
    "apple":     {"price_ticker": "AAPL",  "vol_ticker": "VXAPL"},
    "amazon":    {"price_ticker": "AMZN",  "vol_ticker": "VXAZN"},
    "alphabet":  {"price_ticker": "GOOGL", "vol_ticker": "VXGOG"},
}

# ECI values for models (from CivBench)
ECI_MAP = {
    "claude-3-haiku-20240307": 98,
    "gpt-3.5-turbo-0125": 100,
    "Mixtral-8x7B-Instruct-v0.1": 89,
    "Llama-3.3-70B-Instruct-Turbo": 117,
    "mistral-large-2407": 118,
    "mistral-large-2411": 125,
    "mistral-large-latest": 130,
    "gpt-4o": 132,
    "gemini-2.0-flash-lite-001": 110,
    "gemini-2.5-flash": 135,
    "gemini-2.5-pro": 140,
    "DeepSeek-V3": 128,
    "DeepSeek-V3.1": 135,
    "o3-mini-2025-01-31": 140,
    "gpt-4.1-2025-04-14": 137,
    "claude-sonnet-4-20250514": 140,
    "claude-haiku-4-5-20251001": 138,
    "o4-mini-2025-04-16": 145,
    "gpt-5-nano-2025-08-07": 120,
    "gpt-5-mini-2025-08-07": 142,
    "claude-sonnet-4-5-20250929": 148,
    "o3-2025-04-16": 149,
    "gpt-5-2025-08-07": 150,
    "gpt-5.1-2025-11-13": 152,
    "gemini-3-pro-preview": 154,
    "claude-opus-4-5-20251101": 155,
    "grok-4-fast-non-reasoning": 140,
    "grok-4-fast-reasoning": 148,
    "grok-4-0709": 151,
    "grok-4-1-fast-non-reasoning": 145,
    "grok-4-1-fast-reasoning": 152,
    "claude-opus-4-6": 157,
}


def lognormal_quantiles(price, sigma_ann, days):
    """Generate quantile forecasts from log-normal model.

    Args:
        price: current price
        sigma_ann: annualized volatility (as decimal, e.g. 0.20 for 20%)
        days: forecast horizon in calendar days

    Returns:
        dict of quantile label -> forecasted price
    """
    t = days / 365  # calendar days
    sigma_t = sigma_ann * math.sqrt(t)
    # Use risk-neutral drift for options-implied; for realized vol,
    # drift is ambiguous but small over short horizons, so we use the
    # same model for both (the drift term is tiny for 1-3 week horizons).
    drift = -0.5 * sigma_ann**2 * t
    quantiles = {}
    for label, tau in ALL_QUANTILES.items():
        z = norm.ppf(tau)
        quantiles[label] = price * math.exp(drift + z * sigma_t)
    return quantiles


def compute_realized_vol(prices_df, cutoff_date, lookback_days=60):
    """Compute annualized realized vol from trailing daily log-returns."""
    cutoff_dt = pd.Timestamp(cutoff_date)
    # Get prices up to and including cutoff
    hist = prices_df[prices_df.index <= cutoff_dt].tail(lookback_days + 1)
    if len(hist) < 20:  # need enough data
        return None
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_pit(quantiles_dict, ground_truth):
    """Compute PIT value for a single forecast."""
    qs = []
    vals = []
    for key in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
        if key in quantiles_dict:
            qs.append(ALL_QUANTILES[key])
            vals.append(quantiles_dict[key])
    if len(qs) < 5:
        return None
    qs_arr = np.array(qs)
    vals_arr = np.array(vals)
    if ground_truth <= vals_arr[0]:
        return qs_arr[0] / 2
    elif ground_truth >= vals_arr[-1]:
        return 1 - (1 - qs_arr[-1]) / 2
    else:
        return float(np.interp(ground_truth, vals_arr, qs_arr))


def compute_crps(quantiles_dict, ground_truth):
    """Approximate CRPS from quantile-weighted pinball loss."""
    total = 0
    n = 0
    for key, tau in ALL_QUANTILES.items():
        if key not in quantiles_dict:
            continue
        q = quantiles_dict[key]
        if ground_truth >= q:
            total += tau * (ground_truth - q)
        else:
            total += (1 - tau) * (q - ground_truth)
        n += 1
    return (2 / n) * total if n > 0 else None


def per_quantile_calibration(forecasts_with_gt):
    """Compute observed coverage at each quantile level.

    Args:
        forecasts_with_gt: list of (quantiles_dict, ground_truth) tuples

    Returns:
        dict of quantile label -> observed fraction below
    """
    results = {}
    for key, tau in ALL_QUANTILES.items():
        below = 0
        total = 0
        for qs, gt in forecasts_with_gt:
            if key in qs:
                if gt < qs[key]:
                    below += 1
                total += 1
        if total > 0:
            results[key] = below / total
    return results


def main():
    # ── Load data ──
    print("Loading data...")
    with open(DATA_DIR / "questions.json") as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]

    with open(RESULTS_DIR / "forecasts.json") as f:
        forecasts = json.load(f)

    # Load price histories
    price_dfs = {}
    for asset_id in ASSETS:
        df = pd.read_csv(DATA_DIR / "prices" / f"{asset_id}.csv", parse_dates=["Date"])
        df = df.set_index("Date")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        price_dfs[asset_id] = df

    # Filter to valid forecasts (has percentiles + ground truth)
    valid = [r for r in forecasts
             if r.get("percentiles") and r.get("ground_truth") is not None]

    # Exclude models with >70% parse failure
    MIN_VALID = 300
    models = sorted(set(r["model"] for r in valid))
    model_counts = {m: sum(1 for r in valid if r["model"] == m) for m in models}
    included_models = [m for m in models if model_counts[m] >= MIN_VALID]
    excluded_models = [m for m in models if model_counts[m] < MIN_VALID]
    valid = [r for r in valid if r["model"] in included_models]

    print(f"Valid forecasts: {len(valid)} from {len(included_models)} models")
    if excluded_models:
        print(f"Excluded {len(excluded_models)} models with <{MIN_VALID} valid results")

    # Build question lookup for ground truth + vol index
    q_lookup = {}
    for q in questions:
        key = (q["asset_id"], q["cutoff_date"], q["horizon"])
        q_lookup[key] = q

    # ── Compute baselines for each question ──
    print("\nComputing baselines...")
    hist_vol_forecasts = []  # (quantiles_dict, ground_truth) tuples
    impl_vol_forecasts = []
    llm_forecasts = []

    # Track per-question for later analysis
    question_keys_seen = set()

    for r in valid:
        key = (r["asset_id"], r["cutoff_date"], r["horizon"])
        gt = r["ground_truth"]

        # LLM forecast
        llm_forecasts.append((r["percentiles"], gt, r["model"]))

        # Only compute baselines once per question (not per model)
        if key in question_keys_seen:
            continue
        question_keys_seen.add(key)

        q = q_lookup.get(key)
        if q is None:
            continue

        days = r["days_from_cutoff"]
        price = r["cutoff_price"]

        # Historical vol baseline
        rv = compute_realized_vol(price_dfs[r["asset_id"]], r["cutoff_date"], lookback_days=60)
        if rv is not None and rv > 0:
            hv_qs = lognormal_quantiles(price, rv, days)
            hist_vol_forecasts.append((hv_qs, gt, r["asset_id"], r["cutoff_date"], r["horizon"], rv))

        # Options-implied vol baseline
        vol_idx = q.get("vol_index_at_cutoff")
        if vol_idx is not None and vol_idx > 0:
            iv = vol_idx / 100  # convert from percentage to decimal
            iv_qs = lognormal_quantiles(price, iv, days)
            impl_vol_forecasts.append((iv_qs, gt, r["asset_id"], r["cutoff_date"], r["horizon"], iv))

    print(f"Historical vol baseline: {len(hist_vol_forecasts)} forecasts")
    print(f"Options-implied baseline: {len(impl_vol_forecasts)} forecasts")
    print(f"LLM forecasts: {len(llm_forecasts)} forecasts")

    # ── PIT computation ──
    print("\n" + "=" * 70)
    print("  PIT HISTOGRAMS")
    print("=" * 70)

    llm_pits = [compute_pit(qs, gt) for qs, gt, _ in llm_forecasts]
    llm_pits = np.array([p for p in llm_pits if p is not None])

    hv_pits = [compute_pit(qs, gt) for qs, gt, *_ in hist_vol_forecasts]
    hv_pits = np.array([p for p in hv_pits if p is not None])

    iv_pits = [compute_pit(qs, gt) for qs, gt, *_ in impl_vol_forecasts]
    iv_pits = np.array([p for p in iv_pits if p is not None])

    print(f"\nLLMs:       {len(llm_pits)} PITs, mid-concentration (0.3-0.7): {np.mean((llm_pits >= 0.3) & (llm_pits <= 0.7)):.1%}")
    print(f"Hist vol:   {len(hv_pits)} PITs, mid-concentration: {np.mean((hv_pits >= 0.3) & (hv_pits <= 0.7)):.1%}")
    print(f"Impl vol:   {len(iv_pits)} PITs, mid-concentration: {np.mean((iv_pits >= 0.3) & (iv_pits <= 0.7)):.1%}")
    print(f"(Ideal mid-concentration: 40%)")

    # ── Per-quantile calibration ──
    print("\n" + "=" * 70)
    print("  PER-QUANTILE CALIBRATION")
    print("=" * 70)

    llm_cal = per_quantile_calibration([(qs, gt) for qs, gt, _ in llm_forecasts])
    hv_cal = per_quantile_calibration([(qs, gt) for qs, gt, *_ in hist_vol_forecasts])
    iv_cal = per_quantile_calibration([(qs, gt) for qs, gt, *_ in impl_vol_forecasts])

    print(f"\n  {'Quantile':<10} {'Nominal':>8} {'LLMs':>8} {'Hist Vol':>10} {'Impl Vol':>10}")
    print(f"  {'-'*48}")
    for key, tau in ALL_QUANTILES.items():
        llm_v = llm_cal.get(key, float('nan'))
        hv_v = hv_cal.get(key, float('nan'))
        iv_v = iv_cal.get(key, float('nan'))
        print(f"  {key:<10} {tau:>7.0%} {llm_v:>7.1%} {hv_v:>9.1%} {iv_v:>9.1%}")

    # ── CRPS comparison ──
    print("\n" + "=" * 70)
    print("  CRPS COMPARISON")
    print("=" * 70)

    llm_crps = [compute_crps(qs, gt) for qs, gt, _ in llm_forecasts]
    llm_crps = [c for c in llm_crps if c is not None]

    hv_crps = [compute_crps(qs, gt) for qs, gt, *_ in hist_vol_forecasts]
    hv_crps = [c for c in hv_crps if c is not None]

    iv_crps = [compute_crps(qs, gt) for qs, gt, *_ in impl_vol_forecasts]
    iv_crps = [c for c in iv_crps if c is not None]

    print(f"\n  {'Method':<20} {'Mean CRPS':>12} {'Median CRPS':>14} {'N':>6}")
    print(f"  {'-'*55}")
    print(f"  {'LLMs (pooled)':<20} {np.mean(llm_crps):>11.2f} {np.median(llm_crps):>13.2f} {len(llm_crps):>5}")
    print(f"  {'Hist vol (60d)':<20} {np.mean(hv_crps):>11.2f} {np.median(hv_crps):>13.2f} {len(hv_crps):>5}")
    print(f"  {'Options-implied':<20} {np.mean(iv_crps):>11.2f} {np.median(iv_crps):>13.2f} {len(iv_crps):>5}")

    # Per-model CRPS
    print(f"\n  Per-model CRPS:")
    print(f"  {'Model':<40} {'Mean CRPS':>12} {'N':>6}")
    print(f"  {'-'*60}")
    for model in sorted(included_models, key=lambda m: m.split("/")[-1]):
        model_crps = [compute_crps(r["percentiles"], r["ground_truth"])
                      for r in valid if r["model"] == model]
        model_crps = [c for c in model_crps if c is not None]
        if model_crps:
            print(f"  {model.split('/')[-1]:<40} {np.mean(model_crps):>11.2f} {len(model_crps):>5}")

    # ── Normalized CRPS (per-(asset, horizon) median) ──
    print("\n" + "=" * 70)
    print("  NORMALIZED CRPS (per-(asset, horizon) median)")
    print("=" * 70)
    print("\n  For each (asset, horizon) cell: divide each method's mean CRPS by the")
    print("  cross-method median (hist-vol included), then average across cells.")

    horizons = ["H1", "H2", "H3"]
    assets_list = sorted(ASSETS.keys())

    # Step 1: compute per-model mean CRPS for each (asset, horizon) cell
    # Include hist-vol as a "model" named "_hist_vol"
    # cell_crps_by_method[method][(asset, horizon)] = mean CRPS
    cell_crps_by_method = {}

    # LLM models
    for model in included_models:
        model_name = model.split("/")[-1]
        cell_crps_by_method[model_name] = {}
        for asset in assets_list:
            for hz in horizons:
                vals = [compute_crps(r["percentiles"], r["ground_truth"])
                        for r in valid
                        if r["model"] == model and r["asset_id"] == asset and r["horizon"] == hz]
                vals = [c for c in vals if c is not None]
                if vals:
                    cell_crps_by_method[model_name][(asset, hz)] = np.mean(vals)

    # Hist-vol baseline
    cell_crps_by_method["_hist_vol"] = {}
    for qs, gt, asset_id, cutoff_date, hz, rv in hist_vol_forecasts:
        cell = (asset_id, hz)
        cell_crps_by_method["_hist_vol"].setdefault(cell, [])
        c = compute_crps(qs, gt)
        if c is not None:
            cell_crps_by_method["_hist_vol"][cell].append(c)
    # average within each cell
    for cell in list(cell_crps_by_method["_hist_vol"].keys()):
        vals = cell_crps_by_method["_hist_vol"][cell]
        cell_crps_by_method["_hist_vol"][cell] = np.mean(vals) if vals else None
    # remove None cells
    cell_crps_by_method["_hist_vol"] = {
        k: v for k, v in cell_crps_by_method["_hist_vol"].items() if v is not None
    }

    # Step 2: compute cross-method median (all LLMs + hist-vol) for each cell
    all_methods = list(cell_crps_by_method.keys())
    all_cells_norm = set()
    for m in all_methods:
        all_cells_norm.update(cell_crps_by_method[m].keys())

    cell_medians_norm = {}
    for cell in all_cells_norm:
        method_means = [cell_crps_by_method[m][cell]
                        for m in all_methods if cell in cell_crps_by_method[m]]
        if method_means:
            cell_medians_norm[cell] = np.median(method_means)

    # Step 3: normalize each method, compute mean across cells
    def normalized_crps_for_method(method_name):
        norm_vals = []
        for cell, median in cell_medians_norm.items():
            if cell in cell_crps_by_method[method_name] and median > 0:
                norm_vals.append(cell_crps_by_method[method_name][cell] / median)
        return np.mean(norm_vals) if norm_vals else float('nan'), len(norm_vals)

    # Pooled LLM: average over all LLM model cell means, then normalize
    # (same approach: treat pooled LLM as average of per-model cell means)
    pooled_cell = {}
    llm_method_names = [m.split("/")[-1] for m in included_models]
    for cell in all_cells_norm:
        vals = [cell_crps_by_method[mn][cell]
                for mn in llm_method_names if cell in cell_crps_by_method[mn]]
        if vals:
            pooled_cell[cell] = np.mean(vals)

    pooled_norm_vals = [pooled_cell[cell] / cell_medians_norm[cell]
                        for cell in pooled_cell if cell in cell_medians_norm and cell_medians_norm[cell] > 0]
    pooled_norm = np.mean(pooled_norm_vals) if pooled_norm_vals else float('nan')

    # Best LLM by normalized CRPS
    best_norm = float('inf')
    best_model_norm = None
    for model in included_models:
        mn = model.split("/")[-1]
        ncrps, ncells = normalized_crps_for_method(mn)
        if ncells > 0 and ncrps < best_norm:
            best_norm = ncrps
            best_model_norm = mn

    hv_norm, hv_norm_cells = normalized_crps_for_method("_hist_vol")

    print(f"\n  {'Method':<30} {'Norm CRPS':>12} {'Cells':>8}")
    print(f"  {'-'*52}")
    print(f"  {'LLMs (pooled)':<30} {pooled_norm:>11.4f} {len(pooled_norm_vals):>7}")
    if best_model_norm:
        best_n, best_c = normalized_crps_for_method(best_model_norm)
        print(f"  {f'Best LLM ({best_model_norm})':<30} {best_n:>11.4f} {best_c:>7}")
    print(f"  {'Hist vol (60d)':<30} {hv_norm:>11.4f} {hv_norm_cells:>7}")

    # Per-asset breakdown
    print(f"\n  Per-asset normalized CRPS (hist-vol vs pooled LLM):")
    print(f"  {'Asset':<15} {'HV raw':>10} {'HV norm':>10} {'LLM raw':>10} {'LLM norm':>10}")
    print(f"  {'-'*57}")
    for asset in assets_list:
        hv_raw_vals = [cell_crps_by_method["_hist_vol"][(asset, hz)]
                       for hz in horizons if (asset, hz) in cell_crps_by_method["_hist_vol"]]
        llm_raw_vals = []
        llm_norm_vals_asset = []
        hv_norm_vals_asset = []
        for hz in horizons:
            cell = (asset, hz)
            if cell not in cell_medians_norm or cell_medians_norm[cell] == 0:
                continue
            med = cell_medians_norm[cell]
            if cell in cell_crps_by_method["_hist_vol"]:
                hv_norm_vals_asset.append(cell_crps_by_method["_hist_vol"][cell] / med)
            for mn in llm_method_names:
                if cell in cell_crps_by_method[mn]:
                    llm_raw_vals.append(cell_crps_by_method[mn][cell])
                    llm_norm_vals_asset.append(cell_crps_by_method[mn][cell] / med)
        if hv_raw_vals and llm_raw_vals:
            print(f"  {asset:<15} {np.mean(hv_raw_vals):>9.2f} {np.mean(hv_norm_vals_asset):>9.4f} "
                  f"{np.mean(llm_raw_vals):>9.2f} {np.mean(llm_norm_vals_asset):>9.4f}")

    # ── Coverage rates ──
    print("\n" + "=" * 70)
    print("  COVERAGE RATES (p10-p90: nominal 80%)")
    print("=" * 70)

    def coverage(forecast_list, lo_key="p10", hi_key="p90"):
        inside = 0
        total = 0
        for item in forecast_list:
            qs = item[0]
            gt = item[1]
            if lo_key in qs and hi_key in qs:
                if qs[lo_key] <= gt <= qs[hi_key]:
                    inside += 1
                total += 1
        return inside / total if total > 0 else float('nan')

    llm_cov = coverage([(qs, gt) for qs, gt, _ in llm_forecasts])
    hv_cov = coverage(hist_vol_forecasts)
    iv_cov = coverage(impl_vol_forecasts)

    print(f"\n  LLMs:          {llm_cov:.1%}")
    print(f"  Hist vol (60d): {hv_cov:.1%}")
    print(f"  Options-implied: {iv_cov:.1%}")
    print(f"  (Nominal: 80%)")

    # ── Peakedness × ECI correlation ──
    print("\n" + "=" * 70)
    print("  PEAKEDNESS × ECI CORRELATION")
    print("=" * 70)

    model_peakedness = {}
    model_ecis = {}
    for model in included_models:
        model_name = model.split("/")[-1]
        if model_name not in ECI_MAP:
            continue
        model_pits = []
        for qs, gt, m in llm_forecasts:
            if m == model:
                pit = compute_pit(qs, gt)
                if pit is not None:
                    model_pits.append(pit)
        if len(model_pits) > 50:
            pits_arr = np.array(model_pits)
            model_peakedness[model_name] = float(np.mean((pits_arr >= 0.3) & (pits_arr <= 0.7)))
            model_ecis[model_name] = ECI_MAP[model_name]

    if len(model_peakedness) >= 5:
        names = sorted(model_peakedness.keys())
        ecis = [model_ecis[n] for n in names]
        peaks = [model_peakedness[n] for n in names]
        rho, p = spearmanr(ecis, peaks)
        print(f"\n  ECI × peakedness: ρ={rho:+.3f}, p={p:.4f}, N={len(names)}")
        print(f"\n  {'Model':<40} {'ECI':>5} {'Peak (0.3-0.7)':>15}")
        print(f"  {'-'*62}")
        for n in sorted(names, key=lambda x: model_ecis[x]):
            print(f"  {n:<40} {model_ecis[n]:>5} {model_peakedness[n]:>14.1%}")

    # Compute baselines' peakedness for comparison
    hv_peak = float(np.mean((hv_pits >= 0.3) & (hv_pits <= 0.7)))
    iv_peak = float(np.mean((iv_pits >= 0.3) & (iv_pits <= 0.7)))
    print(f"\n  Baseline peakedness:")
    print(f"    Hist vol (60d):   {hv_peak:.1%}")
    print(f"    Options-implied:  {iv_peak:.1%}")
    print(f"    (Ideal: 40%)")

    # ── Mean realized vs implied vol ──
    print("\n" + "=" * 70)
    print("  MEAN VOLATILITIES")
    print("=" * 70)

    rv_vals = [item[5] for item in hist_vol_forecasts]
    iv_vals = [item[5] for item in impl_vol_forecasts]
    print(f"\n  Mean 60-day realized vol:   {np.mean(rv_vals)*100:.1f}%")
    print(f"  Mean options-implied vol:   {np.mean(iv_vals)*100:.1f}%")

    # Per-asset
    print(f"\n  {'Asset':<15} {'Realized (60d)':>15} {'Implied (CBOE)':>15}")
    print(f"  {'-'*47}")
    for asset_id in sorted(ASSETS.keys()):
        rv_a = [item[5] for item in hist_vol_forecasts if item[2] == asset_id]
        iv_a = [item[5] for item in impl_vol_forecasts if item[2] == asset_id]
        if rv_a and iv_a:
            print(f"  {asset_id:<15} {np.mean(rv_a)*100:>14.1f}% {np.mean(iv_a)*100:>14.1f}%")

    # ── Compute LLM-implied vol for comparison ──
    print("\n" + "=" * 70)
    print("  LLM-IMPLIED VOL (back-solved from p10/p90)")
    print("=" * 70)

    Z_90 = norm.ppf(0.90)
    llm_impl_vols = []
    for r in valid:
        p10 = r["percentiles"].get("p10")
        p90 = r["percentiles"].get("p90")
        if p10 and p90 and p10 > 0 and p90 > p10:
            days = r["days_from_cutoff"]
            t = days / 365
            implied_vol = math.log(p90 / p10) / (2 * Z_90 * math.sqrt(t))
            llm_impl_vols.append({
                "model": r["model"].split("/")[-1],
                "asset_id": r["asset_id"],
                "vol": implied_vol,
            })

    print(f"\n  Mean LLM-implied vol: {np.mean([v['vol'] for v in llm_impl_vols])*100:.1f}%")
    print(f"  Mean hist realized:   {np.mean(rv_vals)*100:.1f}%")
    print(f"  Mean options-implied: {np.mean(iv_vals)*100:.1f}%")
    print(f"  LLM/hist ratio:      {np.mean([v['vol'] for v in llm_impl_vols])/np.mean(rv_vals):.2f}x")
    print(f"  LLM/impl ratio:      {np.mean([v['vol'] for v in llm_impl_vols])/np.mean(iv_vals):.2f}x")

    # ── Figure: 3-panel PIT comparison ──
    print("\nGenerating figure...")

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    # Panel 1: LLMs pooled (by asset class)
    groups = {
        "Equity indices": ["sp500", "nasdaq100", "russell", "em"],
        "Commodities": ["oil", "gold"],
        "Single stocks": ["apple", "amazon", "alphabet"],
        "Bonds": ["treasury"],
    }
    group_colors = {
        "Equity indices": "#2166AC",
        "Commodities": "#B2182B",
        "Single stocks": "#762A83",
        "Bonds": "#1B7837",
    }

    ax = axes[0]
    for group_name, asset_ids in groups.items():
        group_pits = []
        for qs, gt, m in llm_forecasts:
            # need asset_id from original record - rebuild from valid
            pass
        # Use valid records directly for asset filtering
        group_recs = [r for r in valid if r["asset_id"] in asset_ids]
        pits = []
        for r in group_recs:
            pit = compute_pit(r["percentiles"], r["ground_truth"])
            if pit is not None:
                pits.append(pit)
        pits = np.array(pits)
        if len(pits) > 0:
            counts, _ = np.histogram(pits, bins=bins)
            freqs = counts / counts.sum()
            ax.plot(bin_centers, freqs, marker="o", markersize=4, linewidth=2,
                    alpha=0.9, color=group_colors[group_name], label=group_name)

    ax.axhline(0.10, color="#666", linestyle="--", linewidth=1, alpha=0.6, label="Well-calibrated")
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("LLMs (pooled)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.24)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.12)

    # Panel 2: Historical vol baseline
    ax = axes[1]
    hv_counts, _ = np.histogram(hv_pits, bins=bins)
    hv_freqs = hv_counts / hv_counts.sum()
    ax.plot(bin_centers, hv_freqs, marker="s", markersize=5, linewidth=2.5,
            color="#E66100", label=f"60-day realized vol (N={len(hv_pits)})")
    ax.axhline(0.10, color="#666", linestyle="--", linewidth=1, alpha=0.6, label="Well-calibrated")
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_title("Historical vol baseline", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.12)

    # Panel 3: Options-implied baseline
    ax = axes[2]
    iv_counts, _ = np.histogram(iv_pits, bins=bins)
    iv_freqs = iv_counts / iv_counts.sum()
    ax.plot(bin_centers, iv_freqs, marker="D", markersize=5, linewidth=2.5,
            color="#5D3A9B", label=f"CBOE implied vol (N={len(iv_pits)})")
    ax.axhline(0.10, color="#666", linestyle="--", linewidth=1, alpha=0.6, label="Well-calibrated")
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_title("Options-implied baseline", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "baselines_pit_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved PIT figure to {out_path}")
    plt.close()

    # ── Figure 2: Overlay comparison ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(bin_centers, hv_freqs, marker="s", markersize=6, linewidth=2.5,
            color="#E66100", label="Historical vol (60-day realized)", zorder=3)
    ax.plot(bin_centers, iv_freqs, marker="D", markersize=6, linewidth=2.5,
            color="#5D3A9B", label="Options-implied (CBOE indices)", zorder=3)

    # LLM aggregate
    llm_counts, _ = np.histogram(llm_pits, bins=bins)
    llm_freqs = llm_counts / llm_counts.sum()
    ax.plot(bin_centers, llm_freqs, marker="o", markersize=6, linewidth=2.5,
            color="#2166AC", label="LLMs (27 models pooled)", zorder=3)

    ax.axhline(0.10, color="#666", linestyle="--", linewidth=1, alpha=0.6, label="Well-calibrated")
    ax.set_xlabel("PIT value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distributional calibration: LLMs vs. simple baselines", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.24)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    out_path2 = OUTPUT_DIR / "baselines_pit_overlay.png"
    fig.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Saved overlay figure to {out_path2}")
    plt.close()

    # ── Figure 3: ECI × peakedness with baselines ──
    if len(model_peakedness) >= 5:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        names = sorted(model_peakedness.keys(), key=lambda n: model_ecis[n])
        ecis = [model_ecis[n] for n in names]
        peaks = [model_peakedness[n] for n in names]

        ax.scatter(ecis, peaks, c="#2166AC", s=50, alpha=0.7, zorder=3, label="LLMs")

        # Add baseline lines
        ax.axhline(hv_peak, color="#E66100", linestyle="-", linewidth=2,
                    alpha=0.8, label=f"Hist vol: {hv_peak:.1%}")
        ax.axhline(iv_peak, color="#5D3A9B", linestyle="-", linewidth=2,
                    alpha=0.8, label=f"Options-implied: {iv_peak:.1%}")
        ax.axhline(0.40, color="#666", linestyle="--", linewidth=1,
                    alpha=0.6, label="Ideal (40%)")

        # Fit line
        z = np.polyfit(ecis, peaks, 1)
        x_line = np.linspace(min(ecis) - 5, max(ecis) + 5, 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="#2166AC", alpha=0.4)

        rho, p = spearmanr(ecis, peaks)
        ax.set_xlabel("ECI (model capability)", fontsize=12)
        ax.set_ylabel("Peakedness (fraction in 0.3-0.7)", fontsize=12)
        ax.set_title(f"Capability predicts overconfidence (ρ={rho:+.3f}, p={p:.3f})",
                      fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
        ax.grid(alpha=0.12)

        plt.tight_layout()
        out_path3 = OUTPUT_DIR / "realworld_eci_correlation.png"
        fig.savefig(out_path3, dpi=200, bbox_inches="tight")
        print(f"Saved ECI figure to {out_path3}")
        plt.close()

    # ── Save structured results ──
    results_summary = {
        "per_quantile_calibration": {
            "llm": {k: round(v, 4) for k, v in llm_cal.items()},
            "hist_vol": {k: round(v, 4) for k, v in hv_cal.items()},
            "impl_vol": {k: round(v, 4) for k, v in iv_cal.items()},
        },
        "crps": {
            "llm_mean": round(float(np.mean(llm_crps)), 2),
            "hist_vol_mean": round(float(np.mean(hv_crps)), 2),
            "impl_vol_mean": round(float(np.mean(iv_crps)), 2),
        },
        "coverage_p10_p90": {
            "llm": round(llm_cov, 4),
            "hist_vol": round(hv_cov, 4),
            "impl_vol": round(iv_cov, 4),
        },
        "peakedness": {
            "llm_pooled": round(float(np.mean((llm_pits >= 0.3) & (llm_pits <= 0.7))), 4),
            "hist_vol": round(hv_peak, 4),
            "impl_vol": round(iv_peak, 4),
        },
        "n_forecasts": {
            "llm": len(llm_crps),
            "hist_vol": len(hv_crps),
            "impl_vol": len(iv_crps),
        },
        "mean_vol": {
            "realized_60d": round(float(np.mean(rv_vals)) * 100, 1),
            "options_implied": round(float(np.mean(iv_vals)) * 100, 1),
            "llm_implied": round(float(np.mean([v['vol'] for v in llm_impl_vols])) * 100, 1),
        },
    }

    out_json = RESULTS_DIR / "baseline_comparison.json"
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
