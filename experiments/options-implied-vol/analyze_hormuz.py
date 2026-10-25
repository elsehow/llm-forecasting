#!/usr/bin/env python3
"""
Hormuz natural experiment: formalize the pre/post analysis.

Shows three forecasters' response functions to the March 2026 oil shock:
1. Historical vol (60-day realized) — widens mechanically
2. Options-implied vol (OVX) — widens dramatically
3. LLMs (27 models) — barely moves

Also: p50 vs. spread decomposition — did models adjust their central estimate
even if intervals didn't widen?

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/analyze_hormuz.py
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static")

TRADING_DAYS_PER_YEAR = 252


def compute_realized_vol(prices_df, cutoff_date, lookback_days=60):
    cutoff_dt = pd.Timestamp(cutoff_date)
    hist = prices_df[prices_df.index <= cutoff_dt].tail(lookback_days + 1)
    if len(hist) < 20:
        return None
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def lognormal_interval(price, sigma_ann, days, lo_q=0.10, hi_q=0.90):
    """Compute p_lo and p_hi from log-normal model."""
    t = days / 365
    sigma_t = sigma_ann * math.sqrt(t)
    drift = -0.5 * sigma_ann**2 * t
    z_lo = norm.ppf(lo_q)
    z_hi = norm.ppf(hi_q)
    p_lo = price * math.exp(drift + z_lo * sigma_t)
    p_hi = price * math.exp(drift + z_hi * sigma_t)
    return p_lo, p_hi


def main():
    # ── Load data ──
    print("Loading data...")
    with open(DATA_DIR / "questions.json") as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]

    with open(RESULTS_DIR / "forecasts.json") as f:
        forecasts = json.load(f)

    # Load oil prices and OVX
    oil_prices = pd.read_csv(DATA_DIR / "prices" / "oil.csv", parse_dates=["Date"])
    oil_prices = oil_prices.set_index("Date")
    if isinstance(oil_prices.columns, pd.MultiIndex):
        oil_prices.columns = oil_prices.columns.get_level_values(0)

    ovx = pd.read_csv(DATA_DIR / "vol_indices" / "OVX.csv", parse_dates=["Date"])
    ovx = ovx.set_index("Date")

    # Filter to oil forecasts with valid data
    MIN_VALID = 300
    valid_all = [r for r in forecasts
                 if r.get("percentiles") and r.get("ground_truth") is not None]
    models = sorted(set(r["model"] for r in valid_all))
    model_counts = {m: sum(1 for r in valid_all if r["model"] == m) for m in models}
    included = [m for m in models if model_counts[m] >= MIN_VALID]

    oil_forecasts = [r for r in valid_all
                     if r["asset_id"] == "oil" and r["model"] in included]
    print(f"Oil forecasts: {len(oil_forecasts)} from {len(included)} models")

    # Build question lookup
    q_lookup = {}
    for q in questions:
        key = (q["asset_id"], q["cutoff_date"], q["horizon"])
        q_lookup[key] = q

    # ══════════════════════════════════════════════════════════════════
    # 1. THREE-FORECASTER RESPONSE FUNCTION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  THREE-FORECASTER RESPONSE: OIL (H1 = +7 days)")
    print("=" * 70)

    # Get unique cutoff dates for oil H1
    cutoff_data = []
    for r in oil_forecasts:
        if r["horizon"] != "H1":
            continue
        cd = r["cutoff_date"]
        if any(d["cutoff_date"] == cd for d in cutoff_data):
            # Aggregate LLM intervals
            existing = next(d for d in cutoff_data if d["cutoff_date"] == cd)
            existing["llm_p10s"].append(r["percentiles"]["p10"])
            existing["llm_p90s"].append(r["percentiles"]["p90"])
            existing["llm_p50s"].append(r["percentiles"]["p50"])
            continue

        days = r["days_from_cutoff"]
        price = r["cutoff_price"]

        # Historical vol
        rv = compute_realized_vol(oil_prices, cd, lookback_days=60)
        if rv is not None:
            hv_lo, hv_hi = lognormal_interval(price, rv, days)
        else:
            hv_lo, hv_hi = None, None

        # Options-implied vol
        q = q_lookup.get(("oil", cd, "H1"))
        iv = q["vol_index_at_cutoff"] / 100 if q and q.get("vol_index_at_cutoff") else None
        if iv is not None:
            iv_lo, iv_hi = lognormal_interval(price, iv, days)
        else:
            iv_lo, iv_hi = None, None

        cutoff_data.append({
            "cutoff_date": cd,
            "price": price,
            "days": days,
            "ground_truth": r["ground_truth"],
            "realized_return": (r["ground_truth"] - price) / price,
            "realized_vol": rv,
            "implied_vol": iv,
            "hv_lo": hv_lo, "hv_hi": hv_hi,
            "hv_width": (hv_hi - hv_lo) if hv_lo else None,
            "iv_lo": iv_lo, "iv_hi": iv_hi,
            "iv_width": (iv_hi - iv_lo) if iv_lo else None,
            "llm_p10s": [r["percentiles"]["p10"]],
            "llm_p90s": [r["percentiles"]["p90"]],
            "llm_p50s": [r["percentiles"]["p50"]],
        })

    # Sort by date
    cutoff_data.sort(key=lambda d: d["cutoff_date"])

    # Compute LLM aggregates
    for d in cutoff_data:
        d["llm_p10_mean"] = np.mean(d["llm_p10s"])
        d["llm_p90_mean"] = np.mean(d["llm_p90s"])
        d["llm_p50_mean"] = np.mean(d["llm_p50s"])
        d["llm_width"] = d["llm_p90_mean"] - d["llm_p10_mean"]

    print(f"\n  {'Cutoff':<12} {'Price':>8} {'GT':>8} {'Ret':>7} "
          f"{'RV':>6} {'IV':>6} {'HV w':>7} {'IV w':>7} {'LLM w':>7} "
          f"{'LLM p50':>8}")
    print(f"  {'-'*90}")

    for d in cutoff_data:
        rv_str = f"{d['realized_vol']*100:.1f}%" if d['realized_vol'] else "n/a"
        iv_str = f"{d['implied_vol']*100:.1f}%" if d['implied_vol'] else "n/a"
        print(f"  {d['cutoff_date']:<12} {d['price']:>7.1f} {d['ground_truth']:>7.1f} "
              f"{d['realized_return']:>+6.1%} "
              f"{rv_str:>6} {iv_str:>6} "
              f"{d['hv_width']:>6.1f} {d['iv_width']:>6.1f} {d['llm_width']:>6.1f} "
              f"{d['llm_p50_mean']:>7.1f}")

    # ══════════════════════════════════════════════════════════════════
    # 2. PRE/POST HORMUZ COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PRE/POST HORMUZ: OIL INTERVAL WIDTHS")
    print("=" * 70)

    # Define pre-Hormuz (Sep 2025 - Feb 2026) vs during/post (Mar 2026)
    pre = [d for d in cutoff_data if d["cutoff_date"] < "2026-02-23"]
    post = [d for d in cutoff_data if d["cutoff_date"] >= "2026-02-23"]

    def mean_width(data, key):
        vals = [d[key] for d in data if d[key] is not None]
        return np.mean(vals) if vals else float("nan")

    print(f"\n  {'Period':<20} {'HV width':>10} {'IV width':>10} {'LLM width':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Pre (Sep-Feb)':<20} {mean_width(pre, 'hv_width'):>9.1f} "
          f"{mean_width(pre, 'iv_width'):>9.1f} {mean_width(pre, 'llm_width'):>9.1f}")
    print(f"  {'Post (Mar)':<20} {mean_width(post, 'hv_width'):>9.1f} "
          f"{mean_width(post, 'iv_width'):>9.1f} {mean_width(post, 'llm_width'):>9.1f}")

    pre_hv = mean_width(pre, 'hv_width')
    post_hv = mean_width(post, 'hv_width')
    pre_iv = mean_width(pre, 'iv_width')
    post_iv = mean_width(post, 'iv_width')
    pre_llm = mean_width(pre, 'llm_width')
    post_llm = mean_width(post, 'llm_width')

    print(f"\n  Widening ratios (post/pre):")
    print(f"    Historical vol:  {post_hv/pre_hv:.1f}×")
    print(f"    Options-implied: {post_iv/pre_iv:.1f}×")
    print(f"    LLMs:            {post_llm/pre_llm:.1f}×")

    # ══════════════════════════════════════════════════════════════════
    # 3. p50 vs SPREAD DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  p50 vs SPREAD: DO MODELS ADJUST LEVEL BUT NOT UNCERTAINTY?")
    print("=" * 70)

    # Compare p50 tracking vs spread tracking
    # The "correct" p50 for each cutoff is approximately the cutoff price
    # (random walk assumption). The "correct" response to a vol spike is
    # wider spread, not necessarily different p50.

    print(f"\n  {'Cutoff':<12} {'Price':>8} {'LLM p50':>8} {'p50/Price':>10} "
          f"{'LLM spread':>11} {'HV spread':>10} {'LLM/HV':>8}")
    print(f"  {'-'*72}")

    for d in cutoff_data:
        ratio = d["llm_p50_mean"] / d["price"]
        llm_spread = d["llm_width"]
        hv_spread = d["hv_width"] if d["hv_width"] else 0
        spread_ratio = llm_spread / hv_spread if hv_spread > 0 else float("nan")
        print(f"  {d['cutoff_date']:<12} {d['price']:>7.1f} {d['llm_p50_mean']:>7.1f} "
              f"{ratio:>9.3f} {llm_spread:>10.1f} {hv_spread:>9.1f} {spread_ratio:>7.2f}")

    # Summary: how much did p50 and spread each change?
    pre_p50_ratio = np.mean([d["llm_p50_mean"] / d["price"] for d in pre])
    post_p50_ratio = np.mean([d["llm_p50_mean"] / d["price"] for d in post])
    pre_spread_ratio = np.mean([d["llm_width"] / d["hv_width"]
                                 for d in pre if d["hv_width"]])
    post_spread_ratio = np.mean([d["llm_width"] / d["hv_width"]
                                  for d in post if d["hv_width"]])

    print(f"\n  p50 tracking (LLM p50 / cutoff price):")
    print(f"    Pre-Hormuz:  {pre_p50_ratio:.3f}")
    print(f"    Post-Hormuz: {post_p50_ratio:.3f}")
    print(f"    Change:      {(post_p50_ratio - pre_p50_ratio):.3f}")

    print(f"\n  Spread tracking (LLM width / HV width):")
    print(f"    Pre-Hormuz:  {pre_spread_ratio:.2f}×")
    print(f"    Post-Hormuz: {post_spread_ratio:.2f}×")
    print(f"    Change:      {post_spread_ratio - pre_spread_ratio:+.2f}×")

    # ══════════════════════════════════════════════════════════════════
    # 4. ASSET-SPECIFIC BREAKDOWN: HORMUZ IMPACT BY ASSET
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ASSET-SPECIFIC: INTERVAL WIDTH CHANGE AT HORMUZ CUTOFFS")
    print("=" * 70)

    all_assets = sorted(set(r["asset_id"] for r in valid_all if r["model"] in included))

    # Load all price data
    price_dfs = {}
    for asset_id in all_assets:
        df = pd.read_csv(DATA_DIR / "prices" / f"{asset_id}.csv", parse_dates=["Date"])
        df = df.set_index("Date")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        price_dfs[asset_id] = df

    print(f"\n  {'Asset':<12} {'Pre LLM w':>10} {'Post LLM w':>11} {'Ratio':>7} "
          f"{'Pre HV w':>9} {'Post HV w':>10} {'HV Ratio':>9}")
    print(f"  {'-'*72}")

    for asset_id in all_assets:
        asset_recs = [r for r in valid_all
                      if r["asset_id"] == asset_id and r["model"] in included
                      and r["horizon"] == "H1"]
        if not asset_recs:
            continue

        pre_widths_llm = []
        post_widths_llm = []
        pre_widths_hv = []
        post_widths_hv = []

        for r in asset_recs:
            cd = r["cutoff_date"]
            llm_w = r["percentiles"]["p90"] - r["percentiles"]["p10"]
            days = r["days_from_cutoff"]
            price = r["cutoff_price"]

            rv = compute_realized_vol(price_dfs[asset_id], cd, lookback_days=60)
            if rv is not None:
                hv_lo, hv_hi = lognormal_interval(price, rv, days)
                hv_w = hv_hi - hv_lo
            else:
                hv_w = None

            if cd < "2026-02-23":
                pre_widths_llm.append(llm_w)
                if hv_w is not None:
                    pre_widths_hv.append(hv_w)
            else:
                post_widths_llm.append(llm_w)
                if hv_w is not None:
                    post_widths_hv.append(hv_w)

        pre_llm_m = np.mean(pre_widths_llm) if pre_widths_llm else float("nan")
        post_llm_m = np.mean(post_widths_llm) if post_widths_llm else float("nan")
        pre_hv_m = np.mean(pre_widths_hv) if pre_widths_hv else float("nan")
        post_hv_m = np.mean(post_widths_hv) if post_widths_hv else float("nan")

        llm_ratio = post_llm_m / pre_llm_m if pre_llm_m > 0 else float("nan")
        hv_ratio = post_hv_m / pre_hv_m if pre_hv_m > 0 else float("nan")

        print(f"  {asset_id:<12} {pre_llm_m:>9.1f} {post_llm_m:>10.1f} {llm_ratio:>6.2f}× "
              f"{pre_hv_m:>8.1f} {post_hv_m:>9.1f} {hv_ratio:>8.2f}×")

    # ══════════════════════════════════════════════════════════════════
    # 5. PER-MODEL RESPONSE TO HORMUZ
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PER-MODEL: OIL INTERVAL WIDTH CHANGE (H1)")
    print("=" * 70)

    print(f"\n  {'Model':<35} {'Pre width':>10} {'Post width':>11} {'Ratio':>7}")
    print(f"  {'-'*65}")

    for model in sorted(included, key=lambda m: m.split("/")[-1]):
        model_oil = [r for r in oil_forecasts
                     if r["model"] == model and r["horizon"] == "H1"]
        pre_w = [r["percentiles"]["p90"] - r["percentiles"]["p10"]
                 for r in model_oil if r["cutoff_date"] < "2026-02-23"]
        post_w = [r["percentiles"]["p90"] - r["percentiles"]["p10"]
                  for r in model_oil if r["cutoff_date"] >= "2026-02-23"]
        if pre_w and post_w:
            ratio = np.mean(post_w) / np.mean(pre_w)
            print(f"  {model.split('/')[-1]:<35} {np.mean(pre_w):>9.1f} "
                  f"{np.mean(post_w):>10.1f} {ratio:>6.2f}×")

    # ══════════════════════════════════════════════════════════════════
    # 6. COUNTERFACTUAL: WHAT IF LLMs USED HIST VOL DURING HORMUZ?
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COUNTERFACTUAL: LLM CRPS vs HIST-VOL CRPS ON OIL")
    print("=" * 70)

    from analyze_baselines import compute_crps, ALL_QUANTILES, lognormal_quantiles

    # Compute CRPS for LLMs vs hist vol on oil, pre vs post
    for period_label, date_pred in [("Pre-Hormuz", lambda d: d < "2026-02-23"),
                                      ("Post-Hormuz", lambda d: d >= "2026-02-23")]:
        period_recs = [r for r in oil_forecasts if date_pred(r["cutoff_date"])]

        # LLM CRPS
        llm_crps = [compute_crps(r["percentiles"], r["ground_truth"]) for r in period_recs]
        llm_crps = [c for c in llm_crps if c is not None]

        # Hist vol CRPS (one per unique question)
        seen = set()
        hv_crps = []
        for r in period_recs:
            key = (r["cutoff_date"], r["horizon"])
            if key in seen:
                continue
            seen.add(key)
            rv = compute_realized_vol(oil_prices, r["cutoff_date"], lookback_days=60)
            if rv is not None:
                hv_qs = lognormal_quantiles(r["cutoff_price"], rv, r["days_from_cutoff"])
                c = compute_crps(hv_qs, r["ground_truth"])
                if c is not None:
                    hv_crps.append(c)

        print(f"\n  {period_label}:")
        print(f"    LLM CRPS:      {np.mean(llm_crps):.2f} (N={len(llm_crps)})")
        print(f"    Hist vol CRPS: {np.mean(hv_crps):.2f} (N={len(hv_crps)})")
        if llm_crps and hv_crps:
            print(f"    LLM/HV ratio:  {np.mean(llm_crps)/np.mean(hv_crps):.2f}×")

    # ══════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════
    print("\n\nGenerating figures...")

    # ── Figure 1: Three-forecaster response function ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    dates = [pd.Timestamp(d["cutoff_date"]) for d in cutoff_data]
    hv_widths = [d["hv_width"] for d in cutoff_data]
    iv_widths = [d["iv_width"] for d in cutoff_data]
    llm_widths = [d["llm_width"] for d in cutoff_data]

    ax1.plot(dates, iv_widths, "o-", color="#5D3A9B", linewidth=2.5, markersize=6,
             label="Options market (OVX)", zorder=3)
    ax1.plot(dates, hv_widths, "s-", color="#E66100", linewidth=2.5, markersize=6,
             label="Historical vol (60d realized)", zorder=3)
    ax1.plot(dates, llm_widths, "D-", color="#2166AC", linewidth=2.5, markersize=6,
             label="LLMs (27 models, mean)", zorder=3)

    # Mark Hormuz
    hormuz_date = pd.Timestamp("2026-03-05")
    ax1.axvline(hormuz_date, color="#B2182B", linestyle="--", alpha=0.6, linewidth=1.5)
    ax1.text(hormuz_date, ax1.get_ylim()[1] * 0.95, "  Hormuz\n  closure",
             color="#B2182B", fontsize=9, va="top")

    ax1.set_ylabel("p10–p90 interval width ($/bbl)", fontsize=11)
    ax1.set_title("Three forecasters respond to the oil shock", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.12)

    # Bottom panel: oil price
    prices_plot = [d["price"] for d in cutoff_data]
    gt_plot = [d["ground_truth"] for d in cutoff_data]
    ax2.plot(dates, prices_plot, "k-", linewidth=1.5, label="Cutoff price")
    ax2.plot(dates, gt_plot, "k--", linewidth=1, alpha=0.5, label="Realized (+7d)")
    ax2.axvline(hormuz_date, color="#B2182B", linestyle="--", alpha=0.6, linewidth=1.5)
    ax2.set_ylabel("Oil price ($/bbl)", fontsize=11)
    ax2.set_xlabel("Cutoff date", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.12)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out1 = OUTPUT_DIR / "hormuz_three_forecasters.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # ── Figure 2: p50 vs spread decomposition ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: p50 tracking
    llm_p50s = [d["llm_p50_mean"] for d in cutoff_data]
    ax1.plot(dates, prices_plot, "k-", linewidth=2, label="Cutoff price", zorder=2)
    ax1.plot(dates, llm_p50s, "o-", color="#2166AC", linewidth=2, markersize=5,
             label="LLM p50 (mean)", zorder=3)
    ax1.axvline(hormuz_date, color="#B2182B", linestyle="--", alpha=0.6, linewidth=1.5)
    ax1.set_ylabel("Price ($/bbl)", fontsize=11)
    ax1.set_title("Central estimate tracks the level", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right: spread relative to hist vol
    spread_ratios = [d["llm_width"] / d["hv_width"]
                     if d["hv_width"] and d["hv_width"] > 0 else float("nan")
                     for d in cutoff_data]
    ax2.plot(dates, spread_ratios, "D-", color="#2166AC", linewidth=2, markersize=5,
             label="LLM/HV width ratio")
    ax2.axhline(1.0, color="#999", linestyle=":", linewidth=1, alpha=0.6,
                label="Parity (LLM = hist vol)")
    ax2.axvline(hormuz_date, color="#B2182B", linestyle="--", alpha=0.6, linewidth=1.5)
    ax2.set_ylabel("LLM interval / historical vol interval", fontsize=11)
    ax2.set_title("Uncertainty does not", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("LLMs adjust their point estimate but not their uncertainty",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out2 = OUTPUT_DIR / "hormuz_p50_vs_spread.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    # ── Figure 3: Asset-specific widening ratio ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    asset_ratios = []
    for asset_id in all_assets:
        asset_recs = [r for r in valid_all
                      if r["asset_id"] == asset_id and r["model"] in included
                      and r["horizon"] == "H1"]
        pre_w = [r["percentiles"]["p90"] - r["percentiles"]["p10"]
                 for r in asset_recs if r["cutoff_date"] < "2026-02-23"]
        post_w = [r["percentiles"]["p90"] - r["percentiles"]["p10"]
                  for r in asset_recs if r["cutoff_date"] >= "2026-02-23"]

        # Hist vol widening
        pre_hv = []
        post_hv = []
        for r in asset_recs:
            rv = compute_realized_vol(price_dfs[asset_id], r["cutoff_date"], lookback_days=60)
            if rv is None:
                continue
            _, hv_hi = lognormal_interval(r["cutoff_price"], rv, r["days_from_cutoff"])
            hv_lo, _ = lognormal_interval(r["cutoff_price"], rv, r["days_from_cutoff"])
            hv_w = hv_hi - hv_lo
            if r["cutoff_date"] < "2026-02-23":
                pre_hv.append(hv_w)
            else:
                post_hv.append(hv_w)

        if pre_w and post_w and pre_hv and post_hv:
            llm_r = np.mean(post_w) / np.mean(pre_w)
            hv_r = np.mean(post_hv) / np.mean(pre_hv)
            asset_ratios.append({
                "asset_id": asset_id,
                "llm_ratio": llm_r,
                "hv_ratio": hv_r,
            })

    asset_ratios.sort(key=lambda x: x["hv_ratio"])
    names = [a["asset_id"] for a in asset_ratios]
    llm_rs = [a["llm_ratio"] for a in asset_ratios]
    hv_rs = [a["hv_ratio"] for a in asset_ratios]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, hv_rs, width, label="Historical vol", color="#E66100", alpha=0.8)
    ax.bar(x + width/2, llm_rs, width, label="LLMs (27 models)", color="#2166AC", alpha=0.8)
    ax.axhline(1.0, color="#999", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Post/pre width ratio", fontsize=11)
    ax.set_title("Interval widening at Hormuz: historical vol vs LLMs by asset",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    out3 = OUTPUT_DIR / "hormuz_asset_widening.png"
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close()

    # Save structured results
    results = {
        "pre_post_widths": {
            "pre": {"hv": round(mean_width(pre, 'hv_width'), 2),
                    "iv": round(mean_width(pre, 'iv_width'), 2),
                    "llm": round(mean_width(pre, 'llm_width'), 2)},
            "post": {"hv": round(mean_width(post, 'hv_width'), 2),
                     "iv": round(mean_width(post, 'iv_width'), 2),
                     "llm": round(mean_width(post, 'llm_width'), 2)},
        },
        "widening_ratios": {
            "hv": round(mean_width(post, 'hv_width') / mean_width(pre, 'hv_width'), 2),
            "iv": round(mean_width(post, 'iv_width') / mean_width(pre, 'iv_width'), 2),
            "llm": round(mean_width(post, 'llm_width') / mean_width(pre, 'llm_width'), 2),
        },
        "p50_tracking": {
            "pre_ratio": round(pre_p50_ratio, 4),
            "post_ratio": round(post_p50_ratio, 4),
        },
        "spread_tracking": {
            "pre_ratio": round(pre_spread_ratio, 3),
            "post_ratio": round(post_spread_ratio, 3),
        },
    }

    out_json = RESULTS_DIR / "hormuz_analysis.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
