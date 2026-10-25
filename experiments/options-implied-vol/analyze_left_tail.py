#!/usr/bin/env python3
"""
Decompose left-tail miscalibration across asset volatility, market regime,
and forecast horizon. Tests whether the CivBench pattern (p10 miss rate
worsens on disruptable templates at far horizons) transfers to financial data.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/options-implied-vol/analyze_left_tail.py
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

ASSET_CLASSES = {
    "Equity indices": ["sp500", "nasdaq100", "russell", "em"],
    "Commodities": ["oil", "gold"],
    "Single stocks": ["apple", "amazon", "alphabet"],
    "Bonds": ["treasury"],
}

ASSET_TO_CLASS = {}
for cls, ids in ASSET_CLASSES.items():
    for aid in ids:
        ASSET_TO_CLASS[aid] = cls

# Ordered by expected volatility (low to high)
VOLATILITY_TIERS = {
    "Low vol (bonds)": ["treasury"],
    "Med vol (indices)": ["sp500", "nasdaq100", "russell", "em"],
    "High vol (commodities)": ["oil", "gold"],
    "Very high vol (stocks)": ["apple", "amazon", "alphabet"],
}

ECI_MAP = {
    "claude-3-haiku-20240307": 98,
    "Llama-3.3-70B-Instruct-Turbo": 117,
    "mistral-large-2407": 118,
    "mistral-large-2411": 125,
    "DeepSeek-V3": 128,
    "mistral-large-latest": 130,
    "gpt-4o": 132,
    "gemini-2.0-flash-lite-001": 110,
    "DeepSeek-V3.1": 135,
    "gpt-4.1-2025-04-14": 137,
    "claude-haiku-4-5-20251001": 138,
    "claude-sonnet-4-20250514": 140,
    "grok-4-fast-non-reasoning": 140,
    "o3-mini-2025-01-31": 140,
    "gpt-5-mini-2025-08-07": 142,
    "grok-4-1-fast-non-reasoning": 145,
    "o4-mini-2025-04-16": 145,
    "claude-sonnet-4-5-20250929": 148,
    "grok-4-fast-reasoning": 148,
    "o3-2025-04-16": 149,
    "gpt-5-2025-08-07": 150,
    "grok-4-0709": 151,
    "gpt-5.1-2025-11-13": 152,
    "grok-4-1-fast-reasoning": 152,
    "gemini-3-pro-preview": 154,
    "claude-opus-4-5-20251101": 155,
    "claude-opus-4-6": 157,
}


def compute_realized_vol(prices_df, cutoff_date, lookback_days=60):
    cutoff_dt = pd.Timestamp(cutoff_date)
    hist = prices_df[prices_df.index <= cutoff_dt].tail(lookback_days + 1)
    if len(hist) < 20:
        return None
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def p10_miss_rate(records):
    """Fraction of actuals below model's p10 (ideal = 10%)."""
    below = sum(1 for r in records if r["ground_truth"] < r["percentiles"]["p10"])
    return below / len(records) if records else float("nan")


def p5_miss_rate(records):
    below = sum(1 for r in records if r["ground_truth"] < r["percentiles"]["p5"])
    return below / len(records) if records else float("nan")


def pq_miss_rate(records, quantile_key):
    """Generic: fraction of actuals below the stated quantile."""
    below = sum(1 for r in records
                if quantile_key in r["percentiles"]
                and r["ground_truth"] < r["percentiles"][quantile_key])
    total = sum(1 for r in records if quantile_key in r["percentiles"])
    return below / total if total > 0 else float("nan")


def compute_crps(percentiles, ground_truth):
    total = 0
    n = 0
    for key, tau in ALL_QUANTILES.items():
        if key not in percentiles:
            continue
        q = percentiles[key]
        if ground_truth >= q:
            total += tau * (ground_truth - q)
        else:
            total += (1 - tau) * (q - ground_truth)
        n += 1
    return (2 / n) * total if n > 0 else None


def main():
    # ── Load ──
    print("Loading data...")
    with open(DATA_DIR / "questions.json") as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]

    with open(RESULTS_DIR / "forecasts.json") as f:
        forecasts = json.load(f)

    price_dfs = {}
    for asset_id in ASSET_TO_CLASS:
        df = pd.read_csv(DATA_DIR / "prices" / f"{asset_id}.csv", parse_dates=["Date"])
        df = df.set_index("Date")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        price_dfs[asset_id] = df

    # Filter valid
    MIN_VALID = 300
    valid = [r for r in forecasts
             if r.get("percentiles") and r.get("ground_truth") is not None]
    models = sorted(set(r["model"] for r in valid))
    model_counts = {m: sum(1 for r in valid if r["model"] == m) for m in models}
    included = [m for m in models if model_counts[m] >= MIN_VALID]
    valid = [r for r in valid if r["model"] in included]
    print(f"Valid: {len(valid)} forecasts, {len(included)} models")

    # Annotate each record with realized return and realized vol
    q_lookup = {}
    for q in questions:
        key = (q["asset_id"], q["cutoff_date"], q["horizon"])
        q_lookup[key] = q

    rv_cache = {}
    for r in valid:
        # Realized return
        r["realized_return"] = (r["ground_truth"] - r["cutoff_price"]) / r["cutoff_price"]
        r["asset_class"] = ASSET_TO_CLASS.get(r["asset_id"], "unknown")

        # Realized vol (cached by asset+cutoff)
        rv_key = (r["asset_id"], r["cutoff_date"])
        if rv_key not in rv_cache:
            rv_cache[rv_key] = compute_realized_vol(
                price_dfs[r["asset_id"]], r["cutoff_date"], lookback_days=60)
        r["realized_vol"] = rv_cache[rv_key]

        # Implied vol
        q = q_lookup.get((r["asset_id"], r["cutoff_date"], r["horizon"]))
        r["implied_vol"] = q["vol_index_at_cutoff"] / 100 if q and q.get("vol_index_at_cutoff") else None

    # ══════════════════════════════════════════════════════════════════
    # 1. LEFT-TAIL BREAKDOWN BY ASSET CLASS × HORIZON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  1. p10 MISS RATE BY ASSET CLASS × HORIZON")
    print("=" * 80)

    horizons = ["H1", "H2", "H3"]
    classes_ordered = ["Bonds", "Equity indices", "Commodities", "Single stocks"]

    print(f"\n  {'Asset class':<20} {'H1':>8} {'H2':>8} {'H3':>8} {'All':>8} {'N':>6}")
    print(f"  {'-'*56}")

    class_hz_data = {}
    for cls in classes_ordered:
        asset_ids = ASSET_CLASSES[cls]
        row = {}
        for hz in horizons:
            recs = [r for r in valid if r["asset_id"] in asset_ids and r["horizon"] == hz]
            row[hz] = p10_miss_rate(recs)
        all_recs = [r for r in valid if r["asset_id"] in asset_ids]
        row["all"] = p10_miss_rate(all_recs)
        row["n"] = len(all_recs)
        class_hz_data[cls] = row
        print(f"  {cls:<20} {row['H1']:>7.1%} {row['H2']:>7.1%} {row['H3']:>7.1%} "
              f"{row['all']:>7.1%} {row['n']:>5}")

    all_recs_all = valid
    for hz in horizons:
        recs = [r for r in valid if r["horizon"] == hz]
        rate = p10_miss_rate(recs)
        if hz == "H1":
            print(f"  {'All':<20} {rate:>7.1%}", end="")
        else:
            print(f" {rate:>7.1%}", end="")
    print(f" {p10_miss_rate(valid):>7.1%} {len(valid):>5}")
    print(f"\n  Nominal: 10.0%")

    # ══════════════════════════════════════════════════════════════════
    # 2. FULL QUANTILE CALIBRATION BY ASSET CLASS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  2. PER-QUANTILE CALIBRATION BY ASSET CLASS")
    print("=" * 80)

    for cls in classes_ordered:
        asset_ids = ASSET_CLASSES[cls]
        recs = [r for r in valid if r["asset_id"] in asset_ids]
        print(f"\n  {cls} (N={len(recs)}):")
        print(f"    {'Quantile':<8} {'Nominal':>8} {'Observed':>10} {'Gap':>8}")
        print(f"    {'-'*36}")
        for qkey, tau in ALL_QUANTILES.items():
            obs = pq_miss_rate(recs, qkey)
            print(f"    {qkey:<8} {tau:>7.0%} {obs:>9.1%} {(obs-tau)*100:>+6.1f}pp")

    # ══════════════════════════════════════════════════════════════════
    # 3. p10 MISS RATE BY INDIVIDUAL ASSET (sorted by realized vol)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  3. p10 MISS RATE BY ASSET (sorted by realized vol)")
    print("=" * 80)

    asset_stats = []
    for asset_id in ASSET_TO_CLASS:
        recs = [r for r in valid if r["asset_id"] == asset_id]
        if not recs:
            continue
        rvols = [r["realized_vol"] for r in recs if r["realized_vol"] is not None]
        mean_rv = np.mean(rvols) if rvols else 0
        hz_rates = {}
        for hz in horizons:
            hz_recs = [r for r in recs if r["horizon"] == hz]
            hz_rates[hz] = p10_miss_rate(hz_recs)
        asset_stats.append({
            "asset_id": asset_id,
            "class": ASSET_TO_CLASS[asset_id],
            "mean_rv": mean_rv,
            "p10_all": p10_miss_rate(recs),
            "p10_h1": hz_rates["H1"],
            "p10_h2": hz_rates["H2"],
            "p10_h3": hz_rates["H3"],
            "n": len(recs),
        })

    asset_stats.sort(key=lambda x: x["mean_rv"])

    print(f"\n  {'Asset':<12} {'Class':<18} {'60d Vol':>8} {'p10 H1':>8} {'p10 H2':>8} "
          f"{'p10 H3':>8} {'p10 All':>8} {'N':>5}")
    print(f"  {'-'*80}")
    for a in asset_stats:
        print(f"  {a['asset_id']:<12} {a['class']:<18} {a['mean_rv']*100:>7.1f}% "
              f"{a['p10_h1']:>7.1%} {a['p10_h2']:>7.1%} {a['p10_h3']:>7.1%} "
              f"{a['p10_all']:>7.1%} {a['n']:>4}")

    # Correlation: realized vol × p10 miss rate
    vols = [a["mean_rv"] for a in asset_stats]
    p10s = [a["p10_all"] for a in asset_stats]
    rho, p = spearmanr(vols, p10s)
    print(f"\n  Realized vol × p10 miss: ρ={rho:+.3f}, p={p:.3f}, N={len(asset_stats)}")

    # H3-specific
    p10s_h3 = [a["p10_h3"] for a in asset_stats]
    rho3, p3 = spearmanr(vols, p10s_h3)
    print(f"  Realized vol × p10 miss (H3 only): ρ={rho3:+.3f}, p={p3:.3f}")

    # Horizon degradation (H3 - H1)
    deltas = [a["p10_h3"] - a["p10_h1"] for a in asset_stats]
    rho_d, p_d = spearmanr(vols, deltas)
    print(f"  Realized vol × (H3-H1 p10 degradation): ρ={rho_d:+.3f}, p={p_d:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # 4. MARKET REGIME: DRAWDOWN vs CALM
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  4. LEFT-TAIL CALIBRATION BY MARKET REGIME")
    print("=" * 80)

    # Define drawdown: realized return < -3%
    DRAWDOWN_THRESHOLD = -0.03
    drawdown_recs = [r for r in valid if r["realized_return"] < DRAWDOWN_THRESHOLD]
    calm_recs = [r for r in valid if r["realized_return"] >= DRAWDOWN_THRESHOLD]

    print(f"\n  Drawdown (return < {DRAWDOWN_THRESHOLD:.0%}): {len(drawdown_recs)} forecasts "
          f"({len(drawdown_recs)/len(valid):.1%} of total)")
    print(f"  Calm (return ≥ {DRAWDOWN_THRESHOLD:.0%}): {len(calm_recs)} forecasts")

    # p10 miss rate comparison
    print(f"\n  {'Regime':<25} {'p10 miss':>10} {'p5 miss':>10} {'p1 miss':>10} {'N':>6}")
    print(f"  {'-'*65}")
    for label, recs in [("Drawdown (<-3%)", drawdown_recs),
                         ("Calm (≥-3%)", calm_recs),
                         ("All", valid)]:
        p10 = pq_miss_rate(recs, "p10")
        p5 = pq_miss_rate(recs, "p5")
        p1 = pq_miss_rate(recs, "p1")
        print(f"  {label:<25} {p10:>9.1%} {p5:>9.1%} {p1:>9.1%} {len(recs):>5}")

    print(f"\n  Nominal:                    10.0%      5.0%      1.0%")

    # By regime AND asset class
    print(f"\n  p10 miss by regime × asset class:")
    print(f"  {'Asset class':<20} {'Drawdown':>10} {'Calm':>10} {'Δ':>8}")
    print(f"  {'-'*50}")
    for cls in classes_ordered:
        asset_ids = ASSET_CLASSES[cls]
        dd = [r for r in drawdown_recs if r["asset_id"] in asset_ids]
        ca = [r for r in calm_recs if r["asset_id"] in asset_ids]
        dd_rate = p10_miss_rate(dd) if dd else float("nan")
        ca_rate = p10_miss_rate(ca) if ca else float("nan")
        delta = dd_rate - ca_rate if not (math.isnan(dd_rate) or math.isnan(ca_rate)) else float("nan")
        n_dd = len(dd)
        print(f"  {cls:<20} {dd_rate:>9.1%} {ca_rate:>9.1%} {delta*100:>+6.1f}pp  (N_dd={n_dd})")

    # By regime AND horizon
    print(f"\n  p10 miss by regime × horizon:")
    print(f"  {'Horizon':<10} {'Drawdown':>10} {'Calm':>10} {'Δ':>8}")
    print(f"  {'-'*40}")
    for hz in horizons:
        dd = [r for r in drawdown_recs if r["horizon"] == hz]
        ca = [r for r in calm_recs if r["horizon"] == hz]
        dd_rate = p10_miss_rate(dd) if dd else float("nan")
        ca_rate = p10_miss_rate(ca) if ca else float("nan")
        delta = dd_rate - ca_rate
        print(f"  {hz:<10} {dd_rate:>9.1%} {ca_rate:>9.1%} {delta*100:>+6.1f}pp")

    # More granular drawdown thresholds
    print(f"\n  p10 miss by drawdown severity:")
    thresholds = [
        ("Return ≥ 0%", lambda r: r["realized_return"] >= 0),
        ("-3% to 0%", lambda r: -0.03 <= r["realized_return"] < 0),
        ("-5% to -3%", lambda r: -0.05 <= r["realized_return"] < -0.03),
        ("-10% to -5%", lambda r: -0.10 <= r["realized_return"] < -0.05),
        ("< -10%", lambda r: r["realized_return"] < -0.10),
    ]
    print(f"  {'Regime':<20} {'p10 miss':>10} {'p5 miss':>10} {'N':>6} {'Mean ret':>10}")
    print(f"  {'-'*60}")
    for label, pred in thresholds:
        recs = [r for r in valid if pred(r)]
        if recs:
            p10 = pq_miss_rate(recs, "p10")
            p5 = pq_miss_rate(recs, "p5")
            mean_ret = np.mean([r["realized_return"] for r in recs])
            print(f"  {label:<20} {p10:>9.1%} {p5:>9.1%} {len(recs):>5} {mean_ret:>+9.1%}")

    # ══════════════════════════════════════════════════════════════════
    # 5. HORIZON DEGRADATION (CivBench analog)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  5. HORIZON DEGRADATION (CivBench analog)")
    print("=" * 80)

    print(f"\n  p10 miss by horizon (pooled):")
    for hz in horizons:
        recs = [r for r in valid if r["horizon"] == hz]
        rate = p10_miss_rate(recs)
        print(f"    {hz}: {rate:.1%} (N={len(recs)})")

    # Per-model p10 miss rate × ECI at each horizon
    print(f"\n  ECI × p10 miss correlation by horizon:")
    for hz in horizons:
        model_p10s = []
        model_ecis = []
        for model in included:
            mname = model.split("/")[-1]
            if mname not in ECI_MAP:
                continue
            recs = [r for r in valid if r["model"] == model and r["horizon"] == hz]
            if len(recs) < 50:
                continue
            model_p10s.append(p10_miss_rate(recs))
            model_ecis.append(ECI_MAP[mname])
        if len(model_p10s) >= 5:
            rho, p = spearmanr(model_ecis, model_p10s)
            print(f"    {hz}: ρ(ECI, p10_miss) = {rho:+.3f}, p={p:.3f}, N={len(model_p10s)}")

    # Per-model CRPS × ECI at each horizon
    print(f"\n  ECI × CRPS correlation by horizon:")
    for hz in horizons:
        model_crps_list = []
        model_ecis = []
        for model in included:
            mname = model.split("/")[-1]
            if mname not in ECI_MAP:
                continue
            recs = [r for r in valid if r["model"] == model and r["horizon"] == hz]
            if len(recs) < 50:
                continue
            crps_vals = [compute_crps(r["percentiles"], r["ground_truth"]) for r in recs]
            crps_vals = [c for c in crps_vals if c is not None]
            if crps_vals:
                model_crps_list.append(np.mean(crps_vals))
                model_ecis.append(ECI_MAP[mname])
        if len(model_crps_list) >= 5:
            rho, p = spearmanr(model_ecis, model_crps_list)
            print(f"    {hz}: ρ(ECI, CRPS) = {rho:+.3f}, p={p:.3f}, N={len(model_crps_list)}")

    # ECI × p10 miss by asset class
    print(f"\n  ECI × p10 miss by asset class (all horizons):")
    for cls in classes_ordered:
        asset_ids = ASSET_CLASSES[cls]
        model_p10s = []
        model_ecis = []
        for model in included:
            mname = model.split("/")[-1]
            if mname not in ECI_MAP:
                continue
            recs = [r for r in valid if r["model"] == model and r["asset_id"] in asset_ids]
            if len(recs) < 30:
                continue
            model_p10s.append(p10_miss_rate(recs))
            model_ecis.append(ECI_MAP[mname])
        if len(model_p10s) >= 5:
            rho, p = spearmanr(model_ecis, model_p10s)
            print(f"    {cls:<20}: ρ={rho:+.3f}, p={p:.3f}, N={len(model_p10s)}")

    # ══════════════════════════════════════════════════════════════════
    # 6. THE HORMUZ NATURAL EXPERIMENT
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  6. NATURAL EXPERIMENT: EXTREME DRAWDOWN EPISODES")
    print("=" * 80)

    # Find the worst drawdown episodes per asset
    for asset_id in sorted(ASSET_TO_CLASS.keys()):
        recs = [r for r in valid if r["asset_id"] == asset_id]
        if not recs:
            continue
        worst = sorted(recs, key=lambda r: r["realized_return"])[:3]
        best = sorted(recs, key=lambda r: r["realized_return"], reverse=True)[:3]

        # Get all recs where return < -5%
        extreme_dd = [r for r in recs if r["realized_return"] < -0.05]
        if extreme_dd:
            p10_dd = p10_miss_rate(extreme_dd)
            p5_dd = pq_miss_rate(extreme_dd, "p5")
            rest = [r for r in recs if r["realized_return"] >= -0.05]
            p10_rest = p10_miss_rate(rest)
            dates = sorted(set(r["cutoff_date"] for r in extreme_dd))
            print(f"\n  {asset_id} ({ASSET_TO_CLASS[asset_id]}):")
            print(f"    Extreme drawdowns (return < -5%): N={len(extreme_dd)}, "
                  f"cutoffs: {dates[0]} to {dates[-1]}")
            print(f"    p10 miss during drawdown: {p10_dd:.1%} (vs {p10_rest:.1%} at other times)")
            print(f"    p5 miss during drawdown:  {p5_dd:.1%}")
            print(f"    Worst: return={worst[0]['realized_return']:.1%} "
                  f"(cutoff {worst[0]['cutoff_date']}, {worst[0]['horizon']})")

    # ══════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════
    print("\n\nGenerating figures...")

    # ── Figure 1: p10 miss × asset volatility × horizon (3-panel or heatmap) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    colors = {
        "Bonds": "#1B7837",
        "Equity indices": "#2166AC",
        "Commodities": "#B2182B",
        "Single stocks": "#762A83",
    }

    for i, hz in enumerate(horizons):
        ax = axes[i]
        for cls in classes_ordered:
            asset_ids = ASSET_CLASSES[cls]
            for asset_id in sorted(asset_ids):
                recs = [r for r in valid if r["asset_id"] == asset_id and r["horizon"] == hz]
                if not recs:
                    continue
                rv = np.mean([r["realized_vol"] for r in recs if r["realized_vol"] is not None])
                miss = p10_miss_rate(recs)
                ax.scatter(rv * 100, miss * 100, c=colors[cls], s=60, alpha=0.8,
                           edgecolors="white", linewidths=0.5, zorder=3)

        # Fit line
        all_x, all_y = [], []
        for a in asset_stats:
            rv = a["mean_rv"]
            miss = a[f"p10_h{i+1}"]
            all_x.append(rv * 100)
            all_y.append(miss * 100)
        if len(all_x) >= 3:
            z = np.polyfit(all_x, all_y, 1)
            x_line = np.linspace(5, 40, 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color="#666", alpha=0.5, linewidth=1.5)
            rho, p = spearmanr(all_x, all_y)
            ax.set_title(f"{hz} (+{(i+1)*7}d)\nρ={rho:+.2f}, p={p:.2f}",
                         fontsize=11, fontweight="bold")

        ax.axhline(10, color="#999", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Realized vol (%, annualized)", fontsize=10)
        if i == 0:
            ax.set_ylabel("p10 miss rate (%)", fontsize=10)
        ax.set_xlim(5, 40)
        ax.set_ylim(0, 25)
        ax.grid(alpha=0.1)

    # Legend
    for cls in classes_ordered:
        axes[0].scatter([], [], c=colors[cls], s=60, label=cls, edgecolors="white")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)

    fig.suptitle("Left-tail miscalibration by asset volatility and horizon",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out1 = OUTPUT_DIR / "financial_p10_by_vol_horizon.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # ── Figure 2: Asset class × horizon heatmap (CivBench analog) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                     gridspec_kw={"width_ratios": [1, 1]})

    # Left: p10 miss heatmap
    data_matrix = []
    ylabels = []
    for cls in classes_ordered:
        row = []
        for hz in horizons:
            asset_ids = ASSET_CLASSES[cls]
            recs = [r for r in valid if r["asset_id"] in asset_ids and r["horizon"] == hz]
            row.append(p10_miss_rate(recs) * 100)
        data_matrix.append(row)
        ylabels.append(cls)

    data_arr = np.array(data_matrix)
    im = ax1.imshow(data_arr, cmap="YlOrRd", aspect="auto", vmin=5, vmax=20)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["H1 (+7d)", "H2 (+14d)", "H3 (+21d)"])
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(ylabels)

    for i in range(4):
        for j in range(3):
            ax1.text(j, i, f"{data_arr[i,j]:.1f}%", ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     color="white" if data_arr[i,j] > 14 else "black")

    ax1.set_title("p10 miss rate (nom. 10%)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax1, label="Miss rate (%)", shrink=0.8)

    # Right: p10 miss during drawdown vs calm
    cats = []
    dd_vals = []
    calm_vals = []
    for cls in classes_ordered:
        asset_ids = ASSET_CLASSES[cls]
        dd = [r for r in drawdown_recs if r["asset_id"] in asset_ids]
        ca = [r for r in calm_recs if r["asset_id"] in asset_ids]
        dd_rate = p10_miss_rate(dd) * 100 if dd else 0
        ca_rate = p10_miss_rate(ca) * 100 if ca else 0
        cats.append(cls)
        dd_vals.append(dd_rate)
        calm_vals.append(ca_rate)

    x = np.arange(len(cats))
    width = 0.35
    ax2.bar(x - width/2, calm_vals, width, label="Calm (ret ≥ -3%)",
            color="#2166AC", alpha=0.8)
    ax2.bar(x + width/2, dd_vals, width, label="Drawdown (ret < -3%)",
            color="#B2182B", alpha=0.8)
    ax2.axhline(10, color="#999", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("p10 miss rate (%)", fontsize=10)
    ax2.set_title("p10 miss: drawdown vs calm", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.set_ylim(0, 35)
    ax2.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    out2 = OUTPUT_DIR / "financial_p10_heatmap_regime.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    # ── Figure 3: ECI × p10 miss, faceted by horizon ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, hz in enumerate(horizons):
        ax = axes[i]
        model_p10s = []
        model_ecis = []
        model_names = []
        for model in included:
            mname = model.split("/")[-1]
            if mname not in ECI_MAP:
                continue
            recs = [r for r in valid if r["model"] == model and r["horizon"] == hz]
            if len(recs) < 50:
                continue
            model_p10s.append(p10_miss_rate(recs) * 100)
            model_ecis.append(ECI_MAP[mname])
            model_names.append(mname)

        ax.scatter(model_ecis, model_p10s, c="#2166AC", s=40, alpha=0.7, zorder=3)
        ax.axhline(10, color="#999", linestyle=":", linewidth=1, alpha=0.6)

        if len(model_ecis) >= 5:
            z = np.polyfit(model_ecis, model_p10s, 1)
            x_line = np.linspace(min(model_ecis) - 5, max(model_ecis) + 5, 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color="#2166AC", alpha=0.4)
            rho, p = spearmanr(model_ecis, model_p10s)
            ax.set_title(f"{hz} (+{(i+1)*7}d)\nρ={rho:+.2f}, p={p:.3f}",
                         fontsize=11, fontweight="bold")

        ax.set_xlabel("ECI", fontsize=10)
        if i == 0:
            ax.set_ylabel("p10 miss rate (%)", fontsize=10)
        ax.grid(alpha=0.1)

    fig.suptitle("Capability × left-tail miscalibration by horizon",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out3 = OUTPUT_DIR / "financial_eci_p10_by_horizon.png"
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close()

    # ── Figure 4: CivBench-style comparison figure ──
    # p10 miss by horizon for volatile vs stable assets
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    volatile_ids = ["oil", "gold", "apple", "amazon", "alphabet"]
    stable_ids = ["treasury", "sp500", "nasdaq100", "russell", "em"]

    for label, ids, color, marker in [
        ("High vol (commodities + stocks)", volatile_ids, "#B2182B", "o"),
        ("Low vol (bonds + indices)", stable_ids, "#2166AC", "s"),
    ]:
        rates = []
        for hz in horizons:
            recs = [r for r in valid if r["asset_id"] in ids and r["horizon"] == hz]
            rates.append(p10_miss_rate(recs) * 100)
        ax.plot(horizons, rates, marker=marker, markersize=8, linewidth=2.5,
                color=color, label=label)

    # All pooled
    rates_all = []
    for hz in horizons:
        recs = [r for r in valid if r["horizon"] == hz]
        rates_all.append(p10_miss_rate(recs) * 100)
    ax.plot(horizons, rates_all, marker="D", markersize=6, linewidth=1.5,
            color="#666", linestyle="--", label="All assets")

    ax.axhline(10, color="#999", linestyle=":", linewidth=1, alpha=0.6, label="Nominal 10%")
    ax.set_xlabel("Forecast horizon", fontsize=11)
    ax.set_ylabel("p10 miss rate (%)", fontsize=11)
    ax.set_title("Left-tail miscalibration: volatile vs stable assets\n(cf. CivBench disruptable vs predictable templates)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 25)
    ax.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    out4 = OUTPUT_DIR / "financial_volatile_vs_stable_p10.png"
    fig.savefig(out4, dpi=200, bbox_inches="tight")
    print(f"Saved: {out4}")
    plt.close()

    # ── Save structured results ──
    results_json = {
        "p10_miss_by_class_horizon": {
            cls: {hz: class_hz_data[cls][hz] for hz in horizons}
            for cls in classes_ordered
        },
        "p10_miss_by_asset": {
            a["asset_id"]: {
                "realized_vol": round(a["mean_rv"] * 100, 1),
                "p10_h1": round(a["p10_h1"], 4),
                "p10_h2": round(a["p10_h2"], 4),
                "p10_h3": round(a["p10_h3"], 4),
                "p10_all": round(a["p10_all"], 4),
            } for a in asset_stats
        },
        "regime_comparison": {
            "drawdown_p10": round(p10_miss_rate(drawdown_recs), 4),
            "calm_p10": round(p10_miss_rate(calm_recs), 4),
            "n_drawdown": len(drawdown_recs),
            "n_calm": len(calm_recs),
        },
        "vol_p10_correlation": {"rho": round(rho, 3), "p": round(p, 4)},
    }

    out_json = RESULTS_DIR / "left_tail_decomposition.json"
    with open(out_json, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
