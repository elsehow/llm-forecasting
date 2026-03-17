#!/usr/bin/env python3
"""
Analyze quiet-period LLM-implied vol vs OVX.

Compares LLM prediction interval width against options-implied vol across
calm WTI windows (Sep 2025 – Jan 2026) and the crisis window (Feb 2026).
Produces a summary table and figure.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/real-world-tail-risk/analyze_quiet_period.py
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import timedelta
from scipy.stats import norm, spearmanr

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
ARTIFACTS_DIR = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static")

Z_90 = norm.ppf(0.90)  # 1.2816
TRADING_DAYS_PER_YEAR = 252


def annualized_vol_from_quantiles(p10: float, p90: float, t_years: float) -> float:
    """Back out annualized implied vol from p10/p90 under lognormal."""
    if p10 <= 0 or p90 <= 0 or p10 >= p90:
        return np.nan
    return np.log(p90 / p10) / (2 * Z_90 * np.sqrt(t_years))


def download_ovx(date: str) -> float:
    """Get OVX (CBOE Crude Oil Volatility Index) at a specific date."""
    start = (pd.Timestamp(date) - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(date) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download("^OVX", start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return np.nan
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].iloc[:, 0]
    else:
        prices = df["Close"]
    near = prices[prices.index <= pd.Timestamp(date)]
    if near.empty:
        return np.nan
    return float(near.iloc[-1])


def main():
    # Load quiet-period forecasts
    quiet_path = RESULTS_DIR / "quiet_period_forecasts.json"
    if not quiet_path.exists():
        print(f"ERROR: {quiet_path} not found. Run models first:")
        print("  uv run python experiments/real-world-tail-risk/run_models.py \\")
        print("    --questions-file experiments/real-world-tail-risk/data/quiet_period_questions.json \\")
        print("    --output-file experiments/real-world-tail-risk/results/quiet_period_forecasts.json")
        return

    with open(quiet_path) as f:
        quiet_forecasts = json.load(f)

    # Load crisis forecasts for comparison
    crisis_path = RESULTS_DIR / "forecasts.json"
    with open(crisis_path) as f:
        crisis_forecasts = json.load(f)
    # Filter to WTI only
    crisis_wti = [f for f in crisis_forecasts if f["asset_id"] == "wti_crude"]

    # Load quiet-period questions for window metadata
    with open(DATA_DIR / "quiet_period_questions.json") as f:
        quiet_questions = json.load(f)

    # Get unique cutoff dates from quiet-period data
    cutoff_dates = sorted(set(f["cutoff_date"] for f in quiet_forecasts))
    crisis_cutoff = "2026-02-13"

    # Fetch OVX for each cutoff date
    print("Fetching OVX values...")
    all_cutoffs = cutoff_dates + [crisis_cutoff]
    ovx_values = {}
    for d in all_cutoffs:
        ovx = download_ovx(d)
        ovx_values[d] = ovx
        print(f"  OVX at {d}: {ovx:.1f}%")

    # ─── Compute per-window results ──────────────────────────────────────

    print(f"\n{'='*100}")
    print("  QUIET-PERIOD vs CRISIS: LLM-implied vol / OVX")
    print(f"{'='*100}")

    all_rows = []

    # Process quiet-period windows
    for cutoff in cutoff_dates:
        window_forecasts = [f for f in quiet_forecasts if f["cutoff_date"] == cutoff]
        window_label = window_forecasts[0].get("window_label", cutoff)
        # Get label from questions metadata
        for q in quiet_questions["questions"]:
            if q["cutoff_date"] == cutoff:
                window_label = q.get("window_label", cutoff)
                break

        ovx = ovx_values[cutoff] / 100  # convert to decimal

        for hz in ["H1", "H2", "H3"]:
            hz_forecasts = [f for f in window_forecasts if f["horizon"] == hz]
            if not hz_forecasts:
                continue

            days = hz_forecasts[0]["days_from_cutoff"]
            t_years = days / TRADING_DAYS_PER_YEAR
            cutoff_price = hz_forecasts[0]["cutoff_price"]
            actual = hz_forecasts[0]["ground_truth"]
            actual_return = (actual / cutoff_price - 1) * 100

            model_vols = []
            for fc in hz_forecasts:
                if fc["percentiles"] is None:
                    continue
                p = fc["percentiles"]
                llm_vol = annualized_vol_from_quantiles(p["p10"], p["p90"], t_years)
                model_name = fc["model"].split("/")[-1]
                model_vols.append({
                    "model": model_name,
                    "llm_vol": llm_vol,
                    "ratio": llm_vol / ovx if ovx > 0 else np.nan,
                    "p10": p["p10"],
                    "p90": p["p90"],
                })

            if not model_vols:
                continue

            avg_vol = np.nanmean([m["llm_vol"] for m in model_vols])
            avg_ratio = np.nanmean([m["ratio"] for m in model_vols])

            # Check if actual fell inside model intervals
            inside_count = sum(1 for m in model_vols if m["p10"] <= actual <= m["p90"])

            all_rows.append({
                "window": window_label,
                "type": "quiet",
                "cutoff": cutoff,
                "horizon": hz,
                "days": days,
                "cutoff_price": cutoff_price,
                "actual": actual,
                "actual_return": actual_return,
                "ovx": ovx * 100,
                "avg_llm_vol": avg_vol * 100,
                "avg_ratio": avg_ratio,
                "inside_pct": inside_count / len(model_vols) * 100,
                "models": model_vols,
            })

    # Process crisis window
    ovx_crisis = ovx_values[crisis_cutoff] / 100
    for hz in ["H1", "H2", "H3"]:
        hz_forecasts = [f for f in crisis_wti if f["horizon"] == hz]
        if not hz_forecasts:
            continue

        days = hz_forecasts[0]["days_from_cutoff"]
        t_years = days / TRADING_DAYS_PER_YEAR
        cutoff_price = hz_forecasts[0]["cutoff_price"]
        actual = hz_forecasts[0]["ground_truth"]
        actual_return = (actual / cutoff_price - 1) * 100

        model_vols = []
        for fc in hz_forecasts:
            if fc["percentiles"] is None:
                continue
            p = fc["percentiles"]
            llm_vol = annualized_vol_from_quantiles(p["p10"], p["p90"], t_years)
            model_name = fc["model"].split("/")[-1]
            model_vols.append({
                "model": model_name,
                "llm_vol": llm_vol,
                "ratio": llm_vol / ovx_crisis if ovx_crisis > 0 else np.nan,
                "p10": p["p10"],
                "p90": p["p90"],
            })

        if not model_vols:
            continue

        avg_vol = np.nanmean([m["llm_vol"] for m in model_vols])
        avg_ratio = np.nanmean([m["ratio"] for m in model_vols])
        inside_count = sum(1 for m in model_vols if m["p10"] <= actual <= m["p90"])

        all_rows.append({
            "window": "Hormuz crisis",
            "type": "crisis",
            "cutoff": crisis_cutoff,
            "horizon": hz,
            "days": days,
            "cutoff_price": cutoff_price,
            "actual": actual,
            "actual_return": actual_return,
            "ovx": ovx_crisis * 100,
            "avg_llm_vol": avg_vol * 100,
            "avg_ratio": avg_ratio,
            "inside_pct": inside_count / len(model_vols) * 100,
            "models": model_vols,
        })

    # ─── Print summary table ─────────────────────────────────────────────

    print(f"\n{'Window':<18} {'Hz':<4} {'Days':<5} {'OVX':>6} {'LLM vol':>8} "
          f"{'Ratio':>6} {'Return':>8} {'In p10-p90':>10}")
    print("-" * 75)

    for r in all_rows:
        marker = " ←" if r["type"] == "crisis" else ""
        print(f"{r['window']:<18} {r['horizon']:<4} {r['days']:<5} "
              f"{r['ovx']:>5.1f}% {r['avg_llm_vol']:>7.1f}% "
              f"{r['avg_ratio']:>5.2f}x {r['actual_return']:>+7.1f}% "
              f"{r['inside_pct']:>9.0f}%{marker}")

    # ─── Aggregate: quiet vs crisis ──────────────────────────────────────

    quiet_rows = [r for r in all_rows if r["type"] == "quiet"]
    crisis_rows = [r for r in all_rows if r["type"] == "crisis"]

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    quiet_ratios = [r["avg_ratio"] for r in quiet_rows]
    crisis_ratios = [r["avg_ratio"] for r in crisis_rows]
    quiet_inside = [r["inside_pct"] for r in quiet_rows]

    print(f"\nQuiet periods (N={len(quiet_rows)} window×horizon):")
    print(f"  Mean vol ratio (LLM/OVX): {np.mean(quiet_ratios):.2f}x")
    print(f"  Range: {np.min(quiet_ratios):.2f}x – {np.max(quiet_ratios):.2f}x")
    print(f"  Actuals inside p10-p90: {np.mean(quiet_inside):.0f}%")

    print(f"\nCrisis (Hormuz, N={len(crisis_rows)}):")
    print(f"  Mean vol ratio (LLM/OVX): {np.mean(crisis_ratios):.2f}x")
    print(f"  Range: {np.min(crisis_ratios):.2f}x – {np.max(crisis_ratios):.2f}x")

    # ─── Per-model summary at H3 ────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  PER-MODEL VOL RATIO AT H3 (21 days)")
    print(f"{'='*60}")

    h3_quiet = [r for r in quiet_rows if r["horizon"] == "H3"]
    h3_crisis = [r for r in crisis_rows if r["horizon"] == "H3"]

    if h3_quiet and h3_crisis:
        models = sorted(set(m["model"] for r in h3_quiet for m in r["models"]))
        print(f"\n{'Model':<30} {'Quiet avg':>10} {'Crisis':>10}")
        print("-" * 52)
        for model in models:
            quiet_model_ratios = []
            for r in h3_quiet:
                for m in r["models"]:
                    if m["model"] == model:
                        quiet_model_ratios.append(m["ratio"])
            crisis_model_ratio = None
            for r in h3_crisis:
                for m in r["models"]:
                    if m["model"] == model:
                        crisis_model_ratio = m["ratio"]

            q_avg = np.nanmean(quiet_model_ratios) if quiet_model_ratios else np.nan
            c_val = crisis_model_ratio if crisis_model_ratio else np.nan
            print(f"{model:<30} {q_avg:>9.2f}x {c_val:>9.2f}x")

    # ─── ECI × vol ratio correlation ────────────────────────────────────

    # ECI scores from CivBench (Epoch AI + estimates)
    ECI_SCORES = {
        "claude-opus-4-5-20251101": 150.1,
        "claude-opus-4-6": 152.0,
        "claude-sonnet-4-5-20250929": 146.6,
        "claude-sonnet-4-20250514": 145.0,
        "claude-haiku-4-5-20251001": 140.0,
        "claude-3-haiku-20240307": 118.0,
        "gpt-5.1-2025-11-13": 149.7,
        "gpt-5-2025-08-07": 150.0,
        "gpt-5-mini-2025-08-07": 144.3,
        "gpt-5-nano-2025-08-07": 138.0,
        "gpt-4.1-2025-04-14": 137.4,
        "gpt-4o": 132.0,
        "gpt-3.5-turbo-0125": 110.0,
        "o3-2025-04-16": 146.6,
        "o3-mini-2025-01-31": 140.0,
        "o4-mini-2025-04-16": 142.0,
        "gemini-3-pro-preview": 154.2,
        "gemini-2.5-pro": 147.0,
        "gemini-2.5-flash": 145.0,
        "gemini-2.0-flash-lite-001": 125.0,
        "mistral-large-2411": 130.0,
        "mistral-large-2407": 128.0,
        "mistral-large-latest": 130.0,
        "DeepSeek-V3": 135.0,
        "DeepSeek-V3.1": 137.0,
        "Llama-3.3-70B-Instruct-Turbo": 125.0,
        "Mixtral-8x7B-Instruct-v0.1": 110.0,
        "grok-4-0709": 150.0,
        "grok-4-fast-reasoning": 150.0,
        "grok-4-fast-non-reasoning": 145.0,
        "grok-4-1-fast-reasoning": 151.0,
        "grok-4-1-fast-non-reasoning": 146.0,
    }

    print(f"\n{'='*60}")
    print("  ECI × VOL RATIO CORRELATION")
    print(f"{'='*60}")

    # Compute mean vol ratio per model across all quiet H3 windows
    h3_quiet = [r for r in quiet_rows if r["horizon"] == "H3"]
    h3_crisis = [r for r in crisis_rows if r["horizon"] == "H3"]

    # Collect per-model mean ratios
    all_model_names = sorted(set(
        m["model"] for r in h3_quiet for m in r["models"]
        if m["model"] in ECI_SCORES
    ))

    quiet_eci_list = []
    quiet_ratio_list = []
    crisis_eci_list = []
    crisis_ratio_list = []

    print(f"\n{'Model':<35} {'ECI':>5} {'Quiet H3':>10} {'Crisis H3':>10}")
    print("-" * 65)

    for model in sorted(all_model_names, key=lambda m: ECI_SCORES.get(m, 0)):
        eci = ECI_SCORES.get(model)
        if eci is None:
            continue

        # Quiet: mean ratio across 4 windows at H3
        q_ratios = []
        for r in h3_quiet:
            for m in r["models"]:
                if m["model"] == model and not np.isnan(m["ratio"]):
                    q_ratios.append(m["ratio"])
        q_mean = np.mean(q_ratios) if q_ratios else np.nan

        # Crisis: ratio at H3
        c_ratio = np.nan
        for r in h3_crisis:
            for m in r["models"]:
                if m["model"] == model and not np.isnan(m["ratio"]):
                    c_ratio = m["ratio"]

        if not np.isnan(q_mean):
            quiet_eci_list.append(eci)
            quiet_ratio_list.append(q_mean)
        if not np.isnan(c_ratio):
            crisis_eci_list.append(eci)
            crisis_ratio_list.append(c_ratio)

        q_str = f"{q_mean:.2f}x" if not np.isnan(q_mean) else "—"
        c_str = f"{c_ratio:.2f}x" if not np.isnan(c_ratio) else "—"
        print(f"{model:<35} {eci:>5.1f} {q_str:>10} {c_str:>10}")

    # Spearman correlations
    if len(quiet_eci_list) >= 5:
        rho_q, p_q = spearmanr(quiet_eci_list, quiet_ratio_list)
        print(f"\nQuiet H3: ρ(ECI, vol ratio) = {rho_q:+.3f}, p = {p_q:.4f}, N = {len(quiet_eci_list)}")
        if p_q < 0.05:
            direction = "narrower" if rho_q < 0 else "wider"
            print(f"  → Significant: more capable models produce {direction} intervals")
        else:
            print(f"  → Not significant at p<0.05")

    if len(crisis_eci_list) >= 5:
        rho_c, p_c = spearmanr(crisis_eci_list, crisis_ratio_list)
        print(f"\nCrisis H3: ρ(ECI, vol ratio) = {rho_c:+.3f}, p = {p_c:.4f}, N = {len(crisis_eci_list)}")
        if p_c < 0.05:
            direction = "narrower" if rho_c < 0 else "wider"
            print(f"  → Significant: more capable models produce {direction} intervals")
        else:
            print(f"  → Not significant at p<0.05")

    # Also check across all horizons pooled
    print(f"\n--- Pooled across all horizons ---")
    for label, subset in [("Quiet", quiet_rows), ("Crisis", crisis_rows)]:
        ecis, ratios = [], []
        for r in subset:
            for m in r["models"]:
                eci = ECI_SCORES.get(m["model"])
                if eci and not np.isnan(m["ratio"]):
                    ecis.append(eci)
                    ratios.append(m["ratio"])
        if len(ecis) >= 10:
            rho, p = spearmanr(ecis, ratios)
            print(f"{label}: ρ = {rho:+.3f}, p = {p:.4f}, N = {len(ecis)} (model×window×horizon)")

    # ─── Save results ────────────────────────────────────────────────────

    # Clean up for JSON
    for r in all_rows:
        for m in r["models"]:
            m["llm_vol"] = round(m["llm_vol"] * 100, 1) if not np.isnan(m["llm_vol"]) else None
            m["ratio"] = round(m["ratio"], 3) if not np.isnan(m["ratio"]) else None

    out_path = RESULTS_DIR / "quiet_period_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
