#!/usr/bin/env python3
"""
Analyze real-world tail-risk forecasting results and compare to CivBench.

Computes CRPS, p10 miss rates, and generates comparison charts.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/real-world-tail-risk/analyze.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CHART_DIR = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static")

# CivBench reference data: pooled p10 miss rates from disruptable templates
# Source: tailriskbench.org benchmark results
CIVBENCH_P10_MISS = {
    "H1": 0.14, "H2": 0.18, "H3": 0.22, "H4": 0.28,
    "H5": 0.33, "H6": 0.38, "H7": 0.42,
}

MODEL_SHORT_NAMES = {
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "anthropic/claude-opus-4-6": "Opus 4.6",
}

MODEL_COLORS = {
    "GPT-4.1": "#1f77b4",
    "Sonnet 4.5": "#ff7f0e",
    "GPT-5": "#2ca02c",
    "Gemini 3 Pro": "#d62728",
    "Opus 4.6": "#9467bd",
}

ASSET_DISPLAY = {
    "wti_crude": "Oil (WTI)",
    "bitcoin": "Bitcoin",
    "silver": "Silver",
    "natural_gas": "Natural Gas",
}


ALL_QUANTILES = {"p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
                  "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}


def compute_crps(percentiles: dict, true_value: float) -> float:
    """CRPS via quantile-weighted pinball loss. Uses all available quantiles."""
    available = {k: v for k, v in ALL_QUANTILES.items() if k in percentiles}
    if not available:
        return float("nan")
    total_loss = 0.0
    for key, tau in available.items():
        q = percentiles[key]
        residual = true_value - q
        if residual >= 0:
            total_loss += tau * residual
        else:
            total_loss += (tau - 1) * residual
    return (2 / len(available)) * total_loss


def load_results() -> list[dict]:
    """Load forecast results."""
    path = RESULTS_DIR / "forecasts.json"
    with open(path) as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> dict:
    """Compute per-model, per-asset, per-horizon metrics."""
    metrics = defaultdict(lambda: {
        "crps_values": [], "rel_crps_values": [],
        "p1_misses": 0, "p5_misses": 0, "p10_misses": 0,
        "p1_total": 0, "p5_total": 0, "p10_total": 0,
        "p50_errors": [], "forecasts": [],
    })

    for r in results:
        p = r["percentiles"]
        truth = r["ground_truth"]
        if not p or truth is None:
            continue
        if "p50" not in p:
            continue

        model = MODEL_SHORT_NAMES.get(r["model"], r["model"])
        asset = r["asset_id"]
        horizon = r["horizon"]

        crps = compute_crps(p, truth)
        rel_crps = crps / abs(truth) if truth != 0 else crps

        key = (model, asset, horizon)
        m = metrics[key]
        m["crps_values"].append(crps)
        m["rel_crps_values"].append(rel_crps)
        m["p50_errors"].append(abs(p["p50"] - truth) / abs(truth) if truth != 0 else 0)

        for pkey in ["p1", "p5", "p10"]:
            if pkey in p:
                m[f"{pkey}_total"] += 1
                if truth < p[pkey]:
                    m[f"{pkey}_misses"] += 1

        m["forecasts"].append({
            "truth": truth,
            "p1": p.get("p1"), "p5": p.get("p5"), "p10": p.get("p10"),
            "p50": p["p50"], "p90": p.get("p90"),
            "p95": p.get("p95"), "p99": p.get("p99"),
            "cutoff_price": r.get("cutoff_price"),
        })

    return metrics


def print_summary_table(results: list[dict], metrics: dict):
    """Print org-mode formatted summary tables."""
    models = sorted(set(MODEL_SHORT_NAMES.get(r["model"], r["model"]) for r in results))
    assets = sorted(set(r["asset_id"] for r in results))

    print("\n*** Relative CRPS by model × asset (lower is better)")
    print()
    header = "| Model |" + "|".join(f" {ASSET_DISPLAY.get(a, a)} " for a in assets) + "| Pooled |"
    sep = "|" + "+".join(["-" * (len(c) + 2) for c in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    for model in models:
        row = f"| {model:<15} |"
        all_vals = []
        for asset in assets:
            vals = []
            for h_key in sorted(set(k[2] for k in metrics if k[0] == model and k[1] == asset)):
                vals.extend(metrics[(model, asset, h_key)]["rel_crps_values"])
            if vals:
                mean_val = np.mean(vals)
                all_vals.extend(vals)
                row += f" {mean_val:.3f} |"
            else:
                row += " — |"
        pooled = np.mean(all_vals) if all_vals else float("nan")
        row += f" {pooled:.3f} |"
        print(row)

    for pkey, ideal in [("p1", 1), ("p5", 5), ("p10", 10)]:
        print(f"\n*** {pkey.upper()} miss rate by model × asset (ideal = {ideal}%)")
        print()
        print(header)
        print(sep)

        for model in models:
            row = f"| {model:<15} |"
            total_misses = 0
            total_n = 0
            for asset in assets:
                misses = 0
                n = 0
                for h_key in sorted(set(k[2] for k in metrics if k[0] == model and k[1] == asset)):
                    m = metrics[(model, asset, h_key)]
                    misses += m[f"{pkey}_misses"]
                    n += m[f"{pkey}_total"]
                total_misses += misses
                total_n += n
                rate = misses / n * 100 if n > 0 else float("nan")
                row += f" {rate:.0f}% |"
            pooled_rate = total_misses / total_n * 100 if total_n > 0 else float("nan")
            row += f" {pooled_rate:.0f}% |"
            print(row)

    # Per-horizon detail for bitcoin (most horizons)
    print("\n*** P10 miss rate by model × horizon (Bitcoin)")
    print()
    btc_horizons = sorted(set(k[2] for k in metrics if k[1] == "bitcoin"))
    header2 = "| Model |" + "|".join(f" {h} " for h in btc_horizons) + "|"
    sep2 = "|" + "+".join(["-" * (len(c) + 2) for c in header2.split("|")[1:-1]]) + "|"
    print(header2)
    print(sep2)
    for model in models:
        row = f"| {model:<15} |"
        for h in btc_horizons:
            m = metrics.get((model, "bitcoin", h))
            if m and m["p10_total"] > 0:
                rate = m["p10_misses"] / m["p10_total"] * 100
                row += f" {rate:.0f}% |"
            else:
                row += " — |"
        print(row)


def chart_tail_miss_by_asset(results: list[dict], metrics: dict):
    """Panel chart: p1/p5/p10 miss rates per model, grouped by asset."""
    models = sorted(set(MODEL_SHORT_NAMES.get(r["model"], r["model"]) for r in results))
    assets = sorted(set(r["asset_id"] for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    quantile_keys = [("p1", 1), ("p5", 5), ("p10", 10)]

    for ax_idx, (pkey, ideal) in enumerate(quantile_keys):
        ax = axes[ax_idx]
        x = np.arange(len(assets))
        width = 0.15
        offsets = np.arange(len(models)) - (len(models) - 1) / 2

        for i, model in enumerate(models):
            rates = []
            for asset in assets:
                misses = sum(metrics[(model, asset, h)][f"{pkey}_misses"]
                             for h in set(k[2] for k in metrics if k[0] == model and k[1] == asset))
                total = sum(metrics[(model, asset, h)][f"{pkey}_total"]
                            for h in set(k[2] for k in metrics if k[0] == model and k[1] == asset))
                rates.append(misses / total * 100 if total > 0 else 0)
            ax.bar(x + offsets[i] * width, rates, width, label=model,
                   color=MODEL_COLORS.get(model, f"C{i}"))

        ax.axhline(ideal, color="black", linestyle="--", alpha=0.5, label=f"Ideal ({ideal}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([ASSET_DISPLAY.get(a, a) for a in assets], fontsize=8)
        ax.set_title(f"{pkey.upper()} miss rate (ideal = {ideal}%)")
        if ax_idx == 0:
            ax.set_ylabel("Miss rate (%)")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(0, 105)

    fig.suptitle("Tail risk blindness: real-world assets\n"
                 "(% of actuals below model's percentile estimate)", fontsize=13)
    fig.tight_layout()
    path = CHART_DIR / "realworld_tail_miss_by_asset.png"
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    plt.close()


def chart_crps_by_asset(results: list[dict], metrics: dict):
    """Bar chart: relative CRPS per model, grouped by asset."""
    models = sorted(set(MODEL_SHORT_NAMES.get(r["model"], r["model"]) for r in results))
    assets = sorted(set(r["asset_id"] for r in results))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(assets))
    width = 0.15
    offsets = np.arange(len(models)) - (len(models) - 1) / 2

    for i, model in enumerate(models):
        vals = []
        for asset in assets:
            asset_vals = []
            for h in set(k[2] for k in metrics if k[0] == model and k[1] == asset):
                asset_vals.extend(metrics[(model, asset, h)]["rel_crps_values"])
            vals.append(np.mean(asset_vals) if asset_vals else 0)
        ax.bar(x + offsets[i] * width, vals, width, label=model,
               color=MODEL_COLORS.get(model, f"C{i}"))

    ax.set_xticks(x)
    ax.set_xticklabels([ASSET_DISPLAY.get(a, a) for a in assets])
    ax.set_ylabel("Relative CRPS (lower is better)")
    ax.set_title("Forecast accuracy: real-world assets\n(CRPS normalized by true value)")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    path = CHART_DIR / "realworld_crps_by_asset.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def chart_forecast_vs_actual(results: list[dict]):
    """Panel chart: model forecasts vs actual trajectories for each asset."""
    assets = sorted(set(r["asset_id"] for r in results))
    models = sorted(set(MODEL_SHORT_NAMES.get(r["model"], r["model"]) for r in results))

    fig, axes = plt.subplots(1, len(assets), figsize=(5 * len(assets), 5), squeeze=False)

    for col, asset in enumerate(assets):
        ax = axes[0, col]
        asset_results = [r for r in results if r["asset_id"] == asset and r["percentiles"]]
        if not asset_results:
            continue

        # Plot actual trajectory
        horizons = sorted(set(r["horizon"] for r in asset_results))
        cutoff_price = asset_results[0].get("cutoff_price", 0)

        # Ground truth line
        truth_by_h = {}
        for r in asset_results:
            truth_by_h[r["horizon"]] = r["ground_truth"]
        truth_vals = [truth_by_h.get(h, None) for h in horizons]
        x = range(len(horizons))

        # Plot cutoff reference
        ax.axhline(cutoff_price, color="gray", linestyle=":", alpha=0.5, label="Cutoff price")

        # Plot actual
        ax.plot(x, truth_vals, "ko-", linewidth=2, markersize=8, label="Actual", zorder=10)

        # Plot each model's p50 with p1-p99 and p10-p90 bands
        for model in models:
            model_results = [r for r in asset_results
                             if MODEL_SHORT_NAMES.get(r["model"], r["model"]) == model]
            if not model_results:
                continue

            by_h = {r["horizon"]: r for r in model_results}

            def get_pct(h, key):
                return by_h[h]["percentiles"].get(key) if h in by_h else None

            p50s = [get_pct(h, "p50") for h in horizons]
            p10s = [get_pct(h, "p10") for h in horizons]
            p90s = [get_pct(h, "p90") for h in horizons]
            p1s = [get_pct(h, "p1") for h in horizons]
            p99s = [get_pct(h, "p99") for h in horizons]

            color = MODEL_COLORS.get(model, "gray")
            valid_x = [i for i, v in enumerate(p50s) if v is not None]
            valid_p50 = [p50s[i] for i in valid_x]
            valid_p10 = [p10s[i] for i in valid_x]
            valid_p90 = [p90s[i] for i in valid_x]

            ax.plot(valid_x, valid_p50, "o--", color=color, alpha=0.7, markersize=4, label=model)
            ax.fill_between(valid_x, valid_p10, valid_p90, color=color, alpha=0.12)

            # Outer band: p1-p99
            valid_p1 = [p1s[i] for i in valid_x]
            valid_p99 = [p99s[i] for i in valid_x]
            if all(v is not None for v in valid_p1 + valid_p99):
                ax.fill_between(valid_x, valid_p1, valid_p99, color=color, alpha=0.05)

        ax.set_xticks(list(x))
        ax.set_xticklabels(horizons, rotation=45)
        ax.set_title(ASSET_DISPLAY.get(asset, asset))
        ax.set_ylabel("Price" if col == 0 else "")
        if col == 0:
            ax.legend(fontsize=6, loc="best")

    fig.suptitle("Model forecasts (p10–p90 bands) vs actual prices", fontsize=13)
    fig.tight_layout()
    path = CHART_DIR / "realworld_forecast_vs_actual.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def chart_civbench_comparison(results: list[dict], metrics: dict):
    """Compare pooled p10 miss rates: real-world vs CivBench."""
    models = sorted(set(MODEL_SHORT_NAMES.get(r["model"], r["model"]) for r in results))

    # Compute pooled real-world p10 miss rate per model
    rw_rates = {}
    for model in models:
        misses = 0
        total = 0
        for key, m in metrics.items():
            if key[0] == model:
                misses += m["p10_misses"]
                total += m["p10_total"]
        rw_rates[model] = misses / total * 100 if total > 0 else 0

    # CivBench pooled (average across horizons)
    civbench_pooled = np.mean(list(CIVBENCH_P10_MISS.values())) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    rw_vals = [rw_rates[m] for m in models]
    civbench_vals = [civbench_pooled] * len(models)

    ax.bar(x - width / 2, rw_vals, width, label="Real-world assets", color="#2ca02c", alpha=0.8)
    ax.bar(x + width / 2, civbench_vals, width, label="CivBench (disruptable)", color="#1f77b4", alpha=0.8)
    ax.axhline(10, color="black", linestyle="--", alpha=0.5, label="Ideal (10%)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("P10 miss rate (%)")
    ax.set_title("Tail risk blindness transfers to real-world data\n"
                 "(pooled p10 miss rate; ideal = 10%)")
    ax.legend()
    ax.set_ylim(0, max(max(rw_vals), civbench_pooled) * 1.3)

    fig.tight_layout()
    path = CHART_DIR / "realworld_vs_civbench_p10.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# Analyst composite forecasts (point estimates, treated as p50-equivalents).
# Sources: EIA STEO, Goldman Sachs, JPM, Morgan Stanley, Citi, BMI, Reuters,
#          Standard Chartered, Bernstein, VanEck, Fundstrat, Galaxy Digital,
#          HSBC, UBS, Metals Focus, Bank of America.
# All published before the respective cutoff dates.
ANALYST_COMPOSITES = {
    # Oil: median of EIA($58.62), Goldman($52), JPM($56), MS($56.5), Citi($57),
    #      BMI($64), Reuters($60.38) → median ~$57
    # Most analysts forecast Q1 2026 averages; we map to our 3 horizons
    "wti_crude": {
        "H1": {"consensus_p50": 57.0, "range_low": 52.0, "range_high": 64.0,
               "sources": "EIA STEO, Goldman, JPM, MS, Citi, BMI, Reuters (8 sources)"},
        "H2": {"consensus_p50": 56.0, "range_low": 52.0, "range_high": 64.0,
               "sources": "Same — most forecasts are quarterly averages"},
        "H3": {"consensus_p50": 55.0, "range_low": 50.0, "range_high": 64.0,
               "sources": "Q2 estimates trend lower on oversupply thesis"},
    },
    # Bitcoin: pre-Sept 2025 targets. Galaxy ($120-125k), VanEck ($180k),
    # StanChart/Bernstein/Fundstrat ($200-250k). Median ~$180-200k.
    # These are year-end 2025 targets; we map to our horizons roughly.
    # For H1-H3 (Sept-Oct), analysts would expect continued run toward targets.
    # For H5-H7 (Dec-Mar), the year-end targets apply directly.
    "bitcoin": {
        "H1": {"consensus_p50": 115000, "range_low": 100000, "range_high": 200000,
               "sources": "Galaxy ($120k), implied interim from StanChart/VanEck targets"},
        "H2": {"consensus_p50": 120000, "range_low": 100000, "range_high": 200000,
               "sources": "Approaching year-end targets"},
        "H3": {"consensus_p50": 130000, "range_low": 110000, "range_high": 200000,
               "sources": "Galaxy ($120-125k peak), others higher"},
        "H4": {"consensus_p50": 150000, "range_low": 120000, "range_high": 250000,
               "sources": "Approaching year-end; StanChart $200k, VanEck $180k"},
        "H5": {"consensus_p50": 180000, "range_low": 120000, "range_high": 250000,
               "sources": "Year-end 2025 targets: median of StanChart/Bernstein/VanEck/Fundstrat/Galaxy"},
        "H6": {"consensus_p50": 200000, "range_low": 125000, "range_high": 300000,
               "sources": "Into 2026; StanChart $300k YE2026, others extrapolated"},
        "H7": {"consensus_p50": 220000, "range_low": 130000, "range_high": 300000,
               "sources": "Q1 2026; StanChart $300k target, median ~$200-220k"},
    },
    # Silver: HSBC ($44.50 avg, $45-55 H1), Citi ($40-43), UBS ($42-55),
    # Metals Focus ($60). Median ~$48-50 for Q1 2026.
    "silver": {
        "H1": {"consensus_p50": 50.0, "range_low": 40.0, "range_high": 60.0,
               "sources": "HSBC ($45-55 H1), Citi ($40-43), UBS ($42-55), Metals Focus ($60)"},
        "H2": {"consensus_p50": 50.0, "range_low": 40.0, "range_high": 60.0,
               "sources": "Same sources"},
        "H3": {"consensus_p50": 50.0, "range_low": 40.0, "range_high": 60.0,
               "sources": "Same — most give annual or H1 averages"},
        "H4": {"consensus_p50": 48.0, "range_low": 40.0, "range_high": 55.0,
               "sources": "HSBC sees H2 moderation; UBS $52-55 mid-2026"},
        "H5": {"consensus_p50": 48.0, "range_low": 40.0, "range_high": 55.0,
               "sources": "Moving into H2 2026 territory per HSBC"},
    },
    # Natural gas: EIA Feb ($4.31 avg), Goldman ($4.60), MS (>$5), BofA ($4.00),
    # Fitch ($4.10). Our horizons are just 3 weeks out from cutoff.
    "natural_gas": {
        "H1": {"consensus_p50": 4.30, "range_low": 3.38, "range_high": 5.00,
               "sources": "EIA Feb STEO ($4.60 Feb), Goldman ($4.60), MS (>$5), BofA ($4.00)"},
        "H2": {"consensus_p50": 4.20, "range_low": 3.38, "range_high": 5.00,
               "sources": "Same; EIA Feb estimate for Mar ~$4.12"},
        "H3": {"consensus_p50": 4.10, "range_low": 3.38, "range_high": 5.00,
               "sources": "EIA Feb STEO Mar estimate ~$4.12"},
    },
}


def chart_analyst_vs_llm(results: list[dict]):
    """Dumbbell chart: analyst consensus p50 vs LLM pooled p50 vs actual."""
    assets = sorted(set(r["asset_id"] for r in results))
    models = sorted(set(MODEL_SHORT_NAMES.get(r["model"], r["model"]) for r in results))

    fig, axes = plt.subplots(1, len(assets), figsize=(5 * len(assets), 5), squeeze=False)

    for col, asset in enumerate(assets):
        ax = axes[0, col]
        asset_results = [r for r in results if r["asset_id"] == asset and r["percentiles"]]
        if not asset_results:
            continue

        horizons = sorted(set(r["horizon"] for r in asset_results))
        analyst = ANALYST_COMPOSITES.get(asset, {})

        x = np.arange(len(horizons))

        # Ground truth
        truth_by_h = {}
        for r in asset_results:
            truth_by_h[r["horizon"]] = r["ground_truth"]
        truth_vals = [truth_by_h.get(h) for h in horizons]

        # LLM pooled p50 (average across models)
        llm_p50_by_h = {}
        for h in horizons:
            p50s = [r["percentiles"]["p50"] for r in asset_results
                    if r["horizon"] == h and r["percentiles"] and "p50" in r["percentiles"]]
            llm_p50_by_h[h] = np.mean(p50s) if p50s else None
        llm_vals = [llm_p50_by_h.get(h) for h in horizons]

        # Analyst consensus p50
        analyst_vals = [analyst.get(h, {}).get("consensus_p50") for h in horizons]
        analyst_lo = [analyst.get(h, {}).get("range_low") for h in horizons]
        analyst_hi = [analyst.get(h, {}).get("range_high") for h in horizons]

        # Plot
        ax.plot(x, truth_vals, "ko-", linewidth=2.5, markersize=10, label="Actual", zorder=10)
        ax.plot(x, llm_vals, "s--", color="#2ca02c", linewidth=2, markersize=8,
                label="LLM consensus (p50)", zorder=8)

        valid_analyst_x = [i for i, v in enumerate(analyst_vals) if v is not None]
        valid_analyst = [analyst_vals[i] for i in valid_analyst_x]
        valid_lo = [analyst_lo[i] for i in valid_analyst_x]
        valid_hi = [analyst_hi[i] for i in valid_analyst_x]
        ax.plot(valid_analyst_x, valid_analyst, "D--", color="#d62728", linewidth=2,
                markersize=8, label="Analyst consensus", zorder=8)
        if all(v is not None for v in valid_lo + valid_hi):
            ax.fill_between(valid_analyst_x, valid_lo, valid_hi, color="#d62728", alpha=0.1,
                            label="Analyst range")

        ax.set_xticks(list(x))
        ax.set_xticklabels(horizons, rotation=45)
        ax.set_title(ASSET_DISPLAY.get(asset, asset))
        ax.set_ylabel("Price" if col == 0 else "")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Analyst forecasts vs LLM forecasts vs actual prices", fontsize=13)
    fig.tight_layout()
    path = CHART_DIR / "realworld_analyst_vs_llm.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def print_analyst_comparison(results: list[dict]):
    """Print analyst vs LLM comparison table."""
    assets = sorted(set(r["asset_id"] for r in results))

    print("\n*** Analyst consensus vs LLM consensus vs actual")
    print()
    print("| Asset | Horizon | Actual | Analyst p50 | LLM p50 (avg) | Analyst err | LLM err | Winner |")
    print("|" + "+".join(["-" * 17] * 8) + "|")

    for asset in assets:
        asset_results = [r for r in results if r["asset_id"] == asset and r["percentiles"]]
        analyst = ANALYST_COMPOSITES.get(asset, {})
        horizons = sorted(set(r["horizon"] for r in asset_results))

        for h in horizons:
            truth = next((r["ground_truth"] for r in asset_results if r["horizon"] == h), None)
            if truth is None:
                continue

            # LLM pooled p50
            p50s = [r["percentiles"]["p50"] for r in asset_results
                    if r["horizon"] == h and r["percentiles"] and "p50" in r["percentiles"]]
            llm_p50 = np.mean(p50s) if p50s else None

            a_p50 = analyst.get(h, {}).get("consensus_p50")
            if a_p50 is None or llm_p50 is None:
                continue

            a_err = abs(a_p50 - truth) / abs(truth) * 100
            l_err = abs(llm_p50 - truth) / abs(truth) * 100
            winner = "Analyst" if a_err < l_err else "LLM" if l_err < a_err else "Tie"

            name = ASSET_DISPLAY.get(asset, asset)
            print(f"| {name:<15} | {h:<15} | {truth:<15.2f} | {a_p50:<15.2f} | {llm_p50:<15.2f} "
                  f"| {a_err:<15.1f}% | {l_err:<15.1f}% | {winner:<15} |")


def main():
    results = load_results()
    print(f"Loaded {len(results)} forecast results")

    # Filter to valid results
    valid = [r for r in results if r["percentiles"] is not None]
    failed = len(results) - len(valid)
    if failed:
        print(f"  ({failed} parse failures excluded)")

    metrics = compute_metrics(valid)

    # Print summary tables (org-mode formatted for pasting into .org file)
    print_summary_table(valid, metrics)

    # Generate charts
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_tail_miss_by_asset(valid, metrics)
    chart_crps_by_asset(valid, metrics)
    chart_forecast_vs_actual(valid)
    chart_civbench_comparison(valid, metrics)
    chart_analyst_vs_llm(valid)

    # Analyst comparison table
    print_analyst_comparison(valid)

    # Print individual forecasts for .org file
    print("\n*** Individual forecasts")
    for r in sorted(valid, key=lambda x: (x["asset_id"], x["model"], x["horizon"])):
        p = r["percentiles"]
        model = MODEL_SHORT_NAMES.get(r["model"], r["model"])
        pct_err = abs(p["p50"] - r["ground_truth"]) / abs(r["ground_truth"]) * 100
        p1_miss = "MISS" if "p1" in p and r["ground_truth"] < p["p1"] else ""
        p5_miss = "MISS" if "p5" in p and r["ground_truth"] < p["p5"] else ""
        p10_miss = "MISS" if "p10" in p and r["ground_truth"] < p["p10"] else ""
        flags = " ".join(filter(None, [
            f"p1:{p1_miss}" if p1_miss else "",
            f"p5:{p5_miss}" if p5_miss else "",
            f"p10:{p10_miss}" if p10_miss else "",
        ])) or "ok"
        print(f"  {ASSET_DISPLAY.get(r['asset_id'], r['asset_id']):<15} {model:<15} "
              f"{r['horizon']:<4} truth={r['ground_truth']:<10.2f} "
              f"p1={p.get('p1', '?'):<10} p10={p.get('p10', '?'):<10} "
              f"p50={p.get('p50', '?'):<10} p90={p.get('p90', '?'):<10} "
              f"p99={p.get('p99', '?'):<10} err={pct_err:.1f}% {flags}")


if __name__ == "__main__":
    main()
