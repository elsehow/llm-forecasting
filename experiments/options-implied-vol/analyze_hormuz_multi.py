#!/usr/bin/env python3
"""
Hormuz Natural Experiment — Comprehensive Analysis
====================================================
Tests whether LLMs are structurally overconfident on energy commodities
affected by the March 2026 Strait of Hormuz crisis.

Crisis date: Feb 28, 2026.
  - Pre-crisis ("calm"): cutoff_date <= "2026-02-23"
  - Crisis: cutoff_date >= "2026-03-02"

Hormuz-exposed assets: oil, jkm, ttf
Control assets: gold, treasury
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/Users/elsehow/Projects/llm-forecasting/experiments/options-implied-vol")
FIG_DIR = Path("/Users/elsehow/Projects/tailriskbench/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── ECI mapping ───────────────────────────────────────────────────────
ECI_MAP = {
    "anthropic/claude-3-haiku-20240307": 98,
    "openai/gpt-3.5-turbo-0125": 100,
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": 89,
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": 117,
    "mistral/mistral-large-2407": 118,
    "mistral/mistral-large-2411": 125,
    "mistral/mistral-large-latest": 130,
    "google/gemini-2.0-flash-lite-001": 110,
    "openai/gpt-4o": 132,
    "google/gemini-2.5-flash": 135,
    "together_ai/deepseek-ai/DeepSeek-V3": 128,
    "together_ai/deepseek-ai/DeepSeek-V3.1": 135,
    "anthropic/claude-sonnet-4-20250514": 140,
    "anthropic/claude-haiku-4-5-20251001": 138,
    "openai/gpt-4.1-2025-04-14": 137,
    "google/gemini-2.5-pro": 140,
    "openai/o3-mini-2025-01-31": 140,
    "xai/grok-4-fast-non-reasoning": 140,
    "openai/gpt-5-nano-2025-08-07": 120,
    "openai/gpt-5-mini-2025-08-07": 142,
    "openai/o4-mini-2025-04-16": 145,
    "xai/grok-4-fast-reasoning": 148,
    "anthropic/claude-sonnet-4-5-20250929": 148,
    "openai/o3-2025-04-16": 149,
    "openai/gpt-5-2025-08-07": 150,
    "xai/grok-4-0709": 151,
    "openai/gpt-5.1-2025-11-13": 152,
    "xai/grok-4-1-fast-non-reasoning": 145,
    "xai/grok-4-1-fast-reasoning": 152,
    "google/gemini-3-pro-preview": 154,
    "anthropic/claude-opus-4-5-20251101": 155,
    "anthropic/claude-opus-4-6": 157,
}

EXPOSED = {"oil", "jkm", "ttf"}
CONTROLS = {"gold", "treasury"}
CRISIS_DATE = "2026-02-28"
QUANTILES = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
QUANTILE_VALS = {"p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
                 "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
COLOR_EXPOSED = "#c0392b"
COLOR_CONTROL = "#2980b9"
COLOR_NEUTRAL = "#7f8c8d"


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def load_data():
    with open(BASE / "results" / "forecasts.json") as f:
        raw = json.load(f)

    # Filter to valid percentiles only, and to our assets of interest
    assets_of_interest = EXPOSED | CONTROLS
    rows = []
    for r in raw:
        if r["asset_id"] not in assets_of_interest:
            continue
        if r["percentiles"] is None:
            continue
        if r["ground_truth"] is None:
            continue
        rows.append(r)

    df = pd.DataFrame(rows)
    # Flatten percentiles
    pct_df = pd.json_normalize(df["percentiles"])
    pct_df.columns = [c.replace(".", "") for c in pct_df.columns]  # already clean
    df = pd.concat([df.drop(columns=["percentiles"]).reset_index(drop=True),
                     pct_df.reset_index(drop=True)], axis=1)
    df["eci"] = df["model"].map(ECI_MAP)
    df["is_calm"] = df["cutoff_date"] < CRISIS_DATE
    df["is_exposed"] = df["asset_id"].isin(EXPOSED)
    return df


def filter_models_by_parse_rate(df, threshold=0.80):
    """Keep models with >= threshold parse rate on JKM+TTF."""
    with open(BASE / "results" / "forecasts.json") as f:
        raw = json.load(f)
    jkm_ttf = [r for r in raw if r["asset_id"] in ("jkm", "ttf")]
    total = {}
    parsed = {}
    for r in jkm_ttf:
        total[r["model"]] = total.get(r["model"], 0) + 1
        if r["percentiles"] is not None:
            parsed[r["model"]] = parsed.get(r["model"], 0) + 1
    good_models = {m for m in total if parsed.get(m, 0) / total[m] >= threshold}
    return df[df["model"].isin(good_models)].copy()


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def pinball_loss(q_val, predicted, actual):
    """Pinball (quantile) loss."""
    e = actual - predicted
    return np.where(e >= 0, q_val * e, (q_val - 1) * e)


def compute_crps_from_quantiles(row):
    """Approximate CRPS from 9 quantile forecasts using the quantile representation."""
    taus = np.array([QUANTILE_VALS[q] for q in QUANTILES])
    preds = np.array([row[q] for q in QUANTILES])
    y = row["ground_truth"]
    # CRPS = 2 * integral of quantile loss over tau
    # With discrete quantiles, approximate by trapezoidal rule
    losses = pinball_loss(taus, preds, y)
    crps = 2 * np.trapz(losses, taus)
    return crps


def compute_pit(row):
    """Compute PIT value via linear interpolation of the model's implied CDF."""
    taus = np.array([QUANTILE_VALS[q] for q in QUANTILES])
    preds = np.array([row[q] for q in QUANTILES])
    y = row["ground_truth"]
    if y <= preds[0]:
        return taus[0]  # below p1
    if y >= preds[-1]:
        return taus[-1]  # above p99
    return float(np.interp(y, preds, taus))


def short_model(m):
    """Short model name for display."""
    parts = m.split("/")
    return parts[-1] if len(parts) > 1 else m


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: CALM-MARKET PER-QUANTILE PINBALL × ECI
# ══════════════════════════════════════════════════════════════════════
def section1_pinball_eci(df):
    print("\n" + "=" * 72)
    print("SECTION 1: Calm-Market Per-Quantile Pinball × ECI (PRIMARY)")
    print("=" * 72)

    calm = df[df["is_calm"]].copy()

    analysis_quantiles = ["p10", "p25", "p50", "p75", "p90"]

    for group_name, group_assets, color in [
        ("HORMUZ-EXPOSED (oil, jkm, ttf)", EXPOSED, COLOR_EXPOSED),
        ("CONTROLS (gold, treasury)", CONTROLS, COLOR_CONTROL),
    ]:
        print(f"\n--- {group_name} ---")
        subset = calm[calm["asset_id"].isin(group_assets)].copy()

        # Compute normalized pinball for each quantile
        for q in analysis_quantiles:
            tau = QUANTILE_VALS[q]
            subset[f"pinball_{q}"] = pinball_loss(tau, subset[q], subset["ground_truth"]) / subset["cutoff_price"]

        # Aggregate per model: mean normalized pinball at each quantile
        model_agg = subset.groupby("model").agg(
            eci=("eci", "first"),
            **{f"mean_pb_{q}": (f"pinball_{q}", "mean") for q in analysis_quantiles},
            n=("model", "count"),
        ).reset_index()

        for q in analysis_quantiles:
            col = f"mean_pb_{q}"
            rho, p = stats.spearmanr(model_agg["eci"], model_agg[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {q}: Spearman rho = {rho:+.3f}  (p = {p:.4f}) {sig}  [n_models = {len(model_agg)}]")

    # ── Figure: side-by-side per-quantile rho comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))

    rhos_exposed = []
    rhos_control = []
    ps_exposed = []
    ps_control = []
    for q in analysis_quantiles:
        tau = QUANTILE_VALS[q]
        # Exposed
        sub_e = calm[calm["is_exposed"]].copy()
        sub_e[f"pb"] = pinball_loss(tau, sub_e[q], sub_e["ground_truth"]) / sub_e["cutoff_price"]
        agg_e = sub_e.groupby("model").agg(eci=("eci", "first"), mpb=("pb", "mean")).reset_index()
        r_e, p_e = stats.spearmanr(agg_e["eci"], agg_e["mpb"])
        rhos_exposed.append(r_e)
        ps_exposed.append(p_e)
        # Control
        sub_c = calm[~calm["is_exposed"]].copy()
        sub_c[f"pb"] = pinball_loss(tau, sub_c[q], sub_c["ground_truth"]) / sub_c["cutoff_price"]
        agg_c = sub_c.groupby("model").agg(eci=("eci", "first"), mpb=("pb", "mean")).reset_index()
        r_c, p_c = stats.spearmanr(agg_c["eci"], agg_c["mpb"])
        rhos_control.append(r_c)
        ps_control.append(p_c)

    x = np.arange(len(analysis_quantiles))
    w = 0.35
    bars_e = ax.bar(x - w / 2, rhos_exposed, w, label="Hormuz-exposed", color=COLOR_EXPOSED, alpha=0.85)
    bars_c = ax.bar(x + w / 2, rhos_control, w, label="Controls", color=COLOR_CONTROL, alpha=0.85)

    # Significance markers
    for i, (pe, pc) in enumerate(zip(ps_exposed, ps_control)):
        for bars, p_val, offset in [(bars_e, pe, -w / 2), (bars_c, pc, w / 2)]:
            bar_h = bars[i].get_height()
            marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            if marker:
                y_pos = bar_h + 0.02 if bar_h >= 0 else bar_h - 0.04
                ax.text(i + offset, y_pos, marker, ha="center", va="bottom" if bar_h >= 0 else "top", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(analysis_quantiles)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Spearman ρ (ECI vs. normalized pinball loss)")
    ax.set_title("Calm-Market Pinball × ECI: Exposed vs. Controls")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.set_ylim(-0.7, 0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hormuz_pinball_eci_exposed_vs_control.png")
    plt.close(fig)
    print(f"\n  Figure saved: {FIG_DIR / 'hormuz_pinball_eci_exposed_vs_control.png'}")

    return rhos_exposed, rhos_control


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PEAKEDNESS × ECI
# ══════════════════════════════════════════════════════════════════════
def section2_peakedness(df):
    print("\n" + "=" * 72)
    print("SECTION 2: PIT Peakedness × ECI (Calm Market)")
    print("=" * 72)

    calm = df[df["is_calm"]].copy()
    calm["pit"] = calm.apply(compute_pit, axis=1)

    for group_name, is_exp in [("HORMUZ-EXPOSED", True), ("CONTROLS", False)]:
        print(f"\n--- {group_name} ---")
        sub = calm[calm["is_exposed"] == is_exp].copy()

        # Peakedness = excess kurtosis of PIT distribution per model
        # A uniform PIT has kurtosis = -1.2 (platykurtic)
        # More peaked = higher kurtosis
        model_stats = []
        for m, g in sub.groupby("model"):
            pit_vals = g["pit"].values
            if len(pit_vals) < 10:
                continue
            # Peakedness via concentration around median
            # Use fraction of PITs in [0.25, 0.75] — uniform expectation is 0.50
            center_frac = np.mean((pit_vals >= 0.25) & (pit_vals <= 0.75))
            # Also compute fraction in tails [0, 0.1] + [0.9, 1]
            tail_frac = np.mean((pit_vals <= 0.10) | (pit_vals >= 0.90))
            model_stats.append({
                "model": m,
                "eci": ECI_MAP.get(m, np.nan),
                "center_frac": center_frac,
                "tail_frac": tail_frac,
                "n_pits": len(pit_vals),
                "pit_mean": np.mean(pit_vals),
                "pit_std": np.std(pit_vals),
            })
        ms = pd.DataFrame(model_stats)
        if len(ms) == 0:
            print("  No data.")
            continue

        rho_c, p_c = stats.spearmanr(ms["eci"], ms["center_frac"])
        rho_t, p_t = stats.spearmanr(ms["eci"], ms["tail_frac"])
        print(f"  Center concentration [0.25, 0.75]: ρ(ECI) = {rho_c:+.3f}  (p = {p_c:.4f})")
        print(f"  Tail fraction [0, 0.1]∪[0.9, 1]:   ρ(ECI) = {rho_t:+.3f}  (p = {p_t:.4f})")
        print(f"  Mean center_frac = {ms['center_frac'].mean():.3f}  (uniform = 0.50)")
        print(f"  Mean tail_frac   = {ms['tail_frac'].mean():.3f}  (uniform = 0.20)")
        print(f"  PIT mean across models = {ms['pit_mean'].mean():.3f}  (uniform = 0.50)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: CRPS × ECI
# ══════════════════════════════════════════════════════════════════════
def section3_crps(df):
    print("\n" + "=" * 72)
    print("SECTION 3: CRPS × ECI (Calm Market, Normalized)")
    print("=" * 72)

    calm = df[df["is_calm"]].copy()
    calm["crps"] = calm.apply(compute_crps_from_quantiles, axis=1)

    # Normalize CRPS by cross-model median per (asset, cutoff, horizon)
    calm["crps_norm"] = calm.groupby(["asset_id", "cutoff_date", "horizon"])["crps"].transform(
        lambda x: x / x.median() if x.median() > 0 else x
    )

    for group_name, is_exp in [("HORMUZ-EXPOSED", True), ("CONTROLS", False)]:
        print(f"\n--- {group_name} ---")
        sub = calm[calm["is_exposed"] == is_exp]
        model_agg = sub.groupby("model").agg(
            eci=("eci", "first"),
            mean_crps_norm=("crps_norm", "mean"),
            mean_crps_raw=("crps", "mean"),
            n=("model", "count"),
        ).reset_index()

        rho, p = stats.spearmanr(model_agg["eci"], model_agg["mean_crps_norm"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Spearman ρ(ECI, norm CRPS) = {rho:+.3f}  (p = {p:.4f}) {sig}")
        print(f"  Mean raw CRPS: {model_agg['mean_crps_raw'].mean():.4f}")
        print(f"  n_models = {len(model_agg)}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: CRISIS TAIL COVERAGE (DESCRIPTIVE)
# ══════════════════════════════════════════════════════════════════════
def section4_crisis_coverage(df):
    print("\n" + "=" * 72)
    print("SECTION 4: Crisis Tail Coverage (Descriptive)")
    print("=" * 72)

    crisis = df[~df["is_calm"]].copy()
    if len(crisis) == 0:
        print("  No crisis-period data with ground truth.")
        return

    # For each (asset, cutoff, horizon), what fraction of models cover actual?
    coverage_quantiles = ["p1", "p5", "p10", "p90", "p95", "p99"]
    results = []
    for (asset, cutoff, horizon), g in crisis.groupby(["asset_id", "cutoff_date", "horizon"]):
        gt = g["ground_truth"].iloc[0]
        cp = g["cutoff_price"].iloc[0]
        pct_move = (gt - cp) / cp * 100
        row = {
            "asset": asset, "cutoff": cutoff, "horizon": horizon,
            "ground_truth": gt, "cutoff_price": cp,
            "pct_move": pct_move, "n_models": len(g),
        }
        for q in coverage_quantiles:
            tau = QUANTILE_VALS[q]
            if tau < 0.5:
                # Left-tail: what fraction had p_q <= actual (i.e., actual was inside their interval)
                row[f"{q}_covers"] = (g[q] <= gt).mean()
            else:
                # Right-tail: what fraction had p_q >= actual
                row[f"{q}_covers"] = (g[q] >= gt).mean()
        results.append(row)

    rdf = pd.DataFrame(results)
    # Focus on exposed assets
    for asset in ["oil", "jkm", "ttf", "gold", "treasury"]:
        sub = rdf[rdf["asset"] == asset]
        if len(sub) == 0:
            continue
        print(f"\n--- {asset.upper()} ---")
        for _, row in sub.iterrows():
            print(f"  {row['cutoff']} {row['horizon']}: "
                  f"actual={row['ground_truth']:.2f}  cutoff_price={row['cutoff_price']:.2f}  "
                  f"move={row['pct_move']:+.1f}%  n={row['n_models']}")
            for q in coverage_quantiles:
                tau = QUANTILE_VALS[q]
                cov = row[f"{q}_covers"]
                label = "covers actual" if tau < 0.5 else "covers actual"
                print(f"    {q}: {cov:.0%} of models' {q} {'≤' if tau < 0.5 else '≥'} actual")

    # ── Figure: crisis coverage bar chart ──
    # Focus on exposed assets in crisis
    exposed_crisis = rdf[rdf["asset"].isin(EXPOSED)].copy()
    exposed_crisis = exposed_crisis.sort_values(["asset", "cutoff", "horizon"])

    # Show p10 and p1 coverage (left tail) for upside moves, p90 and p99 for downside
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: upside coverage (p90, p95, p99 — did model's right tail contain the actual?)
    ax = axes[0]
    labels = []
    p90_cov = []
    p95_cov = []
    p99_cov = []
    for _, row in exposed_crisis.iterrows():
        if row["pct_move"] > 5:  # upside moves only
            label = f"{row['asset'].upper()}\n{row['cutoff'][-5:]}\n{row['horizon']}"
            labels.append(label)
            p90_cov.append(row["p90_covers"])
            p95_cov.append(row["p95_covers"])
            p99_cov.append(row["p99_covers"])

    if labels:
        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, p99_cov, w, label="p99 covers", color="#e74c3c", alpha=0.9)
        ax.bar(x, p95_cov, w, label="p95 covers", color="#e67e22", alpha=0.9)
        ax.bar(x + w, p90_cov, w, label="p90 covers", color="#f39c12", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Fraction of models whose quantile covers actual")
        ax.set_title("Crisis Upside Coverage (exposed assets, move > +5%)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Right panel: p1/p5/p10 coverage for left-tail (downside or moderate)
    ax = axes[1]
    labels2 = []
    p1_cov = []
    p5_cov = []
    p10_cov = []
    for _, row in exposed_crisis.iterrows():
        label = f"{row['asset'].upper()}\n{row['cutoff'][-5:]}\n{row['horizon']}"
        labels2.append(label)
        p1_cov.append(row["p1_covers"])
        p5_cov.append(row["p5_covers"])
        p10_cov.append(row["p10_covers"])

    if labels2:
        x2 = np.arange(len(labels2))
        w = 0.25
        ax.bar(x2 - w, p1_cov, w, label="p1 covers", color="#2c3e50", alpha=0.9)
        ax.bar(x2, p5_cov, w, label="p5 covers", color="#2980b9", alpha=0.9)
        ax.bar(x2 + w, p10_cov, w, label="p10 covers", color="#3498db", alpha=0.9)
        ax.set_xticks(x2)
        ax.set_xticklabels(labels2, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Fraction of models whose quantile covers actual")
        ax.set_title("Crisis Left-Tail Coverage (all exposed cutoffs)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hormuz_crisis_coverage.png")
    plt.close(fig)
    print(f"\n  Figure saved: {FIG_DIR / 'hormuz_crisis_coverage.png'}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: THREE-FORECASTER RESPONSE (OIL ONLY)
# ══════════════════════════════════════════════════════════════════════
def section5_three_forecaster(df):
    print("\n" + "=" * 72)
    print("SECTION 5: Three-Forecaster Response — Oil H1")
    print("=" * 72)

    oil_prices = pd.read_csv(BASE / "data" / "prices" / "oil.csv", parse_dates=["Date"])
    oil_prices = oil_prices.sort_values("Date").reset_index(drop=True)
    oil_prices["log_ret"] = np.log(oil_prices["Close"] / oil_prices["Close"].shift(1))

    ovx = pd.read_csv(BASE / "data" / "vol_indices" / "OVX.csv", parse_dates=["Date"])
    ovx = ovx.sort_values("Date").reset_index(drop=True)

    oil_h1 = df[(df["asset_id"] == "oil") & (df["horizon"] == "H1")].copy()
    cutoffs = sorted(oil_h1["cutoff_date"].unique())

    results = []
    for cutoff in cutoffs:
        cutoff_dt = pd.Timestamp(cutoff)
        # 60-day realized vol
        mask = (oil_prices["Date"] <= cutoff_dt) & (oil_prices["Date"] > cutoff_dt - pd.Timedelta(days=90))
        recent = oil_prices.loc[mask].tail(60)
        if len(recent) < 20:
            continue
        daily_vol = recent["log_ret"].std()
        ann_vol = daily_vol * np.sqrt(252)
        # 7-day log-normal interval (H1 = 7 days)
        horizon_vol = daily_vol * np.sqrt(7)
        cutoff_price_row = oil_prices.loc[oil_prices["Date"] <= cutoff_dt].iloc[-1]
        cp = cutoff_price_row["Close"]
        hist_p10 = cp * np.exp(-1.2816 * horizon_vol)
        hist_p90 = cp * np.exp(1.2816 * horizon_vol)
        hist_width = hist_p90 - hist_p10

        # OVX at cutoff
        ovx_row = ovx.loc[ovx["Date"] <= cutoff_dt]
        ovx_val = ovx_row.iloc[-1]["Close"] if len(ovx_row) > 0 else np.nan

        # LLM pool
        models_at_cutoff = oil_h1[oil_h1["cutoff_date"] == cutoff]
        if len(models_at_cutoff) == 0:
            continue
        llm_widths = models_at_cutoff["p90"] - models_at_cutoff["p10"]
        llm_mean_width = llm_widths.mean()
        llm_median_width = llm_widths.median()

        # Ground truth
        gt_rows = models_at_cutoff.iloc[0]
        gt = gt_rows["ground_truth"]

        results.append({
            "cutoff": cutoff,
            "cutoff_price": cp,
            "ground_truth": gt,
            "hist_vol_ann": ann_vol,
            "hist_p90_p10_width": hist_width,
            "ovx": ovx_val,
            "llm_mean_width": llm_mean_width,
            "llm_median_width": llm_median_width,
            "n_models": len(models_at_cutoff),
        })

    rdf = pd.DataFrame(results)
    rdf["is_calm"] = rdf["cutoff"] < CRISIS_DATE

    print("\nOil H1 — Interval widths (p90 - p10) over time:")
    print(f"{'Cutoff':<12} {'Price':>8} {'GT':>8} {'HistVol':>10} {'LLM-mean':>10} {'LLM-med':>10} {'OVX':>6} {'Regime':<8}")
    print("-" * 80)
    for _, r in rdf.iterrows():
        regime = "CALM" if r["is_calm"] else "CRISIS"
        gt_str = f"{r['ground_truth']:.1f}" if pd.notna(r["ground_truth"]) else "N/A"
        print(f"{r['cutoff']:<12} {r['cutoff_price']:>8.2f} {gt_str:>8} "
              f"{r['hist_p90_p10_width']:>10.2f} {r['llm_mean_width']:>10.2f} {r['llm_median_width']:>10.2f} "
              f"{r['ovx']:>6.1f} {regime:<8}")

    # Widening ratios
    calm_data = rdf[rdf["is_calm"]]
    crisis_data = rdf[~rdf["is_calm"]]
    if len(calm_data) > 0 and len(crisis_data) > 0:
        print(f"\nCalm-period mean hist width: {calm_data['hist_p90_p10_width'].mean():.2f}")
        print(f"Crisis-period mean hist width: {crisis_data['hist_p90_p10_width'].mean():.2f}")
        print(f"Hist-vol widening ratio: {crisis_data['hist_p90_p10_width'].mean() / calm_data['hist_p90_p10_width'].mean():.2f}x")
        print(f"\nCalm-period mean LLM width: {calm_data['llm_mean_width'].mean():.2f}")
        print(f"Crisis-period mean LLM width: {crisis_data['llm_mean_width'].mean():.2f}")
        print(f"LLM widening ratio: {crisis_data['llm_mean_width'].mean() / calm_data['llm_mean_width'].mean():.2f}x")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cutoff_x = np.arange(len(rdf))
    ax.plot(cutoff_x, rdf["hist_p90_p10_width"], "s-", color="#2c3e50", label="Hist-vol baseline (60d)", markersize=5)
    ax.plot(cutoff_x, rdf["llm_mean_width"], "o-", color=COLOR_EXPOSED, label="LLM pool mean", markersize=5)
    ax.plot(cutoff_x, rdf["llm_median_width"], "^--", color="#e67e22", label="LLM pool median", markersize=4, alpha=0.7)

    # Crisis shading
    crisis_start = None
    for i, (_, r) in enumerate(rdf.iterrows()):
        if not r["is_calm"] and crisis_start is None:
            crisis_start = i - 0.5
    if crisis_start is not None:
        ax.axvspan(crisis_start, len(rdf) - 0.5, color="#e74c3c", alpha=0.08, label="Crisis period")
        ax.axvline(crisis_start, color="#e74c3c", linestyle="--", alpha=0.5)

    ax.set_xticks(cutoff_x)
    ax.set_xticklabels([r["cutoff"][-5:] for _, r in rdf.iterrows()], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Cutoff Date (MM-DD)")
    ax.set_ylabel("Interval Width (p90 − p10, USD)")
    ax.set_title("Oil H1: Three-Forecaster Interval Widths Over Time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hormuz_three_forecaster.png")
    plt.close(fig)
    print(f"\n  Figure saved: {FIG_DIR / 'hormuz_three_forecaster.png'}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: PER-ASSET DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════
def section6_per_asset(df):
    print("\n" + "=" * 72)
    print("SECTION 6: Per-Asset Decomposition — Calm p10 Pinball × ECI")
    print("=" * 72)

    calm = df[df["is_calm"]].copy()
    tau = QUANTILE_VALS["p10"]
    calm["pb_p10"] = pinball_loss(tau, calm["p10"], calm["ground_truth"]) / calm["cutoff_price"]

    for asset in ["oil", "jkm", "ttf", "gold", "treasury"]:
        sub = calm[calm["asset_id"] == asset]
        model_agg = sub.groupby("model").agg(
            eci=("eci", "first"),
            mean_pb=("pb_p10", "mean"),
            n=("model", "count"),
        ).reset_index()
        rho, p = stats.spearmanr(model_agg["eci"], model_agg["mean_pb"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        tag = "[EXPOSED]" if asset in EXPOSED else "[CONTROL]"
        print(f"  {asset:>10} {tag}: ρ = {rho:+.3f}  (p = {p:.4f}) {sig}  n_models={len(model_agg)}")

    # Also show all quantiles per asset
    print("\n  Full quantile decomposition per asset:")
    analysis_quantiles = ["p10", "p25", "p50", "p75", "p90"]
    header = f"  {'Asset':>10} | " + " | ".join(f"{q:>12}" for q in analysis_quantiles)
    print(header)
    print("  " + "-" * len(header))
    for asset in ["oil", "jkm", "ttf", "gold", "treasury"]:
        sub = calm[calm["asset_id"] == asset]
        rho_strs = []
        for q in analysis_quantiles:
            tau_q = QUANTILE_VALS[q]
            sub_q = sub.copy()
            sub_q["pb"] = pinball_loss(tau_q, sub_q[q], sub_q["ground_truth"]) / sub_q["cutoff_price"]
            agg = sub_q.groupby("model").agg(eci=("eci", "first"), mpb=("pb", "mean")).reset_index()
            r, p = stats.spearmanr(agg["eci"], agg["mpb"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "  "
            rho_strs.append(f"{r:+.3f} {sig}")
        print(f"  {asset:>10} | " + " | ".join(f"{s:>12}" for s in rho_strs))


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("Loading data...")
    df = load_data()
    print(f"  Total valid forecasts (exposed + controls with GT): {len(df)}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Assets: {sorted(df['asset_id'].unique())}")

    df = filter_models_by_parse_rate(df, threshold=0.80)
    print(f"  After >=80% JKM+TTF parse filter: {df['model'].nunique()} models, {len(df)} forecasts")
    print(f"  Retained models: {sorted(df['model'].unique())}")

    section1_pinball_eci(df)
    section2_peakedness(df)
    section3_crps(df)
    section4_crisis_coverage(df)
    section5_three_forecaster(df)
    section6_per_asset(df)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
