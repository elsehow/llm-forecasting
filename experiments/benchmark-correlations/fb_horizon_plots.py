"""
Generate figures for the FB Horizon Tail Risk experiment.
Outputs to fri-vault/_artifacts/static/
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUT_DIR = Path(os.path.expanduser("~/Projects/fri-vault/_artifacts/static"))
PROCESSED_DIR = Path(os.path.expanduser(
    "~/Projects/forecastbench-datasets/processed_forecast_sets"
))
QUESTION_SETS_DIR = Path(os.path.expanduser(
    "~/Projects/forecastbench-datasets/datasets/question_sets"
))
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}

ECI_SCORES = {
    "gpt-3.5-turbo": 110.0,
    "claude-3-haiku": 118.0,
    "llama-3.3-70b": 125.0,
    "grok-beta": 130.0,
    "claude-3-opus": 132.0,
    "gpt-4o": 132.0,
    "deepseek-v3": 135.0,
    "gpt-4.1": 137.4,
    "gpt-5-nano": 138.0,
    "deepseek-r1": 138.0,
    "claude-3.5-sonnet-20241022": 139.0,
    "claude-haiku-4.5": 140.0,
    "o3-mini": 140.0,
    "gpt-4.5-preview": 141.0,
    "o4-mini": 142.0,
    "claude-3.7-sonnet": 143.0,
    "gpt-5-mini": 144.3,
    "gemini-3-flash": 145.0,
    "claude-sonnet-4": 145.0,
    "o3": 146.6,
    "sonnet-4.5": 146.6,
    "gemini-2.5-pro": 147.0,
    "claude-opus-4.1": 148.0,
    "gpt-5.1": 149.7,
    "gpt-5": 150.0,
    "grok-4": 150.0,
    "opus-4.5": 150.1,
    "gpt-5.2": 152.0,
    "claude-opus-4.6": 152.0,
    "gemini-3-pro": 154.2,
}

HORIZON_ORDER = [
    "H1 (~17d)", "H2 (~40d)", "H3 (~100d)", "H4 (~190d)", "H5 (~375d)",
]
HORIZON_LABELS_SHORT = ["H1\n17d", "H2\n40d", "H3\n100d", "H4\n190d", "H5\n375d"]
HORIZON_DAYS = [17, 40, 100, 190, 375]


def model_filename_to_eci_key(filename):
    f = filename.lower()
    mappings = [
        ("gpt-5.2", "gpt-5.2"),
        ("gpt-5.1", "gpt-5.1"),
        ("gpt-5-nano", "gpt-5-nano"),
        ("gpt-5-mini", "gpt-5-mini"),
        ("gpt-5-2025", "gpt-5"),
        ("gpt-4.5-preview", "gpt-4.5-preview"),
        ("gpt-4.1", "gpt-4.1"),
        ("gpt-4o-2024", "gpt-4o"),
        ("gpt_4o", "gpt-4o"),
        ("gpt_4_turbo", "gpt-4o"),
        ("gpt_3p5_turbo", "gpt-3.5-turbo"),
        ("o4-mini", "o4-mini"),
        ("o3-mini", "o3-mini"),
        ("o3-2025", "o3"),
        ("claude-opus-4-6", "claude-opus-4.6"),
        ("claude-opus-4-5", "opus-4.5"),
        ("claude-opus-4-1", "claude-opus-4.1"),
        ("claude_3_opus", "claude-3-opus"),
        ("claude-sonnet-4-5", "sonnet-4.5"),
        ("claude-sonnet-4-2025", "claude-sonnet-4"),
        ("claude-3-7-sonnet", "claude-3.7-sonnet"),
        ("claude-3-5-sonnet-20241022", "claude-3.5-sonnet-20241022"),
        ("claude-3-5-sonnet-20240620", "claude-3.5-sonnet-20241022"),
        ("claude_3p5_sonnet", "claude-3.5-sonnet-20241022"),
        ("claude-haiku-4-5", "claude-haiku-4.5"),
        ("claude_3_haiku", "claude-3-haiku"),
        ("deepseek-r1", "deepseek-r1"),
        ("deepseek-v3", "deepseek-v3"),
        ("gemini-3-pro", "gemini-3-pro"),
        ("gemini-3-flash", "gemini-3-flash"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini_1p5_pro", "gemini-2.5-pro"),
        ("grok-4-0709", "grok-4"),
        ("grok-4-fast", "grok-4"),
        ("grok-4-1-fast", "grok-4"),
        ("grok-beta", "grok-beta"),
        ("llama-3p3-70b", "llama-3.3-70b"),
        ("llama_3_70b", "llama-3.3-70b"),
    ]
    for pattern, key in mappings:
        if pattern in f:
            return key
    return None


def load_question_set(date_str):
    path = QUESTION_SETS_DIR / f"{date_str}-llm.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for q in data["questions"]:
        qid = q["id"]
        key = str(qid) if isinstance(qid, list) else qid
        result[key] = q
    return result


def horizon_bin(days):
    if days <= 25: return "H1 (~17d)"
    elif days <= 60: return "H2 (~40d)"
    elif days <= 140: return "H3 (~100d)"
    elif days <= 280: return "H4 (~190d)"
    elif days <= 700: return "H5 (~375d)"
    elif days <= 1400: return "H6 (~1105d)"
    elif days <= 2500: return "H7 (~1835d)"
    else: return "H8 (~3660d)"


def compute_horizon_days(freeze_date_str, resolution_date_str):
    freeze = datetime.strptime(freeze_date_str[:10], "%Y-%m-%d")
    res = datetime.strptime(resolution_date_str[:10], "%Y-%m-%d")
    return (res - freeze).days


def collect_data():
    """Collect per-model, per-source, per-horizon Brier scores."""
    # model -> source -> horizon -> [brier]
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # model -> horizon -> [brier]  (pooled)
    pooled = defaultdict(lambda: defaultdict(list))

    for qs_dir in sorted(PROCESSED_DIR.iterdir()):
        if not qs_dir.is_dir():
            continue
        qs_date = qs_dir.name
        questions = load_question_set(qs_date)
        if not questions:
            continue

        for model_path in sorted(qs_dir.glob("*.json")):
            fname = model_path.stem
            if "ForecastBench" in fname or "external" in fname or "human" in fname:
                continue
            if "zero_shot" not in fname:
                continue
            if "freeze_values" in fname or "news" in fname:
                continue
            eci_key = model_filename_to_eci_key(fname)
            if eci_key is None or eci_key not in ECI_SCORES:
                continue

            with open(model_path) as f:
                data = json.load(f)

            for fc in data["forecasts"]:
                if fc["source"] not in DATASET_SOURCES:
                    continue
                if not fc.get("resolved"):
                    continue
                qid = str(fc["id"])
                q = questions.get(qid) or questions.get(fc["id"])
                if not q:
                    continue
                freeze_dt = q.get("freeze_datetime")
                if not freeze_dt:
                    continue
                res_date = str(fc["resolution_date"])[:10]
                outcome = float(fc["resolved_to"])
                forecast = float(fc["forecast"])
                days = compute_horizon_days(freeze_dt, res_date)
                hbin = horizon_bin(days)
                bs = (forecast - outcome) ** 2
                store[eci_key][fc["source"]][hbin].append(bs)
                pooled[eci_key][hbin].append(bs)

    return store, pooled


def fig1_rho_attenuation(pooled):
    """ECI × Brier rho by horizon, compared to CivBench."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # FB data
    fb_rhos, fb_ps, fb_ns = [], [], []
    for h in HORIZON_ORDER:
        ecis, briers = [], []
        for m in pooled:
            scores = pooled[m].get(h, [])
            if len(scores) >= 10:
                ecis.append(ECI_SCORES[m])
                briers.append(np.mean(scores))
        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, briers)
            fb_rhos.append(rho)
            fb_ps.append(p)
            fb_ns.append(len(ecis))
        else:
            fb_rhos.append(np.nan)
            fb_ps.append(np.nan)
            fb_ns.append(0)

    # CivBench data (from tail-risk-blindness experiment, MAE rho)
    cb_rhos = [-0.619, +0.881, +0.643, +0.667, +0.786, +0.643]
    cb_labels = ["H1\nT90", "H2\nT120", "H3\nT150", "H4\nT180", "H5\nT210", "H6\nT240"]

    x_fb = np.arange(len(HORIZON_ORDER))
    x_cb = np.linspace(0, len(HORIZON_ORDER) - 1, len(cb_rhos))

    ax.plot(x_fb, fb_rhos, "o-", color="#2563eb", linewidth=2.5, markersize=9,
            label="ForecastBench (binary Brier)", zorder=5)
    ax.plot(x_cb, cb_rhos, "s--", color="#dc2626", linewidth=2.5, markersize=9,
            label="CivBench (continuous MAE)", zorder=5)

    # Mark significance
    for i, (rho, p, n) in enumerate(zip(fb_rhos, fb_ps, fb_ns)):
        if not np.isnan(p) and p < 0.05:
            ax.plot(i, rho, "o", color="#2563eb", markersize=14, fillstyle="none",
                    linewidth=2, zorder=6)

    ax.axhline(0, color="gray", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_xticks(x_fb)
    ax.set_xticklabels(HORIZON_LABELS_SHORT)
    ax.set_ylabel("Spearman ρ (ECI vs error)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("G-loading attenuates on ForecastBench, reverses on CivBench")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotations
    ax.annotate("pro-g\n(smarter = better)", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=9, color="#2563eb", alpha=0.6, va="bottom")
    ax.annotate("anti-g\n(smarter = worse)", xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=9, color="#dc2626", alpha=0.6, va="top")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fb_vs_civbench_rho_by_horizon.png", dpi=180)
    print(f"Saved fig1: fb_vs_civbench_rho_by_horizon.png")
    plt.close(fig)


def fig2_scatter_h1_vs_h4(pooled):
    """ECI vs Brier scatter at H1 and H4, showing attenuation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, h, title in zip(axes, ["H1 (~17d)", "H4 (~190d)"],
                             ["H1 (~17 days)", "H4 (~190 days)"]):
        ecis, briers, names = [], [], []
        for m in pooled:
            scores = pooled[m].get(h, [])
            if len(scores) >= 10:
                ecis.append(ECI_SCORES[m])
                briers.append(np.mean(scores))
                names.append(m)

        ecis, briers, names = np.array(ecis), np.array(briers), np.array(names)
        rho, p = stats.spearmanr(ecis, briers)

        ax.scatter(ecis, briers, s=60, c="#2563eb", alpha=0.7, edgecolors="white",
                   linewidth=0.5, zorder=5)

        # Label some interesting models
        for label_name in ["gpt-3.5-turbo", "gpt-5", "gemini-3-pro", "deepseek-r1",
                           "claude-3-haiku", "opus-4.5", "gpt-4.1", "o4-mini"]:
            idx = np.where(names == label_name)[0]
            if len(idx) > 0:
                i = idx[0]
                ax.annotate(label_name, (ecis[i], briers[i]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=7.5, alpha=0.7)

        # Trend line
        z = np.polyfit(ecis, briers, 1)
        x_line = np.linspace(ecis.min() - 2, ecis.max() + 2, 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="#dc2626", alpha=0.5, linewidth=1.5)

        ax.set_xlabel("ECI (capability)")
        ax.set_title(f"{title}\nρ = {rho:+.3f}, p = {p:.4f}, N = {len(ecis)}")

    axes[0].set_ylabel("Brier Score (lower = better)")
    fig.suptitle("Capability predicts forecasting at short horizons, less at long horizons",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fb_eci_brier_scatter_h1_h4.png", dpi=180, bbox_inches="tight")
    print(f"Saved fig2: fb_eci_brier_scatter_h1_h4.png")
    plt.close(fig)


def fig3_per_source_rho(store):
    """Per-source rho attenuation heatmap."""
    sources = ["acled", "fred", "wikipedia", "yfinance", "dbnomics"]
    source_labels = ["ACLED\n(conflict)", "FRED\n(econ)", "Wikipedia\n(rankings)",
                     "YFinance\n(stocks)", "DBnomics\n(weather+)"]

    rho_matrix = np.full((len(sources), len(HORIZON_ORDER)), np.nan)
    sig_matrix = np.full((len(sources), len(HORIZON_ORDER)), False)

    for si, src in enumerate(sources):
        for hi, h in enumerate(HORIZON_ORDER):
            ecis, briers = [], []
            for m in store:
                scores = store[m][src].get(h, [])
                if len(scores) >= 5:
                    ecis.append(ECI_SCORES[m])
                    briers.append(np.mean(scores))
            if len(ecis) >= 4:
                rho, p = stats.spearmanr(ecis, briers)
                rho_matrix[si, hi] = rho
                sig_matrix[si, hi] = p < 0.05

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(HORIZON_ORDER)))
    ax.set_xticklabels(HORIZON_LABELS_SHORT)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(source_labels)

    # Annotate cells
    for si in range(len(sources)):
        for hi in range(len(HORIZON_ORDER)):
            val = rho_matrix[si, hi]
            if np.isnan(val):
                ax.text(hi, si, "—", ha="center", va="center", fontsize=10, color="gray")
            else:
                sig = "**" if sig_matrix[si, hi] else ""
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(hi, si, f"{val:+.2f}{sig}", ha="center", va="center",
                       fontsize=10, fontweight="bold" if sig else "normal", color=color)

    ax.set_xlabel("Forecast horizon")
    ax.set_title("ECI × Brier ρ by source and horizon\n(** = p < 0.05, blue = pro-g, red = anti-g)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fb_source_horizon_rho_heatmap.png", dpi=180)
    print(f"Saved fig3: fb_source_horizon_rho_heatmap.png")
    plt.close(fig)


def fig4_model_brier_lines(pooled):
    """Line plot: Brier by horizon for selected models, colored by ECI."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Select models that have data at H1-H4 minimum
    good_models = []
    for m in pooled:
        has = sum(1 for h in HORIZON_ORDER[:4] if len(pooled[m].get(h, [])) >= 10)
        if has >= 4:
            good_models.append(m)

    good_models.sort(key=lambda x: ECI_SCORES[x])
    ecis = [ECI_SCORES[m] for m in good_models]
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(ecis) - 5, vmax=max(ecis) + 5)

    for m in good_models:
        eci = ECI_SCORES[m]
        xs, ys = [], []
        for hi, h in enumerate(HORIZON_ORDER):
            scores = pooled[m].get(h, [])
            if len(scores) >= 10:
                xs.append(hi)
                ys.append(np.mean(scores))

        color = cmap(norm(eci))
        alpha = 0.35 if eci < 140 else 0.85
        lw = 1.2 if eci < 140 else 2.2
        ax.plot(xs, ys, "o-", color=color, alpha=alpha, linewidth=lw, markersize=6)

    # Label extremes — use last plotted point for each
    label_models = ["gpt-3.5-turbo", "gpt-5", "gemini-3-pro", "deepseek-r1",
                    "claude-3-haiku", "gpt-4.1", "claude-3.5-sonnet-20241022"]
    for m in label_models:
        if m not in pooled:
            continue
        # Find last horizon with data
        last_x, last_y = None, None
        for hi, h in enumerate(HORIZON_ORDER):
            scores = pooled[m].get(h, [])
            if len(scores) >= 10:
                last_x, last_y = hi, np.mean(scores)
        if last_x is not None:
            eci = ECI_SCORES[m]
            ax.annotate(f" {m} ({eci:.0f})", (last_x, last_y),
                       fontsize=8, alpha=0.8, color=cmap(norm(eci)),
                       va="center")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("ECI (capability)")

    ax.set_xticks(range(len(HORIZON_ORDER)))
    ax.set_xticklabels(HORIZON_LABELS_SHORT)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("All models degrade at long horizons, but higher-ECI (red) stays lower")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fb_model_brier_by_horizon.png", dpi=180)
    print(f"Saved fig4: fb_model_brier_by_horizon.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Collecting data...")
    store, pooled = collect_data()
    print(f"Models: {len(pooled)}, Sources: {len(DATASET_SOURCES)}")

    fig1_rho_attenuation(pooled)
    fig2_scatter_h1_vs_h4(pooled)
    fig3_per_source_rho(store)
    fig4_model_brier_lines(pooled)
    print("Done.")
