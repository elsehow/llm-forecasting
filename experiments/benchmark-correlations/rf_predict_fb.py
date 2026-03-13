#!/usr/bin/env python
"""Random Forest: predict ForecastBench from ARC-AGI-2.

Leave-one-out cross-validation (N=27). Reports MAE, RMSE, R² for both
FB Dataset and FB Overall targets. Compares RF to simple linear baseline.

Usage:
    uv run python experiments/benchmark-correlations/rf_predict_fb.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── Matched model data (name, arc1%, arc2%, fb_overall, fb_dataset) ──────────
# ARC scores at DEFAULT reasoning level. FB: best Overall/Dataset score.
DATA = [
    ("GPT-4o",            4.5,   0.0,  59.7, 56.0),
    ("GPT-4.1",           5.5,   0.4,  63.3, 60.1),
    ("GPT-4.5",          10.3,   0.8,  64.3, 60.3),
    ("GPT-5",            56.2,   7.5,  62.5, 61.8),
    ("GPT-5 Mini",       37.3,   4.0,  62.9, 60.8),
    ("GPT-5 Nano",       20.7,   0.9,  60.4, 57.6),
    ("GPT-5.1",           5.8,   0.4,  59.0, 58.7),
    ("o3",               53.8,   3.0,  64.8, 60.4),
    ("o3-mini",          22.3,   2.1,  62.3, 58.9),
    ("o4-mini",          41.8,   2.4,  63.8, 60.7),
    ("Claude 3.7",       13.6,   0.0,  64.6, 60.4),
    ("Claude Son. 4",    23.8,   1.3,  61.7, 59.0),
    ("Claude Son. 4.5",  25.5,   3.8,  64.3, 60.7),
    ("Claude Opus 4",    22.5,   1.3,  63.3, 60.5),
    ("Claude Opus 4.5",  40.0,   7.8,  61.4, 60.5),
    ("Claude Haiku 4.5", 14.3,   1.3,  62.8, 59.4),
    ("Gemini 3 Pro",     75.0,  31.1,  63.3, 61.3),
    ("Gemini 2.5 Pro",   33.0,   3.8,  63.8, 61.0),
    ("Gemini 2.5 Flash", 33.3,   1.7,  63.3, 60.5),
    ("Gemini 1.5 Pro",   None,   0.8,  60.1, 57.5),
    ("DeepSeek R1",      15.8,   1.3,  63.0, 59.3),
    ("DeepSeek V3",      57.0,   4.0,  61.9, 59.1),
    ("Llama 4 Maverick",  4.4,   0.0,  60.1, 57.1),
    ("Llama 4 Scout",     0.5,   0.0,  60.1, 54.2),
    ("Kimi K2.5",        65.3,  11.8,  61.7, 59.8),
    ("Qwen3-235B",       11.0,   1.3,  61.3, 60.1),
    ("Grok 4",           66.7,  16.0,  63.3, 60.4),
]


def provider_color(name: str) -> str:
    if any(k in name for k in ("GPT", "o3", "o4")):
        return "#10a37f"
    if "Claude" in name:
        return "#d97706"
    if "Gemini" in name:
        return "#4285f4"
    return "#6b7280"


def run_loo(X, y, names, target_label, filename):
    """LOO cross-validation: RF vs Linear. Print stats, save actual-vs-predicted chart."""
    loo = LeaveOneOut()

    # Random Forest
    rf = RandomForestRegressor(n_estimators=500, max_features=1.0, random_state=42)
    rf_preds = cross_val_predict(rf, X, y, cv=loo)

    # Linear baseline
    lr = LinearRegression()
    lr_preds = cross_val_predict(lr, X, y, cv=loo)

    # Naive baseline (predict mean)
    naive_preds = np.full_like(y, y.mean())

    print(f"\n{'=' * 60}")
    print(f"  {target_label}  (LOO, N={len(y)})")
    print(f"{'=' * 60}")
    for label, preds in [("Random Forest", rf_preds), ("Linear", lr_preds), ("Mean baseline", naive_preds)]:
        mae = mean_absolute_error(y, preds)
        rmse = root_mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        print(f"  {label:16s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:+.3f}")

    # Feature importance from full-data fit
    rf_full = RandomForestRegressor(n_estimators=500, max_features=1.0, random_state=42)
    rf_full.fit(X, y)

    # ── Chart: actual vs predicted ──
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [provider_color(n) for n in names]

    ax.scatter(y, rf_preds, c=colors, s=60, zorder=3, edgecolors="white", linewidth=0.5,
               label="Random Forest")
    ax.scatter(y, lr_preds, c=colors, s=30, zorder=2, marker="^", alpha=0.5,
               edgecolors="white", linewidth=0.5, label="Linear")

    # Perfect prediction line
    lo, hi = min(y.min(), rf_preds.min()) - 0.5, max(y.max(), rf_preds.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.25, lw=1, label="Perfect")

    for i, n in enumerate(names):
        ax.annotate(n, (y[i], rf_preds[i]), fontsize=6, xytext=(5, 4),
                    textcoords="offset points", alpha=0.8)

    rf_mae = mean_absolute_error(y, rf_preds)
    rf_r2 = r2_score(y, rf_preds)
    lr_mae = mean_absolute_error(y, lr_preds)
    lr_r2 = r2_score(y, lr_preds)

    box = (
        f"Random Forest  MAE={rf_mae:.2f}  R²={rf_r2:+.3f}\n"
        f"Linear         MAE={lr_mae:.2f}  R²={lr_r2:+.3f}\n"
        f"LOO CV, N={len(y)}"
    )
    ax.text(0.02, 0.98, box, transform=ax.transAxes, fontsize=9,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    from matplotlib.lines import Line2D
    legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor="gray", ms=8, label="RF pred"),
        Line2D([], [], marker="^", color="w", markerfacecolor="gray", ms=7, label="Linear pred"),
        Line2D([], [], ls="--", color="k", alpha=0.25, label="Perfect"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#10a37f", ms=8, label="OpenAI"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#d97706", ms=8, label="Anthropic"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#4285f4", ms=8, label="Google"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#6b7280", ms=8, label="Other"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import ScalarFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain")

    ax.set_xlabel(f"Actual {target_label}", fontsize=12)
    ax.set_ylabel(f"LOO Predicted {target_label}", fontsize=12)
    ax.set_title(f"ARC-AGI-2 → {target_label} (LOO Cross-Validation)", fontsize=13)
    ax.grid(True, alpha=0.15, which="both")
    fig.tight_layout()

    path = RESULTS / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart → {path}")

    return rf_preds, lr_preds


def main():
    names = [d[0] for d in DATA]
    arc2 = np.array([d[2] for d in DATA], dtype=float)
    fb_overall = np.array([d[3] for d in DATA], dtype=float)
    fb_dataset = np.array([d[4] for d in DATA], dtype=float)

    X = arc2.reshape(-1, 1)

    run_loo(X, fb_dataset, names, "FB Dataset (Brier Index)", "rf_arcagi2_fb_dataset.png")
    run_loo(X, fb_overall, names, "FB Overall (Brier Index)", "rf_arcagi2_fb_overall.png")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
