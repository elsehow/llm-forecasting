"""
Key comparison figure: CivBench binary vs continuous vs ForecastBench binary
Shows that the anti-g reversal is specific to continuous scoring, not dynamics.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

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

# Data from analyses
# CivBench continuous MAE × ECI (from tail-risk-blindness)
cb_cont_rho = [-0.619, +0.881, +0.643, +0.667, +0.786, +0.643]
cb_cont_labels = ["T90", "T120", "T150", "T180", "T210", "T240"]

# CivBench binary Brier × ECI (just computed)
cb_bin_rho = [-0.353, -0.547, -0.547, -0.650, -0.207, -0.650]
cb_bin_labels = cb_cont_labels

# ForecastBench binary Brier × ECI (from Study B)
fb_rho = [-0.799, -0.752, -0.705, -0.496, -0.382]
fb_labels = ["H1\n17d", "H2\n40d", "H3\n100d", "H4\n190d", "H5\n375d"]

fig, ax = plt.subplots(figsize=(9, 5.5))

x_cb = np.arange(len(cb_cont_rho))
x_fb = np.linspace(0, len(cb_cont_rho) - 1, len(fb_rho))

# Plot three series
ax.plot(x_cb, cb_cont_rho, "s-", color="#dc2626", linewidth=2.5, markersize=10,
        label="CivBench continuous MAE × ECI", zorder=5)
ax.plot(x_cb, cb_bin_rho, "D-", color="#7c3aed", linewidth=2.5, markersize=9,
        label="CivBench binary Brier × ECI", zorder=5)
ax.plot(x_fb, fb_rho, "o-", color="#2563eb", linewidth=2.5, markersize=9,
        label="ForecastBench binary Brier × ECI", zorder=5)

ax.axhline(0, color="gray", linewidth=1.2, linestyle="-", alpha=0.5)

# Shade regions
ax.axhspan(0, 1.05, alpha=0.04, color="#dc2626")
ax.axhspan(-1.05, 0, alpha=0.04, color="#2563eb")

ax.set_xticks(x_cb)
ax.set_xticklabels(cb_cont_labels)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("Spearman ρ (ECI vs error)")
ax.set_ylim(-1.05, 1.05)
ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

ax.annotate("anti-g (smarter = worse)", xy=(0.98, 0.97), xycoords="axes fraction",
            fontsize=9, color="#dc2626", alpha=0.5, ha="right", va="top")
ax.annotate("pro-g (smarter = better)", xy=(0.98, 0.03), xycoords="axes fraction",
            fontsize=9, color="#2563eb", alpha=0.5, ha="right", va="bottom")

ax.set_title("The anti-g reversal is a continuous scoring phenomenon\nBinary Brier stays pro-g on both CivBench and ForecastBench")

fig.tight_layout()
fig.savefig(OUT_DIR / "fb_civbench_binary_vs_continuous_rho.png", dpi=180)
print("Saved: fb_civbench_binary_vs_continuous_rho.png")
plt.close(fig)
