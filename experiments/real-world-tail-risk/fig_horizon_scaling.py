"""
Figure: LLM intervals vs market intervals across horizons.

Shows that LLMs roughly match the market at short horizons but fail to
widen as fast, falling behind by H3. Includes both quiet-period average
and crisis window.
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- constants ----------
z90 = 1.2816

windows_meta = {
    "Sep 2025": {"cutoff_price": 63.30, "ovx": 31.0},
    "Oct 2025": {"cutoff_price": 57.52, "ovx": 36.8},
    "Nov 2025": {"cutoff_price": 58.84, "ovx": 36.3},
    "Jan 2026": {"cutoff_price": 59.50, "ovx": 38.8},
}
crisis_meta = {"cutoff_price": 62.89, "ovx": 42.2}
hz_days = {"H1": 7, "H2": 14, "H3": 21}


def mkt_width(price, ovx_pct, days):
    sigma = ovx_pct / 100
    t = days / 365
    p10 = price * math.exp(-z90 * sigma * math.sqrt(t))
    p90 = price * math.exp(+z90 * sigma * math.sqrt(t))
    return p90 - p10


# ---------- load data ----------
with open("results/quiet_period_analysis.json") as f:
    data = json.load(f)

# Compute per-horizon averages for quiet periods
quiet_by_hz = {"H1": {"mkt": [], "llm": []}, "H2": {"mkt": [], "llm": []}, "H3": {"mkt": [], "llm": []}}
for row in data:
    if row["type"] != "quiet":
        continue
    w = row["window"]
    hz = row["horizon"]
    d = hz_days[hz]
    meta = windows_meta[w]
    mw = mkt_width(meta["cutoff_price"], meta["ovx"], d)
    llm_p10s = [m["p10"] for m in row["models"] if m.get("p10") is not None]
    llm_p90s = [m["p90"] for m in row["models"] if m.get("p90") is not None]
    lw = np.mean(llm_p90s) - np.mean(llm_p10s)
    quiet_by_hz[hz]["mkt"].append(mw)
    quiet_by_hz[hz]["llm"].append(lw)

# Crisis
crisis_by_hz = {}
for row in data:
    if row["type"] != "crisis":
        continue
    hz = row["horizon"]
    d = hz_days[hz]
    mw = mkt_width(crisis_meta["cutoff_price"], crisis_meta["ovx"], d)
    llm_p10s = [m["p10"] for m in row["models"] if m.get("p10") is not None]
    llm_p90s = [m["p90"] for m in row["models"] if m.get("p90") is not None]
    lw = np.mean(llm_p90s) - np.mean(llm_p10s)
    crisis_by_hz[hz] = {"mkt": mw, "llm": lw}

# ---------- build arrays ----------
horizons = ["H1\n(+7d)", "H2\n(+14d)", "H3\n(+21d)"]
x = np.arange(len(horizons))

quiet_mkt = [np.mean(quiet_by_hz[h]["mkt"]) for h in ["H1", "H2", "H3"]]
quiet_llm = [np.mean(quiet_by_hz[h]["llm"]) for h in ["H1", "H2", "H3"]]
crisis_mkt = [crisis_by_hz[h]["mkt"] for h in ["H1", "H2", "H3"]]
crisis_llm = [crisis_by_hz[h]["llm"] for h in ["H1", "H2", "H3"]]

# ---------- plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

bar_width = 0.32
colors_mkt = "#4A90D9"
colors_llm = "#D94A4A"

# Quiet panel
bars1 = ax1.bar(x - bar_width / 2, quiet_mkt, bar_width, label="Market (OVX)", color=colors_mkt, alpha=0.85)
bars2 = ax1.bar(x + bar_width / 2, quiet_llm, bar_width, label="LLM (avg of 28)", color=colors_llm, alpha=0.85)

# Add ratio labels
for i in range(len(horizons)):
    ratio = quiet_llm[i] / quiet_mkt[i]
    y_pos = max(quiet_mkt[i], quiet_llm[i]) + 0.3
    ax1.text(x[i], y_pos, f"{ratio:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_title("Calm markets\n(avg of 4 quiet windows)", fontsize=12)
ax1.set_xlabel("Forecast horizon")
ax1.set_ylabel("p10–p90 interval width ($)")
ax1.set_xticks(x)
ax1.set_xticklabels(horizons)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 20)

# Crisis panel
bars3 = ax2.bar(x - bar_width / 2, crisis_mkt, bar_width, label="Market (OVX)", color=colors_mkt, alpha=0.85)
bars4 = ax2.bar(x + bar_width / 2, crisis_llm, bar_width, label="LLM (avg of 27)", color=colors_llm, alpha=0.85)

for i in range(len(horizons)):
    ratio = crisis_llm[i] / crisis_mkt[i]
    y_pos = max(crisis_mkt[i], crisis_llm[i]) + 0.3
    ax2.text(x[i], y_pos, f"{ratio:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.set_title("Hormuz crisis\n(Feb 2026)", fontsize=12)
ax2.set_xlabel("Forecast horizon")
ax2.set_xticks(x)
ax2.set_xticklabels(horizons)
ax2.legend(fontsize=9)

fig.suptitle("LLM vs market interval width by horizon", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = "/Users/elsehow/Projects/fri-vault/_artifacts/static/realworld_horizon_scaling.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
