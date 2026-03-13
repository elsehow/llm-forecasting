#!/usr/bin/env python
"""Cross-benchmark correlations with ForecastBench.

Tests whether other capability benchmarks predict forecasting ability,
and compares against ARC-AGI-2.

Benchmarks:
  - GPQA Diamond (graduate-level science reasoning)
  - Chatbot Arena ELO (overall perceived quality)
  - SWE-bench Verified (software engineering / coding)

Usage:
    uv run python experiments/benchmark-correlations/multi_benchmark_correlation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from matplotlib.lines import Line2D

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── Model data ───────────────────────────────────────────────────────────
# (name, arc2%, fb_overall, fb_dataset, gpqa_diamond%, arena_elo, swebench%, gdpval_elo, eci)
# None = not available. Scores at default reasoning levels where possible.
# GDPval-AA ELO from artificialanalysis.ai (non-reasoning/default variant).
# ECI from Epoch AI benchmark data (epoch.ai/benchmarks). Same score across thinking levels.
DATA = [
    # name                arc2   fb_ov  fb_ds  gpqa   arena   swe    gdpval  eci
    ("GPT-4o",            0.0,  59.7,  56.0,  53.6,  1346,   33.2,  435,    None),
    ("GPT-4.1",           0.4,  63.3,  60.1,  66.3,  1413,   54.6,  811,    137.43),
    ("GPT-4.5",           0.8,  64.3,  60.3,  71.4,  1444,   38.0,  None,   137.25),
    ("GPT-5",             7.5,  62.5,  61.8,  87.3,  1430,   74.9,  1011,   150.00),  # medium (default)
    ("GPT-5 Mini",        4.0,  62.9,  60.8,  None,  1375,   None,  1028,   144.32),  # medium (default)
    ("GPT-5 Nano",        0.9,  60.4,  57.6,  None,  None,   None,  646,    139.88),  # medium (default)
    ("GPT-5.1",           0.4,  59.0,  58.7,  88.1,  1437,   None,  1000,   149.74),  # non-reasoning (default)
    ("o3",                3.0,  64.8,  60.4,  83.3,  1432,   69.1,  757,    146.62),
    ("o3-mini",           2.1,  62.3,  58.9,  77.0,  1348,   49.3,  787,    141.11),
    ("o4-mini",           2.4,  63.8,  60.7,  81.4,  1391,   68.1,  1015,   145.06),
    ("Claude 3.7",        0.0,  64.6,  60.4,  68.0,  1371,   63.7,  1069,   141.62),  # non-reasoning
    ("Claude Son. 4",     1.3,  61.7,  59.0,  76.1,  1389,   72.7,  1172,   142.33),  # non-reasoning
    ("Claude Son. 4.5",   3.8,  64.3,  60.7,  83.4,  1450,   77.2,  1319,   146.61),  # non-reasoning
    ("Claude Opus 4",     1.3,  63.3,  60.5,  79.6,  1413,   72.5,  None,   143.05),
    ("Claude Opus 4.5",   7.8,  61.4,  60.5,  87.0,  1467,   80.9,  1416,   150.05),  # non-reasoning
    ("Claude Haiku 4.5",  1.3,  62.8,  59.4,  None,  1406,   73.3,  1167,   140.66),  # non-reasoning
    ("Gemini 3 Pro",     31.1,  63.3,  61.3,  91.9,  1486,   None,  1177,   154.22),  # preview (low)
    ("Gemini 2.5 Pro",    3.8,  63.8,  61.0,  84.0,  1449,   63.8,  930,    146.06),
    ("Gemini 2.5 Flash",  1.7,  63.3,  60.5,  78.3,  1410,   60.4,  762,    None),    # not in ECI dataset
    ("Gemini 1.5 Pro",    0.8,  60.1,  57.5,  61.0,  1351,   None,  None,   132.56),
    ("DeepSeek R1",       1.3,  63.0,  59.3,  71.5,  1398,   49.2,  300,    141.75),
    ("DeepSeek V3",       4.0,  61.9,  59.1,  59.1,  1358,   66.2,  474,    132.96),  # original V3
    ("Llama 4 Maverick",  0.0,  60.1,  57.1,  69.8,  1327,   None,  490,    132.81),
    ("Llama 4 Scout",     0.0,  60.1,  54.2,  57.2,  1322,   None,  337,    130.06),
    ("Kimi K2.5",        11.8,  61.7,  59.8,  87.6,  1449,   76.8,  1287,   147.68),  # non-reasoning
    ("Qwen3-235B",        1.3,  61.3,  60.1,  84.0,  1375,   69.6,  None,   139.34),
    ("Grok 4",           16.0,  63.3,  60.4,  88.0,  1463,   73.0,  988,    147.07),
]


def provider_color(name: str) -> str:
    if any(k in name for k in ("GPT", "o3", "o4")):
        return "#10a37f"
    if "Claude" in name:
        return "#d97706"
    if "Gemini" in name:
        return "#4285f4"
    return "#6b7280"


def correlate_and_plot(x, y, names, xlabel, ylabel, title, filename):
    """Compute Spearman + Pearson, scatter plot with log x-axis."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    rho, p_sp = stats.spearmanr(x, y)
    r, p_pe = stats.pearsonr(x, y)

    print(f"  {title:50s}  N={len(x):2d}  ρ={rho:+.3f} (p={p_sp:.4g})  r={r:+.3f} (p={p_pe:.4g})")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [provider_color(n) for n in names]
    ax.scatter(x, y, c=colors, s=60, zorder=3, edgecolors="white", linewidth=0.5)

    for i, n in enumerate(names):
        ax.annotate(n, (x[i], y[i]), fontsize=6.5,
                    xytext=(5, 4), textcoords="offset points", alpha=0.85)

    # regression line
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.linspace(x.min() - 1, x.max() + 1, 200)
    ax.plot(xl, slope * xl + intercept, "k--", alpha=0.25, lw=1)

    box = (
        f"Spearman ρ = {rho:+.3f}  (p = {p_sp:.3g})\n"
        f"Pearson  r = {r:+.3f}  (p = {p_pe:.3g})\n"
        f"N = {len(x)}"
    )
    ax.text(0.02, 0.98, box, transform=ax.transAxes, fontsize=9,
            va="top", bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.15, which="both")

    legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#10a37f", ms=8, label="OpenAI"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#d97706", ms=8, label="Anthropic"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#4285f4", ms=8, label="Google"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#6b7280", ms=8, label="Other"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = RESULTS / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return rho, p_sp, r, p_pe, len(x)


def main():
    names = [d[0] for d in DATA]
    arc2 = [d[1] for d in DATA]
    fb_overall = [d[2] for d in DATA]
    fb_dataset = [d[3] for d in DATA]
    gpqa = [d[4] for d in DATA]
    arena = [d[5] for d in DATA]
    swe = [d[6] for d in DATA]
    gdpval = [d[7] for d in DATA]
    eci = [d[8] for d in DATA]

    print(f"\n{'=' * 100}")
    print(f"  Multi-Benchmark Correlations with ForecastBench")
    print(f"{'=' * 100}")

    results = []

    # For each benchmark × FB target, filter to non-None pairs
    benchmarks = [
        ("ARC-AGI-2 (%)", arc2, "arcagi2"),
        ("GPQA Diamond (%)", gpqa, "gpqa"),
        ("Chatbot Arena ELO", arena, "arena"),
        ("SWE-bench Verified (%)", swe, "swe"),
        ("GDPval-AA ELO", gdpval, "gdpval"),
        ("ECI (Epoch)", eci, "eci"),
    ]
    targets = [
        ("FB Dataset (Brier Index)", fb_dataset, "fb_dataset"),
        ("FB Overall (Brier Index)", fb_overall, "fb_overall"),
    ]

    for bname, bvals, bslug in benchmarks:
        for tname, tvals, tslug in targets:
            # Filter to models with both benchmark and target values
            mask = [i for i in range(len(DATA))
                    if bvals[i] is not None and tvals[i] is not None]
            if len(mask) < 5:
                print(f"  {bname} vs {tname}: skipped (N={len(mask)})")
                continue

            x = [bvals[i] for i in mask]
            y = [tvals[i] for i in mask]
            ns = [names[i] for i in mask]
            fn = f"{bslug}_vs_{tslug}.png"
            title = f"{bname.split(' (')[0]} vs {tname.split(' (')[0]}"

            rho, p_sp, r, p_pe, n = correlate_and_plot(
                x, y, ns, bname, tname, title, fn
            )
            results.append((bname.split(" (")[0], tname.split(" (")[0], n, rho, p_sp, r, p_pe))

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"  Summary")
    print(f"{'=' * 100}")
    print(f"  {'Benchmark':<22s} {'Target':<14s} {'N':>3s} {'Spearman ρ':>11s} {'p':>8s} {'Pearson r':>10s} {'p':>8s}")
    print(f"  {'-'*22} {'-'*14} {'-'*3} {'-'*11} {'-'*8} {'-'*10} {'-'*8}")
    for bname, tname, n, rho, p_sp, r, p_pe in results:
        sig = "***" if p_sp < 0.001 else "**" if p_sp < 0.01 else "*" if p_sp < 0.05 else ""
        print(f"  {bname:<22s} {tname:<14s} {n:3d} {rho:+11.3f} {p_sp:8.4f} {r:+10.3f} {p_pe:8.4f} {sig}")

    print(f"\n  Charts saved to {RESULTS}/")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
