#!/usr/bin/env python
"""ARC-AGI vs ForecastBench correlation analysis.

Computes Spearman and Pearson correlations between ARC-AGI scores and
ForecastBench scores for all model families evaluated on both benchmarks.

Methodology:
  - Each data point = one model family (e.g., "GPT-5", "Claude Opus 4.5")
  - ARC-AGI: best single-system CoT or Base LLM score (excludes Refinement scaffolds)
  - ForecastBench: best Overall/Dataset score across prompting strategies (zero-shot, scratchpad)

Usage:
    uv run python experiments/benchmark-correlations/arcagi_fb_correlation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


# ── Matched model data ──────────────────────────────────────────────────────
# (name, arc_agi_1%, arc_agi_2%, fb_overall, fb_dataset)
# ARC %: percentage correct (0-100). None if N/A.
# FB scores: Brier Index (0-100, higher = better).
#
# ARC: score at DEFAULT reasoning level (matching what FB evaluates).
#   - GPT-5/Mini/Nano: default=medium. GPT-5.1/5.2: default=none.
#   - o3, o3-mini, o4-mini: default=medium.
#   - Claude: default=none (no extended thinking).
#   - Google: ARC-AGI base entry (thinking off or default level).
#   - Others: single ARC entry or base model.
# FB: best Overall/Dataset score from the row with the highest Overall.

DATA = [
    # ── OpenAI ──
    ("GPT-4o",            4.5,   0.0,  59.7, 56.0),   # Base LLM → GPT-4o-2024-05-13 (zero shot)
    ("GPT-4.1",           5.5,   0.4,  63.3, 60.1),   # Base LLM → GPT-4.1-2025-04-14 (scratchpad)
    ("GPT-4.5",          10.3,   0.8,  64.3, 60.3),   # Base LLM → GPT-4.5-Preview-2025-02-27 (zero shot)
    ("GPT-5",            56.2,   7.5,  62.5, 61.8),   # GPT-5 (Medium) default → GPT-5-2025-08-07 (zero shot)
    ("GPT-5 Mini",       37.3,   4.0,  62.9, 60.8),   # GPT-5 Mini (Medium) default → GPT-5-Mini-2025-08-07 (zero shot)
    ("GPT-5 Nano",       20.7,   0.9,  60.4, 57.6),   # GPT-5 Nano (Medium) default → GPT-5-Nano-2025-08-07 (zero shot)
    ("GPT-5.1",           5.8,   0.4,  59.0, 58.7),   # GPT-5.1 (Thinking, None) default → GPT-5.1-2025-11-13 (zero shot)
    ("o3",               53.8,   3.0,  64.8, 60.4),   # o3 (Medium) default → O3-2025-04-16 (scratchpad)
    ("o3-mini",          22.3,   2.1,  62.3, 58.9),   # o3-mini (Medium) default → O3-Mini-2025-01-31 (zero shot)
    ("o4-mini",          41.8,   2.4,  63.8, 60.7),   # o4-mini (Medium) default → O4-Mini-2025-04-16 (scratchpad)

    # ── Anthropic (default = no extended thinking) ──
    ("Claude 3.7",       13.6,   0.0,  64.6, 60.4),   # Claude 3.7 base → Claude-3-7-Sonnet-20250219 (scratchpad)
    ("Claude Son. 4",    23.8,   1.3,  61.7, 59.0),   # Claude Sonnet 4 base → Claude-Sonnet-4-20250514 (zero shot)
    ("Claude Son. 4.5",  25.5,   3.8,  64.3, 60.7),   # Claude Sonnet 4.5 base → Claude-Sonnet-4-5-20250929 (zero shot)
    ("Claude Opus 4",    22.5,   1.3,  63.3, 60.5),   # Claude Opus 4 base → Claude-Opus-4-1-20250805 (zero shot)
    ("Claude Opus 4.5",  40.0,   7.8,  61.4, 60.5),   # Opus 4.5 (Thinking, None) → Claude-Opus-4-5-20251101 (zero shot)
    ("Claude Haiku 4.5", 14.3,   1.3,  62.8, 59.4),   # Claude Haiku 4.5 base → Claude-Haiku-4-5-20251001 (zero shot)

    # ── Google (base/default level) ──
    ("Gemini 3 Pro",     75.0,  31.1,  63.3, 61.3),   # Single entry, default → Gemini-3-Pro-Preview (zero shot)
    ("Gemini 2.5 Pro",   33.0,   3.8,  63.8, 61.0),   # Gemini 2.5 Pro (Preview) base → Gemini-2.5-Pro-Preview-03-25 (zero shot)
    ("Gemini 2.5 Flash", 33.3,   1.7,  63.3, 60.5),   # Gemini 2.5 Flash (Preview) base → Gemini-2.5-Flash-Preview-04-17 (zero shot)
    ("Gemini 1.5 Pro",   None,   0.8,  60.1, 57.5),   # ARC-AGI-1 N/A → Gemini-1.5-Pro (scratchpad)

    # ── Other ──
    ("DeepSeek R1",      15.8,   1.3,  63.0, 59.3),   # Deepseek R1 CoT → DeepSeek-R1 (scratchpad)
    ("DeepSeek V3",      57.0,   4.0,  61.9, 59.1),   # Deepseek V3.2 Base → DeepSeek-V3 (scratchpad). Slight version mismatch.
    ("Llama 4 Maverick",  4.4,   0.0,  60.1, 57.1),   # Base LLM → Llama-4-Maverick (scratchpad)
    ("Llama 4 Scout",     0.5,   0.0,  60.1, 54.2),   # Base LLM → Llama-4-Scout (zero shot)
    ("Kimi K2.5",        65.3,  11.8,  61.7, 59.8),   # Kimi K2.5 CoT → Kimi-K2-Instruct-0905 (zero shot)
    ("Qwen3-235B",       11.0,   1.3,  61.3, 60.1),   # Base LLM → Qwen3-235B-A22B (zero shot)
    ("Grok 4",           66.7,  16.0,  63.3, 60.4),   # Grok 4 (Thinking) → Grok-4-1-Fast-Reasoning (zero shot)
]


def provider_color(name: str) -> str:
    if any(k in name for k in ("GPT", "o3", "o4")):
        return "#10a37f"  # OpenAI green
    if "Claude" in name:
        return "#d97706"  # Anthropic amber
    if "Gemini" in name:
        return "#4285f4"  # Google blue
    return "#6b7280"      # gray


def plot_correlation(x, y, names, xlabel, ylabel, title, filename):
    """Compute correlations, print stats, save scatter plot."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    rho, p_sp = stats.spearmanr(x, y)
    r, p_pe = stats.pearsonr(x, y)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  N         = {len(x)}")
    print(f"  Spearman  = {rho:+.3f}   p = {p_sp:.4g}")
    print(f"  Pearson   = {r:+.3f}   p = {p_pe:.4g}")
    print(f"  R²        = {r**2:.3f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [provider_color(n) for n in names]
    ax.scatter(x, y, c=colors, s=60, zorder=3, edgecolors="white", linewidth=0.5)

    for i, n in enumerate(names):
        ax.annotate(n, (x[i], y[i]), fontsize=6.5,
                    xytext=(5, 4), textcoords="offset points", alpha=0.85)

    # regression line (in linear space, drawn as curve on log axis)
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.linspace(max(x.min() - 0.5, 0), x.max() + 3, 200)
    ax.plot(xl, slope * xl + intercept, "k--", alpha=0.25, lw=1)

    # log scale on ARC axis (symlog handles zeros)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_yscale("log")
    # y-axis: keep readable tick labels in the narrow FB range
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.ticklabel_format(axis="y", style="plain")

    # stats box
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
    print(f"  Chart → {path}")
    return rho, p_sp, r, p_pe


def main():
    names      = [d[0] for d in DATA]
    arc1       = [d[1] for d in DATA]
    arc2       = [d[2] for d in DATA]
    fb_overall = [d[3] for d in DATA]
    fb_dataset = [d[4] for d in DATA]

    # ── ARC-AGI-2 correlations ──
    plot_correlation(
        arc2, fb_overall, names,
        "ARC-AGI-2 (%)", "ForecastBench Overall (Brier Index)",
        "ARC-AGI-2 vs ForecastBench Overall", "arcagi2_vs_fb_overall.png",
    )
    plot_correlation(
        arc2, fb_dataset, names,
        "ARC-AGI-2 (%)", "ForecastBench Dataset (Brier Index)",
        "ARC-AGI-2 vs ForecastBench Dataset", "arcagi2_vs_fb_dataset.png",
    )

    # ── ARC-AGI-1 correlations (exclude N/A) ──
    mask = [i for i, a in enumerate(arc1) if a is not None]
    plot_correlation(
        [arc1[i] for i in mask], [fb_overall[i] for i in mask],
        [names[i] for i in mask],
        "ARC-AGI-1 (%)", "ForecastBench Overall (Brier Index)",
        "ARC-AGI-1 vs ForecastBench Overall", "arcagi1_vs_fb_overall.png",
    )
    plot_correlation(
        [arc1[i] for i in mask], [fb_dataset[i] for i in mask],
        [names[i] for i in mask],
        "ARC-AGI-1 (%)", "ForecastBench Dataset (Brier Index)",
        "ARC-AGI-1 vs ForecastBench Dataset", "arcagi1_vs_fb_dataset.png",
    )

    print(f"\n{'=' * 60}")
    print("  Done. All charts saved to results/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
