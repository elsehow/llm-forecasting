"""Test Opus 4.5 conditional forecasting for within-domain question ranking.

Hypothesis: ForecastBench shows frontier models are good forecasters. The
within-domain failure (r=-0.15 with Sonnet) may be a model capability issue,
not a fundamental limit. Opus 4.5's better forecasting ability may yield
better conditional probability estimates.

Key metric: Conditional Forecast Sensitivity = |P(T|Q=yes) - P(T|Q=no)|

This isolates forecasting capacity: can the model distinguish cruxes that
would cause large vs small belief updates?

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/russell-2000-crux/opus_conditional_forecast.py

Results saved to: experiments/russell-2000-crux/data/opus_experiment/
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Add package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "llm-forecasting" / "src"))

from llm_forecasting.voi import linear_voi, entropy_voi

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "opus_experiment"


# Use Opus 4.5 - the hypothesis is that its superior forecasting calibration
# (per ForecastBench) will produce better conditional probability estimates
OPUS_MODEL = "anthropic/claude-opus-4-20250514"


CONDITIONAL_FORECAST_PROMPT = """You are a forecaster estimating how a signal affects stock price direction.

STOCK: {ticker}
DATE: {date}
TARGET: Will {ticker} close higher than it opened on {date}?

POTENTIAL CRUX (signal that resolves before market close):
{crux}

You need to estimate three probabilities:

1. P(crux = YES): What's the probability the crux resolves YES?
2. P(stock up | crux = YES): If the crux resolves YES, what's the probability the stock closes higher than it opened?
3. P(stock up | crux = NO): If the crux resolves NO, what's the probability the stock closes higher than it opened?

Think carefully about:
- How does this crux causally relate to the stock's price movement?
- What magnitude of belief shift is appropriate given the crux's importance?
- Consider base rates: stocks go up ~52% of days on average

IMPORTANT: Be calibrated about the sensitivity. Most cruxes cause small belief shifts (5-15 percentage points). Only truly pivotal cruxes (like earnings beats/misses) cause large shifts (20-40 percentage points).

Respond with JSON only:
{{"p_crux_yes": <float 0-1>, "p_up_given_yes": <float 0-1>, "p_up_given_no": <float 0-1>, "reasoning": "<brief explanation>"}}"""


async def elicit_conditional_forecast(
    ticker: str,
    date: str,
    crux: str,
) -> dict:
    """Elicit P(crux), P(up|crux=yes), P(up|crux=no) from Opus 4.5.

    Returns:
        Dict with p_crux_yes, p_up_given_yes, p_up_given_no, reasoning
    """
    import litellm

    try:
        response = await litellm.acompletion(
            model=OPUS_MODEL,
            messages=[{
                "role": "user",
                "content": CONDITIONAL_FORECAST_PROMPT.format(
                    ticker=ticker,
                    date=date,
                    crux=crux,
                )
            }],
            max_tokens=500,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return {
            "p_crux_yes": float(result.get("p_crux_yes", 0.5)),
            "p_up_given_yes": float(result.get("p_up_given_yes", 0.5)),
            "p_up_given_no": float(result.get("p_up_given_no", 0.5)),
            "reasoning": result.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "p_crux_yes": 0.5,
            "p_up_given_yes": 0.5,
            "p_up_given_no": 0.5,
            "reasoning": "",
            "error": str(e),
        }


async def run_experiment(
    cruxes_df: pd.DataFrame,
    batch_size: int = 5,
) -> pd.DataFrame:
    """Run Opus 4.5 conditional forecast elicitation on all cruxes.

    Args:
        cruxes_df: DataFrame with ticker, date, crux columns
        batch_size: Number of concurrent API calls

    Returns:
        DataFrame with Opus 4.5 conditional forecast results
    """
    results = []
    total = len(cruxes_df)

    for i in range(0, total, batch_size):
        batch = cruxes_df.iloc[i:i+batch_size]
        print(f"  Processing {i+1}-{min(i+batch_size, total)} of {total}...")

        tasks = [
            elicit_conditional_forecast(
                row["ticker"],
                row["date"],
                row["crux"],
            )
            for _, row in batch.iterrows()
        ]

        forecast_results = await asyncio.gather(*tasks)

        for (_, row), forecast in zip(batch.iterrows(), forecast_results):
            # Compute sensitivity and VOI
            p_crux = forecast["p_crux_yes"]
            p_up_yes = forecast["p_up_given_yes"]
            p_up_no = forecast["p_up_given_no"]

            sensitivity = abs(p_up_yes - p_up_no)

            # Linear VOI using existing function
            # P(stock up) prior = 0.5 (symmetric)
            voi_linear = linear_voi(0.5, p_crux, p_up_yes, p_up_no)
            voi_entropy = entropy_voi(0.5, p_crux, p_up_yes, p_up_no)

            results.append({
                **row.to_dict(),
                "opus_p_crux_yes": p_crux,
                "opus_p_up_given_yes": p_up_yes,
                "opus_p_up_given_no": p_up_no,
                "opus_sensitivity": sensitivity,
                "opus_linear_voi": voi_linear,
                "opus_entropy_voi": voi_entropy,
                "opus_reasoning": forecast["reasoning"],
                "opus_error": forecast["error"],
            })

        # Rate limiting - Opus is expensive, be gentle
        await asyncio.sleep(1.0)

    return pd.DataFrame(results)


def analyze_results(
    results_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> dict:
    """Analyze correlation between Opus VOI/sensitivity and |return|.

    Args:
        results_df: DataFrame with Opus conditional forecast results
        returns_df: DataFrame with stock returns

    Returns:
        Dict with correlation results and interpretation
    """
    # Filter to earnings days
    earnings_returns = returns_df[returns_df["is_earnings_day"]].copy()

    # Aggregate Opus metrics per stock-day
    agg_metrics = results_df.groupby(["ticker", "date"]).agg({
        "opus_sensitivity": ["mean", "max"],
        "opus_linear_voi": ["mean", "max"],
        "opus_entropy_voi": ["mean", "max"],
        # Keep original Sonnet VOI for comparison
        "linear_voi": ["mean", "max"],
    }).reset_index()

    # Flatten column names
    agg_metrics.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in agg_metrics.columns
    ]

    # Merge with returns
    merged = agg_metrics.merge(
        earnings_returns[["ticker", "date", "return"]],
        on=["ticker", "date"],
        how="inner"
    )

    if len(merged) < 5:
        return {"error": f"Too few observations ({len(merged)})"}

    merged["abs_return"] = merged["return"].abs()

    # Compute correlations for all metrics
    metrics = [
        ("opus_sensitivity_mean", "Opus Sensitivity (mean)"),
        ("opus_sensitivity_max", "Opus Sensitivity (max)"),
        ("opus_linear_voi_mean", "Opus Linear VOI (mean)"),
        ("opus_linear_voi_max", "Opus Linear VOI (max)"),
        ("linear_voi_mean", "Sonnet Linear VOI (mean, baseline)"),
    ]

    results = {
        "n_observations": len(merged),
        "metrics": {},
    }

    for col, label in metrics:
        if col not in merged.columns:
            continue
        r, p = stats.pearsonr(merged[col], merged["abs_return"])
        rho, p_rho = stats.spearmanr(merged[col], merged["abs_return"])

        results["metrics"][col] = {
            "label": label,
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
        }

    # Determine verdict
    opus_r = results["metrics"].get("opus_linear_voi_mean", {}).get("pearson_r", 0)
    opus_p = results["metrics"].get("opus_linear_voi_mean", {}).get("pearson_p", 1)
    sonnet_r = results["metrics"].get("linear_voi_mean", {}).get("pearson_r", 0)

    if opus_r > 0.20 and opus_p < 0.10:
        verdict = "STRONG: Opus VOI predicts within-domain magnitude"
    elif opus_r > 0.10:
        verdict = "MODERATE: Opus beats Sonnet baseline"
    elif opus_r > 0:
        verdict = "WEAK: Correct direction but small effect"
    else:
        verdict = "FAIL: No improvement over baseline"

    results["verdict"] = verdict
    results["improvement_over_sonnet"] = float(opus_r - sonnet_r)
    results["merged_df"] = merged

    return results


def create_comparison_plot(results: dict, save_path: Path):
    """Create scatter plots comparing Opus vs Sonnet VOI."""
    if "error" in results or "merged_df" not in results:
        print("Cannot create plot - insufficient data")
        return

    df = results["merged_df"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Sonnet VOI (baseline)
    ax = axes[0]
    ax.scatter(df["linear_voi_mean"], df["abs_return"], alpha=0.6, s=50, c="gray")
    r = results["metrics"]["linear_voi_mean"]["pearson_r"]
    p = results["metrics"]["linear_voi_mean"]["pearson_p"]
    ax.set_xlabel("Sonnet Linear VOI (mean)")
    ax.set_ylabel("|Return| on Earnings Day")
    ax.set_title(f"Sonnet Baseline\nn={len(df)}, r={r:.3f}, p={p:.4f}")
    # Trend line
    z = np.polyfit(df["linear_voi_mean"], df["abs_return"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df["linear_voi_mean"].min(), df["linear_voi_mean"].max(), 100)
    ax.plot(x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2)

    # Plot 2: Opus VOI
    ax = axes[1]
    ax.scatter(df["opus_linear_voi_mean"], df["abs_return"], alpha=0.6, s=50, c="steelblue")
    r = results["metrics"]["opus_linear_voi_mean"]["pearson_r"]
    p = results["metrics"]["opus_linear_voi_mean"]["pearson_p"]
    ax.set_xlabel("Opus 4.5 Linear VOI (mean)")
    ax.set_ylabel("|Return| on Earnings Day")
    ax.set_title(f"Opus 4.5\nn={len(df)}, r={r:.3f}, p={p:.4f}")
    # Trend line
    z = np.polyfit(df["opus_linear_voi_mean"], df["abs_return"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df["opus_linear_voi_mean"].min(), df["opus_linear_voi_mean"].max(), 100)
    ax.plot(x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2)

    # Plot 3: Opus Sensitivity
    ax = axes[2]
    ax.scatter(df["opus_sensitivity_mean"], df["abs_return"], alpha=0.6, s=50, c="darkgreen")
    r = results["metrics"]["opus_sensitivity_mean"]["pearson_r"]
    p = results["metrics"]["opus_sensitivity_mean"]["pearson_p"]
    ax.set_xlabel("Opus Sensitivity |P(up|yes) - P(up|no)|")
    ax.set_ylabel("|Return| on Earnings Day")
    ax.set_title(f"Opus Sensitivity\nn={len(df)}, r={r:.3f}, p={p:.4f}")
    # Trend line
    z = np.polyfit(df["opus_sensitivity_mean"], df["abs_return"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df["opus_sensitivity_mean"].min(), df["opus_sensitivity_mean"].max(), 100)
    ax.plot(x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Opus 4.5 conditional forecasting for within-domain ranking"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="cruxes_with_voi.parquet",
        help="Input cruxes file with existing VOI (default: cruxes_with_voi.parquet)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cruxes for testing (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for API calls (default: 5)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 60)
    print("OPUS 4.5 CONDITIONAL SENSITIVITY EXPERIMENT")
    print("=" * 60)
    print(f"\nHypothesis: Opus 4.5's forecasting calibration (per ForecastBench)")
    print("can improve within-domain question ranking.")
    print(f"\nBaseline: Sonnet r=-0.15 (within-earnings VOI)")
    print(f"Model: {OPUS_MODEL}")

    # Load data
    cruxes_path = DATA_DIR / args.input
    returns_path = DATA_DIR / "stock_returns.parquet"

    if not cruxes_path.exists():
        print(f"\nError: {cruxes_path} not found. Run compute_voi.py first!")
        return
    if not returns_path.exists():
        print(f"\nError: {returns_path} not found. Run fetch_stock_data.py first!")
        return

    cruxes_df = pd.read_parquet(cruxes_path)
    returns_df = pd.read_parquet(returns_path)

    # Apply limit if specified
    if args.limit:
        cruxes_df = cruxes_df.head(args.limit)

    n_cruxes = len(cruxes_df)
    n_stocks = cruxes_df["ticker"].nunique()
    n_days = cruxes_df.groupby(["ticker", "date"]).ngroups

    print(f"\nData:")
    print(f"  Cruxes: {n_cruxes}")
    print(f"  Unique stocks: {n_stocks}")
    print(f"  Stock-days: {n_days}")

    # Estimate cost
    # ~600 input tokens + 200 output tokens per crux
    # Opus: $15/M input, $75/M output
    input_cost = (n_cruxes * 600 / 1_000_000) * 15
    output_cost = (n_cruxes * 200 / 1_000_000) * 75
    total_cost = input_cost + output_cost
    print(f"\nEstimated cost: ${total_cost:.2f}")

    # Run experiment
    print("\n" + "=" * 60)
    print("RUNNING OPUS CONDITIONAL FORECASTS")
    print("=" * 60)

    results_df = await run_experiment(cruxes_df, batch_size=args.batch_size)

    # Check for errors
    n_errors = results_df["opus_error"].notna().sum()
    if n_errors > 0:
        print(f"\nWarning: {n_errors} API errors occurred")

    # Save raw results
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_parquet(OUTPUT_DIR / "opus_cruxes.parquet", index=False)
    print(f"\nSaved raw results to opus_cruxes.parquet")

    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    analysis = analyze_results(results_df, returns_df)

    if "error" in analysis:
        print(f"\nError: {analysis['error']}")
        return

    print(f"\nObservations: {analysis['n_observations']}")

    print("\nCorrelation with |return|:")
    for col, metrics in analysis["metrics"].items():
        print(f"\n  {metrics['label']}:")
        print(f"    Pearson r = {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")
        print(f"    Spearman ρ = {metrics['spearman_rho']:.4f}")

    print(f"\nImprovement over Sonnet: Δr = {analysis['improvement_over_sonnet']:.4f}")
    print(f"\nVerdict: {analysis['verdict']}")

    # Create visualization
    create_comparison_plot(analysis, OUTPUT_DIR / "opus_vs_sonnet.png")

    # Save analysis results (without DataFrame)
    analysis_to_save = {k: v for k, v in analysis.items() if k != "merged_df"}
    analysis_to_save["timestamp"] = datetime.now().isoformat()
    analysis_to_save["model"] = OPUS_MODEL
    analysis_to_save["n_cruxes"] = n_cruxes

    with open(OUTPUT_DIR / "opus_analysis.json", "w") as f:
        json.dump(analysis_to_save, f, indent=2)

    print(f"\nSaved analysis to opus_analysis.json")

    # Print summary for easy copy-paste
    print("\n" + "=" * 60)
    print("SUMMARY FOR PROJECT NOTES")
    print("=" * 60)

    opus_r = analysis["metrics"].get("opus_linear_voi_mean", {}).get("pearson_r", 0)
    opus_p = analysis["metrics"].get("opus_linear_voi_mean", {}).get("pearson_p", 1)
    sonnet_r = analysis["metrics"].get("linear_voi_mean", {}).get("pearson_r", 0)

    print(f"""
| Approach | Result | Source |
|----------|--------|--------|
| **Opus 4.5 conditional** | r={opus_r:.3f}, p={opus_p:.3f} | [[2026-01-26-opus-conditional-sensitivity-experiment]] |
| Sonnet baseline | r={sonnet_r:.3f} | [[2026-01-26-within-earnings-voi-experiment]] |
""")


if __name__ == "__main__":
    asyncio.run(main())
