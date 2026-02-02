"""Analyze backtest results and compare to baseline.

Phase 4: Analysis
1. Compute metrics: total return, Sharpe, max drawdown, win rate
2. Generate random baseline (bootstrap 1000 samples)
3. Statistical comparison (p-value)
4. Parameter sensitivity analysis
5. Output report with success criteria evaluation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from llm_forecasting.market_data.storage import MarketDataStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"


async def generate_random_baseline(
    storage: MarketDataStorage,
    n_traders: int,
    n_samples: int = 1000,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict]:
    """Generate baseline by sampling random traders.

    For each sample:
    1. Select N random traders
    2. Compute their combined performance
    3. Return distribution of results
    """
    all_traders = await storage.get_tracked_traders()

    if len(all_traders) < n_traders:
        logger.warning(f"Only {len(all_traders)} traders available, need {n_traders}")
        return []

    samples = []

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            logger.info(f"Bootstrap sample {i+1}/{n_samples}")

        # Random selection
        selected = np.random.choice(all_traders, size=n_traders, replace=False)

        # Compute combined stats
        total_pnl = 0.0
        total_volume = 0.0
        total_trades = 0

        for trader in selected:
            activities = await storage.get_trader_activity(
                trader,
                start=start_date,
                end=end_date,
                activity_types=["TRADE"],
            )

            for activity in activities:
                total_trades += 1
                if activity.usdc_size:
                    total_volume += activity.usdc_size
                if activity.price and activity.size:
                    # Simplified PNL estimate
                    trade_value = activity.price * activity.size
                    if activity.side == "BUY":
                        total_pnl += (activity.size - trade_value) * 0.5
                    else:
                        total_pnl += trade_value * 0.5

        samples.append({
            "pnl": total_pnl,
            "volume": total_volume,
            "n_trades": total_trades,
        })

    return samples


def compute_significance(strategy_return: float, baseline_returns: list[float]) -> dict:
    """Compute statistical significance of strategy vs baseline."""
    baseline_arr = np.array(baseline_returns)

    # One-sided p-value: what fraction of baseline returns exceed strategy return?
    p_value = np.mean(baseline_arr >= strategy_return)

    return {
        "strategy_return": strategy_return,
        "baseline_mean": float(np.mean(baseline_arr)),
        "baseline_std": float(np.std(baseline_arr)),
        "baseline_median": float(np.median(baseline_arr)),
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "percentile": float(np.mean(baseline_arr < strategy_return) * 100),
    }


def evaluate_success_criteria(metrics: dict, baseline_stats: dict) -> dict:
    """Evaluate whether strategy meets success criteria.

    Success criteria:
    - Sharpe > 1.0 after execution costs
    - Outperforms random baseline (p < 0.05)
    - Edge persists across time periods
    - Max drawdown < 30%

    Abandon criteria:
    - Returns indistinguishable from random
    - All returns from 1-2 traders
    - Prices move > 5% within 1 hour (front-running)
    """
    evaluation = {
        "meets_sharpe": metrics.get("sharpe_ratio", 0) > 1.0,
        "beats_baseline": baseline_stats.get("p_value", 1.0) < 0.05,
        "acceptable_drawdown": metrics.get("max_drawdown", 1.0) < 0.30,
    }

    evaluation["worth_pursuing"] = (
        evaluation["meets_sharpe"]
        and evaluation["beats_baseline"]
        and evaluation["acceptable_drawdown"]
    )

    return evaluation


def generate_report(
    backtest_results: dict,
    baseline_samples: list[dict],
    slippage_analysis: dict | None,
    sensitivity_results: list[dict],
) -> str:
    """Generate human-readable analysis report."""
    metrics = backtest_results.get("metrics", {})

    # Compute baseline comparison
    baseline_pnls = [s["pnl"] for s in baseline_samples] if baseline_samples else []
    strategy_return = metrics.get("total_return", 0)

    if baseline_pnls:
        significance = compute_significance(strategy_return, baseline_pnls)
    else:
        significance = {}

    evaluation = evaluate_success_criteria(metrics, significance)

    report = []
    report.append("=" * 70)
    report.append("POLYMARKET COPY TRADING EXPERIMENT - RESULTS ANALYSIS")
    report.append("=" * 70)
    report.append("")

    # Primary results
    report.append("## PRIMARY BACKTEST RESULTS")
    report.append("")
    report.append(f"  Total Return:     {metrics.get('total_return', 0)*100:>8.2f}%")
    report.append(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>8.2f}")
    report.append(f"  Max Drawdown:     {metrics.get('max_drawdown', 0)*100:>8.2f}%")
    report.append(f"  Win Rate:         {metrics.get('win_rate', 0)*100:>8.1f}%")
    report.append(f"  Total Trades:     {metrics.get('n_trades', 0):>8}")
    report.append("")

    # Baseline comparison
    if significance:
        report.append("## RANDOM BASELINE COMPARISON")
        report.append("")
        report.append(f"  Baseline Mean:    {significance.get('baseline_mean', 0)*100:>8.2f}%")
        report.append(f"  Baseline Std:     {significance.get('baseline_std', 0)*100:>8.2f}%")
        report.append(f"  Strategy Percentile: {significance.get('percentile', 0):>5.1f}%")
        report.append(f"  P-value:          {significance.get('p_value', 1):>8.4f}")
        report.append(f"  Significant (5%): {'YES' if significance.get('significant_05') else 'NO'}")
        report.append("")

    # Slippage analysis
    if slippage_analysis:
        report.append("## SLIPPAGE ANALYSIS")
        report.append("")
        overall = slippage_analysis.get("overall_stats", {})
        for lag, stats in overall.items():
            report.append(f"  {lag} lag: Mean={stats.get('mean', 0)*100:.3f}%, "
                         f"Median={stats.get('median', 0)*100:.3f}%")
        report.append("")

    # Sensitivity analysis
    if sensitivity_results:
        report.append("## PARAMETER SENSITIVITY")
        report.append("")
        for result in sensitivity_results:
            report.append(f"  {result['param']}={result['value']}: "
                         f"Return={result['total_return']*100:.1f}%, "
                         f"Sharpe={result['sharpe']:.2f}")
        report.append("")

    # Success criteria evaluation
    report.append("## SUCCESS CRITERIA EVALUATION")
    report.append("")
    report.append(f"  [{'X' if evaluation['meets_sharpe'] else ' '}] Sharpe > 1.0")
    report.append(f"  [{'X' if evaluation['beats_baseline'] else ' '}] Beats random baseline (p < 0.05)")
    report.append(f"  [{'X' if evaluation['acceptable_drawdown'] else ' '}] Max drawdown < 30%")
    report.append("")
    report.append(f"  VERDICT: {'WORTH PURSUING' if evaluation['worth_pursuing'] else 'DOES NOT MEET CRITERIA'}")
    report.append("")

    # Recommendations
    report.append("## RECOMMENDATIONS")
    report.append("")
    if evaluation["worth_pursuing"]:
        report.append("  Strategy shows promise. Consider:")
        report.append("  - Live paper trading validation")
        report.append("  - Deeper analysis of trader selection criteria")
        report.append("  - Risk management refinement")
    else:
        reasons = []
        if not evaluation["meets_sharpe"]:
            reasons.append("Sharpe ratio below threshold")
        if not evaluation["beats_baseline"]:
            reasons.append("Not significantly better than random")
        if not evaluation["acceptable_drawdown"]:
            reasons.append("Excessive drawdown")
        report.append(f"  Strategy does not meet criteria: {', '.join(reasons)}")
        report.append("  Consider:")
        report.append("  - Alternative trader selection methods")
        report.append("  - Different market categories")
        report.append("  - Hybrid approaches (e.g., ML-based selection)")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


async def main():
    """Run full analysis."""
    storage = MarketDataStorage(DB_PATH)

    try:
        # Load backtest results
        backtest_files = list(DATA_DIR.glob("backtest_*.json"))
        if not backtest_files:
            logger.error("No backtest results found. Run backtest.py first.")
            return

        latest_backtest = max(backtest_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading backtest results from: {latest_backtest}")
        with open(latest_backtest) as f:
            backtest_results = json.load(f)

        # Load slippage analysis if available
        slippage_path = DATA_DIR / "slippage_analysis.json"
        slippage_analysis = None
        if slippage_path.exists():
            with open(slippage_path) as f:
                slippage_analysis = json.load(f)

        # Load sensitivity results if available
        sensitivity_files = list(DATA_DIR.glob("sensitivity_*.json"))
        sensitivity_results = []
        if sensitivity_files:
            latest_sensitivity = max(sensitivity_files, key=lambda p: p.stat().st_mtime)
            with open(latest_sensitivity) as f:
                sensitivity_results = json.load(f)

        # Generate random baseline
        logger.info("Generating random baseline (1000 samples)...")
        n_traders = backtest_results.get("config", {}).get("n_traders", 10)

        # Use date range from backtest
        daily_values = backtest_results.get("daily_values", [])
        if daily_values:
            start_date = datetime.fromisoformat(daily_values[0]["date"])
            end_date = datetime.fromisoformat(daily_values[-1]["date"])
        else:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=180)

        baseline_samples = await generate_random_baseline(
            storage,
            n_traders=n_traders,
            n_samples=1000,
            start_date=start_date,
            end_date=end_date,
        )

        # Generate report
        report = generate_report(
            backtest_results,
            baseline_samples,
            slippage_analysis,
            sensitivity_results,
        )

        # Print report
        print("\n" + report)

        # Save report
        report_path = DATA_DIR / f"analysis_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        # Save detailed analysis
        analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backtest_metrics": backtest_results.get("metrics", {}),
            "baseline_summary": {
                "n_samples": len(baseline_samples),
                "mean_pnl": np.mean([s["pnl"] for s in baseline_samples]) if baseline_samples else 0,
                "std_pnl": np.std([s["pnl"] for s in baseline_samples]) if baseline_samples else 0,
            },
            "success_criteria": evaluate_success_criteria(
                backtest_results.get("metrics", {}),
                compute_significance(
                    backtest_results.get("metrics", {}).get("total_return", 0),
                    [s["pnl"] for s in baseline_samples],
                ) if baseline_samples else {},
            ),
        }

        analysis_path = DATA_DIR / f"analysis_{datetime.now().strftime('%Y%m%d')}.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Detailed analysis saved to: {analysis_path}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
