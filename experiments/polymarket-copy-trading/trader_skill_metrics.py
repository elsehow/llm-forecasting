"""Trader Skill Metrics Experiment

Compute skill metrics for traders in Period 1, measure out-of-sample returns
in Period 2, and test which metrics predict future performance.

Metrics computed:
- Brier score (calibration)
- Hard market win rate (edge on 40-60% markets)
- Trade timing edge (price movement after trade)
- Cross-market consistency
- Concentration (HHI penalty)
- Raw PnL (baseline)
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import stats
from sqlalchemy import select, func

from llm_forecasting.market_data.storage import (
    MarketDataStorage,
    TraderActivityRow,
    LeaderboardSnapshotRow,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"


@dataclass
class TraderMetrics:
    """Skill metrics for a single trader."""
    trader_id: str
    n_trades: int
    n_markets: int

    # Period 1 metrics
    brier_score: float | None  # Lower is better
    hard_market_win_rate: float | None  # Win rate on 40-60% markets
    timing_edge: float | None  # Avg price move in our direction after trade
    consistency: float | None  # Std of returns across markets
    concentration: float  # HHI of market volumes (lower is more diverse)
    period1_pnl: float  # Raw PnL for baseline comparison
    period1_return: float  # Return as percentage

    # Period 2 outcomes
    period2_pnl: float | None
    period2_return: float | None
    period2_brier: float | None


async def load_all_trades(storage: MarketDataStorage) -> list[dict]:
    """Load all trades with timestamps."""
    async with await storage._get_session() as session:
        result = await session.execute(
            select(TraderActivityRow)
            .where(TraderActivityRow.activity_type == "TRADE")
            .where(TraderActivityRow.price.isnot(None))
            .order_by(TraderActivityRow.timestamp)
        )
        rows = result.scalars().all()

        return [
            {
                "trader_id": row.user_address,
                "market_id": row.condition_id,
                "timestamp": row.timestamp,
                "side": row.side,
                "price": row.price,
                "size": row.size or 0,
                "usdc_size": row.usdc_size or 0,
                "outcome_index": row.outcome_index or 0,
                "market_slug": row.market_slug,
            }
            for row in rows
        ]


def load_resolutions() -> dict[str, list[float]]:
    """Load market resolutions."""
    json_path = DATA_DIR / "market_resolutions.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            # Normalize to list format
            resolutions = {}
            for k, v in data.items():
                if isinstance(v, list):
                    resolutions[k] = v
                else:
                    resolutions[k] = [float(v), 1.0 - float(v)]
            return resolutions
    return {}


class MarketPriceIndex:
    """Pre-indexed market prices for fast lookups."""

    def __init__(self, trades: list[dict]):
        """Build index: market_id -> sorted list of (timestamp, price)."""
        self.index = defaultdict(list)
        for t in trades:
            if t["market_id"] and t["price"] is not None:
                self.index[t["market_id"]].append((t["timestamp"], t["price"]))

        # Sort by timestamp
        for market_id in self.index:
            self.index[market_id].sort(key=lambda x: x[0])

    def get_price_at_time(
        self, market_id: str, timestamp: datetime, window_hours: int = 24
    ) -> float | None:
        """Get market price around a timestamp using binary search."""
        if market_id not in self.index:
            return None

        prices = self.index[market_id]
        if not prices:
            return None

        window_start = timestamp - timedelta(hours=window_hours)
        window_end = timestamp + timedelta(hours=window_hours)

        # Binary search for start
        lo, hi = 0, len(prices)
        while lo < hi:
            mid = (lo + hi) // 2
            if prices[mid][0] < window_start:
                lo = mid + 1
            else:
                hi = mid
        start_idx = lo

        # Collect prices in window
        nearby_prices = []
        for i in range(start_idx, len(prices)):
            ts, price = prices[i]
            if ts > window_end:
                break
            nearby_prices.append(price)

        if nearby_prices:
            return np.mean(nearby_prices)
        return None


def compute_trader_metrics(
    trader_id: str,
    trades: list[dict],
    resolutions: dict[str, list[float]],
    period_start: datetime,
    period_end: datetime,
    price_index: MarketPriceIndex | None = None,
) -> dict:
    """Compute all skill metrics for a trader in a time period."""

    # Filter to trader's trades in period
    trader_trades = [
        t for t in trades
        if t["trader_id"] == trader_id
        and period_start <= t["timestamp"] <= period_end
    ]

    if not trader_trades:
        return None

    # Basic counts
    n_trades = len(trader_trades)
    markets = set(t["market_id"] for t in trader_trades)
    n_markets = len(markets)

    # Track positions for PnL calculation
    positions = defaultdict(lambda: {"shares": 0, "cost_basis": 0, "volume": 0})
    realized_pnl = 0.0
    total_invested = 0.0

    # For Brier score: track (implied_prob, outcome) pairs
    brier_samples = []

    # For hard market win rate
    hard_market_trades = []  # (won, market_price_at_trade)

    # For timing edge
    timing_edges = []

    # For consistency: track PnL per market
    market_pnls = defaultdict(float)

    # Process trades
    for trade in sorted(trader_trades, key=lambda x: x["timestamp"]):
        market_id = trade["market_id"]
        side = trade["side"]
        price = trade["price"]
        usdc_size = trade.get("usdc_size") or (trade["size"] * price if price else 0)
        outcome_idx = trade.get("outcome_index", 0) or 0

        if not price or not market_id:
            continue

        pos = positions[market_id]
        pos["volume"] += usdc_size

        # Get market price for "hard market" classification
        market_price = None
        if price_index:
            market_price = price_index.get_price_at_time(market_id, trade["timestamp"], window_hours=1)

        if side == "BUY":
            shares = usdc_size / price if price > 0 else 0
            pos["shares"] += shares
            pos["cost_basis"] += usdc_size
            pos["outcome_idx"] = outcome_idx
            total_invested += usdc_size

            # Record for Brier (implied prob = price they paid)
            resolution = resolutions.get(market_id)
            if resolution is not None:
                try:
                    outcome = resolution[outcome_idx]
                    brier_samples.append((price, outcome))

                    # Hard market check (40-60%)
                    if market_price and 0.4 <= market_price <= 0.6:
                        won = outcome > 0.5  # Did this outcome win?
                        hard_market_trades.append(won)
                except (IndexError, TypeError):
                    pass

            # Timing edge: price 1 hour later
            if price_index:
                price_later = price_index.get_price_at_time(
                    market_id,
                    trade["timestamp"] + timedelta(hours=1),
                    window_hours=1
                )
                if price_later is not None:
                    timing_edges.append(price_later - price)  # Positive = price went up after buy

        elif side == "SELL" and pos["shares"] > 0:
            shares_sold = usdc_size / price if price > 0 else 0
            shares_to_close = min(shares_sold, pos["shares"])
            avg_cost = pos["cost_basis"] / pos["shares"] if pos["shares"] > 0 else 0

            # Realized PnL
            sell_value = shares_to_close * price
            cost_of_sold = shares_to_close * avg_cost
            trade_pnl = sell_value - cost_of_sold
            realized_pnl += trade_pnl
            market_pnls[market_id] += trade_pnl

            # Update position
            pos["shares"] -= shares_to_close
            if pos["shares"] > 0:
                pos["cost_basis"] = pos["shares"] * avg_cost
            else:
                pos["cost_basis"] = 0

    # Close remaining positions at resolution
    for market_id, pos in positions.items():
        if pos["shares"] > 0:
            resolution = resolutions.get(market_id)
            if resolution is not None:
                try:
                    outcome_idx = pos.get("outcome_idx", 0)
                    resolved_price = resolution[outcome_idx]
                    sell_value = pos["shares"] * resolved_price
                    trade_pnl = sell_value - pos["cost_basis"]
                    realized_pnl += trade_pnl
                    market_pnls[market_id] += trade_pnl
                except (IndexError, TypeError):
                    pass

    # Compute metrics

    # Brier score
    brier_score = None
    if brier_samples:
        brier_score = np.mean([(p - o) ** 2 for p, o in brier_samples])

    # Hard market win rate
    hard_market_win_rate = None
    if hard_market_trades:
        hard_market_win_rate = np.mean(hard_market_trades)

    # Timing edge
    timing_edge = None
    if timing_edges:
        timing_edge = np.mean(timing_edges)

    # Consistency (std of per-market returns)
    consistency = None
    if len(market_pnls) >= 3:
        market_returns = list(market_pnls.values())
        consistency = np.std(market_returns) / (np.mean(np.abs(market_returns)) + 1e-6)

    # Concentration (HHI)
    total_volume = sum(pos["volume"] for pos in positions.values())
    if total_volume > 0:
        concentration = sum(
            (pos["volume"] / total_volume) ** 2
            for pos in positions.values()
        )
    else:
        concentration = 1.0

    # Return
    period_return = realized_pnl / total_invested if total_invested > 0 else 0

    return {
        "trader_id": trader_id,
        "n_trades": n_trades,
        "n_markets": n_markets,
        "brier_score": brier_score,
        "hard_market_win_rate": hard_market_win_rate,
        "timing_edge": timing_edge,
        "consistency": consistency,
        "concentration": concentration,
        "period_pnl": realized_pnl,
        "period_return": period_return,
    }


def run_regression(metrics_p1: list[dict], metrics_p2: dict[str, dict]) -> dict:
    """Run regression: Period 2 returns ~ Period 1 metrics."""

    # Build arrays
    traders = []
    X_brier = []
    X_hard = []
    X_timing = []
    X_consistency = []
    X_concentration = []
    X_pnl = []
    X_return = []
    Y_return = []
    Y_brier = []

    for m1 in metrics_p1:
        tid = m1["trader_id"]
        m2 = metrics_p2.get(tid)

        if m2 is None or m2["period_return"] is None:
            continue

        traders.append(tid)
        X_brier.append(m1["brier_score"] if m1["brier_score"] is not None else 0.25)
        X_hard.append(m1["hard_market_win_rate"] if m1["hard_market_win_rate"] is not None else 0.5)
        X_timing.append(m1["timing_edge"] if m1["timing_edge"] is not None else 0)
        X_consistency.append(m1["consistency"] if m1["consistency"] is not None else 1)
        X_concentration.append(m1["concentration"])
        X_pnl.append(m1["period_pnl"])
        X_return.append(m1["period_return"])
        Y_return.append(m2["period_return"])
        Y_brier.append(m2["brier_score"] if m2["brier_score"] is not None else 0.25)

    n = len(traders)
    logger.info(f"Running regression with {n} traders")

    if n < 10:
        logger.warning("Too few traders for meaningful regression")
        return {"error": "insufficient_data", "n": n}

    # Convert to arrays
    X_brier = np.array(X_brier)
    X_hard = np.array(X_hard)
    X_timing = np.array(X_timing)
    X_consistency = np.array(X_consistency)
    X_concentration = np.array(X_concentration)
    X_pnl = np.array(X_pnl)
    X_return = np.array(X_return)
    Y_return = np.array(Y_return)
    Y_brier = np.array(Y_brier)

    # Individual correlations
    results = {
        "n_traders": n,
        "correlations": {},
        "y_stats": {
            "mean_return": float(np.mean(Y_return)),
            "std_return": float(np.std(Y_return)),
            "mean_brier": float(np.mean(Y_brier)),
        }
    }

    predictors = {
        "brier_score": -X_brier,  # Negative because lower Brier is better
        "hard_market_win_rate": X_hard - 0.5,  # Edge over 50%
        "timing_edge": X_timing,
        "consistency": -X_consistency,  # Negative because lower variance is better
        "concentration": -X_concentration,  # Negative because lower HHI is better (more diverse)
        "period1_pnl": X_pnl,
        "period1_return": X_return,
    }

    for name, X in predictors.items():
        # Filter out NaN/inf
        mask = np.isfinite(X) & np.isfinite(Y_return)
        if mask.sum() < 5:
            results["correlations"][name] = {"r": None, "p": None, "n": int(mask.sum())}
            continue

        r, p = stats.pearsonr(X[mask], Y_return[mask])
        results["correlations"][name] = {
            "r": float(r),
            "p": float(p),
            "n": int(mask.sum()),
            "significant": p < 0.05,
        }

    # Multiple regression
    # Standardize predictors
    X_matrix = np.column_stack([
        (X_brier - X_brier.mean()) / (X_brier.std() + 1e-6),
        (X_hard - X_hard.mean()) / (X_hard.std() + 1e-6),
        (X_timing - X_timing.mean()) / (X_timing.std() + 1e-6),
        (X_concentration - X_concentration.mean()) / (X_concentration.std() + 1e-6),
        (X_return - X_return.mean()) / (X_return.std() + 1e-6),
    ])

    # Add intercept
    X_matrix = np.column_stack([np.ones(n), X_matrix])

    # Handle any remaining NaN
    mask = np.all(np.isfinite(X_matrix), axis=1) & np.isfinite(Y_return)
    if mask.sum() >= 10:
        try:
            # OLS
            beta = np.linalg.lstsq(X_matrix[mask], Y_return[mask], rcond=None)[0]
            y_pred = X_matrix[mask] @ beta
            ss_res = np.sum((Y_return[mask] - y_pred) ** 2)
            ss_tot = np.sum((Y_return[mask] - Y_return[mask].mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results["multiple_regression"] = {
                "r_squared": float(r_squared),
                "coefficients": {
                    "intercept": float(beta[0]),
                    "brier": float(beta[1]),
                    "hard_market": float(beta[2]),
                    "timing": float(beta[3]),
                    "concentration": float(beta[4]),
                    "period1_return": float(beta[5]),
                },
                "n": int(mask.sum()),
            }
        except Exception as e:
            results["multiple_regression"] = {"error": str(e)}

    return results


async def main():
    """Run the trader skill metrics experiment."""
    storage = MarketDataStorage(DB_PATH)

    try:
        logger.info("Loading trades...")
        trades = await load_all_trades(storage)
        logger.info(f"Loaded {len(trades)} trades")

        logger.info("Loading resolutions...")
        resolutions = load_resolutions()
        logger.info(f"Loaded {len(resolutions)} resolutions")

        # Find date range
        timestamps = [t["timestamp"] for t in trades]
        min_date = min(timestamps)
        max_date = max(timestamps)
        logger.info(f"Trade range: {min_date.date()} to {max_date.date()}")

        # Split into Period 1 (60%) and Period 2 (40%)
        total_days = (max_date - min_date).days
        split_date = min_date + timedelta(days=int(total_days * 0.6))
        logger.info(f"Split date: {split_date.date()}")
        logger.info(f"Period 1: {min_date.date()} to {split_date.date()}")
        logger.info(f"Period 2: {split_date.date()} to {max_date.date()}")

        # Get all traders
        all_traders = set(t["trader_id"] for t in trades)
        logger.info(f"Total traders: {len(all_traders)}")

        # Build price index for fast lookups
        logger.info("Building price index...")
        price_index = MarketPriceIndex(trades)
        logger.info(f"Indexed {len(price_index.index)} markets")

        # Compute Period 1 metrics
        logger.info("Computing Period 1 metrics...")
        metrics_p1 = []
        for i, trader_id in enumerate(all_traders):
            if i % 100 == 0:
                logger.info(f"  Processing trader {i}/{len(all_traders)}")

            m = compute_trader_metrics(
                trader_id, trades, resolutions,
                min_date, split_date, price_index
            )
            if m and m["n_trades"] >= 10 and m["n_markets"] >= 3:
                metrics_p1.append(m)

        logger.info(f"Period 1: {len(metrics_p1)} traders with sufficient data")

        # Compute Period 2 metrics
        logger.info("Computing Period 2 metrics...")
        metrics_p2 = {}
        for i, trader_id in enumerate(all_traders):
            if i % 100 == 0:
                logger.info(f"  Processing trader {i}/{len(all_traders)}")

            m = compute_trader_metrics(
                trader_id, trades, resolutions,
                split_date, max_date, price_index
            )
            if m and m["n_trades"] >= 5:
                metrics_p2[trader_id] = m

        logger.info(f"Period 2: {len(metrics_p2)} traders with sufficient data")

        # Find overlap
        overlap = [m for m in metrics_p1 if m["trader_id"] in metrics_p2]
        logger.info(f"Traders in both periods: {len(overlap)}")

        # Run regression
        logger.info("Running regression analysis...")
        results = run_regression(metrics_p1, metrics_p2)

        # Print results
        print("\n" + "=" * 70)
        print("TRADER SKILL METRICS EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"\nTraders analyzed: {results.get('n_traders', 0)}")
        print(f"Period 2 mean return: {results.get('y_stats', {}).get('mean_return', 0)*100:.1f}%")
        print(f"Period 2 std return: {results.get('y_stats', {}).get('std_return', 0)*100:.1f}%")

        print("\n" + "-" * 70)
        print("INDIVIDUAL CORRELATIONS (Period 1 metric → Period 2 return)")
        print("-" * 70)
        print(f"{'Metric':<25} {'r':>8} {'p-value':>10} {'n':>6} {'Sig?':>6}")
        print("-" * 70)

        for name, corr in results.get("correlations", {}).items():
            r = corr.get("r")
            p = corr.get("p")
            n = corr.get("n", 0)
            sig = "✓" if corr.get("significant") else ""

            if r is not None:
                print(f"{name:<25} {r:>8.3f} {p:>10.4f} {n:>6} {sig:>6}")
            else:
                print(f"{name:<25} {'N/A':>8} {'N/A':>10} {n:>6}")

        if "multiple_regression" in results:
            mr = results["multiple_regression"]
            print("\n" + "-" * 70)
            print("MULTIPLE REGRESSION")
            print("-" * 70)
            print(f"R²: {mr.get('r_squared', 0):.3f}")
            print(f"N: {mr.get('n', 0)}")
            print("\nStandardized coefficients:")
            for name, coef in mr.get("coefficients", {}).items():
                print(f"  {name}: {coef:.4f}")

        print("\n" + "=" * 70)

        # Interpretation
        print("\nINTERPRETATION:")

        best_predictor = None
        best_r = 0
        for name, corr in results.get("correlations", {}).items():
            r = corr.get("r")
            if r is not None and abs(r) > abs(best_r):
                best_r = r
                best_predictor = name

        if best_predictor:
            if abs(best_r) > 0.3 and results["correlations"][best_predictor].get("significant"):
                print(f"✓ {best_predictor} shows meaningful predictive power (r={best_r:.3f})")
            elif abs(best_r) > 0.1:
                print(f"? Weak signal from {best_predictor} (r={best_r:.3f}), may need more data")
            else:
                print(f"✗ No metric strongly predicts out-of-sample returns (best r={best_r:.3f})")
                print("  This suggests top trader performance is largely luck/survivorship")

        # Save results
        results_file = DATA_DIR / "skill_metrics_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "config": {
                    "split_ratio": 0.6,
                    "min_trades_p1": 10,
                    "min_markets_p1": 3,
                    "min_trades_p2": 5,
                },
                "period1_traders": len(metrics_p1),
                "period2_traders": len(metrics_p2),
                "overlap_traders": len(overlap),
                "results": results,
                "metrics_p1_sample": metrics_p1[:10],  # Save sample for inspection
            }, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
