"""Analyze slippage for top trader trades.

Phase 2: Slippage analysis
For top-50 trader trades, measure price movement at 1m, 5m, 15m, 60m post-trade.
Build empirical slippage model to calibrate backtest.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from llm_forecasting.market_data.polymarket import PolymarketData
from llm_forecasting.market_data.storage import MarketDataStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"


@dataclass
class SlippageObservation:
    """Single slippage observation."""

    trade_price: float
    trade_side: str  # BUY or SELL
    trade_size: float
    prices_after: dict[int, float]  # minutes -> price


def compute_slippage(obs: SlippageObservation, lag_minutes: int) -> float | None:
    """Compute slippage as price change in direction of trade.

    Positive slippage = price moved against us (we bought, price went up before we could)
    """
    if lag_minutes not in obs.prices_after:
        return None

    price_after = obs.prices_after[lag_minutes]
    price_change = price_after - obs.trade_price

    if obs.trade_side == "BUY":
        # Buying: positive price change = slippage (we pay more)
        return price_change
    else:
        # Selling: negative price change = slippage (we receive less)
        return -price_change


async def get_top_traders(storage: MarketDataStorage, n: int = 50) -> list[str]:
    """Get top N traders by PNL from stored leaderboard."""
    entries = await storage.get_leaderboard_snapshot(time_period="ALL", limit=n)
    return [e.user_address for e in entries]


async def analyze_trader_slippage(
    storage: MarketDataStorage,
    user_address: str,
    lag_minutes: list[int] = [1, 5, 15, 60],
) -> list[SlippageObservation]:
    """Analyze slippage for a single trader's trades."""
    observations = []

    # Get trader's trades
    activities = await storage.get_trader_activity(
        user_address,
        activity_types=["TRADE"],
    )

    if not activities:
        return observations

    # Group trades by market
    trades_by_market = defaultdict(list)
    for activity in activities:
        trades_by_market[activity.condition_id].append(activity)

    # For each market, match trades to price history
    for market_id, trades in trades_by_market.items():
        # Get price history for this market
        price_history = await storage.get_price_history("polymarket", market_id)
        if not price_history:
            continue

        # Build price lookup (timestamp -> close price)
        price_by_time = {p.timestamp: p.close for p in price_history}
        sorted_times = sorted(price_by_time.keys())

        for trade in trades:
            if trade.price is None or trade.side is None:
                continue

            # Find prices after trade at each lag
            prices_after = {}
            for lag in lag_minutes:
                target_time = trade.timestamp + timedelta(minutes=lag)

                # Find nearest price at or after target time
                for t in sorted_times:
                    if t >= target_time:
                        prices_after[lag] = price_by_time[t]
                        break

            if prices_after:
                observations.append(
                    SlippageObservation(
                        trade_price=trade.price,
                        trade_side=trade.side,
                        trade_size=trade.size,
                        prices_after=prices_after,
                    )
                )

    return observations


def compute_slippage_stats(
    observations: list[SlippageObservation],
    lag_minutes: list[int] = [1, 5, 15, 60],
) -> dict:
    """Compute slippage statistics across all observations."""
    results = {}

    for lag in lag_minutes:
        slippages = [
            compute_slippage(obs, lag)
            for obs in observations
            if compute_slippage(obs, lag) is not None
        ]

        if slippages:
            slippages_arr = np.array(slippages)
            results[f"{lag}m"] = {
                "n": len(slippages),
                "mean": float(np.mean(slippages_arr)),
                "median": float(np.median(slippages_arr)),
                "std": float(np.std(slippages_arr)),
                "p25": float(np.percentile(slippages_arr, 25)),
                "p75": float(np.percentile(slippages_arr, 75)),
                "p95": float(np.percentile(slippages_arr, 95)),
                "pct_positive": float(np.mean(slippages_arr > 0)),
            }

    return results


def compute_slippage_by_size(
    observations: list[SlippageObservation],
    lag_minutes: int = 60,
    size_buckets: list[float] = [100, 500, 1000, 5000],
) -> dict:
    """Compute slippage stratified by trade size."""
    # Bucket observations by size
    bucketed = defaultdict(list)

    for obs in observations:
        slippage = compute_slippage(obs, lag_minutes)
        if slippage is None:
            continue

        # Find appropriate bucket
        for i, threshold in enumerate(size_buckets):
            if obs.trade_size < threshold:
                bucket = f"<{threshold}"
                break
        else:
            bucket = f">={size_buckets[-1]}"

        bucketed[bucket].append(slippage)

    # Compute stats per bucket
    results = {}
    for bucket, slippages in bucketed.items():
        if slippages:
            slippages_arr = np.array(slippages)
            results[bucket] = {
                "n": len(slippages),
                "mean": float(np.mean(slippages_arr)),
                "median": float(np.median(slippages_arr)),
                "std": float(np.std(slippages_arr)),
            }

    return results


async def main():
    """Run slippage analysis."""
    storage = MarketDataStorage(DB_PATH)

    try:
        # Get top 50 traders
        logger.info("Getting top 50 traders...")
        top_traders = await get_top_traders(storage, n=50)
        logger.info(f"Found {len(top_traders)} traders")

        if not top_traders:
            logger.error("No traders found. Run fetch_data.py first.")
            return

        # Analyze slippage for each trader
        logger.info("Analyzing slippage for top traders...")
        all_observations = []

        for i, user_address in enumerate(top_traders):
            logger.info(f"Processing trader {i+1}/{len(top_traders)}: {user_address[:10]}...")
            observations = await analyze_trader_slippage(storage, user_address)
            all_observations.extend(observations)
            logger.info(f"  {len(observations)} observations")

        logger.info(f"\nTotal observations: {len(all_observations)}")

        if not all_observations:
            logger.error("No observations found. Check that price history was fetched.")
            return

        # Compute overall statistics
        logger.info("\n" + "=" * 60)
        logger.info("SLIPPAGE STATISTICS")
        logger.info("=" * 60)

        stats = compute_slippage_stats(all_observations)
        for lag, lag_stats in stats.items():
            logger.info(f"\n{lag} lag:")
            logger.info(f"  N: {lag_stats['n']}")
            logger.info(f"  Mean: {lag_stats['mean']*100:.3f}%")
            logger.info(f"  Median: {lag_stats['median']*100:.3f}%")
            logger.info(f"  Std: {lag_stats['std']*100:.3f}%")
            logger.info(f"  P95: {lag_stats['p95']*100:.3f}%")
            logger.info(f"  % Positive (adverse): {lag_stats['pct_positive']*100:.1f}%")

        # Compute by size
        logger.info("\n" + "=" * 60)
        logger.info("SLIPPAGE BY TRADE SIZE (60m lag)")
        logger.info("=" * 60)

        size_stats = compute_slippage_by_size(all_observations)
        for bucket, bucket_stats in sorted(size_stats.items()):
            logger.info(f"\n{bucket}:")
            logger.info(f"  N: {bucket_stats['n']}")
            logger.info(f"  Mean: {bucket_stats['mean']*100:.3f}%")
            logger.info(f"  Median: {bucket_stats['median']*100:.3f}%")

        # Save results
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_traders": len(top_traders),
            "n_observations": len(all_observations),
            "overall_stats": stats,
            "by_size": size_stats,
            "recommended_slippage": {
                "1h_lag": stats.get("60m", {}).get("mean", 0.01),
                "15m_lag": stats.get("15m", {}).get("mean", 0.005),
                "5m_lag": stats.get("5m", {}).get("mean", 0.002),
            },
        }

        output_path = DATA_DIR / "slippage_analysis.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
