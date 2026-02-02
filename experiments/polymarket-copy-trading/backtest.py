"""Backtest copy trading strategy.

Phase 3: Backtest simulation
- Rolling window trader selection (avoid lookahead bias)
- Apply quality filters (min trades, min markets, max concentration)
- Simulate copy trades with lag + slippage
- Track portfolio P&L daily
- Support parameter variations
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from llm_forecasting.market_data.storage import MarketDataStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"


@dataclass
class TraderStats:
    """Statistics for a trader over a time window."""

    user_address: str
    pnl: float = 0.0
    volume: float = 0.0
    num_trades: int = 0
    markets_traded: set = field(default_factory=set)
    largest_trade_pnl: float = 0.0

    @property
    def num_markets(self) -> int:
        return len(self.markets_traded)

    @property
    def concentration(self) -> float:
        """Max PNL from single trade as fraction of total."""
        if self.pnl == 0:
            return 0.0
        return abs(self.largest_trade_pnl) / abs(self.pnl)


@dataclass
class Position:
    """A position in a market."""

    market_id: str
    side: str  # BUY or SELL
    size: float
    entry_price: float
    entry_time: datetime


@dataclass
class Portfolio:
    """Portfolio state."""

    cash: float = 10000.0
    positions: dict[str, Position] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def total_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.size * prices.get(pos.market_id, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    # Trader selection
    n_traders: int = 10
    lookback_days: int = 30

    # Quality filters
    min_trades: int = 20
    min_markets: int = 5
    max_concentration: float = 0.5

    # Execution
    execution_lag_minutes: int = 60
    slippage: float = 0.01  # 1%

    # Fees - Polymarket charges ~2% fee on winning positions
    # This is applied when positions resolve profitably
    trading_fee: float = 0.02  # 2% fee on winning positions

    # Position sizing
    position_size: float = 100.0  # USD per trade


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    config: BacktestConfig
    daily_values: list[dict]  # {date, portfolio_value}
    trades: list[dict]  # {date, market_id, side, price, pnl}

    @property
    def total_return(self) -> float:
        if not self.daily_values:
            return 0.0
        initial = self.daily_values[0]["portfolio_value"]
        final = self.daily_values[-1]["portfolio_value"]
        return (final - initial) / initial

    @property
    def sharpe_ratio(self) -> float:
        if len(self.daily_values) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self.daily_values)):
            prev = self.daily_values[i - 1]["portfolio_value"]
            curr = self.daily_values[i]["portfolio_value"]
            returns.append((curr - prev) / prev)
        if not returns or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    @property
    def max_drawdown(self) -> float:
        if not self.daily_values:
            return 0.0
        values = [d["portfolio_value"] for d in self.daily_values]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)


async def compute_trader_stats(
    storage: MarketDataStorage,
    user_address: str,
    start: datetime,
    end: datetime,
) -> TraderStats:
    """Compute statistics for a trader over a time window."""
    stats = TraderStats(user_address=user_address)

    activities = await storage.get_trader_activity(
        user_address,
        start=start,
        end=end,
        activity_types=["TRADE"],
    )

    for activity in activities:
        stats.num_trades += 1
        stats.markets_traded.add(activity.condition_id)

        if activity.usdc_size:
            stats.volume += activity.usdc_size

        # Estimate PNL from trades (simplified - actual P&L requires resolution)
        if activity.price and activity.size:
            trade_value = activity.price * activity.size
            if activity.side == "BUY":
                # Rough estimate: value if resolved YES minus cost
                pnl_if_yes = activity.size - trade_value
                stats.pnl += pnl_if_yes * 0.5  # Expected value at 50%
            else:
                pnl_if_no = trade_value
                stats.pnl += pnl_if_no * 0.5

            stats.largest_trade_pnl = max(
                stats.largest_trade_pnl, abs(trade_value)
            )

    return stats


async def select_traders(
    storage: MarketDataStorage,
    as_of: datetime,
    config: BacktestConfig,
) -> list[str]:
    """Select top N traders using only information available at as_of.

    This is the key to avoiding lookahead bias.
    """
    lookback_start = as_of - timedelta(days=config.lookback_days)

    # Get all traders we have data for
    all_traders = await storage.get_tracked_traders()

    # Compute stats for each trader over lookback window
    trader_stats = []
    for user_address in all_traders:
        stats = await compute_trader_stats(
            storage, user_address, lookback_start, as_of
        )

        # Apply quality filters
        if stats.num_trades < config.min_trades:
            continue
        if stats.num_markets < config.min_markets:
            continue
        if stats.concentration > config.max_concentration:
            continue

        trader_stats.append(stats)

    # Sort by PNL (descending) and take top N
    trader_stats.sort(key=lambda s: s.pnl, reverse=True)
    return [s.user_address for s in trader_stats[: config.n_traders]]


async def get_price_at_time(
    storage: MarketDataStorage,
    market_id: str,
    target_time: datetime,
) -> float | None:
    """Get market price at a specific time."""
    # Get price history around the target time
    start = target_time - timedelta(hours=1)
    end = target_time + timedelta(hours=1)

    prices = await storage.get_price_history("polymarket", market_id, start, end)
    if not prices:
        return None

    # Find closest price to target time
    closest = min(prices, key=lambda p: abs((p.timestamp - target_time).total_seconds()))
    return closest.close


async def simulate_copy_trade(
    storage: MarketDataStorage,
    portfolio: Portfolio,
    original_trade: dict,
    config: BacktestConfig,
) -> dict | None:
    """Simulate copying a trade with lag, slippage, and fees."""
    # Get execution price (after lag)
    exec_time = original_trade["timestamp"] + timedelta(minutes=config.execution_lag_minutes)
    exec_price = await get_price_at_time(
        storage, original_trade["market_id"], exec_time
    )

    if exec_price is None:
        return None

    # Apply slippage
    if original_trade["side"] == "BUY":
        exec_price = exec_price * (1 + config.slippage)
    else:
        exec_price = exec_price * (1 - config.slippage)

    # Calculate position size
    shares = config.position_size / exec_price
    pnl = 0.0
    fee_paid = 0.0

    # Update portfolio
    if original_trade["side"] == "BUY":
        cost = shares * exec_price
        if cost > portfolio.cash:
            # Not enough cash - size down
            shares = portfolio.cash / exec_price
            cost = portfolio.cash

        portfolio.cash -= cost
        position = Position(
            market_id=original_trade["market_id"],
            side="BUY",
            size=shares,
            entry_price=exec_price,
            entry_time=exec_time,
        )
        portfolio.positions[original_trade["market_id"]] = position
    else:
        # For sells, close existing position if any
        if original_trade["market_id"] in portfolio.positions:
            pos = portfolio.positions.pop(original_trade["market_id"])
            proceeds = pos.size * exec_price
            cost_basis = pos.size * pos.entry_price
            pnl = proceeds - cost_basis

            # Apply trading fee on profitable trades (Polymarket charges ~2% on winnings)
            if pnl > 0:
                fee_paid = pnl * config.trading_fee
                proceeds -= fee_paid

            portfolio.cash += proceeds

    return {
        "timestamp": exec_time.isoformat(),
        "market_id": original_trade["market_id"],
        "side": original_trade["side"],
        "price": exec_price,
        "shares": shares,
        "value": shares * exec_price,
        "pnl": pnl,
        "fee_paid": fee_paid,
    }


async def run_backtest(
    storage: MarketDataStorage,
    config: BacktestConfig,
    start_date: datetime,
    end_date: datetime,
) -> BacktestResult:
    """Run the backtest simulation."""
    logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
    logger.info(f"Config: N={config.n_traders}, lookback={config.lookback_days}d, "
                f"lag={config.execution_lag_minutes}m, slippage={config.slippage*100:.1f}%, "
                f"fee={config.trading_fee*100:.1f}%")

    portfolio = Portfolio()
    daily_values = []
    all_trades = []
    current_traders = []
    last_selection_date = None

    # Iterate day by day
    current_date = start_date
    while current_date <= end_date:
        # Re-select traders periodically (e.g., weekly)
        if last_selection_date is None or (current_date - last_selection_date).days >= 7:
            current_traders = await select_traders(storage, current_date, config)
            last_selection_date = current_date
            logger.info(f"{current_date.date()}: Selected {len(current_traders)} traders")

        # Get trades from selected traders for this day
        day_start = current_date.replace(hour=0, minute=0, second=0)
        day_end = current_date.replace(hour=23, minute=59, second=59)

        for trader in current_traders:
            activities = await storage.get_trader_activity(
                trader,
                start=day_start,
                end=day_end,
                activity_types=["TRADE"],
            )

            for activity in activities:
                if activity.price is None or activity.side is None:
                    continue

                original_trade = {
                    "timestamp": activity.timestamp,
                    "market_id": activity.condition_id,
                    "side": activity.side,
                    "price": activity.price,
                    "trader": trader,
                }

                trade_result = await simulate_copy_trade(
                    storage, portfolio, original_trade, config
                )
                if trade_result:
                    all_trades.append(trade_result)

        # Get current prices for all positions
        prices = {}
        for market_id in portfolio.positions:
            price = await get_price_at_time(storage, market_id, day_end)
            if price:
                prices[market_id] = price

        # Record daily value
        daily_values.append({
            "date": current_date.isoformat(),
            "portfolio_value": portfolio.total_value(prices),
            "cash": portfolio.cash,
            "n_positions": len(portfolio.positions),
        })

        current_date += timedelta(days=1)

    return BacktestResult(
        config=config,
        daily_values=daily_values,
        trades=all_trades,
    )


async def main():
    """Run backtest with various parameter configurations."""
    storage = MarketDataStorage(DB_PATH)

    try:
        # Determine date range from data
        # For now, use last 6 months
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=180)

        # Primary configuration
        primary_config = BacktestConfig(
            n_traders=10,
            lookback_days=30,
            execution_lag_minutes=60,
            slippage=0.01,
        )

        # Run primary backtest
        logger.info("=" * 60)
        logger.info("PRIMARY BACKTEST")
        logger.info("=" * 60)
        result = await run_backtest(storage, primary_config, start_date, end_date)

        logger.info(f"\nResults:")
        logger.info(f"  Total Return: {result.total_return*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown*100:.2f}%")
        logger.info(f"  Win Rate: {result.win_rate*100:.1f}%")
        logger.info(f"  Total Trades: {len(result.trades)}")

        # Save primary results
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_traders": primary_config.n_traders,
                "lookback_days": primary_config.lookback_days,
                "execution_lag_minutes": primary_config.execution_lag_minutes,
                "slippage": primary_config.slippage,
                "trading_fee": primary_config.trading_fee,
                "min_trades": primary_config.min_trades,
                "min_markets": primary_config.min_markets,
                "max_concentration": primary_config.max_concentration,
            },
            "metrics": {
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "n_trades": len(result.trades),
            },
            "daily_values": result.daily_values,
        }

        output_path = DATA_DIR / f"backtest_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")

        # Sensitivity tests
        logger.info("\n" + "=" * 60)
        logger.info("SENSITIVITY TESTS")
        logger.info("=" * 60)

        sensitivity_results = []

        # Vary N
        for n in [1, 5, 20, 50]:
            config = BacktestConfig(n_traders=n)
            result = await run_backtest(storage, config, start_date, end_date)
            sensitivity_results.append({
                "param": "n_traders",
                "value": n,
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
            })
            logger.info(f"N={n}: Return={result.total_return*100:.1f}%, Sharpe={result.sharpe_ratio:.2f}")

        # Vary lookback
        for lookback in [7, 90]:
            config = BacktestConfig(lookback_days=lookback)
            result = await run_backtest(storage, config, start_date, end_date)
            sensitivity_results.append({
                "param": "lookback_days",
                "value": lookback,
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
            })
            logger.info(f"Lookback={lookback}d: Return={result.total_return*100:.1f}%, Sharpe={result.sharpe_ratio:.2f}")

        # Vary lag
        for lag in [5, 15]:
            config = BacktestConfig(execution_lag_minutes=lag)
            result = await run_backtest(storage, config, start_date, end_date)
            sensitivity_results.append({
                "param": "execution_lag",
                "value": lag,
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
            })
            logger.info(f"Lag={lag}m: Return={result.total_return*100:.1f}%, Sharpe={result.sharpe_ratio:.2f}")

        # Save sensitivity results
        sensitivity_path = DATA_DIR / f"sensitivity_{datetime.now().strftime('%Y%m%d')}.json"
        with open(sensitivity_path, "w") as f:
            json.dump(sensitivity_results, f, indent=2)
        logger.info(f"\nSensitivity results saved to: {sensitivity_path}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
