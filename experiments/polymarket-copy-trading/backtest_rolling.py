"""Rolling window backtest - no lookahead bias.

At each selection point:
1. Rank traders by realized PNL in the past 60 days (with 7-day lag)
2. Copy their trades for the next 30 days
3. Measure returns
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from sqlalchemy import select, func

from llm_forecasting.market_data.storage import MarketDataStorage, TraderActivityRow, LeaderboardSnapshotRow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"


@dataclass
class RollingBacktestConfig:
    initial_capital: float = 10000.0
    max_position_pct: float = 0.05
    slippage: float = 0.01
    trading_fee: float = 0.02

    # Rolling selection params
    lookback_days: int = 60  # How far back to look for trader performance
    lag_days: int = 7  # Gap between lookback end and trading start (avoid unrealized gains)
    trading_window_days: int = 30  # How long to copy before re-selecting

    # Trader selection
    top_n_traders: int = 50
    min_trades: int = 10  # Min trades in lookback period
    min_markets: int = 3  # Min distinct markets in lookback


@dataclass
class Position:
    market_id: str
    side: str
    entry_price: float
    size: float
    trader: str
    outcome_index: int


def calculate_realized_pnl(
    trades: list[dict],
    trader_id: str,
    start: datetime,
    end: datetime,
) -> tuple[float, int, int]:
    """Calculate realized PNL for a trader in a time window.

    Returns (pnl, trade_count, market_count).
    Uses round-trip trades to calculate actual realized gains.
    """
    # Get trader's trades in window
    trader_trades = [
        t for t in trades
        if t["trader_id"] == trader_id
        and start <= t["timestamp"] <= end
    ]

    if not trader_trades:
        return 0.0, 0, 0

    # Track positions and realized PNL
    positions = defaultdict(lambda: {"shares": 0, "cost_basis": 0})
    realized_pnl = 0.0
    markets = set()

    for trade in sorted(trader_trades, key=lambda x: x["timestamp"]):
        market_id = trade["market_id"]
        side = trade["side"]
        price = trade["price"]
        usdc_size = trade.get("usdc_size") or (trade["size"] * price if price else 0)

        if not price or not market_id:
            continue

        markets.add(market_id)
        pos = positions[market_id]

        if side == "BUY":
            shares = usdc_size / price if price > 0 else 0
            pos["shares"] += shares
            pos["cost_basis"] += usdc_size
        elif side == "SELL" and pos["shares"] > 0:
            shares_sold = usdc_size / price if price > 0 else 0
            shares_to_close = min(shares_sold, pos["shares"])
            avg_cost = pos["cost_basis"] / pos["shares"] if pos["shares"] > 0 else 0

            # Realized PNL from this sale
            sell_value = shares_to_close * price
            cost_of_sold = shares_to_close * avg_cost
            realized_pnl += sell_value - cost_of_sold

            # Update position
            pos["shares"] -= shares_to_close
            if pos["shares"] > 0:
                pos["cost_basis"] = pos["shares"] * avg_cost
            else:
                pos["cost_basis"] = 0

    return realized_pnl, len(trader_trades), len(markets)


def select_traders_rolling(
    trades: list[dict],
    as_of: datetime,
    config: RollingBacktestConfig,
) -> list[str]:
    """Select top traders based on rolling window performance.

    Uses only data from [as_of - lookback - lag, as_of - lag].
    """
    lookback_end = as_of - timedelta(days=config.lag_days)
    lookback_start = lookback_end - timedelta(days=config.lookback_days)

    # Get all traders active in lookback window
    traders_in_window = set(
        t["trader_id"] for t in trades
        if lookback_start <= t["timestamp"] <= lookback_end
    )

    # Calculate PNL for each
    trader_metrics = {}
    for trader_id in traders_in_window:
        pnl, trade_count, market_count = calculate_realized_pnl(
            trades, trader_id, lookback_start, lookback_end
        )

        # Apply quality filters
        if trade_count >= config.min_trades and market_count >= config.min_markets:
            trader_metrics[trader_id] = {
                "pnl": pnl,
                "trades": trade_count,
                "markets": market_count,
            }

    # Sort by PNL and take top N
    sorted_traders = sorted(
        trader_metrics.items(),
        key=lambda x: x[1]["pnl"],
        reverse=True
    )

    return [t[0] for t in sorted_traders[:config.top_n_traders]]


async def load_trades(storage: MarketDataStorage) -> list[dict]:
    """Load all trades."""
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
            }
            for row in rows
        ]


def load_resolutions() -> dict:
    """Load market resolutions."""
    json_path = DATA_DIR / "market_resolutions.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


async def main():
    """Run rolling window backtest."""
    config = RollingBacktestConfig()
    storage = MarketDataStorage(DB_PATH)

    try:
        logger.info("Loading trades...")
        trades = await load_trades(storage)
        logger.info(f"Loaded {len(trades)} trades")

        resolutions = load_resolutions()
        logger.info(f"Loaded {len(resolutions)} resolutions")

        # Find date range
        timestamps = [t["timestamp"] for t in trades]
        min_date = min(timestamps)
        max_date = max(timestamps)
        logger.info(f"Trade range: {min_date.date()} to {max_date.date()}")

        # Start after enough lookback data
        start_date = min_date + timedelta(days=config.lookback_days + config.lag_days)

        # Run rolling backtest
        current_date = start_date
        cash = config.initial_capital
        positions = {}
        current_traders = set()

        total_trades = 0
        total_wins = 0
        period_returns = []

        logger.info(f"Starting rolling backtest from {start_date.date()}")
        logger.info(f"Config: lookback={config.lookback_days}d, lag={config.lag_days}d, window={config.trading_window_days}d")

        while current_date < max_date:
            # Select traders for this period
            current_traders = set(select_traders_rolling(trades, current_date, config))

            if not current_traders:
                logger.warning(f"{current_date.date()}: No traders selected")
                current_date += timedelta(days=config.trading_window_days)
                continue

            # Get trades in this window
            window_end = current_date + timedelta(days=config.trading_window_days)
            window_trades = [
                t for t in trades
                if current_date <= t["timestamp"] < window_end
                and t["trader_id"] in current_traders
            ]

            period_start_cash = cash + sum(p.size for p in positions.values())

            # Process trades
            for trade in sorted(window_trades, key=lambda x: x["timestamp"]):
                market_id = trade["market_id"]
                trader_id = trade["trader_id"]
                price = trade["price"]
                side = trade["side"].lower() if trade["side"] else None

                if not market_id or not price or side not in ("buy", "sell"):
                    continue

                # Exit check (same trader only)
                if market_id in positions:
                    pos = positions[market_id]
                    if pos.trader == trader_id:
                        is_exit = (pos.side == "buy" and side == "sell") or (pos.side == "sell" and side == "buy")
                        if is_exit:
                            if pos.side == "buy":
                                pnl = pos.size * (price * (1 - config.slippage) - pos.entry_price) / pos.entry_price
                            else:
                                pnl = pos.size * (pos.entry_price - price * (1 + config.slippage)) / pos.entry_price

                            if pnl > 0:
                                pnl *= (1 - config.trading_fee)
                                total_wins += 1

                            cash += pos.size + pnl
                            total_trades += 1
                            del positions[market_id]
                            continue

                # Entry
                if market_id not in positions:
                    portfolio_value = cash + sum(p.size for p in positions.values())
                    position_size = min(portfolio_value * config.max_position_pct, cash * 0.9)

                    if position_size < 10:
                        continue

                    entry_price = price * (1 + config.slippage if side == "buy" else 1 - config.slippage)
                    positions[market_id] = Position(
                        market_id=market_id,
                        side=side,
                        entry_price=entry_price,
                        size=position_size,
                        trader=trader_id,
                        outcome_index=trade["outcome_index"],
                    )
                    cash -= position_size

            # Close positions at period end using resolution or mark-to-market
            for market_id in list(positions.keys()):
                pos = positions[market_id]
                outcome_prices = resolutions.get(market_id)

                if outcome_prices and isinstance(outcome_prices, list):
                    try:
                        resolved_price = outcome_prices[pos.outcome_index]
                    except IndexError:
                        resolved_price = outcome_prices[0]

                    if pos.side == "buy":
                        pnl = pos.size * (resolved_price - pos.entry_price) / pos.entry_price
                    else:
                        pnl = pos.size * (pos.entry_price - resolved_price) / pos.entry_price
                else:
                    # Not resolved - keep position open
                    continue

                if pnl > 0:
                    pnl *= (1 - config.trading_fee)
                    total_wins += 1

                cash += pos.size + pnl
                total_trades += 1
                del positions[market_id]

            period_end_value = cash + sum(p.size for p in positions.values())
            period_return = (period_end_value - period_start_cash) / period_start_cash if period_start_cash > 0 else 0
            period_returns.append({
                "date": current_date.date(),
                "traders": len(current_traders),
                "return": period_return,
                "value": period_end_value,
            })

            logger.info(f"{current_date.date()}: {len(current_traders)} traders, {period_return*100:+.1f}% return, value=${period_end_value:,.0f}")

            current_date = window_end

        # Final stats
        final_value = cash + sum(p.size for p in positions.values())
        total_return = (final_value - config.initial_capital) / config.initial_capital
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        logger.info("=" * 60)
        logger.info("ROLLING BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Capital: ${config.initial_capital:,.2f}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return * 100:.1f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate * 100:.1f}%")
        logger.info(f"Open Positions: {len(positions)}")
        logger.info("=" * 60)

        # Save detailed results
        results_file = DATA_DIR / "backtest_rolling_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "config": {
                    "lookback_days": config.lookback_days,
                    "lag_days": config.lag_days,
                    "trading_window_days": config.trading_window_days,
                    "top_n_traders": config.top_n_traders,
                },
                "results": {
                    "initial_capital": config.initial_capital,
                    "final_value": final_value,
                    "total_return": total_return,
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                },
                "periods": [
                    {"date": str(p["date"]), "return": p["return"], "value": p["value"]}
                    for p in period_returns
                ],
            }, f, indent=2)
        logger.info(f"Detailed results saved to {results_file}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
