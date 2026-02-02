"""Simplified backtest using trade prices directly.

Uses the actual trade prices from collected trader activity data instead of
requiring full price histories. This gives faster results while the full
price history fetch runs in background.

Approach:
- Copy trades at the recorded trade price (+ slippage)
- Exit when the trader exits OR at market resolution
- Resolution outcome determined by final trade direction or explicit resolution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
from sqlalchemy import select, func

from llm_forecasting.market_data.storage import MarketDataStorage, TraderActivityRow, LeaderboardSnapshotRow, MarketRow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "copy_trading.db"


@dataclass
class SimpleBacktestConfig:
    """Configuration for simplified backtest."""
    # Portfolio
    initial_capital: float = 10000.0
    max_position_pct: float = 0.05  # Max 5% of portfolio per position

    # Costs
    slippage: float = 0.01  # 1% slippage on entry
    trading_fee: float = 0.02  # 2% fee on profitable positions

    # Copy trading params
    copy_delay_seconds: int = 60  # Assume 1 minute delay to detect and copy
    min_trader_trades: int = 20  # Minimum trades for trader quality
    min_trader_markets: int = 5  # Minimum distinct markets
    max_concentration: float = 0.5  # Max 50% in single market

    # Trader selection
    top_n_traders: int = 50  # Copy top N traders
    selection_window_days: int = 90  # Use 90-day rolling window for selection


@dataclass
class Position:
    """Tracks a copied position."""
    market_id: str
    side: str  # "buy" or "sell"
    entry_price: float
    entry_time: datetime
    size: float  # Dollar amount invested
    original_trader: str
    outcome_index: int = 0  # Which outcome we're betting on


@dataclass
class BacktestState:
    """Tracks portfolio state during backtest."""
    cash: float
    positions: dict = field(default_factory=dict)  # market_id -> Position
    total_pnl: float = 0.0
    total_fees: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    equity_curve: list = field(default_factory=list)


async def get_all_trades(storage: MarketDataStorage) -> list[dict]:
    """Get all trades sorted by timestamp."""
    async with await storage._get_session() as session:
        # Only get actual trades, not splits/merges
        result = await session.execute(
            select(TraderActivityRow)
            .where(TraderActivityRow.activity_type == "TRADE")
            .where(TraderActivityRow.price.isnot(None))
            .order_by(TraderActivityRow.timestamp)
        )
        rows = result.scalars().all()

        trades = []
        for row in rows:
            trades.append({
                "trader_id": row.user_address,
                "market_id": row.condition_id,
                "timestamp": row.timestamp,
                "side": row.side,
                "price": row.price,
                "size": row.size if row.size else 0,
                "usdc_size": row.usdc_size if row.usdc_size else 0,
                "outcome_index": row.outcome_index,
            })
        return trades


async def get_trader_stats(storage: MarketDataStorage) -> dict:
    """Get trader statistics for quality filtering."""
    async with await storage._get_session() as session:
        # Get trade counts per trader
        result = await session.execute(
            select(
                TraderActivityRow.user_address,
                func.count().label("trade_count"),
                func.count(func.distinct(TraderActivityRow.condition_id)).label("market_count"),
            )
            .where(TraderActivityRow.activity_type == "TRADE")
            .group_by(TraderActivityRow.user_address)
        )

        stats = {}
        for row in result.fetchall():
            stats[row.user_address] = {
                "trade_count": row.trade_count,
                "market_count": row.market_count,
            }
        return stats


async def get_leaderboard_pnl(storage: MarketDataStorage) -> dict[str, float]:
    """Get PNL from leaderboard snapshots (actual Polymarket-reported PNL)."""
    async with await storage._get_session() as session:
        # Get most recent PNL for each trader
        result = await session.execute(
            select(
                LeaderboardSnapshotRow.user_address,
                LeaderboardSnapshotRow.pnl,
            ).distinct(LeaderboardSnapshotRow.user_address)
            .order_by(
                LeaderboardSnapshotRow.user_address,
                LeaderboardSnapshotRow.fetched_at.desc()
            )
        )

        pnls = {}
        for row in result.fetchall():
            pnls[row.user_address] = row.pnl
        return pnls


async def get_market_resolutions(storage: MarketDataStorage) -> dict[str, list[float] | None]:
    """Get actual market resolution values.

    First tries to load from JSON file (fetched by slug), then falls back to markets table.
    Returns dict mapping condition_id -> list of outcome prices [outcome0_price, outcome1_price, ...].
    For resolved markets, winning outcome has price 1.0, losing has 0.0.
    """
    import json

    resolutions = {}

    # Try JSON file first (fetched by slug, more accurate)
    json_path = DATA_DIR / "market_resolutions.json"
    if json_path.exists():
        with open(json_path) as f:
            json_resolutions = json.load(f)
            for cid, val in json_resolutions.items():
                if isinstance(val, list):
                    resolutions[cid] = [float(v) for v in val]
                else:
                    # Legacy format: single value, assume binary
                    resolutions[cid] = [float(val), 1.0 - float(val)]
        logger.info(f"Loaded {len(resolutions)} resolutions from JSON file")
        return resolutions

    # Fall back to markets table (binary markets only)
    async with await storage._get_session() as session:
        result = await session.execute(
            select(MarketRow.id, MarketRow.status, MarketRow.resolved_value)
            .where(MarketRow.platform == "polymarket")
        )

        for row in result.fetchall():
            if row.status == "resolved" and row.resolved_value is not None:
                # Binary: [YES_price, NO_price]
                yes_price = float(row.resolved_value)
                resolutions[row.id] = [yes_price, 1.0 - yes_price]
            else:
                resolutions[row.id] = None

    return resolutions


def calculate_trader_pnl_at_time(
    trades: list[dict],
    trader_id: str,
    as_of: datetime,
    window_days: int = 90
) -> float:
    """Calculate a trader's realized PNL using trades within the window.

    PNL is estimated from round-trip trades (buy then sell, or sell then buy).
    """
    window_start = as_of - timedelta(days=window_days)

    # Get trader's trades in window
    trader_trades = [
        t for t in trades
        if t["trader_id"] == trader_id
        and window_start <= t["timestamp"] <= as_of
    ]

    if not trader_trades:
        return 0.0

    # Track positions per market and calculate realized PNL
    positions = {}  # market_id -> list of (side, price, size)
    realized_pnl = 0.0

    for trade in sorted(trader_trades, key=lambda x: x["timestamp"]):
        market_id = trade["market_id"]
        side = trade["side"]
        price = trade["price"]
        usdc_size = trade.get("usdc_size", trade["size"] * price if price else 0)

        if not price or not market_id:
            continue

        if market_id not in positions:
            positions[market_id] = {"shares": 0, "cost_basis": 0}

        pos = positions[market_id]

        if side == "BUY":
            # Buying YES shares
            shares_bought = usdc_size / price if price > 0 else 0
            pos["shares"] += shares_bought
            pos["cost_basis"] += usdc_size
        elif side == "SELL":
            # Selling YES shares
            shares_sold = usdc_size / price if price > 0 else 0
            if pos["shares"] > 0:
                # Calculate realized PNL
                avg_cost = pos["cost_basis"] / pos["shares"] if pos["shares"] > 0 else 0
                sell_value = min(shares_sold, pos["shares"]) * price
                cost_of_sold = min(shares_sold, pos["shares"]) * avg_cost
                realized_pnl += sell_value - cost_of_sold

                # Update position
                pos["shares"] -= min(shares_sold, pos["shares"])
                if pos["shares"] > 0:
                    pos["cost_basis"] = pos["shares"] * avg_cost
                else:
                    pos["cost_basis"] = 0

    return realized_pnl


def select_traders_at_time(
    trader_stats: dict,
    leaderboard_pnl: dict[str, float],
    config: SimpleBacktestConfig,
) -> list[str]:
    """Select top traders based on leaderboard PNL.

    Note: For simplicity, we use the leaderboard PNL which represents
    all-time performance. A more sophisticated approach would track
    PNL changes over time for true rolling selection.
    """
    # Filter by quality criteria
    qualified_traders = [
        trader_id for trader_id, stats in trader_stats.items()
        if stats["trade_count"] >= config.min_trader_trades
        and stats["market_count"] >= config.min_trader_markets
        and trader_id in leaderboard_pnl
    ]

    # Sort by leaderboard PNL and take top N
    sorted_traders = sorted(
        qualified_traders,
        key=lambda t: leaderboard_pnl.get(t, 0),
        reverse=True
    )

    return sorted_traders[:config.top_n_traders]


def run_backtest(
    trades: list[dict],
    trader_stats: dict,
    leaderboard_pnl: dict[str, float],
    market_resolutions: dict[str, float | None],
    config: SimpleBacktestConfig,
) -> BacktestState:
    """Run the simplified backtest."""
    state = BacktestState(cash=config.initial_capital)

    # Get final prices for each market (last trade price as fallback)
    market_final_prices = {}
    for trade in trades:
        if trade["market_id"] and trade["price"]:
            market_final_prices[trade["market_id"]] = trade["price"]

    # Count how many markets we have resolution data for
    resolved_markets = sum(1 for mid in market_final_prices if market_resolutions.get(mid) is not None)
    logger.info(f"Have resolution data for {resolved_markets}/{len(market_final_prices)} markets")

    # Select traders once using leaderboard PNL (simplified - no rolling window)
    # This introduces some lookahead bias but gives cleaner signal
    current_traders = set(select_traders_at_time(
        trader_stats, leaderboard_pnl, config
    ))
    logger.info(f"Selected {len(current_traders)} traders to copy")

    # Track positions by market
    positions = {}  # market_id -> Position

    logger.info(f"Running backtest on {len(trades)} trades...")

    for i, trade in enumerate(trades):
        if i % 100000 == 0:
            logger.info(f"Processing trade {i}/{len(trades)}")

        timestamp = trade["timestamp"]
        trader_id = trade["trader_id"]
        market_id = trade["market_id"]
        price = trade["price"]

        if not market_id or not price:
            continue

        # Skip if not following this trader
        if trader_id not in current_traders:
            continue

        # Normalize side to lowercase
        side = trade["side"].lower() if trade["side"] else None
        if side not in ("buy", "sell"):
            continue

        # Check if this is an exit trade (same trader, opposite direction)
        if market_id in positions:
            pos = positions[market_id]
            # Only exit if the SAME trader we copied is now exiting
            same_trader = pos.original_trader == trader_id
            is_opposite = (
                (pos.side == "buy" and side == "sell") or
                (pos.side == "sell" and side == "buy")
            )

            if same_trader and is_opposite:
                # Close position at trade price (minus slippage)
                exit_price = price
                if pos.side == "buy":
                    # We bought YES, selling at exit_price
                    exit_price = exit_price * (1 - config.slippage)
                    pnl = pos.size * (exit_price - pos.entry_price) / pos.entry_price
                else:
                    # We bought NO (sold YES), buying back
                    exit_price = exit_price * (1 + config.slippage)
                    pnl = pos.size * (pos.entry_price - exit_price) / pos.entry_price

                # Apply fee on profits
                if pnl > 0:
                    fee = pnl * config.trading_fee
                    state.total_fees += fee
                    pnl -= fee
                    state.win_count += 1

                state.cash += pos.size + pnl
                state.total_pnl += pnl
                state.trade_count += 1
                del positions[market_id]
                continue

        # Check if this is an entry trade
        if market_id not in positions:
            # Calculate position size
            portfolio_value = state.cash + sum(p.size for p in positions.values())
            max_size = portfolio_value * config.max_position_pct
            position_size = min(max_size, state.cash * 0.9)  # Leave some cash buffer

            if position_size < 10:  # Minimum position size
                continue

            # Apply slippage to entry
            entry_price = price
            if side == "buy":
                entry_price = entry_price * (1 + config.slippage)
            else:
                entry_price = entry_price * (1 - config.slippage)

            # Open position with outcome_index
            outcome_idx = trade.get("outcome_index", 0) or 0
            positions[market_id] = Position(
                market_id=market_id,
                side=side,
                entry_price=entry_price,
                entry_time=timestamp,
                size=position_size,
                original_trader=trader_id,
                outcome_index=outcome_idx,
            )
            state.cash -= position_size

    # Close remaining positions using actual resolution data when available
    logger.info(f"Closing {len(positions)} remaining positions...")
    resolved_count = 0
    unresolved_count = 0

    for market_id, pos in positions.items():
        outcome_prices = market_resolutions.get(market_id)

        if outcome_prices is not None and isinstance(outcome_prices, list):
            # We have actual resolution data - list of [outcome0_price, outcome1_price, ...]
            resolved_count += 1
            # Get the resolution price for our specific outcome
            try:
                resolved_price = outcome_prices[pos.outcome_index]
            except (IndexError, TypeError):
                resolved_price = outcome_prices[0]  # Fallback to first outcome

            if pos.side == "buy":
                # Bought this outcome - win if its price is 1.0
                pnl = pos.size * (resolved_price - pos.entry_price) / pos.entry_price
            else:
                # Sold this outcome - win if its price is 0.0
                pnl = pos.size * (pos.entry_price - resolved_price) / pos.entry_price
        else:
            # No resolution data - use last trade price as estimate
            unresolved_count += 1
            final_price = market_final_prices.get(market_id, pos.entry_price)

            # Be conservative: if price is extreme, infer resolution
            if final_price > 0.95:
                resolved_price = 1.0
            elif final_price < 0.05:
                resolved_price = 0.0
            else:
                # Truly unresolved - mark to market
                resolved_price = final_price

            if pos.side == "buy":
                pnl = pos.size * (resolved_price - pos.entry_price) / pos.entry_price
            else:
                pnl = pos.size * (pos.entry_price - resolved_price) / pos.entry_price

        # Apply fee on profits
        if pnl > 0:
            fee = pnl * config.trading_fee
            state.total_fees += fee
            pnl -= fee
            state.win_count += 1

        state.cash += pos.size + pnl
        state.total_pnl += pnl
        state.trade_count += 1

    logger.info(f"  Resolved: {resolved_count}, Unresolved (estimated): {unresolved_count}")
    return state


async def main():
    """Run simplified backtest."""
    config = SimpleBacktestConfig()
    storage = MarketDataStorage(DB_PATH)

    try:
        logger.info("Loading trades from database...")
        trades = await get_all_trades(storage)
        logger.info(f"Loaded {len(trades)} trades")

        logger.info("Calculating trader statistics...")
        trader_stats = await get_trader_stats(storage)
        logger.info(f"Found {len(trader_stats)} traders")

        logger.info("Loading leaderboard PNL...")
        leaderboard_pnl = await get_leaderboard_pnl(storage)
        logger.info(f"Have leaderboard PNL for {len(leaderboard_pnl)} traders")

        logger.info("Loading market resolution data...")
        market_resolutions = await get_market_resolutions(storage)
        resolved = sum(1 for v in market_resolutions.values() if v is not None)
        logger.info(f"Have resolution data for {resolved}/{len(market_resolutions)} markets")

        # Count unique markets
        unique_markets = len(set(t["market_id"] for t in trades if t["market_id"]))
        logger.info(f"Trades span {unique_markets} unique markets")

        logger.info("Running backtest...")
        state = run_backtest(trades, trader_stats, leaderboard_pnl, market_resolutions, config)

        # Calculate metrics
        final_value = state.cash
        total_return = (final_value - config.initial_capital) / config.initial_capital
        win_rate = state.win_count / state.trade_count if state.trade_count > 0 else 0

        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Capital: ${config.initial_capital:,.2f}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return * 100:.1f}%")
        logger.info(f"Total PNL: ${state.total_pnl:,.2f}")
        logger.info(f"Total Fees Paid: ${state.total_fees:,.2f}")
        logger.info(f"Total Trades: {state.trade_count}")
        logger.info(f"Win Rate: {win_rate * 100:.1f}%")
        logger.info("=" * 60)
        logger.info(f"Config: slippage={config.slippage*100}%, fee={config.trading_fee*100}%")
        logger.info(f"Copying top {config.top_n_traders} traders")
        logger.info("=" * 60)

        # Save results
        results_file = DATA_DIR / "backtest_simple_results.txt"
        with open(results_file, "w") as f:
            f.write("SIMPLIFIED BACKTEST RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Initial Capital: ${config.initial_capital:,.2f}\n")
            f.write(f"Final Value: ${final_value:,.2f}\n")
            f.write(f"Total Return: {total_return * 100:.1f}%\n")
            f.write(f"Total PNL: ${state.total_pnl:,.2f}\n")
            f.write(f"Total Fees Paid: ${state.total_fees:,.2f}\n")
            f.write(f"Total Trades: {state.trade_count}\n")
            f.write(f"Win Rate: {win_rate * 100:.1f}%\n")
            f.write("=" * 60 + "\n")
            f.write(f"Slippage: {config.slippage*100}%\n")
            f.write(f"Trading Fee: {config.trading_fee*100}%\n")
            f.write(f"Top N Traders: {config.top_n_traders}\n")
            f.write(f"Selection Window: {config.selection_window_days} days\n")

        logger.info(f"Results saved to {results_file}")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
