"""SQLite storage for market data."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_forecasting.market_data.models import (
    Candle,
    LeaderboardEntry,
    Market,
    MarketStatus,
    PricePoint,
    TraderActivity,
)


class MarketDataBase(DeclarativeBase):
    """Base class for market data tables.

    Note: Uses separate metadata from main storage to allow
    independent schema management while sharing the same DB file.
    """

    pass


class MarketRow(MarketDataBase):
    """Raw market data cache."""

    __tablename__ = "markets"

    # Composite primary key: (id, platform)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    platform: Mapped[str] = mapped_column(String, primary_key=True)

    # Core data
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime)
    close_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String, default="open")
    resolved_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Current state
    current_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Market metrics
    liquidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_forecasters: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Platform-specific (JSON-encoded)
    clob_token_ids: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list

    # Categories (JSON-encoded lists, primarily for Metaculus)
    topic_categories: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    tournament_categories: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list

    # Cache management
    fetched_at: Mapped[datetime] = mapped_column(DateTime)

    # Indexes for common queries
    __table_args__ = (
        Index("ix_markets_platform_status", "platform", "status"),
        Index("ix_markets_platform_liquidity", "platform", "liquidity"),
        Index("ix_markets_fetched_at", "fetched_at"),
    )


class PriceHistoryRow(MarketDataBase):
    """Price history cache (OHLC candles or price points)."""

    __tablename__ = "price_history"

    # Composite primary key
    market_id: Mapped[str] = mapped_column(String, primary_key=True)
    platform: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    # OHLC data
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_price_history_market", "platform", "market_id"),)


class LeaderboardSnapshotRow(MarketDataBase):
    """Leaderboard snapshot at a point in time."""

    __tablename__ = "leaderboard_snapshots"

    # Composite primary key
    user_address: Mapped[str] = mapped_column(String, primary_key=True)
    time_period: Mapped[str] = mapped_column(String, primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    # Leaderboard data
    rank: Mapped[int] = mapped_column(Integer)
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    pnl: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String, default="OVERALL")
    profile_image: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_leaderboard_time_period", "time_period", "fetched_at"),
        Index("ix_leaderboard_rank", "time_period", "rank"),
    )


class TraderActivityRow(MarketDataBase):
    """Individual trader activity/trade."""

    __tablename__ = "trader_activity"

    # Composite primary key
    user_address: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    condition_id: Mapped[str] = mapped_column(String, primary_key=True)
    transaction_hash: Mapped[str] = mapped_column(String, primary_key=True)

    # Activity data
    activity_type: Mapped[str] = mapped_column(String)  # TRADE, SPLIT, etc.
    side: Mapped[str | None] = mapped_column(String, nullable=True)  # BUY, SELL
    size: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    usdc_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Market metadata (denormalized)
    market_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    market_slug: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_trader_activity_user", "user_address", "timestamp"),
        Index("ix_trader_activity_market", "condition_id", "timestamp"),
    )


class MarketDataStorage:
    """SQLite storage for market data.

    Can share the same database file as the main storage,
    but uses separate tables (markets, price_history).
    """

    def __init__(self, db_path: str | Path = "forecastbench.db"):
        """Initialize market data storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create tables if they don't exist."""
        if not self._initialized:
            async with self._engine.begin() as conn:
                await conn.run_sync(MarketDataBase.metadata.create_all)
            self._initialized = True

    async def _get_session(self) -> AsyncSession:
        await self._ensure_initialized()
        return self._session_factory()

    # === Market methods ===

    async def save_market(self, market: Market) -> None:
        """Save or update a market."""
        async with await self._get_session() as session:
            row = self._market_to_row(market)
            await session.merge(row)
            await session.commit()

    async def save_markets(self, markets: list[Market]) -> None:
        """Save multiple markets."""
        async with await self._get_session() as session:
            for market in markets:
                row = self._market_to_row(market)
                await session.merge(row)
            await session.commit()

    async def get_market(self, platform: str, market_id: str) -> Market | None:
        """Get a single market."""
        async with await self._get_session() as session:
            result = await session.execute(
                select(MarketRow).where(
                    MarketRow.platform == platform,
                    MarketRow.id == market_id,
                )
            )
            row = result.scalar_one_or_none()
            return self._row_to_market(row) if row else None

    async def get_markets(
        self,
        platform: str | None = None,
        status: MarketStatus | None = None,
        min_liquidity: float | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[Market]:
        """Get markets with optional filters."""
        async with await self._get_session() as session:
            stmt = select(MarketRow)

            if platform:
                stmt = stmt.where(MarketRow.platform == platform)
            if status:
                stmt = stmt.where(MarketRow.status == status.value)
            if min_liquidity is not None:
                stmt = stmt.where(MarketRow.liquidity >= min_liquidity)
            if min_volume is not None:
                stmt = stmt.where(MarketRow.volume_24h >= min_volume)

            stmt = stmt.order_by(MarketRow.liquidity.desc().nullslast())

            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_market(row) for row in rows]

    async def search_by_title(
        self,
        keywords: list[str],
        platform: str | None = None,
        min_liquidity: float | None = None,
        status: MarketStatus | None = None,
        limit: int = 20,
    ) -> list[Market]:
        """Search markets by title keywords (OR logic).

        Args:
            keywords: List of keywords to search for (case-insensitive, OR logic)
            platform: Filter to specific platform (e.g., "polymarket")
            min_liquidity: Minimum liquidity filter
            status: Filter by market status
            limit: Maximum number of results to return

        Returns:
            List of markets matching any keyword, ordered by liquidity
        """
        if not keywords:
            return []

        async with await self._get_session() as session:
            from sqlalchemy import or_

            # Build OR conditions for keywords
            keyword_conditions = [
                MarketRow.title.ilike(f"%{kw}%") for kw in keywords
            ]
            stmt = select(MarketRow).where(or_(*keyword_conditions))

            if platform:
                stmt = stmt.where(MarketRow.platform == platform)
            if status:
                stmt = stmt.where(MarketRow.status == status.value)
            if min_liquidity is not None:
                stmt = stmt.where(MarketRow.liquidity >= min_liquidity)

            stmt = stmt.order_by(MarketRow.liquidity.desc().nullslast())
            stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_market(row) for row in rows]

    # === Price history methods ===

    async def save_price_history(
        self,
        market_id: str,
        platform: str,
        candles: list[Candle] | list[PricePoint],
    ) -> None:
        """Save price history for a market."""
        async with await self._get_session() as session:
            for item in candles:
                if isinstance(item, PricePoint):
                    # Convert PricePoint to row (OHLC all same)
                    row = PriceHistoryRow(
                        market_id=market_id,
                        platform=platform,
                        timestamp=item.timestamp,
                        open=item.price,
                        high=item.price,
                        low=item.price,
                        close=item.price,
                    )
                else:
                    row = PriceHistoryRow(
                        market_id=market_id,
                        platform=platform,
                        timestamp=item.timestamp,
                        open=item.open,
                        high=item.high,
                        low=item.low,
                        close=item.close,
                        volume=item.volume,
                    )
                await session.merge(row)
            await session.commit()

    async def get_price_history(
        self,
        platform: str,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Get price history for a market."""
        async with await self._get_session() as session:
            stmt = select(PriceHistoryRow).where(
                PriceHistoryRow.platform == platform,
                PriceHistoryRow.market_id == market_id,
            )

            if start:
                stmt = stmt.where(PriceHistoryRow.timestamp >= start)
            if end:
                stmt = stmt.where(PriceHistoryRow.timestamp <= end)

            stmt = stmt.order_by(PriceHistoryRow.timestamp)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_candle(row) for row in rows]

    async def has_price_history(self, platform: str, market_id: str) -> bool:
        """Check if we have cached price history for a market."""
        async with await self._get_session() as session:
            result = await session.execute(
                select(PriceHistoryRow.timestamp)
                .where(
                    PriceHistoryRow.platform == platform,
                    PriceHistoryRow.market_id == market_id,
                )
                .limit(1)
            )
            return result.scalar_one_or_none() is not None

    async def get_price_history_count(self, platform: str, market_id: str) -> int:
        """Get the number of price history entries for a market."""
        async with await self._get_session() as session:
            from sqlalchemy import func

            result = await session.execute(
                select(func.count())
                .select_from(PriceHistoryRow)
                .where(
                    PriceHistoryRow.platform == platform,
                    PriceHistoryRow.market_id == market_id,
                )
            )
            return result.scalar_one()

    # === Leaderboard methods ===

    async def save_leaderboard_snapshot(
        self, entries: list[LeaderboardEntry], fetched_at: datetime | None = None
    ) -> None:
        """Save a leaderboard snapshot."""
        if fetched_at is None:
            fetched_at = datetime.now(timezone.utc)

        async with await self._get_session() as session:
            for entry in entries:
                row = LeaderboardSnapshotRow(
                    user_address=entry.user_address,
                    time_period=entry.time_period,
                    fetched_at=fetched_at,
                    rank=entry.rank,
                    username=entry.username,
                    pnl=entry.pnl,
                    volume=entry.volume,
                    category=entry.category,
                    profile_image=entry.profile_image,
                )
                await session.merge(row)
            await session.commit()

    async def get_leaderboard_snapshot(
        self,
        time_period: str,
        fetched_at: datetime | None = None,
        limit: int = 100,
    ) -> list[LeaderboardEntry]:
        """Get leaderboard snapshot, optionally at a specific time.

        If fetched_at is None, returns the most recent snapshot.
        """
        async with await self._get_session() as session:
            if fetched_at is None:
                # Find most recent snapshot time for this period
                from sqlalchemy import func

                result = await session.execute(
                    select(func.max(LeaderboardSnapshotRow.fetched_at)).where(
                        LeaderboardSnapshotRow.time_period == time_period
                    )
                )
                fetched_at = result.scalar_one_or_none()
                if fetched_at is None:
                    return []

            stmt = (
                select(LeaderboardSnapshotRow)
                .where(
                    LeaderboardSnapshotRow.time_period == time_period,
                    LeaderboardSnapshotRow.fetched_at == fetched_at,
                )
                .order_by(LeaderboardSnapshotRow.rank)
                .limit(limit)
            )

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_leaderboard_entry(row) for row in rows]

    async def get_leaderboard_history(
        self,
        user_address: str,
        time_period: str = "ALL",
    ) -> list[LeaderboardEntry]:
        """Get historical leaderboard entries for a user."""
        async with await self._get_session() as session:
            stmt = (
                select(LeaderboardSnapshotRow)
                .where(
                    LeaderboardSnapshotRow.user_address == user_address,
                    LeaderboardSnapshotRow.time_period == time_period,
                )
                .order_by(LeaderboardSnapshotRow.fetched_at)
            )

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_leaderboard_entry(row) for row in rows]

    # === Trader activity methods ===

    async def save_trader_activity(self, activities: list[TraderActivity]) -> None:
        """Save trader activities."""
        async with await self._get_session() as session:
            for activity in activities:
                row = TraderActivityRow(
                    user_address=activity.user_address,
                    timestamp=activity.timestamp,
                    condition_id=activity.condition_id,
                    transaction_hash=activity.transaction_hash or "",
                    activity_type=activity.activity_type,
                    side=activity.side,
                    size=activity.size,
                    price=activity.price,
                    usdc_size=activity.usdc_size,
                    outcome_index=activity.outcome_index,
                    market_title=activity.market_title,
                    market_slug=activity.market_slug,
                )
                await session.merge(row)
            await session.commit()

    async def get_trader_activity(
        self,
        user_address: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        activity_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[TraderActivity]:
        """Get activity for a trader."""
        async with await self._get_session() as session:
            stmt = select(TraderActivityRow).where(
                TraderActivityRow.user_address == user_address
            )

            if start:
                stmt = stmt.where(TraderActivityRow.timestamp >= start)
            if end:
                stmt = stmt.where(TraderActivityRow.timestamp <= end)
            if activity_types:
                stmt = stmt.where(TraderActivityRow.activity_type.in_(activity_types))

            stmt = stmt.order_by(TraderActivityRow.timestamp.desc())

            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_trader_activity(row) for row in rows]

    async def get_market_trades(
        self,
        condition_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[TraderActivity]:
        """Get all trades for a specific market."""
        async with await self._get_session() as session:
            stmt = select(TraderActivityRow).where(
                TraderActivityRow.condition_id == condition_id,
                TraderActivityRow.activity_type == "TRADE",
            )

            if start:
                stmt = stmt.where(TraderActivityRow.timestamp >= start)
            if end:
                stmt = stmt.where(TraderActivityRow.timestamp <= end)

            stmt = stmt.order_by(TraderActivityRow.timestamp)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_trader_activity(row) for row in rows]

    async def get_tracked_traders(self) -> list[str]:
        """Get list of all traders we have activity for."""
        async with await self._get_session() as session:
            from sqlalchemy import distinct

            result = await session.execute(
                select(distinct(TraderActivityRow.user_address))
            )
            return [row[0] for row in result.fetchall()]

    async def get_trader_activity_count(self, user_address: str) -> int:
        """Get count of activities for a trader."""
        async with await self._get_session() as session:
            from sqlalchemy import func

            result = await session.execute(
                select(func.count())
                .select_from(TraderActivityRow)
                .where(TraderActivityRow.user_address == user_address)
            )
            return result.scalar_one()

    # === Conversion helpers ===

    def _market_to_row(self, market: Market) -> MarketRow:
        return MarketRow(
            id=market.id,
            platform=market.platform,
            title=market.title,
            description=market.description,
            url=market.url,
            created_at=market.created_at,
            close_date=market.close_date,
            resolution_date=market.resolution_date,
            status=market.status.value,
            resolved_value=market.resolved_value,
            current_probability=market.current_probability,
            liquidity=market.liquidity,
            volume_24h=market.volume_24h,
            volume_total=market.volume_total,
            num_forecasters=market.num_forecasters,
            clob_token_ids=(
                json.dumps(market.clob_token_ids) if market.clob_token_ids else None
            ),
            topic_categories=(
                json.dumps(market.topic_categories) if market.topic_categories else None
            ),
            tournament_categories=(
                json.dumps(market.tournament_categories)
                if market.tournament_categories
                else None
            ),
            fetched_at=market.fetched_at,
        )

    def _row_to_market(self, row: MarketRow) -> Market:
        return Market(
            id=row.id,
            platform=row.platform,
            title=row.title,
            description=row.description,
            url=row.url,
            created_at=row.created_at,
            close_date=row.close_date,
            resolution_date=row.resolution_date,
            status=MarketStatus(row.status),
            resolved_value=row.resolved_value,
            current_probability=row.current_probability,
            liquidity=row.liquidity,
            volume_24h=row.volume_24h,
            volume_total=row.volume_total,
            num_forecasters=row.num_forecasters,
            clob_token_ids=(
                json.loads(row.clob_token_ids) if row.clob_token_ids else None
            ),
            topic_categories=(
                json.loads(row.topic_categories) if row.topic_categories else None
            ),
            tournament_categories=(
                json.loads(row.tournament_categories)
                if row.tournament_categories
                else None
            ),
            fetched_at=row.fetched_at,
        )

    def _row_to_candle(self, row: PriceHistoryRow) -> Candle:
        return Candle(
            market_id=row.market_id,
            platform=row.platform,
            timestamp=row.timestamp,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )

    def _row_to_leaderboard_entry(self, row: LeaderboardSnapshotRow) -> LeaderboardEntry:
        return LeaderboardEntry(
            rank=row.rank,
            user_address=row.user_address,
            username=row.username,
            pnl=row.pnl,
            volume=row.volume,
            profile_image=row.profile_image,
            time_period=row.time_period,
            category=row.category,
            fetched_at=row.fetched_at,
        )

    def _row_to_trader_activity(self, row: TraderActivityRow) -> TraderActivity:
        return TraderActivity(
            user_address=row.user_address,
            timestamp=row.timestamp,
            condition_id=row.condition_id,
            activity_type=row.activity_type,
            side=row.side,
            size=row.size,
            price=row.price,
            usdc_size=row.usdc_size,
            outcome_index=row.outcome_index,
            transaction_hash=row.transaction_hash if row.transaction_hash else None,
            market_title=row.market_title,
            market_slug=row.market_slug,
        )

    async def close(self) -> None:
        """Close the database connection."""
        await self._engine.dispose()
