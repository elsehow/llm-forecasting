"""SQLite storage for market data."""

import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_forecasting.market_data.models import Candle, Market, MarketStatus, PricePoint


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

    async def close(self) -> None:
        """Close the database connection."""
        await self._engine.dispose()
