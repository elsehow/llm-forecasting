"""Core data models for the market data layer."""

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketStatus(str, Enum):
    """Status of a prediction market."""

    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"


class Market(BaseModel):
    """Raw market data from a prediction market platform.

    This is the canonical representation of market data before
    conversion to a Question. It preserves platform-specific
    fields needed for analysis (e.g., liquidity, volume).
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str  # Platform-specific ID (conditionId, post ID, etc.)
    platform: str  # "polymarket", "metaculus", "manifold", etc.

    # Core data
    title: str  # Question text
    description: str | None = None  # Background/resolution criteria
    url: str | None = None

    # Timing
    created_at: datetime = Field(default_factory=_utc_now)
    close_date: datetime | None = None
    resolution_date: date | None = None

    # Status
    status: MarketStatus = MarketStatus.OPEN
    resolved_value: float | None = None  # 0, 1, or probability for MKT resolution

    # Current state
    current_probability: float | None = None  # Current market price / community prediction

    # Market-specific metadata (platform-dependent)
    liquidity: float | None = None  # USD liquidity (Polymarket, Manifold)
    volume_24h: float | None = None  # 24h volume (Polymarket)
    volume_total: float | None = None  # Total volume (Polymarket, Manifold)
    num_forecasters: int | None = None  # Number of forecasters (Metaculus)

    # Platform-specific IDs for price history
    clob_token_ids: list[str] | None = None  # Polymarket CLOB token IDs

    # Timestamps for cache management
    fetched_at: datetime = Field(default_factory=_utc_now)


class Candle(BaseModel):
    """OHLC price data point."""

    model_config = ConfigDict(frozen=True)

    market_id: str
    platform: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    # Optional volume for this period
    volume: float | None = None


class PricePoint(BaseModel):
    """Simple price/probability at a point in time.

    Simpler than Candle for platforms that only provide point data.
    """

    model_config = ConfigDict(frozen=True)

    market_id: str
    platform: str
    timestamp: datetime
    price: float  # Probability 0-1
