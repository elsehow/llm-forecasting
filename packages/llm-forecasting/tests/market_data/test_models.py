"""Unit tests for market_data models."""

from datetime import datetime, timezone

import pytest

from llm_forecasting.market_data.models import Candle, Market, MarketStatus, PricePoint


class TestMarket:
    """Tests for Market model."""

    def test_create_minimal_market(self):
        """Create a market with only required fields."""
        market = Market(
            id="test-123",
            platform="polymarket",
            title="Will it rain tomorrow?",
        )
        assert market.id == "test-123"
        assert market.platform == "polymarket"
        assert market.status == MarketStatus.OPEN
        assert market.resolved_value is None
        assert market.current_probability is None

    def test_create_market_with_all_fields(self):
        """Create a market with all fields populated."""
        now = datetime.now(timezone.utc)
        market = Market(
            id="test-456",
            platform="polymarket",
            title="Full market test",
            description="Detailed description of the market",
            url="https://example.com/market/test",
            created_at=now,
            close_date=now,
            resolution_date=now.date(),
            status=MarketStatus.RESOLVED,
            resolved_value=1.0,
            current_probability=0.95,
            liquidity=50000.0,
            volume_24h=10000.0,
            volume_total=500000.0,
            num_forecasters=100,
            clob_token_ids=["token1", "token2"],
            fetched_at=now,
        )
        assert market.status == MarketStatus.RESOLVED
        assert market.resolved_value == 1.0
        assert market.liquidity == 50000.0
        assert market.clob_token_ids == ["token1", "token2"]

    def test_market_is_frozen(self):
        """Market model should be immutable."""
        market = Market(
            id="test",
            platform="polymarket",
            title="Test",
        )
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            market.title = "Modified"

    def test_market_status_enum(self):
        """Test MarketStatus enum values."""
        assert MarketStatus.OPEN.value == "open"
        assert MarketStatus.CLOSED.value == "closed"
        assert MarketStatus.RESOLVED.value == "resolved"


class TestCandle:
    """Tests for Candle model."""

    def test_create_candle(self):
        """Create a candle with OHLC data."""
        now = datetime.now(timezone.utc)
        candle = Candle(
            market_id="test",
            platform="polymarket",
            timestamp=now,
            open=0.50,
            high=0.60,
            low=0.45,
            close=0.55,
        )
        assert candle.open == 0.50
        assert candle.high == 0.60
        assert candle.low == 0.45
        assert candle.close == 0.55
        assert candle.volume is None

    def test_create_candle_with_volume(self):
        """Create a candle with volume."""
        now = datetime.now(timezone.utc)
        candle = Candle(
            market_id="test",
            platform="polymarket",
            timestamp=now,
            open=0.50,
            high=0.60,
            low=0.45,
            close=0.55,
            volume=1000.0,
        )
        assert candle.volume == 1000.0


class TestPricePoint:
    """Tests for PricePoint model."""

    def test_create_price_point(self):
        """Create a simple price point."""
        now = datetime.now(timezone.utc)
        point = PricePoint(
            market_id="test",
            platform="polymarket",
            timestamp=now,
            price=0.65,
        )
        assert point.price == 0.65
        assert point.platform == "polymarket"
