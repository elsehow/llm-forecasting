"""Unit tests for MarketDataStorage."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_forecasting.market_data.models import Market, MarketStatus, PricePoint
from llm_forecasting.market_data.storage import MarketDataStorage


class TestMarketDataStorage:
    """Tests for MarketDataStorage."""

    @pytest.fixture
    async def storage(self, tmp_db_path: Path):
        """Create a temporary MarketDataStorage."""
        s = MarketDataStorage(tmp_db_path)
        yield s
        await s.close()

    @pytest.fixture
    def sample_market(self) -> Market:
        """Create a sample market for testing."""
        return Market(
            id="test-market-1",
            platform="polymarket",
            title="Will it rain tomorrow?",
            description="Weather forecast market",
            url="https://polymarket.com/market/test",
            created_at=datetime.now(timezone.utc),
            current_probability=0.65,
            liquidity=100000.0,
            volume_24h=5000.0,
            clob_token_ids=["token1", "token2"],
        )

    async def test_save_and_get_market(
        self, storage: MarketDataStorage, sample_market: Market
    ):
        """Save and retrieve a market."""
        await storage.save_market(sample_market)

        retrieved = await storage.get_market("polymarket", "test-market-1")

        assert retrieved is not None
        assert retrieved.id == sample_market.id
        assert retrieved.title == sample_market.title
        assert retrieved.liquidity == sample_market.liquidity
        assert retrieved.clob_token_ids == sample_market.clob_token_ids

    async def test_save_multiple_markets(self, storage: MarketDataStorage):
        """Save and retrieve multiple markets."""
        markets = [
            Market(
                id=f"m-{i}",
                platform="polymarket",
                title=f"Market {i}",
                liquidity=i * 10000.0,
            )
            for i in range(5)
        ]

        await storage.save_markets(markets)

        retrieved = await storage.get_markets(platform="polymarket")
        assert len(retrieved) == 5

    async def test_get_markets_with_filters(self, storage: MarketDataStorage):
        """Filter markets by various criteria."""
        markets = [
            Market(
                id="m-1",
                platform="polymarket",
                title="High liquidity",
                liquidity=100000.0,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="m-2",
                platform="polymarket",
                title="Low liquidity",
                liquidity=5000.0,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="m-3",
                platform="polymarket",
                title="Resolved market",
                liquidity=50000.0,
                status=MarketStatus.RESOLVED,
            ),
            Market(
                id="m-4",
                platform="metaculus",
                title="Metaculus market",
                liquidity=None,
                status=MarketStatus.OPEN,
            ),
        ]

        await storage.save_markets(markets)

        # Filter by platform
        poly_markets = await storage.get_markets(platform="polymarket")
        assert len(poly_markets) == 3

        # Filter by liquidity
        high_liquidity = await storage.get_markets(
            platform="polymarket", min_liquidity=25000.0
        )
        assert len(high_liquidity) == 2

        # Filter by status
        resolved = await storage.get_markets(
            platform="polymarket", status=MarketStatus.RESOLVED
        )
        assert len(resolved) == 1
        assert resolved[0].id == "m-3"

    async def test_market_update_via_merge(self, storage: MarketDataStorage):
        """Saving a market with same ID updates it."""
        m1 = Market(
            id="update-test",
            platform="polymarket",
            title="Original title",
            current_probability=0.50,
        )
        await storage.save_market(m1)

        m2 = Market(
            id="update-test",
            platform="polymarket",
            title="Updated title",
            current_probability=0.75,
        )
        await storage.save_market(m2)

        retrieved = await storage.get_market("polymarket", "update-test")
        assert retrieved.title == "Updated title"
        assert retrieved.current_probability == 0.75

    async def test_save_and_get_price_history(
        self, storage: MarketDataStorage, sample_market: Market
    ):
        """Save and retrieve price history."""
        await storage.save_market(sample_market)

        # Create price points
        points = [
            PricePoint(
                market_id=sample_market.id,
                platform="polymarket",
                timestamp=datetime(2025, 1, i, tzinfo=timezone.utc),
                price=0.50 + i * 0.01,
            )
            for i in range(1, 8)
        ]

        await storage.save_price_history(sample_market.id, "polymarket", points)

        # Retrieve
        history = await storage.get_price_history("polymarket", sample_market.id)
        assert len(history) == 7
        # Should be sorted by timestamp
        assert abs(history[0].close - 0.51) < 0.001
        assert abs(history[6].close - 0.57) < 0.001

    async def test_has_price_history(
        self, storage: MarketDataStorage, sample_market: Market
    ):
        """Check if price history exists."""
        await storage.save_market(sample_market)

        # No history yet
        assert await storage.has_price_history("polymarket", sample_market.id) is False

        # Add history
        points = [
            PricePoint(
                market_id=sample_market.id,
                platform="polymarket",
                timestamp=datetime.now(timezone.utc),
                price=0.65,
            )
        ]
        await storage.save_price_history(sample_market.id, "polymarket", points)

        # Now has history
        assert await storage.has_price_history("polymarket", sample_market.id) is True

    async def test_get_price_history_count(
        self, storage: MarketDataStorage, sample_market: Market
    ):
        """Get count of price history entries."""
        await storage.save_market(sample_market)

        # No history yet
        count = await storage.get_price_history_count("polymarket", sample_market.id)
        assert count == 0

        # Add 10 points
        points = [
            PricePoint(
                market_id=sample_market.id,
                platform="polymarket",
                timestamp=datetime(2025, 1, i, tzinfo=timezone.utc),
                price=0.50 + i * 0.01,
            )
            for i in range(1, 11)
        ]
        await storage.save_price_history(sample_market.id, "polymarket", points)

        count = await storage.get_price_history_count("polymarket", sample_market.id)
        assert count == 10

    async def test_price_history_update_via_merge(
        self, storage: MarketDataStorage, sample_market: Market
    ):
        """Price history entries with same timestamp are updated."""
        await storage.save_market(sample_market)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        # Save initial point
        p1 = PricePoint(
            market_id=sample_market.id,
            platform="polymarket",
            timestamp=ts,
            price=0.50,
        )
        await storage.save_price_history(sample_market.id, "polymarket", [p1])

        # Save updated point with same timestamp
        p2 = PricePoint(
            market_id=sample_market.id,
            platform="polymarket",
            timestamp=ts,
            price=0.75,
        )
        await storage.save_price_history(sample_market.id, "polymarket", [p2])

        # Should only have one entry
        history = await storage.get_price_history("polymarket", sample_market.id)
        assert len(history) == 1
        assert history[0].close == 0.75

    async def test_get_price_history_with_time_range(
        self, storage: MarketDataStorage, sample_market: Market
    ):
        """Filter price history by time range."""
        await storage.save_market(sample_market)

        # Create 10 days of points
        points = [
            PricePoint(
                market_id=sample_market.id,
                platform="polymarket",
                timestamp=datetime(2025, 1, i, tzinfo=timezone.utc),
                price=0.50 + i * 0.01,
            )
            for i in range(1, 11)
        ]
        await storage.save_price_history(sample_market.id, "polymarket", points)

        # Get subset
        start = datetime(2025, 1, 3, tzinfo=timezone.utc)
        end = datetime(2025, 1, 7, tzinfo=timezone.utc)
        history = await storage.get_price_history(
            "polymarket", sample_market.id, start=start, end=end
        )

        assert len(history) == 5  # Days 3, 4, 5, 6, 7
        assert history[0].timestamp.day == 3
        assert history[-1].timestamp.day == 7
