"""Integration tests for PolymarketData provider."""

import pytest

from llm_forecasting.market_data.models import MarketStatus
from llm_forecasting.market_data.polymarket import PolymarketData


@pytest.mark.integration
class TestPolymarketData:
    """Integration tests for Polymarket data provider."""

    @pytest.fixture
    async def provider(self):
        """Create a PolymarketData provider."""
        p = PolymarketData()
        yield p
        await p.close()

    async def test_fetch_markets(self, provider: PolymarketData):
        """Fetch markets from Polymarket."""
        markets = await provider.fetch_markets(limit=10)

        assert len(markets) > 0
        market = markets[0]
        assert market.platform == "polymarket"
        assert market.id
        assert market.title
        assert market.status in [MarketStatus.OPEN, MarketStatus.CLOSED, MarketStatus.RESOLVED]

    async def test_fetch_markets_with_liquidity_filter(self, provider: PolymarketData):
        """Fetch markets with minimum liquidity."""
        markets = await provider.fetch_markets(
            min_liquidity=50000.0,
            limit=10,
        )

        assert len(markets) > 0
        for market in markets:
            assert market.liquidity is not None
            assert market.liquidity >= 50000.0

    async def test_fetch_single_market(self, provider: PolymarketData):
        """Fetch a single market by ID."""
        # First get a market ID
        markets = await provider.fetch_markets(limit=1)
        assert len(markets) > 0

        market = await provider.fetch_market(markets[0].id)
        assert market is not None
        assert market.id == markets[0].id
        assert market.platform == "polymarket"

    async def test_fetch_nonexistent_market(self, provider: PolymarketData):
        """Fetch a market that doesn't exist."""
        market = await provider.fetch_market("nonexistent-market-id-12345")
        assert market is None

    async def test_fetch_price_history(self, provider: PolymarketData):
        """Fetch price history for a market."""
        # Get a market with CLOB tokens
        markets = await provider.fetch_markets(limit=20)
        market_with_tokens = next(
            (m for m in markets if m.clob_token_ids),
            None,
        )

        if market_with_tokens:
            history = await provider.fetch_price_history_by_token(
                market_with_tokens.clob_token_ids[0]
            )
            # May or may not have history depending on market age
            assert isinstance(history, list)
            if history:
                # Verify sorted chronologically
                for i in range(1, len(history)):
                    assert history[i].timestamp >= history[i - 1].timestamp
        else:
            pytest.skip("No markets with CLOB tokens found")

    async def test_markets_have_expected_fields(self, provider: PolymarketData):
        """Verify markets have expected metadata fields."""
        markets = await provider.fetch_markets(limit=5)

        for market in markets:
            # Required fields
            assert market.id
            assert market.platform == "polymarket"
            assert market.title

            # Should have market metrics
            assert market.liquidity is not None
            assert market.volume_24h is not None

            # Should have URL
            assert market.url is not None
            assert "polymarket.com" in market.url
