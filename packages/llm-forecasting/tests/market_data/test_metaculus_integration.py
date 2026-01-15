"""Integration tests for MetaculusData provider."""

import pytest

from llm_forecasting.market_data.metaculus import MetaculusData
from llm_forecasting.market_data.models import MarketStatus


@pytest.mark.integration
class TestMetaculusData:
    """Integration tests for Metaculus data provider."""

    @pytest.fixture
    async def provider(self):
        """Create a MetaculusData provider."""
        p = MetaculusData()
        yield p
        await p.close()

    async def test_fetch_markets(self, provider: MetaculusData):
        """Fetch markets from Metaculus."""
        markets = await provider.fetch_markets(limit=10, min_forecasters=10)

        assert len(markets) > 0
        market = markets[0]
        assert market.platform == "metaculus"
        assert market.id
        assert market.title
        assert market.status in [MarketStatus.OPEN, MarketStatus.CLOSED, MarketStatus.RESOLVED]

    async def test_fetch_single_market(self, provider: MetaculusData):
        """Fetch a single market by ID."""
        # First get a market ID
        markets = await provider.fetch_markets(limit=1, min_forecasters=10)
        assert len(markets) > 0

        market = await provider.fetch_market(markets[0].id)
        assert market is not None
        assert market.id == markets[0].id
        assert market.platform == "metaculus"

    async def test_fetch_nonexistent_market(self, provider: MetaculusData):
        """Fetch a market that doesn't exist."""
        market = await provider.fetch_market("999999999")
        assert market is None

    async def test_markets_have_expected_fields(self, provider: MetaculusData):
        """Verify markets have expected metadata fields."""
        markets = await provider.fetch_markets(limit=5, min_forecasters=10)

        for market in markets:
            # Required fields
            assert market.id
            assert market.platform == "metaculus"
            assert market.title

            # Should have community prediction (current_probability)
            # (may be None if not yet revealed)

            # Should have URL
            assert market.url is not None
            assert "metaculus.com" in market.url

    async def test_markets_with_community_prediction(self, provider: MetaculusData):
        """Verify markets have community predictions where available."""
        markets = await provider.fetch_markets(limit=20, min_forecasters=50)

        # At least some markets should have community predictions
        markets_with_cp = [m for m in markets if m.current_probability is not None]
        assert len(markets_with_cp) > 0

        for market in markets_with_cp:
            assert 0.0 <= market.current_probability <= 1.0
