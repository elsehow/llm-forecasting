"""Integration tests for Yahoo Finance source."""

import pytest

from llm_forecasting.models import SourceType
from llm_forecasting.sources import registry
from llm_forecasting.sources.yfinance import YahooFinanceSource


@pytest.mark.integration
class TestYahooFinanceSource:
    """Integration tests for Yahoo Finance source."""

    @pytest.fixture
    def source(self):
        return YahooFinanceSource()

    async def test_fetch_questions_returns_questions(self, source: YahooFinanceSource):
        """Fetch real data from Yahoo Finance."""
        questions = await source.fetch_questions()

        assert len(questions) > 0, "Should fetch at least one question"

        q = questions[0]
        assert q.id, "Question should have an ID (ticker symbol)"
        assert q.source == "yfinance"
        assert q.source_type == SourceType.DATA
        assert q.base_rate is not None, "Should have current price"
        assert q.base_rate > 0, "Price should be positive"

    async def test_fetch_resolution_for_ticker(self, source: YahooFinanceSource):
        """Fetch current price for a ticker."""
        resolution = await source.fetch_resolution("^GSPC")  # S&P 500

        assert resolution is not None
        assert resolution.source == "yfinance"
        assert resolution.value > 0

    async def test_source_is_registered(self):
        """Verify Yahoo Finance source is in the registry."""
        assert "yfinance" in registry
        assert registry.get("yfinance") is YahooFinanceSource
