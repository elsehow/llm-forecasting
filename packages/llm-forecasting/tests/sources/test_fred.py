"""Integration tests for FRED (Federal Reserve Economic Data) source."""

import os

import pytest

from llm_forecasting.models import SourceType
from llm_forecasting.sources import registry
from llm_forecasting.sources.fred import FREDSource


@pytest.mark.integration
class TestFREDSource:
    """Integration tests for FRED source."""

    @pytest.fixture
    def source(self):
        api_key = os.environ.get("FRED_API_KEY")
        return FREDSource(api_key=api_key)

    @pytest.mark.skipif(
        not os.environ.get("FRED_API_KEY"),
        reason="FRED_API_KEY not set",
    )
    async def test_fetch_questions_returns_questions(self, source: FREDSource):
        """Fetch real questions from FRED API."""
        questions = await source.fetch_questions()

        assert len(questions) > 0, "Should fetch at least one question"

        q = questions[0]
        assert q.id, "Question should have an ID (series ID)"
        assert q.source == "fred"
        assert q.source_type == SourceType.DATA
        assert q.base_rate is not None, "Should have current value"

    @pytest.mark.skipif(
        not os.environ.get("FRED_API_KEY"),
        reason="FRED_API_KEY not set",
    )
    async def test_fetch_resolution_for_series(self, source: FREDSource):
        """Fetch current value for a series."""
        resolution = await source.fetch_resolution("DFF")  # Fed funds rate

        assert resolution is not None
        assert resolution.source == "fred"
        assert resolution.value is not None

    async def test_source_is_registered(self):
        """Verify FRED source is in the registry."""
        assert "fred" in registry
        assert registry.get("fred") is FREDSource
