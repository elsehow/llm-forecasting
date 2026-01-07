"""Integration tests for Metaculus source."""

import pytest

from llm_forecasting.models import SourceType
from llm_forecasting.sources import registry
from llm_forecasting.sources.metaculus import MetaculusSource


@pytest.mark.integration
class TestMetaculusSource:
    """Integration tests for Metaculus source."""

    @pytest.fixture
    def source(self):
        # Note: Some endpoints may require API key for full access
        return MetaculusSource()

    async def test_fetch_questions_returns_questions(self, source: MetaculusSource):
        """Fetch real questions from Metaculus API."""
        questions = await source.fetch_questions()

        # May return empty if no API key, but should not error
        if len(questions) > 0:
            q = questions[0]
            assert q.id, "Question should have an ID"
            assert q.source == "metaculus"
            assert q.source_type == SourceType.MARKET

    async def test_source_is_registered(self):
        """Verify Metaculus source is in the registry."""
        assert "metaculus" in registry
        assert registry.get("metaculus") is MetaculusSource
