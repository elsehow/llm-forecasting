"""Integration tests for Polymarket source."""

import pytest

from llm_forecasting.models import QuestionType, SourceType
from llm_forecasting.sources import registry
from llm_forecasting.sources.polymarket import PolymarketSource


@pytest.mark.integration
class TestPolymarketSource:
    """Integration tests for Polymarket source."""

    @pytest.fixture
    def source(self):
        return PolymarketSource()

    async def test_fetch_questions_returns_questions(self, source: PolymarketSource):
        """Fetch real questions from Polymarket API."""
        questions = await source.fetch_questions()

        assert len(questions) > 0, "Should fetch at least one question"

        q = questions[0]
        assert q.id, "Question should have an ID"
        assert q.source == "polymarket"
        assert q.source_type == SourceType.MARKET
        assert q.question_type == QuestionType.BINARY

    async def test_fetch_questions_have_valid_base_rates(self, source: PolymarketSource):
        """Verify fetched questions have base rates (current prices)."""
        questions = await source.fetch_questions()

        for q in questions[:5]:  # Check first 5
            if q.base_rate is not None:
                assert 0 <= q.base_rate <= 1, "Base rate should be a probability"

    async def test_source_is_registered(self):
        """Verify Polymarket source is in the registry."""
        assert "polymarket" in registry
        assert registry.get("polymarket") is PolymarketSource
