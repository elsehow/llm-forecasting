"""Integration tests for INFER (RAND Forecasting Initiative) source."""

import os

import pytest

from llm_forecasting.models import QuestionType, SourceType
from llm_forecasting.sources import registry
from llm_forecasting.sources.infer import INFERSource


@pytest.mark.integration
class TestINFERSource:
    """Integration tests for INFER source."""

    @pytest.fixture
    def source(self):
        api_key = os.environ.get("INFER_API_KEY")
        return INFERSource(api_key=api_key)

    @pytest.mark.skipif(
        not os.environ.get("INFER_API_KEY"),
        reason="INFER_API_KEY not set",
    )
    async def test_fetch_questions_returns_questions(self, source: INFERSource):
        """Fetch real questions from INFER API."""
        questions = await source.fetch_questions()

        assert len(questions) > 0, "Should fetch at least one question"

        q = questions[0]
        assert q.id, "Question should have an ID"
        assert q.source == "infer"
        assert q.source_type == SourceType.MARKET
        assert q.question_type == QuestionType.BINARY

    @pytest.mark.skipif(
        not os.environ.get("INFER_API_KEY"),
        reason="INFER_API_KEY not set",
    )
    async def test_fetch_questions_have_valid_data(self, source: INFERSource):
        """Verify fetched questions have valid data."""
        questions = await source.fetch_questions()

        for q in questions[:5]:  # Check first 5
            assert q.id is not None
            assert len(q.text) > 0
            assert q.url is not None and q.url.startswith("http")
            if q.base_rate is not None:
                assert 0 <= q.base_rate <= 1, "Base rate should be a probability"

    @pytest.mark.skipif(
        not os.environ.get("INFER_API_KEY"),
        reason="INFER_API_KEY not set",
    )
    async def test_fetch_resolution_for_question(self, source: INFERSource):
        """Fetch resolution for a real question."""
        questions = await source.fetch_questions()
        assert len(questions) > 0

        resolution = await source.fetch_resolution(questions[0].id)

        if resolution is not None:
            assert 0 <= resolution.value <= 1
            assert resolution.source == "infer"
            assert resolution.question_id == questions[0].id

    async def test_source_is_registered(self):
        """Verify INFER source is in the registry."""
        assert "infer" in registry
        assert registry.get("infer") is INFERSource
