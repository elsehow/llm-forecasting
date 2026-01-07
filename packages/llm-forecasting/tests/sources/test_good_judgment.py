"""Integration tests for Good Judgment Open source."""

import pytest

from llm_forecasting.models import QuestionType, SourceType
from llm_forecasting.sources import registry
from llm_forecasting.sources.good_judgment import GoodJudgmentSource


@pytest.mark.integration
class TestGoodJudgmentSource:
    """Integration tests for Good Judgment Open source (HTML scraping)."""

    @pytest.fixture
    def source(self):
        return GoodJudgmentSource()

    async def test_fetch_questions_returns_questions(self, source: GoodJudgmentSource):
        """Fetch real questions from Good Judgment Open."""
        questions = await source.fetch_questions(max_questions=3)

        assert len(questions) > 0, "Should fetch at least one question"

        q = questions[0]
        assert q.id, "Question should have an ID"
        assert q.source == "good_judgment"
        assert q.source_type == SourceType.MARKET
        assert q.text, "Question should have text"

    async def test_fetch_questions_have_valid_data(self, source: GoodJudgmentSource):
        """Verify fetched questions have valid data."""
        questions = await source.fetch_questions(max_questions=3)

        for q in questions:
            assert q.id.isdigit(), f"Question ID should be numeric: {q.id}"
            assert len(q.text) > 20, f"Question text too short: {q.text}"
            assert q.url.startswith("https://www.gjopen.com/questions/")

            # Base rate should be between 0 and 1 if present
            if q.base_rate is not None:
                assert 0 <= q.base_rate <= 1, f"Invalid base rate: {q.base_rate}"

    async def test_fetch_resolution_returns_value(self, source: GoodJudgmentSource):
        """Fetch resolution for a specific question."""
        # First get a question ID
        questions = await source.fetch_questions(max_questions=1)
        assert len(questions) > 0

        question_id = questions[0].id
        resolution = await source.fetch_resolution(question_id)

        # Should return a resolution with crowd forecast
        if resolution:
            assert resolution.question_id == question_id
            assert resolution.source == "good_judgment"
            assert 0 <= resolution.value <= 1

    async def test_source_is_registered(self):
        """Verify Good Judgment source is in the registry."""
        source_class = registry.get("good_judgment")
        assert source_class is GoodJudgmentSource

    async def test_close_client(self, source: GoodJudgmentSource):
        """Test that client can be closed."""
        # Trigger client creation
        await source.fetch_questions(max_questions=1)

        # Close should not error
        await source.close()

        # Client should be None after close
        assert source._client is None
