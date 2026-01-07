"""Integration tests for storage backends."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from llm_forecasting.models import Forecast, Question, QuestionType, Resolution
from llm_forecasting.storage.sqlite import SQLiteStorage


class TestSQLiteStorage:
    """Integration tests for SQLite storage."""

    @pytest.fixture
    async def storage(self, tmp_db_path: Path):
        """Create a temporary SQLite storage."""
        s = SQLiteStorage(tmp_db_path)
        yield s
        await s.close()

    @pytest.fixture
    def sample_question(self) -> Question:
        return Question(
            id="test-123",
            source="test_source",
            text="Will it rain tomorrow?",
            background="Weather forecast question",
            question_type=QuestionType.BINARY,
            created_at=datetime.now(timezone.utc),
            resolution_date=date(2025, 1, 10),
        )

    @pytest.fixture
    def sample_forecast(self) -> Forecast:
        return Forecast(
            question_id="test-123",
            source="test_source",
            forecaster="gpt-4o",
            probability=0.65,
            reasoning="Based on weather patterns...",
        )

    @pytest.fixture
    def sample_resolution(self) -> Resolution:
        return Resolution(
            question_id="test-123",
            source="test_source",
            date=date(2025, 1, 10),
            value=1.0,
        )

    async def test_save_and_get_question(self, storage: SQLiteStorage, sample_question: Question):
        """Save and retrieve a question."""
        await storage.save_question(sample_question)

        retrieved = await storage.get_question(sample_question.source, sample_question.id)

        assert retrieved is not None
        assert retrieved.id == sample_question.id
        assert retrieved.text == sample_question.text
        assert retrieved.source == sample_question.source

    async def test_save_multiple_questions(self, storage: SQLiteStorage):
        """Save and retrieve multiple questions."""
        questions = [
            Question(
                id=f"q-{i}",
                source="test",
                text=f"Question {i}?",
                created_at=datetime.now(timezone.utc),
            )
            for i in range(5)
        ]

        await storage.save_questions(questions)
        retrieved = await storage.get_questions(source="test")

        assert len(retrieved) == 5

    async def test_get_questions_with_filters(self, storage: SQLiteStorage):
        """Filter questions by various criteria."""
        questions = [
            Question(
                id="q-1",
                source="source_a",
                text="Question 1",
                resolved=True,
                category="science",
                created_at=datetime.now(timezone.utc),
            ),
            Question(
                id="q-2",
                source="source_a",
                text="Question 2",
                resolved=False,
                category="politics",
                created_at=datetime.now(timezone.utc),
            ),
            Question(
                id="q-3",
                source="source_b",
                text="Question 3",
                resolved=True,
                category="science",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        await storage.save_questions(questions)

        # Filter by source
        source_a = await storage.get_questions(source="source_a")
        assert len(source_a) == 2

        # Filter by resolved
        resolved = await storage.get_questions(resolved=True)
        assert len(resolved) == 2

        # Filter by category
        science = await storage.get_questions(category="science")
        assert len(science) == 2

    async def test_save_and_get_forecast(self, storage: SQLiteStorage, sample_forecast: Forecast):
        """Save and retrieve a forecast."""
        await storage.save_forecast(sample_forecast)

        forecasts = await storage.get_forecasts(question_id=sample_forecast.question_id)

        assert len(forecasts) == 1
        assert forecasts[0].probability == sample_forecast.probability
        assert forecasts[0].forecaster == sample_forecast.forecaster

    async def test_save_and_get_resolution(
        self, storage: SQLiteStorage, sample_resolution: Resolution
    ):
        """Save and retrieve a resolution."""
        await storage.save_resolution(sample_resolution)

        resolution = await storage.get_resolution(
            sample_resolution.source, sample_resolution.question_id
        )

        assert resolution is not None
        assert resolution.value == sample_resolution.value
        assert resolution.date == sample_resolution.date

    async def test_question_update_via_merge(self, storage: SQLiteStorage):
        """Saving a question with same ID updates it."""
        q1 = Question(
            id="update-test",
            source="test",
            text="Original text",
            resolved=False,
            created_at=datetime.now(timezone.utc),
        )
        await storage.save_question(q1)

        q2 = Question(
            id="update-test",
            source="test",
            text="Updated text",
            resolved=True,
            created_at=datetime.now(timezone.utc),
        )
        await storage.save_question(q2)

        retrieved = await storage.get_question("test", "update-test")
        assert retrieved.text == "Updated text"
        assert retrieved.resolved is True

    async def test_forecast_returns_id(self, storage: SQLiteStorage, sample_forecast: Forecast):
        """Saving a forecast returns its ID."""
        forecast_id = await storage.save_forecast(sample_forecast)

        assert forecast_id is not None
        assert isinstance(forecast_id, int)
        assert forecast_id > 0

        # Verify the ID is on the retrieved forecast
        forecasts = await storage.get_forecasts(question_id=sample_forecast.question_id)
        assert len(forecasts) == 1
        assert forecasts[0].id == forecast_id

    async def test_forecast_with_question_set_id(self, storage: SQLiteStorage, sample_forecast: Forecast):
        """Forecasts can be saved with a question_set_id."""
        # First create a question set
        q = Question(
            id="qs-test-q",
            source="test",
            text="Test question",
            created_at=datetime.now(timezone.utc),
        )
        await storage.save_question(q)

        qs_id = await storage.create_question_set(
            name="test-qs",
            freeze_date=date.today(),
            forecast_due_date=date.today(),
            resolution_dates=[date.today()],
            questions=[q],
        )

        # Save forecast with question_set_id
        forecast = Forecast(
            question_id="qs-test-q",
            source="test",
            forecaster="test-model",
            probability=0.7,
        )
        forecast_id = await storage.save_forecast(forecast, question_set_id=qs_id)

        # Verify we can filter by question_set_id
        forecasts = await storage.get_forecasts(question_set_id=qs_id)
        assert len(forecasts) == 1
        assert forecasts[0].id == forecast_id
        assert forecasts[0].question_set_id == qs_id

