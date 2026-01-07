"""End-to-end pipeline tests.

These tests verify the full forecasting pipeline works correctly:
1. Questions are stored
2. Forecasts are made and stored
3. Resolutions are fetched/stored
4. Scores are computed and stored
5. Leaderboard is generated with statistical significance
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from llm_forecasting.models import (
    Forecast,
    Question,
    QuestionType,
    Resolution,
    SourceType,
)
from llm_forecasting.resolution import (
    ResolutionResult,
    resolve_market_question,
    resolve_question,
    score_forecast,
)
from llm_forecasting.eval.scoring import (
    build_leaderboard,
    compute_pairwise_significance,
    format_leaderboard,
)
from llm_forecasting.storage.sqlite import SQLiteStorage


class TestEndToEndPipeline:
    """End-to-end tests for the full forecasting pipeline."""

    @pytest.fixture
    async def storage(self, tmp_db_path: Path):
        """Create a temporary SQLite storage."""
        s = SQLiteStorage(tmp_db_path)
        yield s
        await s.close()

    @pytest.fixture
    def market_questions(self) -> list[Question]:
        """Sample market questions (like Manifold/Polymarket)."""
        return [
            Question(
                id="market-q1",
                source="manifold",
                source_type=SourceType.MARKET,
                text="Will GPT-5 be released before July 2025?",
                question_type=QuestionType.BINARY,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                resolved=True,
                resolution_value=1.0,
                resolution_date=date(2025, 6, 15),
            ),
            Question(
                id="market-q2",
                source="manifold",
                source_type=SourceType.MARKET,
                text="Will Bitcoin reach $100k in 2025?",
                question_type=QuestionType.BINARY,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                resolved=True,
                resolution_value=0.0,
                resolution_date=date(2025, 12, 31),
            ),
            Question(
                id="market-q3",
                source="manifold",
                source_type=SourceType.MARKET,
                text="Will there be a major AI safety incident in 2025?",
                question_type=QuestionType.BINARY,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                resolved=True,
                resolution_value=0.0,
                resolution_date=date(2025, 12, 31),
            ),
            Question(
                id="market-q4",
                source="manifold",
                source_type=SourceType.MARKET,
                text="Will OpenAI release a reasoning model in Q1 2025?",
                question_type=QuestionType.BINARY,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                resolved=True,
                resolution_value=1.0,
                resolution_date=date(2025, 3, 1),
            ),
        ]

    @pytest.fixture
    def data_questions(self) -> list[Question]:
        """Sample data questions (like FRED/Yahoo Finance)."""
        return [
            Question(
                id="SP500",
                source="yfinance",
                source_type=SourceType.DATA,
                text="Will S&P 500 increase from forecast date to horizon?",
                question_type=QuestionType.BINARY,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                base_rate=4500.0,  # Price at creation
            ),
            Question(
                id="DFF",
                source="fred",
                source_type=SourceType.DATA,
                text="Will Fed Funds Rate increase from forecast date to horizon?",
                question_type=QuestionType.BINARY,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                base_rate=4.5,  # Rate at creation
            ),
        ]

    async def test_full_market_pipeline(self, storage: SQLiteStorage, market_questions: list[Question]):
        """Test the full pipeline for market-based questions."""
        # Step 1: Save questions
        await storage.save_questions(market_questions)

        # Verify questions are saved
        saved = await storage.get_questions(source="manifold")
        assert len(saved) == 4

        # Step 2: Create a question set
        forecast_due_date = date(2025, 1, 15)
        resolution_dates = [date(2025, 3, 15), date(2025, 6, 15), date(2025, 12, 31)]

        qs_id = await storage.create_question_set(
            name="2025-q1-llm-test",
            freeze_date=date(2025, 1, 10),
            forecast_due_date=forecast_due_date,
            resolution_dates=resolution_dates,
            questions=market_questions,
        )
        assert qs_id > 0

        # Step 3: Generate forecasts from two "models"
        # Model A: Good forecaster - high confidence on correct outcomes
        model_a_forecasts = [
            Forecast(
                question_id="market-q1",
                source="manifold",
                forecaster="model-a",
                probability=0.85,  # Resolved YES -> good forecast
                reasoning="Strong signals GPT-5 is coming",
            ),
            Forecast(
                question_id="market-q2",
                source="manifold",
                forecaster="model-a",
                probability=0.20,  # Resolved NO -> good forecast
                reasoning="Bitcoin momentum slowing",
            ),
            Forecast(
                question_id="market-q3",
                source="manifold",
                forecaster="model-a",
                probability=0.15,  # Resolved NO -> good forecast
                reasoning="AI safety improving",
            ),
            Forecast(
                question_id="market-q4",
                source="manifold",
                forecaster="model-a",
                probability=0.90,  # Resolved YES -> good forecast
                reasoning="OpenAI o1 hints",
            ),
        ]

        # Model B: Mediocre forecaster - less calibrated
        model_b_forecasts = [
            Forecast(
                question_id="market-q1",
                source="manifold",
                forecaster="model-b",
                probability=0.50,  # Resolved YES -> mediocre
                reasoning="Uncertain",
            ),
            Forecast(
                question_id="market-q2",
                source="manifold",
                forecaster="model-b",
                probability=0.70,  # Resolved NO -> bad forecast
                reasoning="Bitcoin to the moon",
            ),
            Forecast(
                question_id="market-q3",
                source="manifold",
                forecaster="model-b",
                probability=0.50,  # Resolved NO -> mediocre
                reasoning="Hard to say",
            ),
            Forecast(
                question_id="market-q4",
                source="manifold",
                forecaster="model-b",
                probability=0.60,  # Resolved YES -> mediocre
                reasoning="Maybe",
            ),
        ]

        # Save all forecasts
        model_a_ids = await storage.save_forecasts(model_a_forecasts, question_set_id=qs_id)
        model_b_ids = await storage.save_forecasts(model_b_forecasts, question_set_id=qs_id)

        assert len(model_a_ids) == 4
        assert len(model_b_ids) == 4

        # Step 4: Resolve and score forecasts
        question_map = {q.id: q for q in market_questions}
        scores_by_forecaster: dict[str, list[float]] = {"model-a": [], "model-b": []}

        for forecast, forecast_id in zip(model_a_forecasts + model_b_forecasts, model_a_ids + model_b_ids):
            question = question_map[forecast.question_id]

            # For market questions, resolution is the official resolution_value
            resolution = Resolution(
                question_id=question.id,
                source=question.source,
                date=question.resolution_date,
                value=question.resolution_value,
            )

            # Resolve the question
            result = resolve_market_question(question, resolution, forecast_due_date)

            assert result.resolved, f"Question {question.id} should resolve"
            assert result.resolution_value is not None

            # Score the forecast
            brier = score_forecast(forecast, result)
            assert brier is not None

            scores_by_forecaster[forecast.forecaster].append(brier)

        # Step 5: Generate leaderboard using pure functions (scores computed on-the-fly)
        leaderboard = build_leaderboard(scores_by_forecaster, with_confidence=True)

        assert len(leaderboard) == 2
        # Model A should be ranked first (lower Brier = better)
        assert leaderboard[0].forecaster == "model-a"
        assert leaderboard[1].forecaster == "model-b"

        # Model A's Brier scores:
        # q1: (0.85 - 1.0)^2 = 0.0225
        # q2: (0.20 - 0.0)^2 = 0.04
        # q3: (0.15 - 0.0)^2 = 0.0225
        # q4: (0.90 - 1.0)^2 = 0.01
        # Mean: 0.02375
        assert leaderboard[0].mean_brier_score == pytest.approx(0.02375, rel=0.01)

        # Model B's Brier scores:
        # q1: (0.50 - 1.0)^2 = 0.25
        # q2: (0.70 - 0.0)^2 = 0.49
        # q3: (0.50 - 0.0)^2 = 0.25
        # q4: (0.60 - 1.0)^2 = 0.16
        # Mean: 0.2875
        assert leaderboard[1].mean_brier_score == pytest.approx(0.2875, rel=0.01)

        # Step 6: Statistical significance
        entries = build_leaderboard(scores_by_forecaster, with_confidence=True)
        comparisons = compute_pairwise_significance(
            scores_by_forecaster,
            question_ids=[q.id for q in market_questions],
        )

        assert len(entries) == 2
        assert entries[0].forecaster == "model-a"
        assert entries[0].std_error is not None
        assert entries[0].confidence_interval_95 is not None

        # With only 4 questions, might not reach significance
        # but the comparison should still be generated
        assert len(comparisons) == 1
        assert comparisons[0].forecaster_a == "model-a"
        assert comparisons[0].forecaster_b == "model-b"
        assert comparisons[0].score_diff < 0  # model-a is better (lower score)

    async def test_data_question_resolution(self, storage: SQLiteStorage, data_questions: list[Question]):
        """Test resolution logic for data-based questions (FRED, Yahoo Finance)."""
        await storage.save_questions(data_questions)

        # Create resolutions at two time points
        forecast_due_date = date(2025, 1, 15)
        horizon_date = date(2025, 3, 15)

        # S&P 500: went up (4500 -> 4800)
        resolution_sp500_due = Resolution(
            question_id="SP500",
            source="yfinance",
            date=forecast_due_date,
            value=4500.0,
        )
        resolution_sp500_horizon = Resolution(
            question_id="SP500",
            source="yfinance",
            date=horizon_date,
            value=4800.0,  # Increased
        )

        # Fed Funds Rate: went down (4.5 -> 4.25)
        resolution_dff_due = Resolution(
            question_id="DFF",
            source="fred",
            date=forecast_due_date,
            value=4.5,
        )
        resolution_dff_horizon = Resolution(
            question_id="DFF",
            source="fred",
            date=horizon_date,
            value=4.25,  # Decreased
        )

        await storage.save_resolution(resolution_sp500_due)
        await storage.save_resolution(resolution_sp500_horizon)
        await storage.save_resolution(resolution_dff_due)
        await storage.save_resolution(resolution_dff_horizon)

        # Resolve data questions
        sp500_q = data_questions[0]
        dff_q = data_questions[1]

        sp500_result = resolve_question(
            sp500_q,
            resolution=resolution_sp500_horizon,
            forecast_due_date=forecast_due_date,
            resolution_at_due_date=resolution_sp500_due,
        )
        assert sp500_result.resolved
        assert sp500_result.resolution_value == 1.0  # Increased

        dff_result = resolve_question(
            dff_q,
            resolution=resolution_dff_horizon,
            forecast_due_date=forecast_due_date,
            resolution_at_due_date=resolution_dff_due,
        )
        assert dff_result.resolved
        assert dff_result.resolution_value == 0.0  # Decreased

        # Test scoring
        forecast_sp500 = Forecast(
            question_id="SP500",
            source="yfinance",
            forecaster="test-model",
            probability=0.70,  # Predicted increase -> correct
        )
        forecast_dff = Forecast(
            question_id="DFF",
            source="fred",
            forecaster="test-model",
            probability=0.30,  # Predicted decrease -> correct
        )

        brier_sp500 = score_forecast(forecast_sp500, sp500_result)
        brier_dff = score_forecast(forecast_dff, dff_result)

        # S&P: (0.70 - 1.0)^2 = 0.09
        assert brier_sp500 == pytest.approx(0.09)
        # DFF: (0.30 - 0.0)^2 = 0.09
        assert brier_dff == pytest.approx(0.09)

    async def test_question_set_workflow(self, storage: SQLiteStorage, market_questions: list[Question]):
        """Test the full question set workflow: create, forecast, update status."""
        await storage.save_questions(market_questions)

        # Create question set
        qs_id = await storage.create_question_set(
            name="workflow-test",
            freeze_date=date(2025, 1, 10),
            forecast_due_date=date(2025, 1, 15),
            resolution_dates=[date(2025, 3, 15)],
            questions=market_questions,
        )

        # Check initial status
        qs = await storage.get_question_set(qs_id)
        assert qs["status"] == "pending"

        # Update to forecasting
        await storage.update_question_set_status(qs_id, "forecasting")
        qs = await storage.get_question_set(qs_id)
        assert qs["status"] == "forecasting"

        # Add forecasts
        for q in market_questions:
            forecast = Forecast(
                question_id=q.id,
                source=q.source,
                forecaster="test-model",
                probability=0.5,
            )
            await storage.save_forecast(forecast, question_set_id=qs_id)

        # Verify forecasts are linked to question set
        forecasts = await storage.get_forecasts(question_set_id=qs_id)
        assert len(forecasts) == 4

        # Update to resolving
        await storage.update_question_set_status(qs_id, "resolving")
        qs = await storage.get_question_set(qs_id)
        assert qs["status"] == "resolving"

        # Update to completed
        await storage.update_question_set_status(qs_id, "completed")
        qs = await storage.get_question_set(qs_id)
        assert qs["status"] == "completed"

    async def test_leaderboard_formatting(self, storage: SQLiteStorage, market_questions: list[Question]):
        """Test leaderboard formatting with significance."""
        # Create scores for multiple models
        scores_by_forecaster = {
            "claude-3.5-sonnet": [0.05, 0.08, 0.03, 0.06, 0.04, 0.07],  # Good
            "gpt-4o": [0.12, 0.15, 0.10, 0.14, 0.11, 0.13],  # Mediocre
            "gemini-pro": [0.25, 0.30, 0.22, 0.28, 0.24, 0.26],  # Poor
        }

        # Build leaderboard with confidence intervals
        entries = build_leaderboard(scores_by_forecaster, with_confidence=True)

        assert len(entries) == 3
        assert entries[0].forecaster == "claude-3.5-sonnet"
        assert entries[1].forecaster == "gpt-4o"
        assert entries[2].forecaster == "gemini-pro"

        # All should have confidence intervals
        for entry in entries:
            assert entry.std_error is not None
            assert entry.confidence_interval_95 is not None

        # Compute pairwise significance
        question_ids = ["q1", "q2", "q3", "q4", "q5", "q6"]
        comparisons = compute_pairwise_significance(scores_by_forecaster, question_ids)

        # 3 models = 3 pairwise comparisons
        assert len(comparisons) == 3

        # With these clear differences, some comparisons should be significant
        sig_comparisons = [c for c in comparisons if c.is_significant]
        assert len(sig_comparisons) >= 1  # At least claude vs gemini should be significant

        # Format and verify output
        output = format_leaderboard(entries, comparisons)
        assert "Leaderboard" in output
        assert "claude-3.5-sonnet" in output
        assert "gpt-4o" in output
        assert "gemini-pro" in output

        # If there are significant differences, they should be shown
        if sig_comparisons:
            assert "Significant" in output
            assert "beats" in output

    async def test_ambiguous_resolution_skipped(self, storage: SQLiteStorage):
        """Questions with ambiguous resolution (e.g., 0.5 for MKT) should not be scored."""
        question = Question(
            id="ambiguous-q",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Will X happen?",
            question_type=QuestionType.BINARY,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            resolved=True,
            resolution_value=0.5,  # Ambiguous (e.g., N/A or MKT)
            resolution_date=date(2025, 6, 1),
        )

        resolution = Resolution(
            question_id="ambiguous-q",
            source="manifold",
            date=date(2025, 6, 1),
            value=0.5,
        )

        result = resolve_market_question(question, resolution, date(2025, 1, 15))

        assert not result.resolved
        assert "Ambiguous" in result.reason

    async def test_early_resolution_skipped(self, storage: SQLiteStorage):
        """Questions that resolved before forecast due date should not be scored."""
        question = Question(
            id="early-q",
            source="manifold",
            source_type=SourceType.MARKET,
            text="Will early thing happen?",
            question_type=QuestionType.BINARY,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            resolved=True,
            resolution_value=1.0,
            resolution_date=date(2025, 1, 10),  # Before forecast due date
        )

        resolution = Resolution(
            question_id="early-q",
            source="manifold",
            date=date(2025, 1, 10),
            value=1.0,
        )

        # Forecast due date is after resolution
        result = resolve_market_question(question, resolution, date(2025, 1, 15))

        assert not result.resolved
        assert "before forecast due" in result.reason

    async def test_multiple_question_sets(self, storage: SQLiteStorage, market_questions: list[Question]):
        """Different question sets can track forecasts independently."""
        await storage.save_questions(market_questions)

        # Create two question sets
        qs1_id = await storage.create_question_set(
            name="qs1-test",
            freeze_date=date(2025, 1, 10),
            forecast_due_date=date(2025, 1, 15),
            resolution_dates=[date(2025, 3, 15)],
            questions=market_questions[:2],
        )

        qs2_id = await storage.create_question_set(
            name="qs2-test",
            freeze_date=date(2025, 2, 10),
            forecast_due_date=date(2025, 2, 15),
            resolution_dates=[date(2025, 4, 15)],
            questions=market_questions[2:],
        )

        # Add forecasts to each set
        for q in market_questions[:2]:
            f = Forecast(question_id=q.id, source=q.source, forecaster="model", probability=0.5)
            await storage.save_forecast(f, question_set_id=qs1_id)

        for q in market_questions[2:]:
            f = Forecast(question_id=q.id, source=q.source, forecaster="model", probability=0.5)
            await storage.save_forecast(f, question_set_id=qs2_id)

        # Verify independent tracking
        qs1_forecasts = await storage.get_forecasts(question_set_id=qs1_id)
        qs2_forecasts = await storage.get_forecasts(question_set_id=qs2_id)

        assert len(qs1_forecasts) == 2
        assert len(qs2_forecasts) == 2

        # Each set's forecasts should be for different questions
        qs1_qids = {f.question_id for f in qs1_forecasts}
        qs2_qids = {f.question_id for f in qs2_forecasts}
        assert qs1_qids.isdisjoint(qs2_qids)
