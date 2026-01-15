"""SQLite storage backend using SQLAlchemy."""

import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_forecasting.models import Forecast, Question, QuestionType, Resolution, SourceType
from llm_forecasting.storage.base import Storage


class Base(DeclarativeBase):
    pass


class QuestionRow(Base):
    """Questions from all sources."""

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    source_type: Mapped[str] = mapped_column(String, default="market")
    text: Mapped[str] = mapped_column(Text)
    background: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    question_type: Mapped[str] = mapped_column(String, default="binary")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    resolution_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_rate: Mapped[float | None] = mapped_column(Float, nullable=True)


class QuestionSetRow(Base):
    """Curated question sets for evaluation."""

    __tablename__ = "question_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True)  # e.g., "2025-01-15-llm"
    created_at: Mapped[datetime] = mapped_column(DateTime)
    freeze_date: Mapped[date] = mapped_column(Date)  # When questions were frozen
    forecast_due_date: Mapped[date] = mapped_column(Date)  # When forecasts are due
    resolution_dates: Mapped[str] = mapped_column(Text)  # JSON list of dates
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, forecasting, resolving, completed


class QuestionSetItemRow(Base):
    """Links questions to question sets."""

    __tablename__ = "question_set_items"

    question_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("question_sets.id"), primary_key=True
    )
    question_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    # Snapshot of base_rate at freeze time (for data source resolution)
    base_rate_at_freeze: Mapped[float | None] = mapped_column(Float, nullable=True)


class ForecastRow(Base):
    """Model predictions."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question_set_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("question_sets.id"), nullable=True)
    question_id: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    forecaster: Mapped[str] = mapped_column(String)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    point_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantile_values: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    created_at: Mapped[datetime] = mapped_column(DateTime)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)


class ResolutionRow(Base):
    """Historical price/value snapshots for resolution."""

    __tablename__ = "resolutions"

    question_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class SQLiteStorage(Storage):
    """SQLite storage backend using async SQLAlchemy."""

    def __init__(self, db_path: str | Path = "forecastbench.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create tables if they don't exist."""
        if not self._initialized:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True

    async def _get_session(self) -> AsyncSession:
        await self._ensure_initialized()
        return self._session_factory()

    def _question_to_row(self, question: Question) -> QuestionRow:
        now = datetime.now()
        return QuestionRow(
            id=question.id,
            source=question.source,
            source_type=question.source_type.value,
            text=question.text,
            background=question.background,
            url=question.url,
            question_type=question.question_type.value,
            created_at=question.created_at or now,
            updated_at=now,
            resolution_date=question.resolution_date,
            category=question.category,
            resolved=question.resolved,
            resolution_value=question.resolution_value,
            base_rate=question.base_rate,
        )

    def _row_to_question(self, row: QuestionRow) -> Question:
        return Question(
            id=row.id,
            source=row.source,
            source_type=SourceType(row.source_type),
            text=row.text,
            background=row.background,
            url=row.url,
            question_type=QuestionType(row.question_type),
            created_at=row.created_at,
            resolution_date=row.resolution_date,
            category=row.category,
            resolved=row.resolved,
            resolution_value=row.resolution_value,
            base_rate=row.base_rate,
        )

    def _forecast_to_row(self, forecast: Forecast, question_set_id: int | None = None) -> ForecastRow:
        import json

        quantile_json = None
        if forecast.quantile_values:
            quantile_json = json.dumps(forecast.quantile_values)

        return ForecastRow(
            question_set_id=question_set_id or forecast.question_set_id,
            question_id=forecast.question_id,
            source=forecast.source,
            forecaster=forecast.forecaster,
            probability=forecast.probability,
            point_estimate=forecast.point_estimate,
            quantile_values=quantile_json,
            created_at=forecast.created_at,
            reasoning=forecast.reasoning,
        )

    def _row_to_forecast(self, row: ForecastRow) -> Forecast:
        import json

        quantile_values = None
        if row.quantile_values:
            quantile_values = json.loads(row.quantile_values)

        return Forecast(
            id=row.id,
            question_set_id=row.question_set_id,
            question_id=row.question_id,
            source=row.source,
            forecaster=row.forecaster,
            probability=row.probability,
            point_estimate=row.point_estimate,
            quantile_values=quantile_values,
            created_at=row.created_at,
            reasoning=row.reasoning,
        )

    def _resolution_to_row(self, resolution: Resolution) -> ResolutionRow:
        return ResolutionRow(
            question_id=resolution.question_id,
            source=resolution.source,
            date=resolution.date,
            value=resolution.value,
        )

    def _row_to_resolution(self, row: ResolutionRow) -> Resolution:
        return Resolution(
            question_id=row.question_id,
            source=row.source,
            date=row.date,
            value=row.value,
        )

    def _question_set_row_to_dict(self, row: QuestionSetRow) -> dict:
        """Convert a QuestionSetRow to a dictionary."""
        return {
            "id": row.id,
            "name": row.name,
            "created_at": row.created_at,
            "freeze_date": row.freeze_date,
            "forecast_due_date": row.forecast_due_date,
            "resolution_dates": json.loads(row.resolution_dates),
            "status": row.status,
        }

    async def save_question(self, question: Question) -> None:
        async with await self._get_session() as session:
            row = self._question_to_row(question)
            await session.merge(row)
            await session.commit()

    async def save_questions(self, questions: list[Question]) -> None:
        async with await self._get_session() as session:
            for question in questions:
                row = self._question_to_row(question)
                await session.merge(row)
            await session.commit()

    async def get_question(self, source: str, question_id: str) -> Question | None:
        async with await self._get_session() as session:
            result = await session.execute(
                select(QuestionRow).where(
                    QuestionRow.source == source,
                    QuestionRow.id == question_id,
                )
            )
            row = result.scalar_one_or_none()
            return self._row_to_question(row) if row else None

    async def get_questions(
        self,
        source: str | None = None,
        resolved: bool | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Question]:
        async with await self._get_session() as session:
            stmt = select(QuestionRow)

            if source is not None:
                stmt = stmt.where(QuestionRow.source == source)
            if resolved is not None:
                stmt = stmt.where(QuestionRow.resolved == resolved)
            if category is not None:
                stmt = stmt.where(QuestionRow.category == category)

            stmt = stmt.order_by(QuestionRow.created_at.desc())

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_question(row) for row in rows]

    async def save_forecast(
        self, forecast: Forecast, question_set_id: int | None = None
    ) -> int:
        """Save a forecast and return its ID."""
        async with await self._get_session() as session:
            row = self._forecast_to_row(forecast, question_set_id)
            session.add(row)
            await session.flush()  # Get the ID
            forecast_id = row.id
            await session.commit()
            return forecast_id

    async def save_forecasts(
        self, forecasts: list[Forecast], question_set_id: int | None = None
    ) -> list[int]:
        """Save multiple forecasts and return their IDs."""
        async with await self._get_session() as session:
            ids = []
            for forecast in forecasts:
                row = self._forecast_to_row(forecast, question_set_id)
                session.add(row)
                await session.flush()
                ids.append(row.id)
            await session.commit()
            return ids

    async def get_forecasts(
        self,
        question_id: str | None = None,
        source: str | None = None,
        forecaster: str | None = None,
        question_set_id: int | None = None,
        limit: int | None = None,
    ) -> list[Forecast]:
        async with await self._get_session() as session:
            stmt = select(ForecastRow)

            if question_id is not None:
                stmt = stmt.where(ForecastRow.question_id == question_id)
            if source is not None:
                stmt = stmt.where(ForecastRow.source == source)
            if forecaster is not None:
                stmt = stmt.where(ForecastRow.forecaster == forecaster)
            if question_set_id is not None:
                stmt = stmt.where(ForecastRow.question_set_id == question_set_id)

            stmt = stmt.order_by(ForecastRow.created_at.desc())

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_forecast(row) for row in rows]

    async def save_resolution(self, resolution: Resolution) -> None:
        async with await self._get_session() as session:
            row = self._resolution_to_row(resolution)
            await session.merge(row)
            await session.commit()

    async def get_resolution(
        self, source: str, question_id: str, resolution_date: date | None = None
    ) -> Resolution | None:
        async with await self._get_session() as session:
            stmt = select(ResolutionRow).where(
                ResolutionRow.source == source,
                ResolutionRow.question_id == question_id,
            )

            if resolution_date is not None:
                stmt = stmt.where(ResolutionRow.date == resolution_date)
            else:
                # Get the latest resolution
                stmt = stmt.order_by(ResolutionRow.date.desc())

            result = await session.execute(stmt)
            row = result.scalars().first()
            return self._row_to_resolution(row) if row else None

    async def get_resolutions(
        self,
        question_id: str | None = None,
        source: str | None = None,
    ) -> list[Resolution]:
        async with await self._get_session() as session:
            stmt = select(ResolutionRow)

            if question_id is not None:
                stmt = stmt.where(ResolutionRow.question_id == question_id)
            if source is not None:
                stmt = stmt.where(ResolutionRow.source == source)

            stmt = stmt.order_by(ResolutionRow.date.desc())

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_resolution(row) for row in rows]

    # === Question Set Methods ===

    async def create_question_set(
        self,
        name: str,
        freeze_date: date,
        forecast_due_date: date,
        resolution_dates: list[date],
        questions: list[Question],
    ) -> int:
        """Create a new question set with questions.

        Args:
            name: Unique name for the set (e.g., "2025-01-15-llm")
            freeze_date: When questions were frozen
            forecast_due_date: When forecasts are due
            resolution_dates: List of evaluation dates
            questions: Questions to include in the set

        Returns:
            The ID of the created question set
        """
        async with await self._get_session() as session:
            # Create the question set
            qs_row = QuestionSetRow(
                name=name,
                created_at=datetime.now(),
                freeze_date=freeze_date,
                forecast_due_date=forecast_due_date,
                resolution_dates=json.dumps([d.isoformat() for d in resolution_dates]),
                status="pending",
            )
            session.add(qs_row)
            await session.flush()  # Get the ID

            # Add items
            for q in questions:
                item_row = QuestionSetItemRow(
                    question_set_id=qs_row.id,
                    question_id=q.id,
                    source=q.source,
                    base_rate_at_freeze=q.base_rate,
                )
                session.add(item_row)

            await session.commit()
            return qs_row.id

    async def get_question_set(self, question_set_id: int) -> dict | None:
        """Get a question set by ID."""
        async with await self._get_session() as session:
            result = await session.execute(
                select(QuestionSetRow).where(QuestionSetRow.id == question_set_id)
            )
            row = result.scalar_one_or_none()
            if not row:
                return None

            return self._question_set_row_to_dict(row)

    async def get_question_set_by_name(self, name: str) -> dict | None:
        """Get a question set by name."""
        async with await self._get_session() as session:
            result = await session.execute(
                select(QuestionSetRow).where(QuestionSetRow.name == name)
            )
            row = result.scalar_one_or_none()
            if not row:
                return None

            return self._question_set_row_to_dict(row)

    async def get_question_sets(self, status: str | None = None) -> list[dict]:
        """Get all question sets, optionally filtered by status."""
        async with await self._get_session() as session:
            stmt = select(QuestionSetRow)
            if status:
                stmt = stmt.where(QuestionSetRow.status == status)
            stmt = stmt.order_by(QuestionSetRow.created_at.desc())

            result = await session.execute(stmt)
            rows = result.scalars().all()

            return [self._question_set_row_to_dict(row) for row in rows]

    async def get_question_set_items(self, question_set_id: int) -> list[dict]:
        """Get all items in a question set."""
        async with await self._get_session() as session:
            result = await session.execute(
                select(QuestionSetItemRow).where(
                    QuestionSetItemRow.question_set_id == question_set_id
                )
            )
            rows = result.scalars().all()

            return [
                {
                    "question_id": row.question_id,
                    "source": row.source,
                    "base_rate_at_freeze": row.base_rate_at_freeze,
                }
                for row in rows
            ]

    async def update_question_set_status(self, question_set_id: int, status: str) -> None:
        """Update the status of a question set."""
        async with await self._get_session() as session:
            result = await session.execute(
                select(QuestionSetRow).where(QuestionSetRow.id == question_set_id)
            )
            row = result.scalar_one_or_none()
            if row:
                row.status = status
                await session.commit()

    async def close(self) -> None:
        await self._engine.dispose()
