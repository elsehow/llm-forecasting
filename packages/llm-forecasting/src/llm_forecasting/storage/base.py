"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from datetime import date

from llm_forecasting.models import Forecast, Question, Resolution


class Storage(ABC):
    """Abstract base class for storage backends.

    Storage backends handle persistence of questions, forecasts, and resolutions.
    """

    @abstractmethod
    async def save_question(self, question: Question) -> None:
        """Save a question to storage."""
        ...

    @abstractmethod
    async def save_questions(self, questions: list[Question]) -> None:
        """Save multiple questions to storage."""
        ...

    @abstractmethod
    async def get_question(self, source: str, question_id: str) -> Question | None:
        """Get a question by source and ID."""
        ...

    @abstractmethod
    async def get_questions(
        self,
        source: str | None = None,
        resolved: bool | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Question]:
        """Get questions with optional filters."""
        ...

    @abstractmethod
    async def save_forecast(self, forecast: Forecast) -> None:
        """Save a forecast to storage."""
        ...

    @abstractmethod
    async def save_forecasts(self, forecasts: list[Forecast]) -> None:
        """Save multiple forecasts to storage."""
        ...

    @abstractmethod
    async def get_forecasts(
        self,
        question_id: str | None = None,
        source: str | None = None,
        forecaster: str | None = None,
        limit: int | None = None,
    ) -> list[Forecast]:
        """Get forecasts with optional filters."""
        ...

    @abstractmethod
    async def save_resolution(self, resolution: Resolution) -> None:
        """Save a resolution to storage."""
        ...

    @abstractmethod
    async def get_resolution(
        self, source: str, question_id: str, resolution_date: date | None = None
    ) -> Resolution | None:
        """Get the resolution for a question, optionally at a specific date."""
        ...

    @abstractmethod
    async def get_resolutions(
        self,
        question_id: str | None = None,
        source: str | None = None,
    ) -> list[Resolution]:
        """Get resolutions with optional filters."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        ...
