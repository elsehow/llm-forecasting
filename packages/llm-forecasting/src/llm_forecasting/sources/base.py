"""Abstract base class for question sources."""

from abc import ABC, abstractmethod
from typing import ClassVar

from llm_forecasting.models import Question, Resolution


class SourceRegistry:
    """Registry of available question sources."""

    def __init__(self):
        self._sources: dict[str, type["QuestionSource"]] = {}

    def register(self, cls: type["QuestionSource"]) -> type["QuestionSource"]:
        """Register a question source class."""
        self._sources[cls.name] = cls
        return cls

    def get(self, name: str) -> type["QuestionSource"]:
        """Get a source class by name."""
        if name not in self._sources:
            raise KeyError(f"Unknown source: {name}. Available: {list(self._sources.keys())}")
        return self._sources[name]

    def list(self) -> list[str]:
        """List all registered source names."""
        return list(self._sources.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._sources


registry = SourceRegistry()


class QuestionSource(ABC):
    """Abstract base class for question sources.

    To create a new source:
    1. Subclass QuestionSource
    2. Set the `name` class variable
    3. Implement fetch_questions() and fetch_resolution()
    4. Decorate with @registry.register

    Example:
        @registry.register
        class MySource(QuestionSource):
            name = "mysource"

            async def fetch_questions(self) -> list[Question]:
                ...

            async def fetch_resolution(self, question_id: str) -> Resolution | None:
                ...
    """

    name: ClassVar[str]

    @abstractmethod
    async def fetch_questions(self) -> list[Question]:
        """Fetch all available questions from this source.

        Returns:
            List of Question objects.
        """
        ...

    @abstractmethod
    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch the resolution for a specific question.

        Args:
            question_id: The question ID to fetch resolution for.

        Returns:
            Resolution if the question has resolved, None otherwise.
        """
        ...

    async def fetch_questions_with_resolutions(self) -> list[Question]:
        """Fetch questions and update them with resolution status.

        This is a convenience method that fetches questions and then
        checks for resolutions, updating the Question objects accordingly.
        """
        questions = await self.fetch_questions()
        updated = []
        for q in questions:
            resolution = await self.fetch_resolution(q.id)
            if resolution is not None:
                # Create new Question with resolution data (since Question is frozen)
                q = Question(
                    **{
                        **q.model_dump(),
                        "resolved": True,
                        "resolution_value": resolution.value,
                    }
                )
            updated.append(q)
        return updated
