"""Abstract base class for question sources."""

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING, ClassVar

from llm_forecasting.models import Question, QuestionType, Resolution, SourceType

if TYPE_CHECKING:
    from llm_forecasting.market_data.base import MarketDataProvider
    from llm_forecasting.market_data.models import Market, MarketStatus


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


class MarketSource(QuestionSource):
    """Base class for sources that use the market_data layer.

    This class provides shared logic for converting Market objects to Questions
    and fetching resolutions. Subclasses should:

    1. Set the `name` class variable
    2. Initialize `_data_provider` in __init__
    3. Implement `fetch_questions()` to call the provider and filter results
    4. Optionally override `_should_include_market()` for custom filtering

    Example:
        @registry.register
        class MyMarketSource(MarketSource):
            name = "mymarket"

            def __init__(self):
                self._data_provider = MyMarketData()

            async def fetch_questions(self) -> list[Question]:
                markets = await self._data_provider.fetch_markets()
                return self._markets_to_questions(markets)
    """

    _data_provider: "MarketDataProvider"

    def _should_include_market(self, market: "Market") -> bool:
        """Return True if this market should be included as a question.

        Override this method to filter out unwanted markets.
        Default implementation includes all markets.
        """
        return True

    def _market_to_question(self, market: "Market") -> Question:
        """Convert a Market model to a Question model.

        This is the shared conversion logic used by all MarketSource subclasses.
        """
        from llm_forecasting.market_data.models import MarketStatus

        return Question(
            id=market.id,
            source=self.name,
            source_type=SourceType.MARKET,
            text=market.title,
            background=market.description,
            url=market.url,
            question_type=QuestionType.BINARY,
            created_at=market.created_at,
            resolution_date=market.resolution_date,
            resolved=market.status == MarketStatus.RESOLVED,
            resolution_value=market.resolved_value,
            base_rate=market.current_probability,
        )

    def _markets_to_questions(self, markets: list["Market"]) -> list[Question]:
        """Convert a list of markets to questions, applying filtering.

        This handles the common pattern of converting markets and filtering
        out unwanted ones.
        """
        questions = []
        for market in markets:
            if self._should_include_market(market):
                questions.append(self._market_to_question(market))
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific market.

        Returns resolved value if market is resolved, otherwise current probability.
        """
        from llm_forecasting.market_data.models import MarketStatus

        market = await self._data_provider.fetch_market(question_id)
        if not market:
            return None

        if market.status == MarketStatus.RESOLVED and market.resolved_value is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=market.resolved_value,
            )

        # Return current probability as interim value
        if market.current_probability is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=market.current_probability,
            )

        return None

    async def close(self):
        """Close the underlying data provider."""
        await self._data_provider.close()
