"""Metaculus forecasting platform source."""

import logging
from datetime import date

import httpx

from llm_forecasting.market_data.metaculus import MetaculusData
from llm_forecasting.market_data.models import Market, MarketStatus
from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

MIN_FORECASTERS = 50


@registry.register
class MetaculusSource(QuestionSource):
    """Fetch questions from Metaculus forecasting platform.

    Uses the market_data.MetaculusData provider internally for
    fetching raw market data, then converts to Question objects.
    """

    name = "metaculus"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize Metaculus source.

        Args:
            api_key: Metaculus API key. If not provided, uses settings.metaculus_api_key.
                    Public API works without a key but has rate limits.
            http_client: Optional httpx client.
        """
        self._data_provider = MetaculusData(
            api_key=api_key,
            http_client=http_client,
        )

    def _market_to_question(self, market: Market) -> Question:
        """Convert Market model to Question model."""
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

    async def fetch_questions(self) -> list[Question]:
        """Fetch popular binary questions from Metaculus."""
        markets = await self._data_provider.fetch_markets(
            active_only=True,
            min_forecasters=MIN_FORECASTERS,
        )

        questions = [self._market_to_question(m) for m in markets]
        logger.info(f"Fetched {len(questions)} questions from Metaculus")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific question."""
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

        # Return community prediction as interim value
        if market.current_probability is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=market.current_probability,
            )

        return None

    async def close(self):
        await self._data_provider.close()
