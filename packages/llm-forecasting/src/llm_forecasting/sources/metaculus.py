"""Metaculus forecasting platform source."""

import logging

import httpx

from llm_forecasting.market_data.metaculus import MetaculusData
from llm_forecasting.models import Question
from llm_forecasting.sources.base import MarketSource, registry

logger = logging.getLogger(__name__)

MIN_FORECASTERS = 50


@registry.register
class MetaculusSource(MarketSource):
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

    async def fetch_questions(self) -> list[Question]:
        """Fetch popular binary questions from Metaculus."""
        markets = await self._data_provider.fetch_markets(
            active_only=True,
            min_forecasters=MIN_FORECASTERS,
        )

        questions = self._markets_to_questions(markets)
        logger.info(f"Fetched {len(questions)} questions from Metaculus")
        return questions
