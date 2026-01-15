"""Polymarket prediction market source."""

import logging

import httpx

from llm_forecasting.market_data.models import Market
from llm_forecasting.market_data.polymarket import PolymarketData
from llm_forecasting.models import Question
from llm_forecasting.sources.base import MarketSource, registry

logger = logging.getLogger(__name__)

MIN_LIQUIDITY = 25000


@registry.register
class PolymarketSource(MarketSource):
    """Fetch questions from Polymarket prediction market.

    Uses the market_data.PolymarketData provider internally for
    fetching raw market data, then converts to Question objects.
    """

    name = "polymarket"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._data_provider = PolymarketData(http_client=http_client)

    def _should_include_market(self, market: Market) -> bool:
        """Skip 'catch-all' markets that have 'other' in the URL."""
        if market.url and "other" in market.url.lower():
            return False
        return True

    async def fetch_questions(self) -> list[Question]:
        """Fetch open markets from Polymarket."""
        markets = await self._data_provider.fetch_markets(
            active_only=True,
            min_liquidity=MIN_LIQUIDITY,
        )

        questions = self._markets_to_questions(markets)
        logger.info(f"Fetched {len(questions)} questions from Polymarket")
        return questions
