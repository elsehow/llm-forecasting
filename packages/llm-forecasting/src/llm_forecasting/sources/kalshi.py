"""Kalshi prediction market source."""

import logging
from datetime import date

import httpx

from llm_forecasting.market_data.models import Market, MarketStatus
from llm_forecasting.market_data.kalshi import KalshiData
from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

# Minimum liquidity to filter out illiquid markets
MIN_LIQUIDITY = 1000


@registry.register
class KalshiSource(QuestionSource):
    """Fetch questions from Kalshi prediction market.

    Uses the market_data.KalshiData provider internally for
    fetching raw market data, then converts to Question objects.
    """

    name = "kalshi"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._data_provider = KalshiData(http_client=http_client)

    def _market_to_question(self, market: Market) -> Question | None:
        """Convert Market model to Question model."""
        # Skip markets without meaningful titles
        if not market.title or len(market.title) < 10:
            return None

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
        """Fetch open markets from Kalshi."""
        markets = await self._data_provider.fetch_markets(
            active_only=True,
            min_liquidity=MIN_LIQUIDITY,
        )

        questions = []
        for market in markets:
            q = self._market_to_question(market)
            if q:
                questions.append(q)

        logger.info(f"Fetched {len(questions)} questions from Kalshi")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific market."""
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
        await self._data_provider.close()
