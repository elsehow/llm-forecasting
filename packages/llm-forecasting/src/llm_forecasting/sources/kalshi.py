"""Kalshi prediction market source.

DEPRECATED: This source is disabled until we obtain API access permissions.
Kalshi's API requires authentication and we don't currently have credentials.

Kalshi is a regulated prediction market in the US.
API docs: https://trading-api.readme.io/reference/getting-started
"""

import logging
from datetime import date, datetime, timezone

import httpx

from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


# NOTE: Not registered - source is deprecated until we get API permissions
# @registry.register
class KalshiSource(QuestionSource):
    """Fetch questions from Kalshi prediction market.

    Kalshi offers regulated prediction markets on events like elections,
    economics, weather, and more.
    """

    name = "kalshi"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _get_markets(self, limit: int = 100, cursor: str | None = None) -> dict:
        """Get markets from Kalshi API."""
        client = await self._get_client()
        params = {"limit": limit, "status": "open"}
        if cursor:
            params["cursor"] = cursor

        response = await client.get(f"{BASE_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    async def _get_market(self, ticker: str) -> dict:
        """Get a specific market by ticker."""
        client = await self._get_client()
        response = await client.get(f"{BASE_URL}/markets/{ticker}")
        response.raise_for_status()
        return response.json()["market"]

    def _market_to_question(self, market: dict) -> Question:
        """Convert a Kalshi market to a Question."""
        # Parse timestamps
        created_at = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
        close_time = market.get("close_time")
        resolution_date = None
        if close_time:
            resolution_date = datetime.fromisoformat(
                close_time.replace("Z", "+00:00")
            ).date()

        # Determine resolution status
        resolved = market.get("status") == "finalized"
        resolution_value = None
        if resolved:
            result = market.get("result")
            resolution_value = 1.0 if result == "yes" else 0.0 if result == "no" else None

        return Question(
            id=market["ticker"],
            source=self.name,
            source_type=SourceType.MARKET,
            text=market["title"],
            background=market.get("subtitle"),
            url=f"https://kalshi.com/markets/{market['ticker']}",
            question_type=QuestionType.BINARY,
            created_at=created_at,
            resolution_date=resolution_date,
            category=market.get("category"),
            resolved=resolved,
            resolution_value=resolution_value,
            # Store current market probability as base_rate for sampling
            base_rate=market.get("last_price"),
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch open markets from Kalshi.

        Note: Kalshi API requires authentication for some endpoints.
        This implementation fetches public market data.
        """
        questions = []
        cursor = None

        while True:
            data = await self._get_markets(cursor=cursor)
            markets = data.get("markets", [])

            for market in markets:
                try:
                    questions.append(self._market_to_question(market))
                except Exception as e:
                    logger.warning(f"Failed to parse market {market.get('ticker')}: {e}")

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

        logger.info(f"Fetched {len(questions)} questions from Kalshi")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific market."""
        market = await self._get_market(question_id)

        if market.get("status") == "finalized":
            result = market.get("result")
            if result in ("yes", "no"):
                return Resolution(
                    question_id=question_id,
                    source=self.name,
                    date=date.today(),
                    value=1.0 if result == "yes" else 0.0,
                )

        # Return current price as interim resolution
        last_price = market.get("last_price")
        if last_price is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=last_price,
            )

        return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
