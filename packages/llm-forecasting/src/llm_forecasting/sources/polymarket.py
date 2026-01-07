"""Polymarket prediction market source."""

import json
import logging
from datetime import date, datetime, timezone

import httpx

from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
MIN_LIQUIDITY = 25000


@registry.register
class PolymarketSource(QuestionSource):
    """Fetch questions from Polymarket prediction market."""

    name = "polymarket"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get open markets from Polymarket."""
        client = await self._get_client()
        params = {
            "limit": limit,
            "offset": offset,
            "archived": False,
            "active": True,
            "closed": False,
            "order": "liquidity",
            "ascending": False,
        }
        response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    async def _get_market(self, condition_id: str) -> dict | None:
        """Get a specific market by condition ID."""
        client = await self._get_client()
        params = {"condition_ids": condition_id}
        response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        markets = response.json()
        return markets[0] if markets else None

    def _is_binary_market(self, market: dict) -> bool:
        """Check if market is binary (Yes/No)."""
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            return {s.lower() for s in outcomes} == {"yes", "no"}
        except (json.JSONDecodeError, TypeError):
            return False

    def _get_yes_index(self, market: dict) -> int:
        """Get the index of the 'Yes' outcome."""
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            return 0 if outcomes[0].lower() == "yes" else 1
        except (json.JSONDecodeError, TypeError, IndexError):
            return 0

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse ISO datetime string."""
        if not dt_str:
            return None
        try:
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str)
        except ValueError:
            return None

    def _market_to_question(self, market: dict) -> Question | None:
        """Convert a Polymarket market to our Question model."""
        # Only process binary markets with sufficient liquidity
        if not self._is_binary_market(market):
            return None

        liquidity = market.get("liquidityNum", 0)
        if liquidity < MIN_LIQUIDITY:
            return None

        # Skip "catch-all" markets
        if "other" in market.get("slug", "").lower():
            return None

        # Parse timestamps
        created_at = self._parse_datetime(market.get("startDateIso"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        # Get end date
        end_date_str = market.get("endDate")
        if not end_date_str:
            events = market.get("events", [])
            if events:
                end_date_str = events[0].get("endDate")

        resolution_date = None
        if end_date_str:
            end_dt = self._parse_datetime(end_date_str)
            if end_dt:
                resolution_date = end_dt.date()

        # Resolution status
        resolved = market.get("umaResolutionStatus", "") == "resolved"
        resolution_value = None
        if resolved:
            try:
                yes_idx = self._get_yes_index(market)
                prices = json.loads(market.get("outcomePrices", "[]"))
                resolution_value = float(prices[yes_idx])
            except (json.JSONDecodeError, TypeError, IndexError, ValueError):
                pass

        # Get current price as base rate
        base_rate = None
        try:
            yes_idx = self._get_yes_index(market)
            prices = json.loads(market.get("outcomePrices", "[]"))
            base_rate = float(prices[yes_idx])
        except (json.JSONDecodeError, TypeError, IndexError, ValueError):
            pass

        return Question(
            id=market.get("conditionId", ""),
            source=self.name,
            source_type=SourceType.MARKET,
            text=market.get("question", ""),
            background=market.get("description"),
            url=f"https://polymarket.com/market/{market.get('slug', '')}",
            question_type=QuestionType.BINARY,
            created_at=created_at,
            resolution_date=resolution_date,
            resolved=resolved,
            resolution_value=resolution_value,
            base_rate=base_rate,
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch open markets from Polymarket."""
        questions = []
        offset = 0
        limit = 100

        while True:
            markets = await self._get_markets(limit=limit, offset=offset)
            if not markets:
                break

            for market in markets:
                q = self._market_to_question(market)
                if q:
                    questions.append(q)

            if len(markets) < limit:
                break
            offset += limit

        logger.info(f"Fetched {len(questions)} questions from Polymarket")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific market."""
        market = await self._get_market(question_id)
        if not market:
            return None

        if market.get("umaResolutionStatus", "") == "resolved":
            try:
                yes_idx = self._get_yes_index(market)
                prices = json.loads(market.get("outcomePrices", "[]"))
                value = float(prices[yes_idx])
                return Resolution(
                    question_id=question_id,
                    source=self.name,
                    date=date.today(),
                    value=value,
                )
            except (json.JSONDecodeError, TypeError, IndexError, ValueError):
                pass

        # Return current price as interim value
        try:
            yes_idx = self._get_yes_index(market)
            prices = json.loads(market.get("outcomePrices", "[]"))
            value = float(prices[yes_idx])
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=value,
            )
        except (json.JSONDecodeError, TypeError, IndexError, ValueError):
            pass

        return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
