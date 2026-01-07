"""Manifold Markets question source."""

import logging
from datetime import date, datetime, timezone

import httpx

from llm_forecasting.models import Question, QuestionType, Resolution
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.manifold.markets/v0"
TOPIC_SLUGS = ["entertainment", "sports-default", "technology-default"]


@registry.register
class ManifoldSource(QuestionSource):
    """Fetch questions from Manifold Markets prediction market."""

    name = "manifold"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _search_markets(
        self,
        topic_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search for binary markets on Manifold."""
        client = await self._get_client()
        params = {
            "sort": "most-popular",
            "contractType": "BINARY",
            "filter": "open",
            "limit": limit,
        }
        if topic_slug:
            params["topicSlug"] = topic_slug

        response = await client.get(f"{BASE_URL}/search-markets", params=params)
        response.raise_for_status()
        return response.json()

    async def _get_market(self, market_id: str) -> dict:
        """Get details for a specific market."""
        client = await self._get_client()
        response = await client.get(f"{BASE_URL}/market/{market_id}")
        response.raise_for_status()
        return response.json()

    async def _get_market_bets(self, market_id: str, limit: int = 1000) -> list[dict]:
        """Get betting history for a market."""
        client = await self._get_client()
        params = {"contractId": market_id, "limit": limit}
        all_bets = []

        while True:
            response = await client.get(f"{BASE_URL}/bets", params=params)
            response.raise_for_status()
            bets = response.json()

            if not bets:
                break

            all_bets.extend(bets)
            if len(bets) < limit:
                break
            params["before"] = bets[-1]["id"]

        return all_bets

    def _epoch_ms_to_datetime(self, epoch_ms: int) -> datetime:
        return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)

    def _market_to_question(self, market: dict) -> Question:
        """Convert a Manifold market to a Question."""
        created_at = self._epoch_ms_to_datetime(market["createdTime"])
        close_time = market.get("closeTime")
        resolution_date = None
        if close_time:
            resolution_date = self._epoch_ms_to_datetime(close_time).date()

        resolved = market.get("isResolved", False)
        resolution_value = None
        if resolved:
            res = market.get("resolution")
            if res == "YES":
                resolution_value = 1.0
            elif res == "NO":
                resolution_value = 0.0
            elif res == "MKT":
                resolution_value = market.get("resolutionProbability")
            # CANCEL -> None (nullified)

        return Question(
            id=market["id"],
            source=self.name,
            text=market["question"],
            background=market.get("textDescription"),
            url=market.get("url"),
            question_type=QuestionType.BINARY,
            created_at=created_at,
            resolution_date=resolution_date,
            resolved=resolved,
            resolution_value=resolution_value,
            base_rate=market.get("probability"),
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch popular binary markets from Manifold."""
        seen_ids = set()
        questions = []

        # Fetch from general + specific topics
        for topic in [None] + TOPIC_SLUGS:
            markets = await self._search_markets(topic_slug=topic)
            for market in markets:
                if market["id"] not in seen_ids:
                    seen_ids.add(market["id"])
                    questions.append(self._market_to_question(market))

        logger.info(f"Fetched {len(questions)} questions from Manifold")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific market.

        Returns the current market probability or resolved value.
        """
        market = await self._get_market(question_id)

        if market.get("isResolved"):
            res = market.get("resolution")
            if res == "YES":
                value = 1.0
            elif res == "NO":
                value = 0.0
            elif res == "MKT":
                value = market.get("resolutionProbability")
            else:
                return None  # CANCEL/N/A

            res_time = market.get("resolutionTime")
            res_date = self._epoch_ms_to_datetime(res_time).date() if res_time else date.today()

            return Resolution(
                question_id=question_id,
                source=self.name,
                date=res_date,
                value=value,
            )

        # Not resolved - return current probability
        prob = market.get("probability")
        if prob is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=prob,
            )

        return None

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
