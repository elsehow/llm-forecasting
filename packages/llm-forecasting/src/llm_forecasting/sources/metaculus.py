"""Metaculus forecasting platform source."""

import logging
from datetime import date, datetime, timedelta, timezone

import httpx

from llm_forecasting.config import settings
from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

BASE_URL = "https://www.metaculus.com/api"
MIN_FORECASTERS = 50


@registry.register
class MetaculusSource(QuestionSource):
    """Fetch questions from Metaculus forecasting platform."""

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
        self._api_key = api_key if api_key is not None else settings.metaculus_api_key
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Token {self._api_key}"
            self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self._client

    async def _search_questions(
        self,
        statuses: str = "open",
        forecast_type: str = "binary",
        order_by: str = "-hotness",
        limit: int = 100,
        categories: str | None = None,
    ) -> list[dict]:
        """Search for questions on Metaculus."""
        client = await self._get_client()

        # Only get questions that resolve at least 10 days from now
        min_resolve_date = (date.today() + timedelta(days=10)).isoformat()

        params = {
            "statuses": statuses,
            "with_cp": "false",
            "scheduled_resolve_time__gt": min_resolve_date,
            "forecast_type": forecast_type,
            "order_by": order_by,
            "limit": limit,
            "for_main_feed": "true",
        }
        if categories:
            params["categories"] = categories

        response = await client.get(f"{BASE_URL}/posts/", params=params)
        response.raise_for_status()
        return response.json().get("results", [])

    async def _get_question(self, question_id: str) -> dict:
        """Get a specific question by ID."""
        client = await self._get_client()
        response = await client.get(f"{BASE_URL}/posts/{question_id}")
        response.raise_for_status()
        return response.json()

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse ISO datetime string."""
        if not dt_str:
            return None
        try:
            # Handle Zulu time format
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str)
        except ValueError:
            return None

    def _extract_community_prediction(self, question_data: dict) -> float | None:
        """Extract community prediction from question data."""
        try:
            cp = question_data.get("community_prediction", {})
            if isinstance(cp, dict):
                full = cp.get("full", {})
                if isinstance(full, dict):
                    return full.get("q2")  # Median prediction
            return None
        except (KeyError, TypeError):
            return None

    def _market_to_question(self, market: dict) -> Question | None:
        """Convert a Metaculus question to our Question model."""
        question_data = market.get("question", {})

        # Skip questions with too few forecasters
        if market.get("nr_forecasters", 0) < MIN_FORECASTERS:
            return None

        # Check if community prediction is revealed
        cp_reveal_time = question_data.get("cp_reveal_time")
        if cp_reveal_time:
            reveal_dt = self._parse_datetime(cp_reveal_time)
            if reveal_dt and reveal_dt > datetime.now(timezone.utc):
                return None  # CP not yet revealed

        # Parse timestamps
        created_at = self._parse_datetime(question_data.get("open_time"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        close_time = question_data.get("actual_close_time") or question_data.get(
            "scheduled_close_time"
        )
        resolution_date = None
        if close_time:
            close_dt = self._parse_datetime(close_time)
            if close_dt:
                resolution_date = close_dt.date()

        # Resolution status
        resolved = market.get("resolved", False)
        resolution_value = None
        if resolved:
            resolution = question_data.get("resolution", "").lower()
            if resolution == "yes":
                resolution_value = 1.0
            elif resolution == "no":
                resolution_value = 0.0
            # ambiguous/annulled -> None

        # Get community prediction as base rate
        base_rate = self._extract_community_prediction(market)

        return Question(
            id=str(market["id"]),
            source=self.name,
            source_type=SourceType.MARKET,
            text=market.get("title", ""),
            background=question_data.get("description"),
            url=f"https://www.metaculus.com/questions/{market['id']}",
            question_type=QuestionType.BINARY,
            created_at=created_at,
            resolution_date=resolution_date,
            resolved=resolved,
            resolution_value=resolution_value,
            base_rate=base_rate,
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch popular binary questions from Metaculus."""
        questions = []
        seen_ids = set()

        # Fetch from main feed
        markets = await self._search_questions()
        for market in markets:
            q = self._market_to_question(market)
            if q and q.id not in seen_ids:
                seen_ids.add(q.id)
                questions.append(q)

        # Fetch from categories
        # Note: Metaculus API category slugs change over time.
        # Only include categories that are confirmed to work.
        categories = [
            "geopolitics",
            "technology",
        ]
        for category in categories:
            try:
                markets = await self._search_questions(categories=category)
                for market in markets:
                    q = self._market_to_question(market)
                    if q and q.id not in seen_ids:
                        seen_ids.add(q.id)
                        questions.append(q)
            except httpx.HTTPError as e:
                logger.warning(f"Failed to fetch category {category}: {e}")

        logger.info(f"Fetched {len(questions)} questions from Metaculus")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific question."""
        try:
            data = await self._get_question(question_id)
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch question {question_id}: {e}")
            return None

        question_data = data.get("question", {})

        if data.get("resolved"):
            resolution = question_data.get("resolution", "").lower()
            if resolution == "yes":
                value = 1.0
            elif resolution == "no":
                value = 0.0
            else:
                return None  # Ambiguous/annulled

            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=value,
            )

        # Return community prediction as interim value
        cp = self._extract_community_prediction(data)
        if cp is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=cp,
            )

        return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
