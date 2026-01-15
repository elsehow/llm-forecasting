"""Metaculus market data provider."""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from llm_forecasting.config import settings
from llm_forecasting.market_data.base import MarketDataProvider, market_data_registry
from llm_forecasting.market_data.models import Market, MarketStatus

logger = logging.getLogger(__name__)

BASE_URL = "https://www.metaculus.com/api"


@market_data_registry.register
class MetaculusData(MarketDataProvider):
    """Metaculus market data provider."""

    name = "metaculus"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize Metaculus data provider.

        Args:
            api_key: Metaculus API key. If not provided, uses settings.metaculus_api_key.
                    Public API works without a key but has rate limits.
            http_client: Optional httpx client for connection reuse.
        """
        self._api_key = api_key if api_key is not None else settings.metaculus_api_key
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Token {self._api_key}"
            self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self._client

    async def fetch_markets(
        self,
        *,
        active_only: bool = True,
        min_liquidity: float | None = None,  # Ignored for Metaculus
        min_volume: float | None = None,  # Ignored for Metaculus
        min_forecasters: int = 50,
        limit: int | None = 100,
        categories: list[str] | None = None,
    ) -> list[Market]:
        """Fetch questions from Metaculus.

        Args:
            active_only: Only fetch open questions
            min_liquidity: Ignored (Metaculus doesn't have liquidity)
            min_volume: Ignored (Metaculus doesn't have volume)
            min_forecasters: Minimum number of forecasters
            limit: Maximum number of markets per category
            categories: Categories to fetch from (default: main feed + geopolitics, technology)

        Returns:
            List of Market objects
        """
        client = await self._get_client()
        all_markets = []
        seen_ids = set()

        # Default categories
        if categories is None:
            categories = [None, "geopolitics", "technology"]

        for category in categories:
            params = {
                "statuses": "open" if active_only else "open,resolved",
                "with_cp": "false",
                "forecast_type": "binary",
                "order_by": "-hotness",
                "limit": limit or 100,
                "for_main_feed": "true",
            }
            if active_only:
                min_resolve = (datetime.now().date() + timedelta(days=10)).isoformat()
                params["scheduled_resolve_time__gt"] = min_resolve
            if category:
                params["categories"] = category

            try:
                response = await client.get(f"{BASE_URL}/posts/", params=params)
                response.raise_for_status()
                results = response.json().get("results", [])

                for raw in results:
                    if raw["id"] in seen_ids:
                        continue

                    # Filter by forecaster count
                    if raw.get("nr_forecasters", 0) < min_forecasters:
                        continue

                    market = self._parse_market(raw)
                    if market:
                        seen_ids.add(raw["id"])
                        all_markets.append(market)
            except httpx.HTTPError as e:
                logger.warning(f"Failed to fetch category {category}: {e}")

        logger.info(f"Fetched {len(all_markets)} markets from Metaculus")
        return all_markets

    async def fetch_market(self, market_id: str) -> Market | None:
        """Fetch a single question by ID."""
        client = await self._get_client()
        try:
            response = await client.get(f"{BASE_URL}/posts/{market_id}")
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data)
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch market {market_id}: {e}")
            return None

    def _parse_market(self, raw: dict) -> Market | None:
        """Parse raw API response into Market model."""
        question_data = raw.get("question", {})

        # Check CP reveal time
        cp_reveal = question_data.get("cp_reveal_time")
        if cp_reveal:
            reveal_dt = self._parse_datetime(cp_reveal)
            if reveal_dt and reveal_dt > datetime.now(timezone.utc):
                return None  # CP not yet revealed

        # Parse timestamps
        created_at = self._parse_datetime(question_data.get("open_time"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        close_time = question_data.get("actual_close_time") or question_data.get(
            "scheduled_close_time"
        )
        close_date = self._parse_datetime(close_time)
        resolution_date = close_date.date() if close_date else None

        # Status and resolution
        status = MarketStatus.OPEN
        resolved_value = None
        if raw.get("resolved"):
            status = MarketStatus.RESOLVED
            resolution = question_data.get("resolution", "").lower()
            if resolution == "yes":
                resolved_value = 1.0
            elif resolution == "no":
                resolved_value = 0.0
            # ambiguous/annulled -> None

        # Community prediction
        current_prob = self._extract_community_prediction(raw)

        return Market(
            id=str(raw["id"]),
            platform=self.name,
            title=raw.get("title", ""),
            description=question_data.get("description"),
            url=f"https://www.metaculus.com/questions/{raw['id']}",
            created_at=created_at,
            close_date=close_date,
            resolution_date=resolution_date,
            status=status,
            resolved_value=resolved_value,
            current_probability=current_prob,
            num_forecasters=raw.get("nr_forecasters"),
        )

    def _extract_community_prediction(self, data: dict) -> float | None:
        """Extract community prediction from question data."""
        try:
            cp = data.get("community_prediction", {})
            if isinstance(cp, dict):
                full = cp.get("full", {})
                if isinstance(full, dict):
                    return full.get("q2")  # Median prediction
        except (KeyError, TypeError):
            pass
        return None

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

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._client and self._owns_client:
            await self._client.aclose()
            self._client = None
