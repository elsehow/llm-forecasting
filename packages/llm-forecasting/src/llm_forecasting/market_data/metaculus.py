"""Metaculus market data provider."""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from llm_forecasting.config import settings
from llm_forecasting.date_utils import parse_iso_datetime
from llm_forecasting.http_utils import HTTPClientMixin
from llm_forecasting.market_data.base import MarketDataProvider, market_data_registry
from llm_forecasting.market_data.models import Market, MarketStatus

logger = logging.getLogger(__name__)

BASE_URL = "https://www.metaculus.com/api"


@market_data_registry.register
class MetaculusData(MarketDataProvider, HTTPClientMixin):
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
        headers = {"Authorization": f"Token {self._api_key}"} if self._api_key else None
        self._init_client(http_client, headers=headers)

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
                "with_cp": "true",
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
            response = await client.get(f"{BASE_URL}/posts/{market_id}/")
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
            reveal_dt = parse_iso_datetime(cp_reveal)
            if reveal_dt and reveal_dt > datetime.now(timezone.utc):
                return None  # CP not yet revealed

        # Parse timestamps
        created_at = parse_iso_datetime(question_data.get("open_time"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        close_time = question_data.get("actual_close_time") or question_data.get(
            "scheduled_close_time"
        )
        close_date = parse_iso_datetime(close_time)
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

        # Extract categories from projects
        topic_categories, tournament_categories = self._extract_categories(raw)

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
            topic_categories=topic_categories,
            tournament_categories=tournament_categories,
        )

    def _extract_community_prediction(self, data: dict) -> float | None:
        """Extract community prediction from question data.

        The API returns aggregations in question.aggregations with recency_weighted
        and unweighted methods. Each has a 'latest' object with 'centers' array.
        For binary questions, centers[0] is the median probability.
        """
        try:
            question = data.get("question", {})
            aggregations = question.get("aggregations", {})

            # Try recency_weighted first, then unweighted
            for method in ["recency_weighted", "unweighted"]:
                agg = aggregations.get(method, {})
                latest = agg.get("latest")
                if latest and isinstance(latest, dict):
                    centers = latest.get("centers")
                    if centers and isinstance(centers, list) and len(centers) > 0:
                        return centers[0]  # Median for binary questions

            # Fallback to old format for backwards compatibility
            cp = data.get("community_prediction", {})
            if isinstance(cp, dict):
                full = cp.get("full", {})
                if isinstance(full, dict):
                    return full.get("q2")
        except (KeyError, TypeError, IndexError):
            pass
        return None

    def _extract_categories(
        self, data: dict
    ) -> tuple[list[str] | None, list[str] | None]:
        """Extract topic and tournament categories from projects field.

        Returns:
            Tuple of (topic_categories, tournament_categories)
        """
        # Keywords that indicate tournament/competition vs topic categories
        tournament_keywords = [
            "tournament",
            "benchmark",
            "leaderboard",
            "minibench",
            "cup",
            "contest",
            "learning",
            "academy",
            "warmup",
            "experiment",
            "pro forecasters",
            "quarterly",
            "university",
            "community",
        ]

        topics: list[str] = []
        tournaments: list[str] = []

        projects = data.get("projects", {})
        for key, val in projects.items():
            items = []
            if isinstance(val, list):
                items = val
            elif isinstance(val, dict):
                items = [val]

            for item in items:
                if isinstance(item, dict) and "name" in item:
                    name = item["name"]
                    name_lower = name.lower()
                    if any(kw in name_lower for kw in tournament_keywords):
                        tournaments.append(name)
                    else:
                        topics.append(name)

        return (
            topics if topics else None,
            tournaments if tournaments else None,
        )
