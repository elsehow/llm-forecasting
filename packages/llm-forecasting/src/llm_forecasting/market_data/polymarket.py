"""Polymarket market data provider."""

import json
import logging
from datetime import datetime, timedelta, timezone

import httpx

from llm_forecasting.date_utils import parse_iso_datetime
from llm_forecasting.http_utils import HTTPClientMixin
from llm_forecasting.market_data.base import MarketDataProvider, market_data_registry
from llm_forecasting.market_data.models import Market, MarketStatus, PricePoint

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"


@market_data_registry.register
class PolymarketData(MarketDataProvider, HTTPClientMixin):
    """Polymarket market data provider.

    Uses two APIs:
    - Gamma API for market metadata
    - CLOB API for price history
    """

    name = "polymarket"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        """Initialize Polymarket data provider.

        Args:
            http_client: Optional httpx client for connection reuse.
        """
        self._init_client(http_client)

    async def fetch_markets(
        self,
        *,
        active_only: bool = True,
        min_liquidity: float | None = None,
        min_volume: float | None = None,
        limit: int | None = 500,
    ) -> list[Market]:
        """Fetch markets from Polymarket Gamma API."""
        client = await self._get_client()
        all_markets = []
        offset = 0
        batch_size = 100

        while limit is None or len(all_markets) < limit:
            params = {
                "limit": batch_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }
            if active_only:
                params["active"] = "true"
                params["closed"] = "false"

            response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            for raw in batch:
                market = self._parse_market(raw)
                if market is None:
                    continue

                # Apply filters
                if min_liquidity and (market.liquidity or 0) < min_liquidity:
                    continue
                if min_volume and (market.volume_24h or 0) < min_volume:
                    continue

                all_markets.append(market)

                if limit and len(all_markets) >= limit:
                    break

            if len(batch) < batch_size:
                break
            offset += batch_size

        logger.info(f"Fetched {len(all_markets)} markets from Polymarket")
        return all_markets

    async def fetch_market(self, market_id: str) -> Market | None:
        """Fetch a single market by condition ID."""
        client = await self._get_client()
        params = {"condition_ids": market_id}
        response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        markets = response.json()

        if not markets:
            return None
        return self._parse_market(markets[0])

    async def fetch_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> list[PricePoint]:
        """Fetch price history from CLOB API.

        Note: Requires the CLOB token ID, not the condition ID.
        This method first fetches the market to get the token ID.
        """
        # First get the market to find the token ID
        market = await self.fetch_market(market_id)
        if not market or not market.clob_token_ids:
            return []

        token_id = market.clob_token_ids[0]  # Use YES token
        return await self.fetch_price_history_by_token(
            token_id, start=start, end=end, interval=interval
        )

    async def fetch_price_history_by_token(
        self,
        token_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> list[PricePoint]:
        """Fetch price history directly by CLOB token ID.

        Args:
            token_id: CLOB token ID (from market.clob_token_ids)
            start: Start of time range (default: 60 days ago)
            end: End of time range (default: now)
            interval: Candle interval ("1m", "5m", "1h", "4h", "1d")

        Returns:
            List of PricePoint objects sorted chronologically
        """
        client = await self._get_client()

        # Defaults
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=60)

        # Fidelity mapping (minutes per candle)
        fidelity_map = {"1m": 1, "5m": 5, "1h": 60, "4h": 240, "1d": 1440}
        fidelity = fidelity_map.get(interval, 1440)

        all_points = []
        chunk_days = 7  # API limit - fetch in 7-day chunks

        chunk_start = int(start.timestamp())
        end_ts = int(end.timestamp())

        while chunk_start < end_ts:
            chunk_end = min(chunk_start + (chunk_days * 86400), end_ts)

            try:
                params = {
                    "market": token_id,
                    "interval": interval,
                    "fidelity": fidelity,
                    "startTs": chunk_start,
                    "endTs": chunk_end,
                }
                response = await client.get(
                    f"{CLOB_API_URL}/prices-history", params=params
                )
                response.raise_for_status()
                data = response.json()

                for point in data.get("history", []):
                    ts = point.get("t")
                    price = point.get("p")
                    if ts and price:
                        all_points.append(
                            PricePoint(
                                market_id=token_id,
                                platform=self.name,
                                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                                price=float(price),
                            )
                        )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return []  # Token not found
                logger.warning(f"Error fetching price history chunk: {e}")
            except Exception as e:
                logger.warning(f"Error fetching price history: {e}")

            chunk_start = chunk_end

        # Deduplicate by timestamp and sort
        seen = set()
        unique = []
        for p in sorted(all_points, key=lambda x: x.timestamp):
            if p.timestamp not in seen:
                seen.add(p.timestamp)
                unique.append(p)

        return unique

    def _parse_market(self, raw: dict) -> Market | None:
        """Parse raw API response into Market model."""
        condition_id = raw.get("conditionId")
        if not condition_id:
            return None

        # Parse outcomes to check if binary
        try:
            outcomes = json.loads(raw.get("outcomes", "[]"))
            is_binary = {s.lower() for s in outcomes} == {"yes", "no"}
        except (json.JSONDecodeError, TypeError):
            is_binary = False

        if not is_binary:
            return None  # Skip non-binary markets

        # Determine YES index
        try:
            yes_idx = 0 if outcomes[0].lower() == "yes" else 1
        except (IndexError, TypeError):
            yes_idx = 0

        # Parse timestamps
        created_at = parse_iso_datetime(raw.get("startDateIso"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        end_date_str = raw.get("endDate")
        if not end_date_str:
            events = raw.get("events", [])
            if events:
                end_date_str = events[0].get("endDate")

        close_date = parse_iso_datetime(end_date_str)
        resolution_date = close_date.date() if close_date else None

        # Status and resolution
        status = MarketStatus.OPEN
        resolved_value = None
        if raw.get("umaResolutionStatus") == "resolved":
            status = MarketStatus.RESOLVED
            try:
                prices = json.loads(raw.get("outcomePrices", "[]"))
                resolved_value = float(prices[yes_idx])
            except (json.JSONDecodeError, TypeError, IndexError, ValueError):
                pass
        elif raw.get("closed") or not raw.get("active"):
            status = MarketStatus.CLOSED

        # Current probability
        current_prob = None
        try:
            prices = json.loads(raw.get("outcomePrices", "[]"))
            current_prob = float(prices[yes_idx])
        except (json.JSONDecodeError, TypeError, IndexError, ValueError):
            pass

        # CLOB token IDs
        clob_token_ids = raw.get("clobTokenIds", "[]")
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except json.JSONDecodeError:
                clob_token_ids = None

        return Market(
            id=condition_id,
            platform=self.name,
            title=raw.get("question", ""),
            description=raw.get("description"),
            url=f"https://polymarket.com/market/{raw.get('slug', '')}",
            created_at=created_at,
            close_date=close_date,
            resolution_date=resolution_date,
            status=status,
            resolved_value=resolved_value,
            current_probability=current_prob,
            liquidity=float(raw.get("liquidityNum", 0) or 0),
            volume_24h=float(raw.get("volume24hr", 0) or 0),
            volume_total=float(raw.get("volume", 0) or 0),
            clob_token_ids=clob_token_ids if clob_token_ids else None,
        )
