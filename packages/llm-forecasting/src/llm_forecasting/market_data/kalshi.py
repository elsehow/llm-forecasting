"""Kalshi market data provider."""

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from llm_forecasting.market_data.base import MarketDataProvider, market_data_registry
from llm_forecasting.market_data.models import Market, MarketStatus

logger = logging.getLogger(__name__)

# Kalshi API - despite "elections" subdomain, this serves ALL markets
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3


@market_data_registry.register
class KalshiData(MarketDataProvider):
    """Kalshi market data provider.

    Fetches from Kalshi's public API (no auth required for market data).
    """

    name = "kalshi"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        """Initialize Kalshi data provider.

        Args:
            http_client: Optional httpx client for connection reuse.
        """
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _request_with_retry(
        self, url: str, params: dict | None = None
    ) -> dict | None:
        """Make a request with retry logic for rate limits."""
        client = await self._get_client()

        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limit
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - back off exponentially
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except httpx.HTTPError:
                raise

        logger.error(f"Max retries exceeded for {url}")
        return None

    async def fetch_events(
        self,
        *,
        exclude_sports: bool = True,
        limit: int | None = 500,
    ) -> list[dict]:
        """Fetch events from Kalshi API.

        Events contain metadata about market groups (politics, economics, etc.)

        Args:
            exclude_sports: Filter out sports betting events (KXM* prefix)
            limit: Maximum number of events to fetch

        Returns:
            List of event dicts with event_ticker, title, category, etc.
        """
        all_events = []
        cursor = None
        batch_size = 200

        while limit is None or len(all_events) < limit:
            params = {"limit": batch_size}
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._request_with_retry(
                    f"{KALSHI_API_URL}/events", params=params
                )
                if data is None:
                    break
            except httpx.HTTPError as e:
                logger.error(f"Error fetching Kalshi events: {e}")
                break

            events_data = data.get("events", [])
            if not events_data:
                break

            for event in events_data:
                ticker = event.get("event_ticker", "")
                # Skip sports events (KXM prefix = sports multi-game, single-game, etc.)
                if exclude_sports and ticker.startswith("KXM"):
                    continue

                all_events.append(event)

                if limit and len(all_events) >= limit:
                    break

            cursor = data.get("cursor")
            if not cursor:
                break

        logger.info(f"Fetched {len(all_events)} non-sports events from Kalshi")
        return all_events

    async def fetch_markets(
        self,
        *,
        active_only: bool = True,
        min_liquidity: float | None = None,
        min_volume: float | None = None,
        limit: int | None = 500,
        exclude_sports: bool = True,
    ) -> list[Market]:
        """Fetch markets from Kalshi API.

        When exclude_sports=True, fetches non-sports events first then
        gets markets for each event (since /markets is sorted by sports).

        Args:
            active_only: Only fetch open markets
            min_liquidity: Minimum liquidity filter (in dollars)
            min_volume: Minimum 24h volume filter
            limit: Maximum number of markets to fetch
            exclude_sports: Filter out sports betting markets

        Returns:
            List of Market objects
        """
        all_markets = []

        if exclude_sports:
            # Fetch via events for non-sports (more efficient)
            events = await self.fetch_events(exclude_sports=True, limit=None)

            for event in events:
                if limit and len(all_markets) >= limit:
                    break

                event_ticker = event.get("event_ticker")
                if not event_ticker:
                    continue

                params = {"event_ticker": event_ticker}
                if active_only:
                    params["status"] = "open"

                try:
                    data = await self._request_with_retry(
                        f"{KALSHI_API_URL}/markets", params=params
                    )
                    if data is None:
                        continue
                except httpx.HTTPError as e:
                    logger.warning(f"Error fetching markets for {event_ticker}: {e}")
                    continue

                for raw in data.get("markets", []):
                    market = self._parse_market(raw)
                    if market is None:
                        continue

                    if min_liquidity and (market.liquidity or 0) < min_liquidity:
                        continue
                    if min_volume and (market.volume_24h or 0) < min_volume:
                        continue

                    all_markets.append(market)

                    if limit and len(all_markets) >= limit:
                        break
        else:
            # Include all markets (sports + non-sports)
            cursor = None
            batch_size = 200

            while limit is None or len(all_markets) < limit:
                params = {"limit": batch_size}

                if active_only:
                    params["status"] = "open"

                if cursor:
                    params["cursor"] = cursor

                try:
                    data = await self._request_with_retry(
                        f"{KALSHI_API_URL}/markets", params=params
                    )
                    if data is None:
                        break
                except httpx.HTTPError as e:
                    logger.error(f"Error fetching Kalshi markets: {e}")
                    break

                markets_data = data.get("markets", [])
                if not markets_data:
                    break

                for raw in markets_data:
                    market = self._parse_market(raw)
                    if market is None:
                        continue

                    if min_liquidity and (market.liquidity or 0) < min_liquidity:
                        continue
                    if min_volume and (market.volume_24h or 0) < min_volume:
                        continue

                    all_markets.append(market)

                    if limit and len(all_markets) >= limit:
                        break

                cursor = data.get("cursor")
                if not cursor:
                    break

        logger.info(f"Fetched {len(all_markets)} markets from Kalshi")
        return all_markets

    async def fetch_market(self, market_id: str) -> Market | None:
        """Fetch a single market by ticker.

        Args:
            market_id: Market ticker (e.g., "INXD-26JAN17-T7850")

        Returns:
            Market if found, None otherwise
        """
        try:
            data = await self._request_with_retry(
                f"{KALSHI_API_URL}/markets/{market_id}"
            )
            if data is None:
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Error fetching Kalshi market {market_id}: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"Error fetching Kalshi market {market_id}: {e}")
            return None

        return self._parse_market(data.get("market", {}))

    def _parse_market(self, raw: dict) -> Market | None:
        """Parse raw API response into Market model."""
        ticker = raw.get("ticker")
        if not ticker:
            return None

        # Only handle binary markets
        market_type = raw.get("market_type", "binary")
        if market_type != "binary":
            return None

        # Parse timestamps
        created_at = self._parse_datetime(raw.get("created_time"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        close_time = self._parse_datetime(raw.get("close_time"))
        resolution_date = close_time.date() if close_time else None

        # Determine status
        status_str = raw.get("status", "")
        if status_str == "settled":
            status = MarketStatus.RESOLVED
        elif status_str == "closed":
            status = MarketStatus.CLOSED
        elif status_str in ("open", "active"):
            status = MarketStatus.OPEN
        else:
            status = MarketStatus.OPEN

        # Resolution value
        resolved_value = None
        result = raw.get("result", "")
        if result == "yes":
            resolved_value = 1.0
        elif result == "no":
            resolved_value = 0.0

        # Current probability from yes_ask (best ask price)
        # Kalshi prices are in dollars (cents), so 0.65 means 65%
        current_prob = None
        try:
            yes_ask = raw.get("yes_ask")
            if yes_ask is not None:
                current_prob = float(yes_ask) / 100.0  # Convert cents to probability
        except (ValueError, TypeError):
            pass

        # If no ask, try last price
        if current_prob is None:
            try:
                last_price = raw.get("last_price")
                if last_price is not None:
                    current_prob = float(last_price) / 100.0
            except (ValueError, TypeError):
                pass

        # Liquidity
        liquidity = None
        try:
            liq_str = raw.get("liquidity")
            if liq_str:
                liquidity = float(liq_str)
        except (ValueError, TypeError):
            pass

        # Volume
        volume_24h = raw.get("volume_24h")
        volume_total = raw.get("volume")

        # Build title from title + subtitle or rules
        title = raw.get("title", "")
        subtitle = raw.get("subtitle", "")
        if subtitle and subtitle not in title:
            title = f"{title}: {subtitle}" if title else subtitle

        # Description from rules
        description = raw.get("rules_primary", "")
        if raw.get("rules_secondary"):
            description = f"{description}\n\n{raw['rules_secondary']}"

        return Market(
            id=ticker,
            platform=self.name,
            title=title.strip(),
            description=description.strip() if description else None,
            url=f"https://kalshi.com/markets/{ticker}",
            created_at=created_at,
            close_date=close_time,
            resolution_date=resolution_date,
            status=status,
            resolved_value=resolved_value,
            current_probability=current_prob,
            liquidity=liquidity,
            volume_24h=float(volume_24h) if volume_24h else None,
            volume_total=float(volume_total) if volume_total else None,
        )

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
