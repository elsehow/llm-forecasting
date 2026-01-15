"""Abstract base class for market data providers."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import ClassVar

from llm_forecasting.market_data.models import Candle, Market, PricePoint


class MarketDataRegistry:
    """Registry of available market data providers."""

    def __init__(self):
        self._providers: dict[str, type["MarketDataProvider"]] = {}

    def register(self, cls: type["MarketDataProvider"]) -> type["MarketDataProvider"]:
        """Register a provider class."""
        self._providers[cls.name] = cls
        return cls

    def get(self, name: str) -> type["MarketDataProvider"]:
        """Get a provider class by name."""
        if name not in self._providers:
            raise KeyError(
                f"Unknown provider: {name}. Available: {list(self._providers.keys())}"
            )
        return self._providers[name]

    def list(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._providers


market_data_registry = MarketDataRegistry()


class MarketDataProvider(ABC):
    """Abstract base for market data providers.

    Implementations fetch raw market data from prediction platforms.
    This is lower-level than QuestionSource - it provides raw market
    data that can be cached and later converted to Questions.

    To create a new provider:
    1. Subclass MarketDataProvider
    2. Set the `name` class variable
    3. Implement fetch_markets() and fetch_market()
    4. Optionally implement fetch_price_history() for platforms with history
    5. Decorate with @market_data_registry.register

    Example:
        @market_data_registry.register
        class MyPlatformData(MarketDataProvider):
            name = "myplatform"

            async def fetch_markets(self, ...) -> list[Market]:
                ...

            async def fetch_market(self, market_id: str) -> Market | None:
                ...
    """

    name: ClassVar[str]

    @abstractmethod
    async def fetch_markets(
        self,
        *,
        active_only: bool = True,
        min_liquidity: float | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[Market]:
        """Fetch markets from the platform.

        Args:
            active_only: Only fetch open/active markets
            min_liquidity: Minimum liquidity filter (platform-dependent)
            min_volume: Minimum volume filter (platform-dependent)
            limit: Maximum number of markets to fetch

        Returns:
            List of Market objects
        """
        ...

    @abstractmethod
    async def fetch_market(self, market_id: str) -> Market | None:
        """Fetch a single market by ID.

        Args:
            market_id: Platform-specific market identifier

        Returns:
            Market if found, None otherwise
        """
        ...

    async def fetch_price_history(
        self,
        market_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> list[Candle] | list[PricePoint]:
        """Fetch historical price data for a market.

        Args:
            market_id: Market identifier
            start: Start of time range (default: 60 days ago)
            end: End of time range (default: now)
            interval: Candle interval ("1m", "5m", "1h", "4h", "1d")

        Returns:
            List of Candle or PricePoint objects (platform-dependent)

        Note: Not all platforms support this. Default raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.name} does not support price history")

    async def close(self) -> None:
        """Clean up resources (HTTP client, etc.)."""
        pass
