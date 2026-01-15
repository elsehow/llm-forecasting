"""Market data layer for raw market data fetching and caching.

This layer provides:
- MarketDataProvider ABC for platform-specific implementations
- Market/Candle/PricePoint models for raw market data
- SQLite storage with caching
- Polymarket and Metaculus implementations

Usage:
    from llm_forecasting.market_data import PolymarketData, MarketDataStorage

    async def example():
        provider = PolymarketData()
        storage = MarketDataStorage()

        # Fetch and cache markets
        markets = await provider.fetch_markets(min_liquidity=25000)
        await storage.save_markets(markets)

        # Fetch price history
        for market in markets[:10]:
            if market.clob_token_ids:
                history = await provider.fetch_price_history_by_token(
                    market.clob_token_ids[0]
                )
                await storage.save_price_history(market.id, "polymarket", history)

        # Later: retrieve from cache
        cached_markets = await storage.get_markets(platform="polymarket")
"""

from llm_forecasting.market_data.base import (
    MarketDataProvider,
    market_data_registry,
)
from llm_forecasting.market_data.models import (
    Candle,
    Market,
    MarketStatus,
    PricePoint,
)
from llm_forecasting.market_data.storage import MarketDataStorage

# Import implementations to register them
from llm_forecasting.market_data.metaculus import MetaculusData
from llm_forecasting.market_data.polymarket import PolymarketData

__all__ = [
    # ABC and registry
    "MarketDataProvider",
    "market_data_registry",
    # Models
    "Market",
    "MarketStatus",
    "Candle",
    "PricePoint",
    # Storage
    "MarketDataStorage",
    # Implementations
    "PolymarketData",
    "MetaculusData",
]
