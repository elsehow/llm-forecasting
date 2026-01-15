"""Question sources for ForecastBench."""

from llm_forecasting.sources.base import QuestionSource, registry

# Import all sources to trigger registration
from llm_forecasting.sources.fred import FREDSource
from llm_forecasting.sources.good_judgment import GoodJudgmentSource
from llm_forecasting.sources.infer import INFERSource
from llm_forecasting.sources.kalshi import KalshiSource
from llm_forecasting.sources.manifold import ManifoldSource
from llm_forecasting.sources.metaculus import MetaculusSource
from llm_forecasting.sources.polymarket import PolymarketSource
from llm_forecasting.sources.yfinance import YahooFinanceSource


def get_all_sources() -> dict[str, type[QuestionSource]]:
    """Get all registered source classes.

    Returns:
        Dict mapping source name to source class.
    """
    return {name: registry.get(name) for name in registry.list()}


__all__ = [
    "QuestionSource",
    "registry",
    "get_all_sources",
    "FREDSource",
    "GoodJudgmentSource",
    "INFERSource",
    "KalshiSource",
    "ManifoldSource",
    "MetaculusSource",
    "PolymarketSource",
    "YahooFinanceSource",
]
