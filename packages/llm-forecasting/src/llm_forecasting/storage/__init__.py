"""Storage backends for ForecastBench."""

from llm_forecasting.storage.base import Storage
from llm_forecasting.storage.sqlite import SQLiteStorage

__all__ = ["Storage", "SQLiteStorage"]
