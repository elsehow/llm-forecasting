"""Forecasters for ForecastBench."""

from llm_forecasting.agents.base import Forecaster
from llm_forecasting.agents.llm import LLMForecaster

__all__ = ["Forecaster", "LLMForecaster"]
