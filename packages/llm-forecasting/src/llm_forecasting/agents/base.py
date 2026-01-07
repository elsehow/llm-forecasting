"""Abstract base class for forecasters."""

from abc import ABC, abstractmethod

from llm_forecasting.models import Forecast, Question


class Forecaster(ABC):
    """Abstract base class for forecasters.

    A forecaster takes a Question and produces a Forecast with a probability.
    """

    name: str

    @abstractmethod
    async def forecast(self, question: Question) -> Forecast:
        """Generate a forecast for a question.

        Args:
            question: The question to forecast.

        Returns:
            A Forecast with the predicted probability.
        """
        ...

    async def forecast_many(self, questions: list[Question]) -> list[Forecast]:
        """Generate forecasts for multiple questions.

        Default implementation calls forecast() for each question sequentially.
        Subclasses may override for batch optimization.

        Args:
            questions: List of questions to forecast.

        Returns:
            List of Forecast objects.
        """
        return [await self.forecast(q) for q in questions]
