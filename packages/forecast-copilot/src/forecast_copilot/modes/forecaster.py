"""Forecaster mode - provides direct forecasts.

This mode provides forecasts directly to users.

Design considerations:
- Risk of anchoring: users may over-weight AI forecasts
- Risk of homogenization: aggregate accuracy may degrade if all users anchor similarly
- Useful for: quick estimates, unfamiliar domains, baseline comparisons

TODO: Implement after user research validates this mode is wanted.
"""

from dataclasses import dataclass


@dataclass
class ForecasterMode:
    """Copilot mode that provides direct forecasts."""

    model: str = "claude-sonnet-4-20250514"

    async def forecast(self, question: str, context: str | None = None) -> dict:
        """Generate a forecast for the given question.

        TODO: Implement
        """
        raise NotImplementedError("Forecaster mode not yet implemented")

    async def explain(self, question: str, forecast: float) -> str:
        """Explain the reasoning behind a forecast.

        TODO: Implement
        """
        raise NotImplementedError("Forecaster mode not yet implemented")
