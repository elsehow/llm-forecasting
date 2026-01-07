"""Copilot interaction modes.

Different modes for user interaction:
- forecaster: Provides direct forecasts (risk: anchoring, homogenization)
- sounding_board: Asks questions, surfaces info, challenges reasoning (no forecasts)
"""

from forecast_copilot.modes.forecaster import ForecasterMode
from forecast_copilot.modes.sounding_board import SoundingBoardMode

__all__ = ["ForecasterMode", "SoundingBoardMode"]
