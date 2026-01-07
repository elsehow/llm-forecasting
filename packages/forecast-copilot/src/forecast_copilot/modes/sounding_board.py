"""Sounding board mode - questions and challenges without providing forecasts.

This mode helps users think through their forecasts without anchoring them.

Design rationale:
- Avoids anchoring: never provides a number for users to anchor on
- Preserves diversity: users develop independent reasoning
- Supports experts: research suggests experts benefit from sounding boards
- Challenges assumptions: can take adversarial stance to stress-test reasoning

Key behaviors:
- Asks clarifying questions about the question itself
- Surfaces relevant information without interpreting it
- Challenges stated reasoning and assumptions
- Identifies cruxes and scenarios to consider
- Never provides a probability estimate

TODO: Implement after user research validates this mode is wanted.
"""

from dataclasses import dataclass


@dataclass
class SoundingBoardMode:
    """Copilot mode that helps users think without providing forecasts."""

    model: str = "claude-sonnet-4-20250514"
    adversarial: bool = False  # Whether to actively challenge user reasoning

    async def analyze_question(self, question: str) -> dict:
        """Analyze a question and surface key considerations.

        Returns:
            Dict with:
            - clarifying_questions: Questions about the question itself
            - key_uncertainties: Major sources of uncertainty
            - relevant_scenarios: Scenarios to consider
            - information_gaps: What information would help

        TODO: Implement
        """
        raise NotImplementedError("Sounding board mode not yet implemented")

    async def challenge(self, question: str, user_forecast: float, user_reasoning: str) -> dict:
        """Challenge the user's forecast and reasoning.

        Returns:
            Dict with:
            - challenges: Specific challenges to the reasoning
            - alternative_scenarios: Scenarios the user may have underweighted
            - cruxes: Key beliefs that drive the forecast

        TODO: Implement
        """
        raise NotImplementedError("Sounding board mode not yet implemented")

    async def surface_info(self, question: str, query: str) -> dict:
        """Surface relevant information without interpretation.

        Returns:
            Dict with:
            - sources: Relevant sources found
            - facts: Key facts (without probabilistic interpretation)
            - caveats: Limitations of the information

        TODO: Implement
        """
        raise NotImplementedError("Sounding board mode not yet implemented")
