"""Tests for structure generation module.

These tests use mocked LLM responses to test the structure parsing logic.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cc_builder.structure import (
    generate_logical_structure,
    LogicalStructure,
    NecessityConstraint,
    ExclusivityConstraint,
    CausalPathway,
    structure_to_dict,
)


class TestLogicalStructureDataclasses:
    """Tests for the dataclass definitions."""

    def test_necessity_constraint(self):
        """NecessityConstraint should have prerequisite and reasoning."""
        constraint = NecessityConstraint(
            prerequisite="Must be nominated",
            reasoning="Cannot win without nomination",
        )
        assert constraint.prerequisite == "Must be nominated"
        assert constraint.reasoning == "Cannot win without nomination"

    def test_exclusivity_constraint(self):
        """ExclusivityConstraint should have competitor, prize, and reasoning."""
        constraint = ExclusivityConstraint(
            competitor="Sinners",
            prize="Best Picture",
            reasoning="Only one film can win",
        )
        assert constraint.competitor == "Sinners"
        assert constraint.prize == "Best Picture"
        assert constraint.reasoning == "Only one film can win"

    def test_causal_pathway(self):
        """CausalPathway should have upstream_event, mechanism, and effect."""
        pathway = CausalPathway(
            upstream_event="Golden Globe win",
            mechanism="Industry momentum",
            effect_on_target="positive",
        )
        assert pathway.upstream_event == "Golden Globe win"
        assert pathway.mechanism == "Industry momentum"
        assert pathway.effect_on_target == "positive"

    def test_logical_structure(self):
        """LogicalStructure should contain all constraint types."""
        structure = LogicalStructure(
            target="Will X win?",
            necessity_constraints=[
                NecessityConstraint("Must be nominated", "Required")
            ],
            exclusivity_constraints=[
                ExclusivityConstraint("Y", "Prize", "Competition")
            ],
            causal_pathways=[
                CausalPathway("Event A", "Mechanism", "positive")
            ],
        )
        assert structure.target == "Will X win?"
        assert len(structure.necessity_constraints) == 1
        assert len(structure.exclusivity_constraints) == 1
        assert len(structure.causal_pathways) == 1


class TestStructureToDict:
    """Tests for structure serialization."""

    def test_structure_to_dict(self):
        """structure_to_dict should serialize all fields."""
        structure = LogicalStructure(
            target="Will X win?",
            necessity_constraints=[
                NecessityConstraint("Nomination", "Required")
            ],
            exclusivity_constraints=[
                ExclusivityConstraint("Y", "Prize", "Competition")
            ],
            causal_pathways=[
                CausalPathway("Event", "Mechanism", "positive")
            ],
        )

        result = structure_to_dict(structure)

        assert result["target"] == "Will X win?"
        assert len(result["necessity_constraints"]) == 1
        assert result["necessity_constraints"][0]["prerequisite"] == "Nomination"
        assert len(result["exclusivity_constraints"]) == 1
        assert result["exclusivity_constraints"][0]["competitor"] == "Y"
        assert len(result["causal_pathways"]) == 1
        assert result["causal_pathways"][0]["effect_on_target"] == "positive"

    def test_structure_to_dict_empty(self):
        """structure_to_dict should handle empty constraint lists."""
        structure = LogicalStructure(
            target="Test",
            necessity_constraints=[],
            exclusivity_constraints=[],
            causal_pathways=[],
        )

        result = structure_to_dict(structure)

        assert result["necessity_constraints"] == []
        assert result["exclusivity_constraints"] == []
        assert result["causal_pathways"] == []


class TestGenerateLogicalStructure:
    """Tests for LLM-based structure generation (mocked)."""

    @pytest.mark.asyncio
    async def test_generate_logical_structure_parses_response(self):
        """Should parse LLM response into LogicalStructure."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""{
                        "necessity": [
                            {"prerequisite": "Must be nominated", "reasoning": "Required"}
                        ],
                        "exclusivity": [
                            {"competitor": "Sinners", "prize": "Best Picture", "reasoning": "Only one winner"}
                        ],
                        "causal": [
                            {"upstream_event": "Golden Globe", "mechanism": "Momentum", "effect_on_target": "positive"}
                        ]
                    }"""
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await generate_logical_structure(
                "Will One Battle win Best Picture?"
            )

            assert isinstance(result, LogicalStructure)
            assert len(result.necessity_constraints) == 1
            assert result.necessity_constraints[0].prerequisite == "Must be nominated"
            assert len(result.exclusivity_constraints) == 1
            assert result.exclusivity_constraints[0].competitor == "Sinners"
            assert len(result.causal_pathways) == 1
            assert result.causal_pathways[0].effect_on_target == "positive"

    @pytest.mark.asyncio
    async def test_generate_logical_structure_with_context(self):
        """Should include context in prompt."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""{
                        "necessity": [],
                        "exclusivity": [],
                        "causal": []
                    }"""
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            await generate_logical_structure(
                "Will X win?",
                context="X is currently leading in polls",
            )

            # Check that context was included in the prompt
            call_args = mock_llm.call_args
            messages = call_args.kwargs["messages"]
            assert "X is currently leading in polls" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_logical_structure_empty_response(self):
        """Should handle empty constraint lists."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""{
                        "necessity": [],
                        "exclusivity": [],
                        "causal": []
                    }"""
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await generate_logical_structure("Simple question")

            assert len(result.necessity_constraints) == 0
            assert len(result.exclusivity_constraints) == 0
            assert len(result.causal_pathways) == 0
