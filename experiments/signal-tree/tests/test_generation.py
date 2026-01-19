"""Tests for signal generation (LLM-based).

These tests require LLM calls - run with --integration flag.
"""

from datetime import date, timedelta

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.decomposition import generate_signals_for_node
from shared.tree import SignalNode, TreeGenerationConfig


@pytest.mark.integration
class TestGenerateSignalsForNode:
    """Tests for generate_signals_for_node function (requires LLM)."""

    @pytest.mark.asyncio
    async def test_generates_n_signals(self, today: date):
        """Should generate exactly signals_per_node signals."""
        parent = SignalNode(
            id="target",
            text="Will a Democrat win the 2028 presidential election?",
            depth=0,
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=3,  # Request 3 signals
            actionable_horizon_days=365,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        assert len(signals) == 3

    @pytest.mark.asyncio
    async def test_signals_have_valid_resolution_dates(self, today: date):
        """All signals should have resolution dates."""
        parent = SignalNode(
            id="target",
            text="Will SpaceX land humans on Mars by 2030?",
            depth=0,
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=3,
            actionable_horizon_days=365,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        for signal in signals:
            assert signal.resolution_date is not None
            # Resolution date should be in the future
            assert signal.resolution_date > today

    @pytest.mark.asyncio
    async def test_signals_have_valid_base_rates(self, today: date):
        """All signals should have base_rate in [0,1]."""
        parent = SignalNode(
            id="target",
            text="Will AI achieve AGI by 2030?",
            depth=0,
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=3,
            actionable_horizon_days=180,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        for signal in signals:
            assert signal.base_rate is not None
            assert 0.0 <= signal.base_rate <= 1.0

    @pytest.mark.asyncio
    async def test_parent_child_relationship_correct(self, today: date):
        """All signals should have correct parent_id and depth."""
        parent = SignalNode(
            id="root_question",
            text="Will Bitcoin reach $200k by end of 2026?",
            depth=2,  # Parent is at depth 2
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=3,
            actionable_horizon_days=365,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        for signal in signals:
            assert signal.parent_id == parent.id
            assert signal.depth == parent.depth + 1  # Should be 3

    @pytest.mark.asyncio
    async def test_signals_have_unique_ids(self, today: date):
        """All signals should have unique IDs."""
        parent = SignalNode(
            id="target",
            text="Will the UK rejoin the EU by 2035?",
            depth=0,
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=5,
            actionable_horizon_days=365,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        ids = [s.id for s in signals]
        assert len(ids) == len(set(ids))  # All unique

    @pytest.mark.asyncio
    async def test_signals_have_probability_source_llm(self, today: date):
        """All signals should have probability_source='llm'."""
        parent = SignalNode(
            id="target",
            text="Will Tesla release a sub-$25k car by 2027?",
            depth=0,
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=3,
            actionable_horizon_days=365,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        for signal in signals:
            assert signal.probability_source == "llm"

    @pytest.mark.asyncio
    async def test_signals_have_meaningful_text(self, today: date):
        """Signals should have non-empty, question-like text."""
        parent = SignalNode(
            id="target",
            text="Will Mark Carney become Prime Minister of Canada?",
            depth=0,
        )
        config = TreeGenerationConfig(
            target_question=parent.text,
            signals_per_node=3,
            actionable_horizon_days=180,
            minimum_resolution_days=7,
        )

        signals = await generate_signals_for_node(parent, config, today)

        for signal in signals:
            assert len(signal.text) > 10  # Meaningful length
            # Should be phrased as a question or statement about an outcome
            assert any(
                word in signal.text.lower()
                for word in ["will", "?", "whether", "if"]
            )
