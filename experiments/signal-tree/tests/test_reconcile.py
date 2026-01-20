"""Tests for reconciliation module.

Tests the mapping of markets onto logical structure.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import date

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cc_builder.structure import (
    LogicalStructure,
    NecessityConstraint,
    ExclusivityConstraint,
    CausalPathway,
)
from cc_builder.markets import MarketSignal
from cc_builder.reconcile import (
    find_best_match,
    reconcile,
    identify_uncertain_signals,
    create_signal_id,
)
from shared.tree import SignalNode, SignalTree


class TestFindBestMatch:
    """Tests for market matching."""

    def test_find_best_match_with_exact_match(self):
        """Should find market with matching title."""
        markets = [
            MarketSignal(
                market_id="1",
                platform="polymarket",
                title="Will X be nominated for Best Picture?",
                similarity=0.9,
                current_probability=0.8,
            ),
            MarketSignal(
                market_id="2",
                platform="polymarket",
                title="Will Y win Best Actor?",
                similarity=0.5,
                current_probability=0.6,
            ),
        ]

        match = find_best_match("X be nominated", markets)

        assert match is not None
        assert match.market_id == "1"

    def test_find_best_match_no_good_match(self):
        """Should return None if no match above threshold."""
        markets = [
            MarketSignal(
                market_id="1",
                platform="polymarket",
                title="Completely unrelated question",
                similarity=0.2,
                current_probability=0.5,
            ),
        ]

        match = find_best_match("Something totally different", markets, min_similarity=0.5)

        assert match is None

    def test_find_best_match_combines_similarity_and_word_overlap(self):
        """Should consider both embedding similarity and word overlap."""
        markets = [
            MarketSignal(
                market_id="1",
                platform="polymarket",
                title="Best Picture winner prediction",
                similarity=0.8,  # High embedding similarity
                current_probability=0.5,
            ),
            MarketSignal(
                market_id="2",
                platform="polymarket",
                title="Will One Battle win Best Picture at 2026 Oscars?",
                similarity=0.7,  # Lower embedding similarity but better word match
                current_probability=0.5,
            ),
        ]

        # Search for something with specific words
        match = find_best_match("One Battle Best Picture 2026", markets)

        # Market 2 has better word overlap even with lower embedding similarity
        assert match is not None
        assert match.market_id == "2"


class TestCreateSignalId:
    """Tests for signal ID generation."""

    def test_create_signal_id_format(self):
        """Signal IDs should have correct format."""
        signal_id = create_signal_id()
        assert signal_id.startswith("sig_")
        assert len(signal_id) == 12  # "sig_" + 8 hex chars

    def test_create_signal_id_unique(self):
        """Signal IDs should be unique."""
        ids = [create_signal_id() for _ in range(100)]
        assert len(ids) == len(set(ids))


class TestReconcile:
    """Tests for reconciliation function."""

    @pytest.mark.asyncio
    async def test_reconcile_with_necessity_constraint(self):
        """Should create necessity signal for necessity constraint."""
        structure = LogicalStructure(
            target="Will X win Best Picture?",
            necessity_constraints=[
                NecessityConstraint("Must be nominated", "Required to win")
            ],
            exclusivity_constraints=[],
            causal_pathways=[],
        )
        markets = [
            MarketSignal(
                market_id="1",
                platform="polymarket",
                title="Will X be nominated for Best Picture?",
                similarity=0.9,
                current_probability=0.8,
                url="https://polymarket.com/1",
            ),
        ]

        # Mock estimate_base_rate to avoid LLM call
        with patch("cc_builder.reconcile.estimate_base_rate", new_callable=AsyncMock) as mock_estimate:
            mock_estimate.return_value = (0.5, "Default estimate")

            tree = await reconcile(structure, markets)

            assert len(tree.signals) == 1
            signal = tree.signals[0]
            assert signal.relationship_type == "necessity"
            assert signal.market_price == 0.8  # From matched market
            assert signal.market_url == "https://polymarket.com/1"

    @pytest.mark.asyncio
    async def test_reconcile_with_exclusivity_constraint(self):
        """Should create exclusivity signal for exclusivity constraint."""
        structure = LogicalStructure(
            target="Will X win?",
            necessity_constraints=[],
            exclusivity_constraints=[
                ExclusivityConstraint("Y", "the prize", "Competition")
            ],
            causal_pathways=[],
        )
        markets = [
            MarketSignal(
                market_id="1",
                platform="polymarket",
                title="Will Y win the prize?",
                similarity=0.8,
                current_probability=0.3,
            ),
        ]

        with patch("cc_builder.reconcile.estimate_base_rate", new_callable=AsyncMock) as mock_estimate:
            mock_estimate.return_value = (0.5, "Default estimate")

            tree = await reconcile(structure, markets)

            assert len(tree.signals) == 1
            signal = tree.signals[0]
            assert signal.relationship_type == "exclusivity"
            assert signal.p_target_given_yes == 0.01  # Default exclusivity value
            assert signal.p_target_given_no is not None

    @pytest.mark.asyncio
    async def test_reconcile_with_causal_pathway(self):
        """Should create correlation signal for causal pathway."""
        structure = LogicalStructure(
            target="Will X win?",
            necessity_constraints=[],
            exclusivity_constraints=[],
            causal_pathways=[
                CausalPathway("Event A", "Mechanism", "positive")
            ],
        )
        markets = []  # No markets

        with patch("cc_builder.reconcile.estimate_base_rate", new_callable=AsyncMock) as mock_estimate:
            mock_estimate.return_value = (0.6, "Based on historical data")

            tree = await reconcile(structure, markets)

            assert len(tree.signals) == 1
            signal = tree.signals[0]
            assert signal.relationship_type == "correlation"
            assert signal.rho == 0.4  # Default positive rho for "positive" effect
            assert signal.base_rate == 0.6  # From LLM estimate

    @pytest.mark.asyncio
    async def test_reconcile_builds_valid_tree(self):
        """Should build a valid SignalTree with all metadata."""
        structure = LogicalStructure(
            target="Test target",
            necessity_constraints=[
                NecessityConstraint("Prereq", "Required")
            ],
            exclusivity_constraints=[
                ExclusivityConstraint("Comp", "Prize", "Competition")
            ],
            causal_pathways=[
                CausalPathway("Event", "Mech", "negative")
            ],
        )
        markets = []

        with patch("cc_builder.reconcile.estimate_base_rate", new_callable=AsyncMock) as mock_estimate:
            mock_estimate.return_value = (0.5, "Estimate")

            tree = await reconcile(structure, markets)

            assert tree.target.text == "Test target"
            assert tree.target.id == "target"
            assert len(tree.signals) == 3
            assert tree.max_depth == 1
            assert tree.leaf_count == 3


class TestIdentifyUncertainSignals:
    """Tests for identifying signals that need decomposition."""

    def test_identifies_uncertain_signals(self):
        """Should identify signals with base_rate between 0.2 and 0.8."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        certain = SignalNode(
            id="certain",
            text="Very likely",
            base_rate=0.95,  # Too certain
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        uncertain = SignalNode(
            id="uncertain",
            text="Uncertain",
            base_rate=0.50,  # Uncertain
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [certain, uncertain]
        tree = SignalTree(target=root, signals=[certain, uncertain])

        result = identify_uncertain_signals(tree)

        assert len(result) == 1
        assert result[0].id == "uncertain"

    def test_respects_resolution_date_filter(self):
        """Should exclude signals resolving soon."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        soon = SignalNode(
            id="soon",
            text="Resolves soon",
            base_rate=0.50,
            resolution_date=date(2026, 1, 25),  # 5 days away
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        later = SignalNode(
            id="later",
            text="Resolves later",
            base_rate=0.50,
            resolution_date=date(2026, 2, 15),  # 26 days away
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [soon, later]
        tree = SignalTree(target=root, signals=[soon, later])

        result = identify_uncertain_signals(
            tree,
            min_resolution_days=14,
            today=date(2026, 1, 20),
        )

        # Only "later" should be returned
        assert len(result) == 1
        assert result[0].id == "later"

    def test_excludes_non_leaf_signals(self):
        """Should only return leaf signals."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        internal = SignalNode(
            id="internal",
            text="Internal",
            base_rate=0.50,
            parent_id="target",
            depth=1,
            is_leaf=False,  # Not a leaf
        )
        leaf = SignalNode(
            id="leaf",
            text="Leaf",
            base_rate=0.50,
            parent_id="internal",
            depth=2,
            is_leaf=True,
        )
        root.children = [internal]
        internal.children = [leaf]
        tree = SignalTree(target=root, signals=[internal, leaf])

        result = identify_uncertain_signals(tree)

        assert len(result) == 1
        assert result[0].id == "leaf"
