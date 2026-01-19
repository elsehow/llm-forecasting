"""Integration tests for full tree construction E2E.

These tests require LLM calls - run with --integration flag.
"""

from datetime import date, timedelta

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.decomposition import build_signal_tree
from shared.rollup import rollup_tree, analyze_tree
from shared.tree import TreeGenerationConfig


@pytest.mark.integration
class TestBuildSignalTree:
    """E2E tests for build_signal_tree function."""

    @pytest.mark.asyncio
    async def test_builds_tree_for_short_horizon_question(self, today: date):
        """Short horizon question should produce shallow tree."""
        config = TreeGenerationConfig(
            target_question="Will it snow in New York City next week?",
            target_id="snow_nyc",
            minimum_resolution_days=7,
            max_signals=50,
            actionable_horizon_days=14,  # Short horizon
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)

        # Should have some signals
        assert len(tree.signals) > 0
        # Most/all should be leaves (short horizon, signals resolve soon)
        leaves = tree.get_leaves()
        assert len(leaves) > 0
        # Tree should be shallow for short-horizon question
        assert tree.max_depth <= 2

    @pytest.mark.asyncio
    async def test_builds_tree_for_long_horizon_question(self, today: date):
        """Long horizon question should produce deeper tree."""
        config = TreeGenerationConfig(
            target_question="Will a Democrat win the 2028 US presidential election?",
            target_id="dem_2028",
            minimum_resolution_days=7,
            max_signals=20,  # Limit for test speed
            actionable_horizon_days=365,  # Long horizon
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)

        # Should have signals
        assert len(tree.signals) > 0
        # Should have some leaves
        leaves = tree.get_leaves()
        assert len(leaves) > 0

    @pytest.mark.asyncio
    async def test_respects_max_signals_budget(self, today: date):
        """Tree should stop when max_signals is reached."""
        config = TreeGenerationConfig(
            target_question="Will humanity establish a permanent base on Mars by 2050?",
            target_id="mars_base",
            minimum_resolution_days=7,
            max_signals=10,  # Very small budget
            actionable_horizon_days=365,
            signals_per_node=5,
        )

        tree = await build_signal_tree(config, today)

        # Should not exceed max_signals
        assert len(tree.signals) <= config.max_signals

    @pytest.mark.asyncio
    async def test_all_signals_have_rho_estimated(self, today: date):
        """All signals should have rho estimated after tree build."""
        config = TreeGenerationConfig(
            target_question="Will Apple release AR glasses in 2026?",
            target_id="apple_ar",
            minimum_resolution_days=7,
            max_signals=15,
            actionable_horizon_days=180,
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)

        for signal in tree.signals:
            assert signal.rho is not None, f"Signal {signal.id} has no rho"
            assert -1.0 <= signal.rho <= 1.0

    @pytest.mark.asyncio
    async def test_tree_structure_is_valid(self, today: date):
        """Tree should have valid structure (parent-child relationships)."""
        config = TreeGenerationConfig(
            target_question="Will the Fed cut interest rates in Q1 2026?",
            target_id="fed_rates",
            minimum_resolution_days=7,
            max_signals=15,
            actionable_horizon_days=90,
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)

        # All signals should have valid parent
        for signal in tree.signals:
            assert signal.parent_id is not None
            # Parent should exist (either target or another signal)
            parent = tree.get_node(signal.parent_id)
            assert parent is not None, f"Signal {signal.id} has invalid parent {signal.parent_id}"

        # Depth should be consistent
        for signal in tree.signals:
            path = tree.get_path_to_root(signal.id)
            assert len(path) - 1 == signal.depth

    @pytest.mark.asyncio
    async def test_leaf_signals_resolve_within_minimum_days(self, today: date):
        """Leaf signals should resolve within minimum_resolution_days or have no date."""
        min_days = 14
        config = TreeGenerationConfig(
            target_question="Will Tesla stock reach $500 by mid-2026?",
            target_id="tsla_500",
            minimum_resolution_days=min_days,
            max_signals=15,
            actionable_horizon_days=180,
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)
        leaves = tree.get_leaves()

        min_date = today + timedelta(days=min_days)
        for leaf in leaves:
            if leaf.resolution_date is not None:
                # Leaf should resolve within minimum_resolution_days
                assert leaf.resolution_date <= min_date, (
                    f"Leaf {leaf.id} resolves on {leaf.resolution_date}, "
                    f"but minimum is {min_date}"
                )


@pytest.mark.integration
class TestTreeRollupIntegration:
    """E2E tests for rollup on real trees."""

    @pytest.mark.asyncio
    async def test_rollup_produces_valid_probability(self, today: date):
        """Rollup should produce a valid probability."""
        config = TreeGenerationConfig(
            target_question="Will the S&P 500 hit a new all-time high in Q1 2026?",
            target_id="sp500_ath",
            minimum_resolution_days=7,
            max_signals=12,
            actionable_horizon_days=90,
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)
        probability = rollup_tree(tree, target_prior=0.5)

        assert 0.0 < probability < 1.0
        assert tree.computed_probability == probability

    @pytest.mark.asyncio
    async def test_analyze_tree_returns_valid_analysis(self, today: date):
        """analyze_tree should return comprehensive analysis."""
        config = TreeGenerationConfig(
            target_question="Will Nvidia remain the most valuable AI chip company in 2026?",
            target_id="nvda_lead",
            minimum_resolution_days=7,
            max_signals=12,
            actionable_horizon_days=180,
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)
        analysis = analyze_tree(tree, target_prior=0.5)

        # Check analysis structure
        assert "target" in analysis
        assert "computed_probability" in analysis
        assert "max_depth" in analysis
        assert "total_signals" in analysis
        assert "leaf_count" in analysis
        assert "top_contributors" in analysis
        assert "all_contributions" in analysis

        # Check values
        assert 0.0 < analysis["computed_probability"] < 1.0
        assert analysis["total_signals"] == len(tree.signals)
        assert analysis["leaf_count"] == len(tree.get_leaves())

    @pytest.mark.asyncio
    async def test_top_contributors_sorted_by_contribution(self, today: date):
        """Top contributors should be sorted by contribution (descending)."""
        config = TreeGenerationConfig(
            target_question="Will OpenAI release GPT-5 in 2026?",
            target_id="gpt5",
            minimum_resolution_days=7,
            max_signals=15,
            actionable_horizon_days=180,
            signals_per_node=3,
        )

        tree = await build_signal_tree(config, today)
        analysis = analyze_tree(tree, target_prior=0.5)

        contributions = analysis["top_contributors"]
        if len(contributions) > 1:
            for i in range(len(contributions) - 1):
                assert contributions[i]["contribution"] >= contributions[i + 1]["contribution"]
