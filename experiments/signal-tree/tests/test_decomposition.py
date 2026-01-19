"""Tests for decomposition decision logic.

These are pure math tests - no LLM calls required.
"""

from datetime import date, timedelta

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.decomposition import needs_decomposition
from shared.tree import SignalNode, TreeGenerationConfig


class TestNeedsDecomposition:
    """Tests for needs_decomposition function."""

    def test_short_horizon_signal_no_decomposition(
        self, short_horizon_signal: SignalNode, default_config: TreeGenerationConfig, today: date
    ):
        """Signal resolving in 3 days should NOT be decomposed (below minimum_resolution_days=7)."""
        result = needs_decomposition(short_horizon_signal, default_config, today)
        assert result is False

    def test_long_horizon_signal_needs_decomposition(
        self, long_horizon_signal: SignalNode, default_config: TreeGenerationConfig, today: date
    ):
        """Signal resolving in 30 days SHOULD be decomposed (above minimum_resolution_days=7)."""
        result = needs_decomposition(long_horizon_signal, default_config, today)
        assert result is True

    def test_no_date_signal_needs_decomposition(
        self, no_date_signal: SignalNode, default_config: TreeGenerationConfig, today: date
    ):
        """Signal with no resolution date SHOULD be decomposed (assume needs it)."""
        result = needs_decomposition(no_date_signal, default_config, today)
        assert result is True

    def test_at_max_signals_no_decomposition(
        self, long_horizon_signal: SignalNode, today: date
    ):
        """Signal should NOT decompose if tree already has max_signals."""
        config = TreeGenerationConfig(
            target_question="Will X?",
            max_signals=100,
            minimum_resolution_days=7,
        )
        # Pass current_signal_count = 100 (at limit)
        result = needs_decomposition(long_horizon_signal, config, today, current_signal_count=100)
        assert result is False

    def test_below_max_signals_can_decompose(
        self, long_horizon_signal: SignalNode, today: date
    ):
        """Signal CAN decompose if tree is below max_signals."""
        config = TreeGenerationConfig(
            target_question="Will X?",
            max_signals=100,
            minimum_resolution_days=7,
        )
        # Pass current_signal_count = 50 (below limit)
        result = needs_decomposition(long_horizon_signal, config, today, current_signal_count=50)
        assert result is True

    def test_boundary_exactly_at_minimum_days(self, today: date):
        """Signal resolving exactly at minimum_resolution_days should NOT decompose."""
        config = TreeGenerationConfig(
            target_question="Will X?",
            minimum_resolution_days=7,
            max_signals=100,
        )
        signal = SignalNode(
            id="boundary",
            text="Boundary signal",
            resolution_date=today + timedelta(days=7),  # Exactly at boundary
            base_rate=0.5,
            parent_id="target",
            depth=1,
        )
        result = needs_decomposition(signal, config, today)
        # At minimum_resolution_days: signal.resolution_date (7 days) > min_date (7 days) is False
        assert result is False

    def test_one_day_over_minimum(self, today: date):
        """Signal resolving one day over minimum SHOULD decompose."""
        config = TreeGenerationConfig(
            target_question="Will X?",
            minimum_resolution_days=7,
            max_signals=100,
        )
        signal = SignalNode(
            id="one_over",
            text="One over signal",
            resolution_date=today + timedelta(days=8),  # One day over
            base_rate=0.5,
            parent_id="target",
            depth=1,
        )
        result = needs_decomposition(signal, config, today)
        assert result is True

    def test_different_minimum_resolution_days(self, today: date):
        """Test with different minimum_resolution_days values."""
        # With 30-day minimum
        config_30 = TreeGenerationConfig(
            target_question="Will X?",
            minimum_resolution_days=30,
            max_signals=100,
        )
        signal_25_days = SignalNode(
            id="s25",
            text="Signal at 25 days",
            resolution_date=today + timedelta(days=25),
            base_rate=0.5,
            parent_id="target",
            depth=1,
        )
        signal_35_days = SignalNode(
            id="s35",
            text="Signal at 35 days",
            resolution_date=today + timedelta(days=35),
            base_rate=0.5,
            parent_id="target",
            depth=1,
        )

        # 25 days < 30 minimum: no decomposition
        assert needs_decomposition(signal_25_days, config_30, today) is False
        # 35 days > 30 minimum: needs decomposition
        assert needs_decomposition(signal_35_days, config_30, today) is True


class TestResolutionBasedTermination:
    """Integration tests verifying resolution-based termination replaces max_depth."""

    def test_depth_is_no_longer_checked(self, today: date):
        """Verify that depth alone does not prevent decomposition."""
        config = TreeGenerationConfig(
            target_question="Will X?",
            minimum_resolution_days=7,
            max_signals=100,
        )
        # Deep signal (depth=10) but resolves in 30 days
        deep_signal = SignalNode(
            id="deep",
            text="Deep signal",
            resolution_date=today + timedelta(days=30),
            base_rate=0.5,
            parent_id="parent",
            depth=10,  # Very deep
        )
        # Should still decompose because resolution date is beyond minimum
        result = needs_decomposition(deep_signal, config, today)
        assert result is True

    def test_shallow_signal_can_be_leaf(self, today: date):
        """Shallow signal can be a leaf if it resolves soon."""
        config = TreeGenerationConfig(
            target_question="Will X?",
            minimum_resolution_days=7,
            max_signals=100,
        )
        # Shallow signal (depth=1) but resolves in 3 days
        shallow_signal = SignalNode(
            id="shallow",
            text="Shallow signal",
            resolution_date=today + timedelta(days=3),
            base_rate=0.5,
            parent_id="target",
            depth=1,  # Very shallow
        )
        # Should NOT decompose because resolution date is soon
        result = needs_decomposition(shallow_signal, config, today)
        assert result is False
