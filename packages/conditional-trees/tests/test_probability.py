"""Tests for probability normalization."""

import pytest

from conditional_trees.probability import (
    ProbabilityResult,
    force_normalize,
    format_probabilities_for_retry,
    handle_probability_sum,
)


class TestProbabilitySumHandling:
    """Tests for tiered probability sum handling."""

    def test_sum_within_tolerance_ok(self):
        """Probabilities summing to 95-105% should normalize silently."""
        scenarios = {"a": 0.30, "b": 0.35, "c": 0.37}  # Sum = 1.02
        result = handle_probability_sum(scenarios)

        assert result.status == "ok"
        assert result.action_taken == "normalized"
        assert result.normalized is not None
        assert abs(sum(result.normalized.values()) - 1.0) < 0.001

    def test_sum_exactly_one(self):
        """Probabilities summing to exactly 1.0 should pass."""
        scenarios = {"a": 0.30, "b": 0.30, "c": 0.40}  # Sum = 1.0
        result = handle_probability_sum(scenarios)

        assert result.status == "ok"
        assert result.raw_sum == 1.0

    def test_sum_95_percent_ok(self):
        """Lower bound of tolerance (95%) should be ok."""
        # Use values that sum to exactly 0.96 to avoid floating point edge cases
        scenarios = {"a": 0.32, "b": 0.32, "c": 0.32}  # Sum = 0.96
        result = handle_probability_sum(scenarios)

        assert result.status == "ok"

    def test_sum_105_percent_ok(self):
        """Upper bound of tolerance (105%) should be ok."""
        scenarios = {"a": 0.35, "b": 0.35, "c": 0.35}  # Sum = 1.05
        result = handle_probability_sum(scenarios)

        assert result.status == "ok"

    def test_sum_warning_range(self):
        """Probabilities summing to 70-130% should warn but normalize."""
        scenarios = {"a": 0.40, "b": 0.40, "c": 0.40}  # Sum = 1.20
        result = handle_probability_sum(scenarios)

        assert result.status == "warning"
        assert result.action_taken == "normalized_with_warning"
        assert result.normalized is not None
        assert abs(sum(result.normalized.values()) - 1.0) < 0.001

    def test_sum_70_percent_warning(self):
        """70% sum should be at warning boundary."""
        scenarios = {"a": 0.23, "b": 0.23, "c": 0.24}  # Sum = 0.70
        result = handle_probability_sum(scenarios)

        assert result.status == "warning"

    def test_sum_130_percent_warning(self):
        """130% sum should be at warning boundary."""
        scenarios = {"a": 0.43, "b": 0.43, "c": 0.44}  # Sum = 1.30
        result = handle_probability_sum(scenarios)

        assert result.status == "warning"

    def test_sum_too_low_retry(self):
        """Probabilities summing to <70% should request retry."""
        scenarios = {"a": 0.20, "b": 0.20, "c": 0.20}  # Sum = 0.60
        result = handle_probability_sum(scenarios)

        assert result.status == "retry_needed"
        assert result.action_taken == "requesting_retry"
        assert result.normalized is None

    def test_sum_too_high_retry(self):
        """Probabilities summing to >130% should request retry."""
        scenarios = {"a": 0.50, "b": 0.50, "c": 0.50}  # Sum = 1.50
        result = handle_probability_sum(scenarios)

        assert result.status == "retry_needed"
        assert result.normalized is None

    def test_overlapping_scenarios_retry(self):
        """191% sum (like original pipeline output) should request retry."""
        scenarios = {
            "ai_transform": 0.25,
            "climate_crisis": 0.18,
            "fragmented_world": 0.20,
            "status_quo_plus": 0.22,
            "authoritarian_resurgence": 0.12,
            "green_transition": 0.15,
            "tech_backlash": 0.10,
            "demographic_decline": 0.19,
            "resource_conflicts": 0.14,
            "biotech_revolution": 0.16,
        }  # Sum ≈ 1.71
        result = handle_probability_sum(scenarios)

        assert result.status == "retry_needed"

    def test_preserves_raw_probabilities(self):
        """Result should preserve original probabilities for diagnostics."""
        scenarios = {"a": 0.30, "b": 0.35, "c": 0.37}
        result = handle_probability_sum(scenarios)

        assert result.raw_probabilities == scenarios
        assert result.raw_sum == pytest.approx(1.02)


class TestForceNormalize:
    """Tests for force normalization fallback."""

    def test_force_normalize_basic(self):
        """Force normalize should work on any input."""
        scenarios = {"a": 0.50, "b": 0.50, "c": 0.50}  # Sum = 1.50
        normalized = force_normalize(scenarios)

        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert all(v == pytest.approx(1 / 3) for v in normalized.values())

    def test_force_normalize_zero_sum(self):
        """Force normalize should handle zero sum gracefully."""
        scenarios = {"a": 0.0, "b": 0.0, "c": 0.0}
        normalized = force_normalize(scenarios)

        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert all(v == pytest.approx(1 / 3) for v in normalized.values())


class TestFormatForRetry:
    """Tests for retry prompt formatting."""

    def test_format_basic(self):
        """Format should produce readable output."""
        scenarios = {"a": 0.30, "b": 0.50}
        formatted = format_probabilities_for_retry(scenarios)

        assert "a:" in formatted
        assert "30" in formatted  # Could be "30%" or "30.0%"
        assert "b:" in formatted
        assert "50" in formatted

    def test_format_sorted(self):
        """Output should be sorted by scenario ID."""
        scenarios = {"z": 0.30, "a": 0.50, "m": 0.20}
        formatted = format_probabilities_for_retry(scenarios)

        # 'a' should appear before 'm' which should appear before 'z'
        assert formatted.index("a:") < formatted.index("m:")
        assert formatted.index("m:") < formatted.index("z:")
