"""Tests for VOI (Value of Information) calculations."""

import math

import pytest

from llm_forecasting.voi import (
    entropy,
    entropy_voi,
    entropy_voi_from_rho,
    linear_voi,
    linear_voi_from_rho,
    rho_to_posteriors,
)


class TestLinearVOI:
    """Tests for linear_voi function."""

    def test_no_shift_gives_zero_voi(self):
        """If posteriors equal prior, VOI should be zero."""
        result = linear_voi(0.5, 0.5, 0.5, 0.5)
        assert result == 0.0

    def test_symmetric_shift(self):
        """Test symmetric shift around prior."""
        # Prior 0.5, signal prob 0.5, posteriors 0.7 and 0.3
        result = linear_voi(0.5, 0.5, 0.7, 0.3)
        # Expected: 0.5 * |0.7 - 0.5| + 0.5 * |0.3 - 0.5| = 0.5 * 0.2 + 0.5 * 0.2 = 0.2
        assert result == pytest.approx(0.2)

    def test_asymmetric_signal_probability(self):
        """Test with asymmetric signal probability."""
        # Prior 0.5, signal prob 0.8 (likely to fire), posteriors 0.6 and 0.2
        result = linear_voi(0.5, 0.8, 0.6, 0.2)
        # Expected: 0.8 * |0.6 - 0.5| + 0.2 * |0.2 - 0.5| = 0.8 * 0.1 + 0.2 * 0.3 = 0.14
        assert result == pytest.approx(0.14)

    def test_maximum_voi(self):
        """Maximum VOI when posteriors are 0 and 1."""
        # Prior 0.5, signal prob 0.5, posteriors 1.0 and 0.0
        result = linear_voi(0.5, 0.5, 1.0, 0.0)
        # Expected: 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        assert result == pytest.approx(0.5)

    def test_always_non_negative(self):
        """VOI should always be non-negative."""
        test_cases = [
            (0.1, 0.5, 0.2, 0.05),
            (0.9, 0.3, 0.95, 0.8),
            (0.5, 0.5, 0.5, 0.5),
        ]
        for p_x, p_q, p_yes, p_no in test_cases:
            assert linear_voi(p_x, p_q, p_yes, p_no) >= 0


class TestEntropy:
    """Tests for entropy function."""

    def test_maximum_entropy_at_half(self):
        """Maximum entropy at p=0.5."""
        assert entropy(0.5) == pytest.approx(1.0)

    def test_zero_entropy_at_extremes(self):
        """Entropy is zero at p=0 and p=1."""
        assert entropy(0.0) == 0.0
        assert entropy(1.0) == 0.0

    def test_symmetry(self):
        """Entropy is symmetric around 0.5."""
        assert entropy(0.3) == pytest.approx(entropy(0.7))
        assert entropy(0.1) == pytest.approx(entropy(0.9))


class TestEntropyVOI:
    """Tests for entropy_voi function."""

    def test_no_information_gain_same_posteriors(self):
        """No information gain if posteriors equal prior."""
        result = entropy_voi(0.5, 0.5, 0.5, 0.5)
        assert result == pytest.approx(0.0)

    def test_positive_information_gain(self):
        """Information gain should be positive when posteriors differ from prior."""
        result = entropy_voi(0.5, 0.5, 0.8, 0.2)
        assert result > 0

    def test_bounded_by_prior_entropy(self):
        """Information gain cannot exceed prior entropy."""
        p_x = 0.3
        result = entropy_voi(p_x, 0.5, 0.9, 0.1)
        assert result <= entropy(p_x)


class TestRhoToPosteriors:
    """Tests for rho_to_posteriors function."""

    def test_zero_correlation_gives_prior(self):
        """With rho=0, posteriors should equal prior."""
        p_a, p_b = 0.6, 0.4
        p_yes, p_no = rho_to_posteriors(0.0, p_a, p_b, clamp_posteriors=False)
        assert p_yes == pytest.approx(p_a, abs=0.01)
        assert p_no == pytest.approx(p_a, abs=0.01)

    def test_positive_correlation(self):
        """Positive correlation: P(A|B=yes) > P(A) > P(A|B=no)."""
        p_a, p_b = 0.5, 0.5
        p_yes, p_no = rho_to_posteriors(0.5, p_a, p_b, clamp_posteriors=False)
        assert p_yes > p_a
        assert p_no < p_a

    def test_negative_correlation(self):
        """Negative correlation: P(A|B=yes) < P(A) < P(A|B=no)."""
        p_a, p_b = 0.5, 0.5
        p_yes, p_no = rho_to_posteriors(-0.5, p_a, p_b, clamp_posteriors=False)
        assert p_yes < p_a
        assert p_no > p_a

    def test_perfect_positive_correlation(self):
        """Perfect positive correlation (rho=1)."""
        p_a, p_b = 0.5, 0.5
        p_yes, p_no = rho_to_posteriors(1.0, p_a, p_b, clamp_posteriors=False)
        # With rho=1 and equal priors, P(A|B=yes) should be 1
        assert p_yes == pytest.approx(1.0)
        assert p_no == pytest.approx(0.0)

    def test_perfect_negative_correlation(self):
        """Perfect negative correlation (rho=-1)."""
        p_a, p_b = 0.5, 0.5
        p_yes, p_no = rho_to_posteriors(-1.0, p_a, p_b, clamp_posteriors=False)
        # With rho=-1 and equal priors, P(A|B=yes) should be 0
        assert p_yes == pytest.approx(0.0)
        assert p_no == pytest.approx(1.0)

    def test_frechet_hoeffding_bounds(self):
        """Posteriors should respect Frechet-Hoeffding bounds."""
        # Test various edge cases
        test_cases = [
            (0.9, 0.1, 0.1),  # Extreme priors
            (-0.9, 0.9, 0.9),  # High priors with negative correlation
            (0.5, 0.2, 0.8),  # Asymmetric priors
        ]
        for rho, p_a, p_b in test_cases:
            p_yes, p_no = rho_to_posteriors(rho, p_a, p_b, clamp_posteriors=False)
            assert 0.0 <= p_yes <= 1.0, f"p_yes={p_yes} out of bounds"
            assert 0.0 <= p_no <= 1.0, f"p_no={p_no} out of bounds"

    def test_nan_handling(self):
        """NaN inputs should return prior as both posteriors."""
        p_yes, p_no = rho_to_posteriors(float("nan"), 0.5, 0.5)
        assert p_yes == 0.5
        assert p_no == 0.5

    def test_clamping(self):
        """Test that clamping keeps posteriors in [0.01, 0.99]."""
        # Perfect correlation with clamping
        p_yes, p_no = rho_to_posteriors(1.0, 0.5, 0.5, clamp_posteriors=True)
        assert p_yes == 0.99
        assert p_no == 0.01


class TestLinearVOIFromRho:
    """Tests for linear_voi_from_rho convenience function."""

    def test_zero_correlation_gives_zero_voi(self):
        """Zero correlation should give zero VOI."""
        result = linear_voi_from_rho(0.0, 0.5, 0.5)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_positive_correlation_gives_positive_voi(self):
        """Non-zero correlation should give positive VOI."""
        result = linear_voi_from_rho(0.5, 0.5, 0.5)
        assert result > 0

    def test_negative_correlation_gives_positive_voi(self):
        """Negative correlation also gives positive VOI (direction doesn't matter for VOI)."""
        result = linear_voi_from_rho(-0.5, 0.5, 0.5)
        assert result > 0

    def test_symmetric_in_rho_sign(self):
        """VOI should be symmetric with respect to rho sign."""
        voi_pos = linear_voi_from_rho(0.6, 0.4, 0.5)
        voi_neg = linear_voi_from_rho(-0.6, 0.4, 0.5)
        assert voi_pos == pytest.approx(voi_neg, rel=0.01)

    def test_higher_correlation_higher_voi(self):
        """Higher absolute correlation should give higher VOI."""
        voi_low = linear_voi_from_rho(0.2, 0.5, 0.5)
        voi_high = linear_voi_from_rho(0.8, 0.5, 0.5)
        assert voi_high > voi_low


class TestEntropyVOIFromRho:
    """Tests for entropy_voi_from_rho convenience function."""

    def test_zero_correlation_gives_zero_voi(self):
        """Zero correlation should give zero entropy VOI."""
        result = entropy_voi_from_rho(0.0, 0.5, 0.5)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_positive_voi_for_nonzero_correlation(self):
        """Non-zero correlation should give positive entropy VOI."""
        result = entropy_voi_from_rho(0.5, 0.5, 0.5)
        assert result > 0


class TestLinearVsEntropyStability:
    """Tests comparing linear vs entropy VOI stability properties."""

    def test_linear_more_stable_at_extremes(self):
        """Linear VOI should have more predictable behavior at extreme priors.

        This is a qualitative test - the key insight is that linear VOI
        has constant gradient while entropy VOI has steep gradients at extremes.
        """
        # At extreme prior (0.05), small changes in rho should produce
        # proportionally small changes in VOI for linear

        p_a = 0.05  # Extreme prior
        p_b = 0.5

        rho_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        linear_vois = [linear_voi_from_rho(r, p_a, p_b) for r in rho_values]
        entropy_vois = [entropy_voi_from_rho(r, p_a, p_b) for r in rho_values]

        # Linear VOI should be monotonically increasing with rho
        assert all(linear_vois[i] <= linear_vois[i+1] for i in range(len(linear_vois)-1))

        # Both methods should give positive VOI for non-zero correlation
        assert all(v > 0 for v in linear_vois)
        assert all(v >= 0 for v in entropy_vois)

        # Linear VOI should show more consistent increments (full analysis in experiments)
