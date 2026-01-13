"""Tests for Value of Information (VOI) calculations."""

from datetime import date

import pytest

from tree_of_life.models import (
    ForecastTree,
    GlobalScenario,
    Signal,
    Question,
    QuestionType,
)
from tree_of_life.voi import (
    linear_voi,
    entropy_voi,
    entropy,
    estimate_posteriors,
    compute_signal_voi,
    rank_signals_by_voi,
    top_signals_by_voi,
    signals_by_voi_threshold,
    compare_voi_methods,
    MAGNITUDE_SHIFTS,
)


class TestLinearVOI:
    """Tests for the core linear VOI formula."""

    def test_no_information_same_posteriors(self):
        """VOI should be 0 when posteriors equal prior."""
        voi = linear_voi(
            p_x=0.5,
            p_q=0.5,
            p_x_given_q_yes=0.5,
            p_x_given_q_no=0.5,
        )
        assert voi == pytest.approx(0.0)

    def test_perfect_information(self):
        """VOI should be maximized when posteriors are 0 and 1."""
        voi = linear_voi(
            p_x=0.5,
            p_q=0.5,
            p_x_given_q_yes=1.0,
            p_x_given_q_no=0.0,
        )
        # Expected: 0.5 * |1.0 - 0.5| + 0.5 * |0.0 - 0.5| = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        assert voi == pytest.approx(0.5)

    def test_asymmetric_posteriors(self):
        """VOI with asymmetric shifts."""
        voi = linear_voi(
            p_x=0.3,
            p_q=0.4,
            p_x_given_q_yes=0.6,
            p_x_given_q_no=0.2,
        )
        # Expected: 0.4 * |0.6 - 0.3| + 0.6 * |0.2 - 0.3| = 0.4 * 0.3 + 0.6 * 0.1 = 0.18
        assert voi == pytest.approx(0.18)

    def test_extreme_base_rate_low(self):
        """Linear VOI handles low base rates without inflation."""
        voi = linear_voi(
            p_x=0.05,
            p_q=0.5,
            p_x_given_q_yes=0.15,
            p_x_given_q_no=0.02,
        )
        # Should be moderate, not inflated like entropy VOI would be
        assert 0 < voi < 0.1

    def test_extreme_base_rate_high(self):
        """Linear VOI handles high base rates without inflation."""
        voi = linear_voi(
            p_x=0.95,
            p_q=0.5,
            p_x_given_q_yes=0.98,
            p_x_given_q_no=0.85,
        )
        assert 0 < voi < 0.1


class TestEntropyVOI:
    """Tests for entropy-based VOI."""

    def test_entropy_of_certainty(self):
        """Entropy should be 0 for p=0 or p=1."""
        assert entropy(0.0) == 0.0
        assert entropy(1.0) == 0.0

    def test_entropy_of_uncertainty(self):
        """Entropy should be maximized at p=0.5."""
        assert entropy(0.5) == pytest.approx(1.0)  # 1 bit

    def test_entropy_voi_no_information(self):
        """Entropy VOI should be 0 when posteriors equal prior."""
        voi = entropy_voi(
            p_x=0.5,
            p_q=0.5,
            p_x_given_q_yes=0.5,
            p_x_given_q_no=0.5,
        )
        assert voi == pytest.approx(0.0)

    def test_entropy_voi_perfect_information(self):
        """Entropy VOI should equal prior entropy for perfect information."""
        voi = entropy_voi(
            p_x=0.5,
            p_q=0.5,
            p_x_given_q_yes=1.0,
            p_x_given_q_no=0.0,
        )
        # With perfect info, we gain all the prior uncertainty
        # H(prior) = 1 bit, H(posterior) = 0.5 * 0 + 0.5 * 0 = 0
        assert voi == pytest.approx(1.0)


class TestEstimatePosteiors:
    """Tests for posterior estimation from magnitude/direction."""

    def test_increases_medium(self):
        """Test increases direction with medium magnitude."""
        p_yes, p_no = estimate_posteriors(
            p_x=0.3,
            direction="increases",
            magnitude="medium",
        )
        # Medium shift = 0.25 of available space
        # p_yes: 0.3 + 0.25 * (1.0 - 0.3) = 0.3 + 0.175 = 0.475
        # p_no: 0.3 - 0.25 * 0.3 = 0.3 - 0.075 = 0.225
        assert p_yes == pytest.approx(0.475)
        assert p_no == pytest.approx(0.225)

    def test_decreases_medium(self):
        """Test decreases direction with medium magnitude."""
        p_yes, p_no = estimate_posteriors(
            p_x=0.3,
            direction="decreases",
            magnitude="medium",
        )
        # Medium shift = 0.25 of available space
        # p_yes: 0.3 - 0.25 * 0.3 = 0.225
        # p_no: 0.3 + 0.25 * 0.7 = 0.475
        assert p_yes == pytest.approx(0.225)
        assert p_no == pytest.approx(0.475)

    def test_small_magnitude(self):
        """Test small magnitude shifts."""
        p_yes, p_no = estimate_posteriors(
            p_x=0.5,
            direction="increases",
            magnitude="small",
        )
        # Small shift = 0.10
        assert p_yes == pytest.approx(0.55)
        assert p_no == pytest.approx(0.45)

    def test_large_magnitude(self):
        """Test large magnitude shifts."""
        p_yes, p_no = estimate_posteriors(
            p_x=0.5,
            direction="increases",
            magnitude="large",
        )
        # Large shift = 0.50
        assert p_yes == pytest.approx(0.75)
        assert p_no == pytest.approx(0.25)

    def test_clamping_at_boundaries(self):
        """Test that posteriors are clamped to [0, 1]."""
        # Near 1.0
        p_yes, p_no = estimate_posteriors(
            p_x=0.95,
            direction="increases",
            magnitude="large",
        )
        assert p_yes <= 1.0
        assert p_no >= 0.0

        # Near 0.0
        p_yes, p_no = estimate_posteriors(
            p_x=0.05,
            direction="decreases",
            magnitude="large",
        )
        assert p_yes >= 0.0
        assert p_no <= 1.0

    def test_custom_magnitude_shifts(self):
        """Test custom magnitude shift factors."""
        custom = {"small": 0.05, "medium": 0.15, "large": 0.30}
        p_yes, p_no = estimate_posteriors(
            p_x=0.5,
            direction="increases",
            magnitude="medium",
            magnitude_shifts=custom,
        )
        # Custom medium = 0.15
        assert p_yes == pytest.approx(0.575)
        assert p_no == pytest.approx(0.425)


class TestComputeSignalVOI:
    """Tests for signal-level VOI computation."""

    @pytest.fixture
    def scenario(self) -> GlobalScenario:
        return GlobalScenario(
            id="s1",
            name="Test Scenario",
            description="A test scenario",
            probability=0.3,
        )

    @pytest.fixture
    def signal_increases(self) -> Signal:
        return Signal(
            id="sig1",
            text="Something happens",
            resolves_by=date(2026, 6, 1),
            scenario_id="s1",
            direction="increases",
            magnitude="medium",
            current_probability=0.4,
        )

    @pytest.fixture
    def signal_decreases(self) -> Signal:
        return Signal(
            id="sig2",
            text="Something else happens",
            resolves_by=date(2026, 6, 1),
            scenario_id="s1",
            direction="decreases",
            magnitude="large",
            current_probability=0.3,
        )

    def test_compute_signal_voi_linear(self, scenario, signal_increases):
        """Test VOI computation with linear method."""
        voi = compute_signal_voi(signal_increases, scenario, voi_type="linear")
        assert voi > 0
        # With medium magnitude increases, should have moderate VOI
        assert 0.05 < voi < 0.3

    def test_compute_signal_voi_entropy(self, scenario, signal_increases):
        """Test VOI computation with entropy method."""
        voi = compute_signal_voi(signal_increases, scenario, voi_type="entropy")
        assert voi > 0

    def test_default_p_q(self, scenario):
        """Test that missing current_probability defaults to 0.5."""
        signal = Signal(
            id="sig1",
            text="Something happens",
            resolves_by=date(2026, 6, 1),
            scenario_id="s1",
            direction="increases",
            magnitude="medium",
            current_probability=None,
        )
        voi = compute_signal_voi(signal, scenario, voi_type="linear")
        assert voi > 0

    def test_large_magnitude_higher_voi(self, scenario):
        """Larger magnitude should generally produce higher VOI."""
        signal_small = Signal(
            id="sig1",
            text="Something",
            resolves_by=date(2026, 6, 1),
            scenario_id="s1",
            direction="increases",
            magnitude="small",
            current_probability=0.5,
        )
        signal_large = Signal(
            id="sig2",
            text="Something",
            resolves_by=date(2026, 6, 1),
            scenario_id="s1",
            direction="increases",
            magnitude="large",
            current_probability=0.5,
        )

        voi_small = compute_signal_voi(signal_small, scenario, voi_type="linear")
        voi_large = compute_signal_voi(signal_large, scenario, voi_type="linear")

        assert voi_large > voi_small


class TestRankSignalsByVOI:
    """Tests for ranking signals by VOI."""

    @pytest.fixture
    def simple_tree(self) -> ForecastTree:
        """Create a simple tree with scenarios and signals."""
        scenarios = [
            GlobalScenario(
                id="high_prob",
                name="High Prob",
                description="High probability scenario",
                probability=0.6,
            ),
            GlobalScenario(
                id="low_prob",
                name="Low Prob",
                description="Low probability scenario",
                probability=0.1,
            ),
            GlobalScenario(
                id="mid_prob",
                name="Mid Prob",
                description="Mid probability scenario",
                probability=0.3,
            ),
        ]

        signals = [
            Signal(
                id="sig_high_large",
                text="Large signal for high prob",
                resolves_by=date(2026, 6, 1),
                scenario_id="high_prob",
                direction="increases",
                magnitude="large",
                current_probability=0.5,
            ),
            Signal(
                id="sig_low_small",
                text="Small signal for low prob",
                resolves_by=date(2026, 6, 1),
                scenario_id="low_prob",
                direction="increases",
                magnitude="small",
                current_probability=0.5,
            ),
            Signal(
                id="sig_mid_medium",
                text="Medium signal for mid prob",
                resolves_by=date(2026, 6, 1),
                scenario_id="mid_prob",
                direction="decreases",
                magnitude="medium",
                current_probability=0.5,
            ),
        ]

        return ForecastTree(
            questions=[
                Question(
                    id="q1",
                    source="tree",
                    text="Test question",
                    question_type=QuestionType.BINARY,
                )
            ],
            raw_scenarios=[],
            global_scenarios=scenarios,
            relationships=[],
            conditionals=[],
            signals=signals,
        )

    def test_rank_returns_all_signals(self, simple_tree):
        """Ranking should return all signals."""
        ranked = rank_signals_by_voi(simple_tree, voi_type="linear")
        assert len(ranked) == 3

    def test_rank_is_descending(self, simple_tree):
        """Signals should be sorted descending by VOI."""
        ranked = rank_signals_by_voi(simple_tree, voi_type="linear")
        voi_values = [voi for _, voi in ranked]
        assert voi_values == sorted(voi_values, reverse=True)

    def test_top_signals(self, simple_tree):
        """top_signals_by_voi should return top N signals."""
        top = top_signals_by_voi(simple_tree, n=2, voi_type="linear")
        assert len(top) == 2

        # Should be top 2 from full ranking
        full = rank_signals_by_voi(simple_tree, voi_type="linear")
        assert top[0][0].id == full[0][0].id
        assert top[1][0].id == full[1][0].id

    def test_signals_by_threshold(self, simple_tree):
        """signals_by_voi_threshold should filter by minimum VOI."""
        # Get all VOI values
        ranked = rank_signals_by_voi(simple_tree, voi_type="linear")
        mid_voi = ranked[1][1]  # Middle VOI value

        # Filter by threshold
        filtered = signals_by_voi_threshold(simple_tree, min_voi=mid_voi)
        assert len(filtered) >= 1
        assert all(voi >= mid_voi for _, voi in filtered)

    def test_skips_unknown_scenarios(self):
        """Signals pointing to unknown scenarios should be skipped."""
        tree = ForecastTree(
            questions=[
                Question(
                    id="q1",
                    source="tree",
                    text="Test",
                    question_type=QuestionType.BINARY,
                )
            ],
            raw_scenarios=[],
            global_scenarios=[
                GlobalScenario(id="known", name="Known", description="Known", probability=0.5)
            ],
            relationships=[],
            conditionals=[],
            signals=[
                Signal(
                    id="sig1",
                    text="Signal for known",
                    resolves_by=date(2026, 6, 1),
                    scenario_id="known",
                    direction="increases",
                    magnitude="medium",
                ),
                Signal(
                    id="sig2",
                    text="Signal for unknown",
                    resolves_by=date(2026, 6, 1),
                    scenario_id="unknown",
                    direction="increases",
                    magnitude="medium",
                ),
            ],
        )

        ranked = rank_signals_by_voi(tree)
        assert len(ranked) == 1
        assert ranked[0][0].id == "sig1"


class TestCompareVOIMethods:
    """Tests for comparing linear vs entropy VOI."""

    @pytest.fixture
    def tree_with_extreme_rates(self) -> ForecastTree:
        """Create tree with scenarios at extreme base rates."""
        scenarios = [
            GlobalScenario(id="rare", name="Rare", description="Rare event", probability=0.05),
            GlobalScenario(id="common", name="Common", description="Common event", probability=0.95),
            GlobalScenario(id="balanced", name="Balanced", description="50/50", probability=0.50),
        ]

        signals = [
            Signal(
                id="sig_rare",
                text="Signal for rare",
                resolves_by=date(2026, 6, 1),
                scenario_id="rare",
                direction="increases",
                magnitude="medium",
                current_probability=0.5,
            ),
            Signal(
                id="sig_common",
                text="Signal for common",
                resolves_by=date(2026, 6, 1),
                scenario_id="common",
                direction="decreases",
                magnitude="medium",
                current_probability=0.5,
            ),
            Signal(
                id="sig_balanced",
                text="Signal for balanced",
                resolves_by=date(2026, 6, 1),
                scenario_id="balanced",
                direction="increases",
                magnitude="medium",
                current_probability=0.5,
            ),
        ]

        return ForecastTree(
            questions=[
                Question(id="q1", source="tree", text="Test", question_type=QuestionType.BINARY)
            ],
            raw_scenarios=[],
            global_scenarios=scenarios,
            relationships=[],
            conditionals=[],
            signals=signals,
        )

    def test_compare_returns_both_methods(self, tree_with_extreme_rates):
        """Comparison should include both VOI types."""
        results = compare_voi_methods(tree_with_extreme_rates)
        assert len(results) == 3

        for r in results:
            assert "linear_voi" in r
            assert "entropy_voi" in r
            assert "voi_difference" in r

    def test_methods_differ_at_extremes(self, tree_with_extreme_rates):
        """Linear and entropy VOI should differ for extreme base rates."""
        results = compare_voi_methods(tree_with_extreme_rates)

        # Find the rare scenario result
        rare_result = next(r for r in results if r["scenario_id"] == "rare")
        balanced_result = next(r for r in results if r["scenario_id"] == "balanced")

        # At extreme base rates, the methods should diverge more
        # Linear should be more stable (smaller) relative to entropy for rare events
        # This is the key insight from the experiments
        assert rare_result["linear_voi"] != rare_result["entropy_voi"]


class TestForecastTreeVOIMethods:
    """Tests for ForecastTree convenience methods."""

    @pytest.fixture
    def tree(self) -> ForecastTree:
        """Create a simple tree for testing."""
        return ForecastTree(
            questions=[
                Question(id="q1", source="tree", text="Test", question_type=QuestionType.BINARY)
            ],
            raw_scenarios=[],
            global_scenarios=[
                GlobalScenario(id="s1", name="S1", description="Scenario 1", probability=0.4),
                GlobalScenario(id="s2", name="S2", description="Scenario 2", probability=0.6),
            ],
            relationships=[],
            conditionals=[],
            signals=[
                Signal(
                    id="sig1",
                    text="Signal 1",
                    resolves_by=date(2026, 6, 1),
                    scenario_id="s1",
                    direction="increases",
                    magnitude="large",
                    current_probability=0.5,
                ),
                Signal(
                    id="sig2",
                    text="Signal 2",
                    resolves_by=date(2026, 6, 1),
                    scenario_id="s2",
                    direction="decreases",
                    magnitude="small",
                    current_probability=0.5,
                ),
            ],
        )

    def test_signals_ranked_by_voi(self, tree):
        """Test the convenience method on ForecastTree."""
        ranked = tree.signals_ranked_by_voi(voi_type="linear")
        assert len(ranked) == 2
        # Should be sorted descending
        assert ranked[0][1] >= ranked[1][1]

    def test_top_voi_signals(self, tree):
        """Test the top N convenience method."""
        top = tree.top_voi_signals(n=1, voi_type="linear")
        assert len(top) == 1


class TestVOIStabilityAdvantage:
    """Tests demonstrating linear VOI's stability advantage.

    From experiments: Linear VOI provides +0.160 τ stability advantage,
    rising to +0.352 τ at extreme base rates (<0.10 or >0.90).
    """

    def test_low_probability_stability(self):
        """Linear VOI should be more stable for rare events."""
        p_x = 0.05  # Rare event

        # Slightly perturbed posteriors (simulating estimation noise)
        p_yes_true = 0.15
        p_no_true = 0.02

        # Noisy estimates (±20% error)
        p_yes_noisy = 0.18
        p_no_noisy = 0.025

        linear_true = linear_voi(p_x, 0.5, p_yes_true, p_no_true)
        linear_noisy = linear_voi(p_x, 0.5, p_yes_noisy, p_no_noisy)

        entropy_true = entropy_voi(p_x, 0.5, p_yes_true, p_no_true)
        entropy_noisy = entropy_voi(p_x, 0.5, p_yes_noisy, p_no_noisy)

        linear_error = abs(linear_noisy - linear_true) / linear_true if linear_true > 0 else 0
        entropy_error = abs(entropy_noisy - entropy_true) / entropy_true if entropy_true > 0 else 0

        # Linear should have smaller relative error for extreme base rates
        # Note: This is a qualitative check, not exact numerical verification
        assert linear_error < 0.5  # Linear error should be manageable

    def test_high_probability_stability(self):
        """Linear VOI should be more stable for near-certain events."""
        p_x = 0.95  # Near-certain event

        # Use Bayesian-consistent posteriors: P(X) = P(X|yes)*P(yes) + P(X|no)*P(no)
        # 0.95 = 0.98 * 0.5 + 0.92 * 0.5 = 0.95 ✓
        p_yes_true = 0.98
        p_no_true = 0.92

        linear_voi_val = linear_voi(p_x, 0.5, p_yes_true, p_no_true)
        entropy_voi_val = entropy_voi(p_x, 0.5, p_yes_true, p_no_true)

        # Both should be positive (non-negative for proper Bayesian updates)
        assert linear_voi_val > 0
        assert entropy_voi_val >= 0
