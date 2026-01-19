"""Tests for probability rollup functions.

These are pure math tests - no LLM calls required.

The rollup module uses a "soft evidence" model where signal base_rates
are treated as implicit evidence about the target:

    evidence = (base_rate - 0.5) * spread
    spread = P(parent|signal=YES) - P(parent|signal=NO)

This captures: "how certain are we about the signal" × "how much would it move belief"
"""

from datetime import date, timedelta
import math

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.rollup import (
    compute_signal_evidence,
    compute_node_probability,
    rollup_tree,
    compute_signal_contribution,
    analyze_tree,
)
from shared.tree import SignalNode, SignalTree


class TestRhoToPosteriors:
    """Tests for rho_to_posteriors (from llm_forecasting.voi)."""

    def test_positive_rho_increases_posterior(self):
        """Positive rho: P(parent|YES) > prior."""
        from llm_forecasting.voi import rho_to_posteriors

        rho = 0.6
        prior = 0.5
        p_signal = 0.7

        p_yes, p_no = rho_to_posteriors(rho, prior, p_signal)

        # With positive correlation, signal YES should increase parent prob
        assert p_yes > prior
        # And signal NO should decrease it
        assert p_no < prior

    def test_negative_rho_decreases_posterior(self):
        """Negative rho: P(parent|YES) < prior."""
        from llm_forecasting.voi import rho_to_posteriors

        rho = -0.4
        prior = 0.5
        p_signal = 0.8

        p_yes, p_no = rho_to_posteriors(rho, prior, p_signal)

        # With negative correlation, signal YES should decrease parent prob
        assert p_yes < prior
        # And signal NO should increase it
        assert p_no > prior

    def test_zero_rho_unchanged_posterior(self):
        """Zero rho: P(parent|YES) = P(parent|NO) = prior."""
        from llm_forecasting.voi import rho_to_posteriors

        rho = 0.0
        prior = 0.5
        p_signal = 0.7

        p_yes, p_no = rho_to_posteriors(rho, prior, p_signal)

        # Independent signals don't change the posterior
        assert abs(p_yes - prior) < 0.01
        assert abs(p_no - prior) < 0.01


class TestComputeSignalEvidence:
    """Tests for compute_signal_evidence function."""

    def test_high_base_rate_positive_rho_gives_positive_evidence(self):
        """High base_rate + positive rho → positive evidence for parent."""
        signal = SignalNode(
            id="test",
            text="Test signal",
            base_rate=0.8,  # Likely to be YES
            rho=0.6,  # Positive correlation
            parent_id="target",
            depth=1,
        )

        evidence, spread, direction = compute_signal_evidence(signal, parent_prior=0.5)

        assert evidence > 0  # Positive evidence
        assert spread > 0  # Signal YES increases parent prob
        assert direction > 0  # base_rate > 0.5

    def test_low_base_rate_positive_rho_gives_negative_evidence(self):
        """Low base_rate + positive rho → negative evidence for parent."""
        signal = SignalNode(
            id="test",
            text="Test signal",
            base_rate=0.2,  # Likely to be NO
            rho=0.6,  # Positive correlation
            parent_id="target",
            depth=1,
        )

        evidence, spread, direction = compute_signal_evidence(signal, parent_prior=0.5)

        assert evidence < 0  # Negative evidence (signal likely NO, but rho positive)
        assert spread > 0  # Signal YES would increase parent prob
        assert direction < 0  # base_rate < 0.5

    def test_high_base_rate_negative_rho_gives_negative_evidence(self):
        """High base_rate + negative rho → negative evidence (competitor succeeding)."""
        signal = SignalNode(
            id="test",
            text="Competitor likely to win",
            base_rate=0.8,  # Competitor likely to succeed
            rho=-0.5,  # Negative correlation (competition)
            parent_id="target",
            depth=1,
        )

        evidence, spread, direction = compute_signal_evidence(signal, parent_prior=0.5)

        assert evidence < 0  # Negative evidence (competitor doing well hurts us)
        assert spread < 0  # Signal YES decreases parent prob
        assert direction > 0  # base_rate > 0.5

    def test_low_base_rate_negative_rho_gives_positive_evidence(self):
        """Low base_rate + negative rho → positive evidence (competitor failing)."""
        signal = SignalNode(
            id="test",
            text="Competitor likely to fail",
            base_rate=0.2,  # Competitor likely to fail
            rho=-0.5,  # Negative correlation (competition)
            parent_id="target",
            depth=1,
        )

        evidence, spread, direction = compute_signal_evidence(signal, parent_prior=0.5)

        assert evidence > 0  # Positive evidence (competitor failing helps us)
        assert spread < 0  # Signal YES would decrease parent prob
        assert direction < 0  # base_rate < 0.5, so likely NO

    def test_base_rate_at_half_gives_zero_evidence(self):
        """base_rate = 0.5 → zero evidence (maximum uncertainty)."""
        signal = SignalNode(
            id="test",
            text="Uncertain signal",
            base_rate=0.5,
            rho=0.8,  # Strong correlation but...
            parent_id="target",
            depth=1,
        )

        evidence, spread, direction = compute_signal_evidence(signal, parent_prior=0.5)

        assert evidence == 0.0  # No evidence when maximally uncertain
        assert direction == 0.0

    def test_zero_rho_gives_zero_evidence(self):
        """rho = 0 → zero evidence (independent signal)."""
        signal = SignalNode(
            id="test",
            text="Independent signal",
            base_rate=0.9,  # Very certain, but...
            rho=0.0,  # Independent
            parent_id="target",
            depth=1,
        )

        evidence, spread, direction = compute_signal_evidence(signal, parent_prior=0.5)

        assert abs(evidence) < 0.01  # Essentially zero evidence
        assert abs(spread) < 0.01  # No spread for independent signal

    def test_missing_values_return_zero(self):
        """Missing rho or base_rate → zero evidence."""
        signal_no_rho = SignalNode(
            id="no_rho",
            text="No rho",
            base_rate=0.8,
            rho=None,
            parent_id="target",
            depth=1,
        )
        signal_no_base = SignalNode(
            id="no_base",
            text="No base_rate",
            base_rate=None,
            rho=0.5,
            parent_id="target",
            depth=1,
        )

        ev1, _, _ = compute_signal_evidence(signal_no_rho, 0.5)
        ev2, _, _ = compute_signal_evidence(signal_no_base, 0.5)

        assert ev1 == 0.0
        assert ev2 == 0.0


class TestComputeNodeProbability:
    """Tests for compute_node_probability function."""

    def test_leaf_node_returns_base_rate(self, leaf_node: SignalNode):
        """Leaf node with no children returns its base_rate."""
        tree = SignalTree(
            target=SignalNode(id="target", text="Root", depth=0, is_leaf=False),
            signals=[leaf_node],
        )

        result = compute_node_probability(leaf_node, tree, prior=0.5)
        assert result == leaf_node.base_rate

    def test_node_with_zero_rho_child_unchanged(self, today: date):
        """Node with rho=0 child should have probability unchanged from prior."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        child = SignalNode(
            id="child",
            text="Independent signal",
            base_rate=0.8,
            rho=0.0,  # Independent
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [child]
        tree = SignalTree(target=root, signals=[child])

        result = compute_node_probability(root, tree, prior=0.5)

        # With rho=0, no evidence is contributed
        assert abs(result - 0.5) < 0.05

    def test_positive_rho_high_prob_increases_parent(self, today: date):
        """Positive rho with p > 0.5 should increase parent probability."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        child = SignalNode(
            id="child",
            text="Supportive signal",
            base_rate=0.8,
            rho=0.6,
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [child]
        tree = SignalTree(target=root, signals=[child])

        result = compute_node_probability(root, tree, prior=0.5)

        assert result > 0.5  # Should increase from prior

    def test_negative_rho_high_prob_decreases_parent(self, today: date):
        """Negative rho with p > 0.5 should decrease parent probability."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        child = SignalNode(
            id="child",
            text="Competitor signal",
            base_rate=0.8,
            rho=-0.6,
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [child]
        tree = SignalTree(target=root, signals=[child])

        result = compute_node_probability(root, tree, prior=0.5)

        assert result < 0.5  # Should decrease from prior

    def test_negative_rho_low_prob_increases_parent(self, today: date):
        """Negative rho with p < 0.5 should increase parent probability."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)
        child = SignalNode(
            id="child",
            text="Competitor failing signal",
            base_rate=0.2,  # Competitor likely to fail
            rho=-0.6,  # Negative correlation
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [child]
        tree = SignalTree(target=root, signals=[child])

        result = compute_node_probability(root, tree, prior=0.5)

        # Competitor failing (low base_rate) + negative rho = positive evidence
        assert result > 0.5


class TestRollupTree:
    """Tests for rollup_tree function."""

    def test_flat_tree_uses_direct_children(self, sample_tree: SignalTree):
        """Flat tree (depth 1) should use direct child contributions."""
        target_prior = 0.5
        result = rollup_tree(sample_tree, target_prior)

        # Result should be a valid probability
        assert 0.0 < result < 1.0

    def test_deep_tree_propagates_correctly(self, deep_tree: SignalTree):
        """Deep tree (depth 3) should propagate probabilities from leaves up."""
        target_prior = 0.5
        result = rollup_tree(deep_tree, target_prior)

        # Result should be a valid probability
        assert 0.0 < result < 1.0
        assert deep_tree.computed_probability == result

    def test_tree_with_mixed_rho_balances(self, today: date):
        """Tree with opposing signals should partially balance out."""
        root = SignalNode(id="target", text="Root", depth=0, is_leaf=False)

        # Two children with opposite effects
        child_pos = SignalNode(
            id="pos",
            text="Positive evidence",
            base_rate=0.7,  # Likely
            rho=0.5,  # Positive correlation
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        child_neg = SignalNode(
            id="neg",
            text="Negative evidence (competitor)",
            base_rate=0.7,  # Competitor also likely
            rho=-0.5,  # Negative correlation
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        root.children = [child_pos, child_neg]
        tree = SignalTree(target=root, signals=[child_pos, child_neg])

        result = rollup_tree(tree, target_prior=0.5)

        # Evidence should partially cancel out, result near prior
        assert abs(result - 0.5) < 0.15

    def test_conditional_probabilities_populated(self, sample_tree: SignalTree):
        """Rollup should populate p_parent_given_yes/no on signals."""
        rollup_tree(sample_tree, target_prior=0.5)

        for signal in sample_tree.signals:
            if signal.rho is not None and signal.base_rate is not None:
                assert signal.p_parent_given_yes is not None
                assert signal.p_parent_given_no is not None
                assert 0.0 <= signal.p_parent_given_yes <= 1.0
                assert 0.0 <= signal.p_parent_given_no <= 1.0


class TestComputeSignalContribution:
    """Tests for compute_signal_contribution function."""

    def test_returns_evidence_field(self, today: date):
        """Should return 'evidence' field with actual contribution."""
        signal = SignalNode(
            id="test",
            text="Test signal",
            base_rate=0.8,
            rho=0.6,
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        tree = SignalTree(
            target=SignalNode(id="target", text="Root", depth=0),
            signals=[signal],
        )

        result = compute_signal_contribution(signal, tree, parent_prior=0.5)

        assert "evidence" in result
        assert "spread" in result
        assert "certainty" in result
        assert result["evidence"] > 0  # High base_rate + positive rho

    def test_evidence_matches_formula(self, today: date):
        """Evidence should equal (base_rate - 0.5) * spread."""
        signal = SignalNode(
            id="test",
            text="Test signal",
            base_rate=0.9,
            rho=0.8,
            parent_id="target",
            depth=1,
            is_leaf=True,
        )
        tree = SignalTree(
            target=SignalNode(id="target", text="Root", depth=0),
            signals=[signal],
        )

        result = compute_signal_contribution(signal, tree, parent_prior=0.5)

        # Verify formula
        expected_evidence = (signal.base_rate - 0.5) * result["spread"]
        assert abs(result["evidence"] - expected_evidence) < 0.001

    def test_direction_based_on_rho_sign(self, today: date):
        """Direction should be 'enhances' for positive rho, 'suppresses' for negative."""
        pos_signal = SignalNode(
            id="pos", text="Positive", base_rate=0.7, rho=0.5, parent_id="t", depth=1
        )
        neg_signal = SignalNode(
            id="neg", text="Negative", base_rate=0.7, rho=-0.5, parent_id="t", depth=1
        )
        tree = SignalTree(
            target=SignalNode(id="t", text="Root", depth=0),
            signals=[pos_signal, neg_signal],
        )

        pos_result = compute_signal_contribution(pos_signal, tree)
        neg_result = compute_signal_contribution(neg_signal, tree)

        assert pos_result["direction"] == "enhances"
        assert neg_result["direction"] == "suppresses"

    def test_certainty_is_distance_from_half(self, today: date):
        """Certainty should be |base_rate - 0.5|."""
        signal = SignalNode(
            id="test", text="Test", base_rate=0.8, rho=0.5, parent_id="t", depth=1
        )
        tree = SignalTree(
            target=SignalNode(id="t", text="Root", depth=0),
            signals=[signal],
        )

        result = compute_signal_contribution(signal, tree)

        assert result["certainty"] == abs(0.8 - 0.5)


class TestAnalyzeTree:
    """Tests for analyze_tree function."""

    def test_returns_evidence_breakdown(self, sample_tree: SignalTree):
        """Should return evidence breakdown with positive/negative/net."""
        analysis = analyze_tree(sample_tree, target_prior=0.5)

        assert "evidence_breakdown" in analysis
        breakdown = analysis["evidence_breakdown"]
        assert "positive" in breakdown
        assert "negative" in breakdown
        assert "net" in breakdown
        assert abs(breakdown["net"] - (breakdown["positive"] + breakdown["negative"])) < 0.001

    def test_top_contributors_sorted_by_evidence(self, sample_tree: SignalTree):
        """Top contributors should be sorted by |evidence|."""
        analysis = analyze_tree(sample_tree, target_prior=0.5)

        contributions = analysis["top_contributors"]
        if len(contributions) > 1:
            for i in range(len(contributions) - 1):
                assert abs(contributions[i]["evidence"]) >= abs(contributions[i + 1]["evidence"])

    def test_computed_probability_stored(self, sample_tree: SignalTree):
        """Computed probability should be stored in analysis and tree."""
        analysis = analyze_tree(sample_tree, target_prior=0.5)

        assert "computed_probability" in analysis
        assert 0.0 < analysis["computed_probability"] < 1.0
        assert sample_tree.computed_probability == analysis["computed_probability"]
