"""Tests for rollup with cross-tree references.

Tests that rollup_tree correctly resolves refs to other trees
and uses their probabilities in calculations.
No LLM calls required.
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.registry import TreeRegistry
from shared.tree import SignalNode, SignalTree
from shared.rollup import (
    rollup_tree,
    compute_signal_evidence,
    compute_node_probability,
    compute_signal_contribution,
    analyze_tree,
)


@pytest.fixture
def source_tree() -> SignalTree:
    """A source tree that will be referenced by other trees."""
    root = SignalNode(
        id="target",
        text="Will Democrats win the House in 2026?",
        depth=0,
        is_leaf=False,
        base_rate=0.5,
    )

    economy = SignalNode(
        id="sig_economy",
        text="Will the economy be in recession by mid-2026?",
        depth=1,
        is_leaf=True,
        base_rate=0.25,  # 25% chance of recession
        rho=-0.4,  # Negative: recession hurts incumbent party
        parent_id="target",
    )

    approval = SignalNode(
        id="sig_approval",
        text="Will Biden's approval be above 45% in Oct 2026?",
        depth=1,
        is_leaf=True,
        base_rate=0.35,  # 35% chance
        rho=0.5,  # Positive: higher approval helps
        parent_id="target",
    )

    root.children = [economy, approval]

    tree = SignalTree(
        target=root,
        signals=[economy, approval],
        max_depth=1,
        leaf_count=2,
    )

    # Compute and store probability
    rollup_tree(tree, target_prior=0.5)

    return tree


@pytest.fixture
def registry_with_source(source_tree: SignalTree) -> TreeRegistry:
    """Registry with source tree already loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Save source tree
        house_dir = results_dir / "house_2026"
        house_dir.mkdir()
        with open(house_dir / "tree_20260120_test.json", "w") as f:
            json.dump(source_tree.model_dump(mode="json"), f)

        registry = TreeRegistry(results_dir)
        registry.load_tree("house_2026")

        yield registry


class TestRollupWithRefs:
    """Tests for rollup_tree with cross-tree refs."""

    def test_rollup_without_registry_ignores_refs(self, source_tree: SignalTree):
        """Ref signals are ignored if no registry provided."""
        # Create a tree with a ref signal
        root = SignalNode(
            id="target",
            text="Will Democrats win the Senate in 2026?",
            depth=0,
            is_leaf=False,
        )

        ref_signal = SignalNode(
            id="ref_economy",
            text="Will the economy be in recession by mid-2026?",
            ref="house_2026:sig_economy",  # Reference to another tree
            depth=1,
            is_leaf=True,
            rho=-0.35,
            parent_id="target",
            # base_rate intentionally None
        )

        root.children = [ref_signal]
        tree = SignalTree(target=root, signals=[ref_signal], max_depth=1, leaf_count=1)

        # Without registry, ref can't be resolved - uses fallback (prior)
        prob = rollup_tree(tree, target_prior=0.5)

        # Should return approximately the prior since ref can't be resolved
        assert abs(prob - 0.5) < 0.01

    def test_rollup_with_registry_resolves_refs(
        self, source_tree: SignalTree, registry_with_source: TreeRegistry
    ):
        """Ref signals pull base_rate from referenced tree."""
        # Create a new tree that references the source tree
        root = SignalNode(
            id="target",
            text="Will Democrats win the Senate in 2026?",
            depth=0,
            is_leaf=False,
        )

        # Reference the economy signal from house tree
        ref_signal = SignalNode(
            id="ref_economy",
            text="Will the economy be in recession by mid-2026?",
            ref="house_2026:sig_economy",
            depth=1,
            is_leaf=True,
            rho=-0.35,  # Similar correlation
            parent_id="target",
            # base_rate will be pulled from ref (0.25)
        )

        root.children = [ref_signal]
        tree = SignalTree(target=root, signals=[ref_signal], max_depth=1, leaf_count=1)

        prob = rollup_tree(tree, target_prior=0.5, registry=registry_with_source)

        # With registry, ref resolves to base_rate=0.25
        # Low recession probability + negative rho = positive evidence for Dems
        # So probability should be above 0.5
        assert prob > 0.5

    def test_rollup_with_root_ref(
        self, source_tree: SignalTree, registry_with_source: TreeRegistry
    ):
        """Can reference another tree's root (computed probability)."""
        root = SignalNode(
            id="target",
            text="Will Democrats win the presidency in 2028?",
            depth=0,
            is_leaf=False,
        )

        # Reference the entire House 2026 tree outcome
        house_ref = SignalNode(
            id="ref_house",
            text="Will Democrats win the House in 2026?",
            ref="house_2026",  # Reference to tree root
            depth=1,
            is_leaf=True,
            rho=0.3,  # Winning House is positive signal for 2028
            parent_id="target",
        )

        root.children = [house_ref]
        tree = SignalTree(target=root, signals=[house_ref], max_depth=1, leaf_count=1)

        prob = rollup_tree(tree, target_prior=0.5, registry=registry_with_source)

        # Result depends on house_2026's computed probability
        # Just verify it computed something reasonable
        assert 0.0 < prob < 1.0

    def test_compute_signal_evidence_with_ref(
        self, registry_with_source: TreeRegistry
    ):
        """compute_signal_evidence resolves refs correctly."""
        signal = SignalNode(
            id="ref_test",
            text="Economy signal",
            ref="house_2026:sig_economy",
            depth=1,
            is_leaf=True,
            rho=-0.4,
            parent_id="target",
        )

        evidence, spread, direction = compute_signal_evidence(
            signal, parent_prior=0.5, registry=registry_with_source
        )

        # Ref resolves to 0.25 base_rate
        # Low recession (0.25) with negative rho (-0.4) gives positive evidence
        # direction = 0.25 - 0.5 = -0.25 (signal likely NO)
        # spread < 0 (negative rho)
        # evidence = -0.25 * negative = positive
        assert direction < 0  # Signal likely to be NO (recession unlikely)
        assert spread < 0  # Negative rho
        assert evidence > 0  # Positive evidence for parent


class TestContributionWithRefs:
    """Tests for compute_signal_contribution with refs."""

    def test_contribution_includes_ref_info(
        self, source_tree: SignalTree, registry_with_source: TreeRegistry
    ):
        """compute_signal_contribution returns ref in result dict."""
        root = SignalNode(id="target", text="Test", depth=0, is_leaf=False)
        ref_signal = SignalNode(
            id="ref_test",
            text="Economy",
            ref="house_2026:sig_economy",
            depth=1,
            is_leaf=True,
            rho=-0.3,
            parent_id="target",
        )

        root.children = [ref_signal]
        tree = SignalTree(target=root, signals=[ref_signal], max_depth=1, leaf_count=1)

        contrib = compute_signal_contribution(
            ref_signal, tree, parent_prior=0.5, registry=registry_with_source
        )

        assert "ref" in contrib
        assert contrib["ref"] == "house_2026:sig_economy"
        assert contrib["base_rate"] == 0.25  # Resolved from ref

    def test_contribution_without_ref_has_none(self):
        """Regular signals have ref=None in contribution."""
        signal = SignalNode(
            id="test",
            text="Test",
            base_rate=0.6,
            rho=0.4,
            depth=1,
            is_leaf=True,
            parent_id="target",
        )
        tree = SignalTree(
            target=SignalNode(id="target", text="Root", depth=0),
            signals=[signal],
        )

        contrib = compute_signal_contribution(signal, tree, parent_prior=0.5)

        assert contrib.get("ref") is None


class TestAnalyzeTreeWithRefs:
    """Tests for analyze_tree with refs."""

    def test_analyze_tree_with_refs(
        self, source_tree: SignalTree, registry_with_source: TreeRegistry
    ):
        """analyze_tree works with ref signals."""
        root = SignalNode(id="target", text="Senate 2026", depth=0, is_leaf=False)

        ref_signal = SignalNode(
            id="ref_economy",
            text="Economy",
            ref="house_2026:sig_economy",
            depth=1,
            is_leaf=True,
            rho=-0.3,
            parent_id="target",
        )

        direct_signal = SignalNode(
            id="sig_polls",
            text="Senate polls favorable?",
            base_rate=0.55,
            rho=0.6,
            depth=1,
            is_leaf=True,
            parent_id="target",
        )

        root.children = [ref_signal, direct_signal]
        tree = SignalTree(
            target=root,
            signals=[ref_signal, direct_signal],
            max_depth=1,
            leaf_count=2,
        )

        analysis = analyze_tree(tree, target_prior=0.5, registry=registry_with_source)

        assert "computed_probability" in analysis
        assert 0.0 < analysis["computed_probability"] < 1.0
        assert len(analysis["all_contributions"]) == 2

        # Find the ref signal in contributions
        ref_contrib = next(
            c for c in analysis["all_contributions"] if c["signal_id"] == "ref_economy"
        )
        assert ref_contrib["ref"] == "house_2026:sig_economy"
        assert ref_contrib["base_rate"] == 0.25


class TestNecessitySufficiencyWithRefs:
    """Tests for necessity/sufficiency relationship types with refs."""

    def test_necessity_with_ref(self, registry_with_source: TreeRegistry):
        """Necessity constraint works with ref signals."""
        root = SignalNode(id="target", text="Will X win?", depth=0, is_leaf=False)

        # Necessity: Must have economy recovery (ref to inverse of recession)
        necessity_signal = SignalNode(
            id="ref_necessity",
            text="Economy signal",
            ref="house_2026:sig_economy",  # base_rate=0.25
            relationship_type="necessity",
            depth=1,
            is_leaf=True,
            parent_id="target",
        )

        root.children = [necessity_signal]
        tree = SignalTree(
            target=root, signals=[necessity_signal], max_depth=1, leaf_count=1
        )

        prob = rollup_tree(tree, target_prior=0.5, registry=registry_with_source)

        # Necessity caps at signal's base_rate (0.25)
        assert prob <= 0.26  # Small tolerance


class TestEdgeCases:
    """Edge case tests for ref handling."""

    def test_missing_ref_returns_fallback(self):
        """Ref to non-existent tree uses fallback."""
        registry = TreeRegistry()

        signal = SignalNode(
            id="test",
            text="Test",
            ref="nonexistent_tree:sig_abc",
            depth=1,
            is_leaf=True,
            rho=0.5,
            parent_id="target",
        )

        evidence, spread, direction = compute_signal_evidence(
            signal, parent_prior=0.5, registry=registry
        )

        # Can't resolve ref, returns zeros
        assert evidence == 0.0
        assert spread == 0.0
        assert direction == 0.0

    def test_ref_with_none_base_rate_in_source(self, registry_with_source: TreeRegistry):
        """Handles case where referenced node has None base_rate."""
        # Manually set a node's base_rate to None
        tree = registry_with_source.get_tree("house_2026")
        for signal in tree.signals:
            if signal.id == "sig_economy":
                signal.base_rate = None

        ref_signal = SignalNode(
            id="ref_test",
            text="Test",
            ref="house_2026:sig_economy",
            depth=1,
            is_leaf=True,
            rho=0.5,
            parent_id="target",
        )

        evidence, spread, direction = compute_signal_evidence(
            ref_signal, parent_prior=0.5, registry=registry_with_source
        )

        # None base_rate returns zeros
        assert evidence == 0.0

    def test_ref_signal_without_rho(self, registry_with_source: TreeRegistry):
        """Ref signal without rho returns zero evidence."""
        signal = SignalNode(
            id="ref_test",
            text="Test",
            ref="house_2026:sig_economy",
            depth=1,
            is_leaf=True,
            rho=None,  # No rho
            parent_id="target",
        )

        evidence, spread, direction = compute_signal_evidence(
            signal, parent_prior=0.5, registry=registry_with_source
        )

        assert evidence == 0.0
