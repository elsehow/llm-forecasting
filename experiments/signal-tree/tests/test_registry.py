"""Tests for TreeRegistry - cross-tree reference management.

Tests tree loading, node resolution, and similarity search.
No LLM calls required.
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.registry import TreeRegistry, parse_ref, make_ref
from shared.tree import SignalNode, SignalTree


@pytest.fixture
def sample_tree_data() -> dict:
    """Sample tree data for testing."""
    return {
        "target": {
            "id": "target",
            "text": "Will Democrats win the House in 2026?",
            "depth": 0,
            "is_leaf": False,
            "base_rate": 0.5,
            "children": [],
        },
        "signals": [
            {
                "id": "sig_economy",
                "text": "Will the economy be in recession by mid-2026?",
                "depth": 1,
                "is_leaf": True,
                "base_rate": 0.25,
                "rho": -0.4,
                "parent_id": "target",
                "children": [],
            },
            {
                "id": "sig_approval",
                "text": "Will Biden's approval be above 45% in Oct 2026?",
                "depth": 1,
                "is_leaf": True,
                "base_rate": 0.35,
                "rho": 0.5,
                "parent_id": "target",
                "children": [],
            },
        ],
        "max_depth": 1,
        "leaf_count": 2,
        "computed_probability": 0.48,
    }


@pytest.fixture
def temp_results_dir(sample_tree_data: dict) -> Path:
    """Create a temporary results directory with a sample tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create house_2026 tree
        house_dir = results_dir / "house_2026"
        house_dir.mkdir()
        with open(house_dir / "tree_20260120_test.json", "w") as f:
            json.dump(sample_tree_data, f)

        # Create senate_2026 tree (slightly different)
        senate_data = sample_tree_data.copy()
        senate_data["target"] = {
            "id": "target",
            "text": "Will Democrats win the Senate in 2026?",
            "depth": 0,
            "is_leaf": False,
            "base_rate": 0.5,
            "children": [],
        }
        senate_data["signals"] = [
            {
                "id": "sig_economy",  # Same ID as house tree
                "text": "Will the economy be in recession by mid-2026?",
                "depth": 1,
                "is_leaf": True,
                "base_rate": 0.25,
                "rho": -0.35,
                "parent_id": "target",
                "children": [],
            },
        ]
        senate_data["computed_probability"] = 0.52

        senate_dir = results_dir / "senate_2026"
        senate_dir.mkdir()
        with open(senate_dir / "tree_20260120_test.json", "w") as f:
            json.dump(senate_data, f)

        yield results_dir


class TestTreeRegistry:
    """Tests for TreeRegistry class."""

    def test_init_with_default_dir(self):
        """Registry initializes with default results directory."""
        registry = TreeRegistry()
        assert registry.results_dir.name == "results"

    def test_init_with_custom_dir(self, temp_results_dir: Path):
        """Registry can use custom results directory."""
        registry = TreeRegistry(temp_results_dir)
        assert registry.results_dir == temp_results_dir

    def test_load_tree_success(self, temp_results_dir: Path):
        """Registry can load tree from disk."""
        registry = TreeRegistry(temp_results_dir)
        tree = registry.load_tree("house_2026")

        assert tree is not None
        assert tree.target.text == "Will Democrats win the House in 2026?"
        assert len(tree.signals) == 2
        assert tree.computed_probability == 0.48

    def test_load_tree_not_found(self, temp_results_dir: Path):
        """Registry returns None for non-existent tree."""
        registry = TreeRegistry(temp_results_dir)
        tree = registry.load_tree("nonexistent_tree")
        assert tree is None

    def test_load_tree_caching(self, temp_results_dir: Path):
        """Registry caches loaded trees."""
        registry = TreeRegistry(temp_results_dir)

        tree1 = registry.load_tree("house_2026")
        tree2 = registry.load_tree("house_2026")

        assert tree1 is tree2  # Same object

    def test_register_tree_directly(self, temp_results_dir: Path):
        """Registry can register trees directly without disk."""
        registry = TreeRegistry(temp_results_dir)

        tree = SignalTree(
            target=SignalNode(id="target", text="Test", depth=0, is_leaf=False),
            signals=[],
            computed_probability=0.6,
        )
        registry.register_tree("test_tree", tree)

        assert registry.get_tree("test_tree") is tree

    def test_list_trees(self, temp_results_dir: Path):
        """Registry lists available trees."""
        registry = TreeRegistry(temp_results_dir)
        trees = registry.list_trees()

        assert "house_2026" in trees
        assert "senate_2026" in trees
        assert len(trees) == 2


class TestNodeResolution:
    """Tests for get_node() resolution."""

    def test_get_node_with_tree_id_only(self, temp_results_dir: Path):
        """tree_id alone resolves to root node."""
        registry = TreeRegistry(temp_results_dir)
        node = registry.get_node("house_2026")

        assert node is not None
        assert node.id == "target"
        assert "House" in node.text

    def test_get_node_with_full_ref(self, temp_results_dir: Path):
        """tree_id:node_id resolves to specific node."""
        registry = TreeRegistry(temp_results_dir)
        node = registry.get_node("house_2026:sig_economy")

        assert node is not None
        assert node.id == "sig_economy"
        assert "recession" in node.text.lower()

    def test_get_node_not_found(self, temp_results_dir: Path):
        """Returns None for non-existent ref."""
        registry = TreeRegistry(temp_results_dir)

        assert registry.get_node("house_2026:nonexistent") is None
        assert registry.get_node("nonexistent:sig_abc") is None

    def test_get_node_lazy_loads_tree(self, temp_results_dir: Path):
        """get_node() loads tree if not already loaded."""
        registry = TreeRegistry(temp_results_dir)

        # Tree not loaded yet
        assert "house_2026" not in registry._trees

        # get_node triggers lazy load
        node = registry.get_node("house_2026:sig_approval")

        assert node is not None
        assert "house_2026" in registry._trees


class TestProbabilityLookup:
    """Tests for get_probability()."""

    def test_get_probability_root(self, temp_results_dir: Path):
        """get_probability returns computed_probability for root."""
        registry = TreeRegistry(temp_results_dir)
        prob = registry.get_probability("house_2026")

        assert prob == 0.48

    def test_get_probability_signal(self, temp_results_dir: Path):
        """get_probability returns base_rate for non-root nodes."""
        registry = TreeRegistry(temp_results_dir)
        prob = registry.get_probability("house_2026:sig_economy")

        assert prob == 0.25

    def test_get_probability_not_found(self, temp_results_dir: Path):
        """get_probability returns None for non-existent refs."""
        registry = TreeRegistry(temp_results_dir)
        prob = registry.get_probability("nonexistent")

        assert prob is None


class TestSimilaritySearch:
    """Tests for search_similar()."""

    def test_search_similar_exact_match(self, temp_results_dir: Path):
        """search_similar finds exact text matches."""
        registry = TreeRegistry(temp_results_dir)
        registry.load_tree("house_2026")

        results = registry.search_similar("Will the economy be in recession")

        assert len(results) > 0
        tree_id, node, score = results[0]
        assert "recession" in node.text.lower()
        assert score > 0.3

    def test_search_similar_word_overlap(self, temp_results_dir: Path):
        """search_similar finds partial word matches."""
        registry = TreeRegistry(temp_results_dir)
        registry.load_tree("house_2026")

        # Use words that actually appear in the signal text
        results = registry.search_similar("economy recession 2026 mid")

        assert len(results) > 0
        # Should match the economy signal
        found_economy = any("economy" in r[1].text.lower() for r in results)
        assert found_economy

    def test_search_similar_no_matches(self, temp_results_dir: Path):
        """search_similar returns empty for unrelated text."""
        registry = TreeRegistry(temp_results_dir)
        registry.load_tree("house_2026")

        results = registry.search_similar("weather forecast temperature")

        # Should have no high-quality matches
        high_score_matches = [r for r in results if r[2] > 0.3]
        assert len(high_score_matches) == 0

    def test_search_similar_limit(self, temp_results_dir: Path):
        """search_similar respects limit parameter."""
        registry = TreeRegistry(temp_results_dir)
        registry.load_tree("house_2026")
        registry.load_tree("senate_2026")

        results = registry.search_similar("economy", limit=1)

        assert len(results) <= 1


class TestHelperFunctions:
    """Tests for parse_ref and make_ref helpers."""

    def test_parse_ref_with_node_id(self):
        """parse_ref splits tree_id:node_id correctly."""
        tree_id, node_id = parse_ref("house_2026:sig_abc")
        assert tree_id == "house_2026"
        assert node_id == "sig_abc"

    def test_parse_ref_without_node_id(self):
        """parse_ref handles tree_id alone."""
        tree_id, node_id = parse_ref("house_2026")
        assert tree_id == "house_2026"
        assert node_id is None

    def test_parse_ref_with_colon_in_node_id(self):
        """parse_ref handles colons in node_id."""
        tree_id, node_id = parse_ref("house_2026:sig:with:colons")
        assert tree_id == "house_2026"
        assert node_id == "sig:with:colons"

    def test_make_ref_with_node_id(self):
        """make_ref creates tree_id:node_id format."""
        ref = make_ref("house_2026", "sig_abc")
        assert ref == "house_2026:sig_abc"

    def test_make_ref_without_node_id(self):
        """make_ref returns tree_id alone when no node_id."""
        ref = make_ref("house_2026")
        assert ref == "house_2026"

        ref = make_ref("house_2026", None)
        assert ref == "house_2026"
