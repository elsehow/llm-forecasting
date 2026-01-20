"""Registry for loading, caching, and searching signal trees.

Enables cross-tree references by providing:
- Tree loading from disk (results/{tree_id}/tree_*.json)
- Node lookup by reference string (tree_id:node_id or tree_id for root)
- Probability lookup from referenced nodes
- Simple text search for finding similar existing signals
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .tree import SignalNode, SignalTree


class TreeRegistry:
    """Load, cache, and search signal trees."""

    def __init__(self, results_dir: Path | str | None = None):
        """Initialize registry.

        Args:
            results_dir: Path to results directory containing tree subdirectories.
                         Defaults to experiments/signal-tree/results.
        """
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results"
        self.results_dir = Path(results_dir)
        self._trees: dict[str, SignalTree] = {}
        self._computed_probs: dict[str, float] = {}  # tree_id -> computed_probability
        self._node_index: dict[str, tuple[str, SignalNode]] = {}  # "tree_id:node_id" -> (tree_id, node)

    def load_tree(self, tree_id: str) -> SignalTree | None:
        """Load tree from results/{tree_id}/tree_*.json (most recent).

        Args:
            tree_id: The tree identifier (directory name in results/)

        Returns:
            The loaded SignalTree, or None if not found
        """
        if tree_id in self._trees:
            return self._trees[tree_id]

        tree_dir = self.results_dir / tree_id
        if not tree_dir.exists():
            return None

        # Find most recent tree file
        tree_files = sorted(tree_dir.glob("tree_*.json"), reverse=True)
        if not tree_files:
            return None

        # Load the tree
        with open(tree_files[0]) as f:
            data = json.load(f)

        tree = SignalTree.model_validate(data)
        self._trees[tree_id] = tree

        # Store computed probability if present
        if tree.computed_probability is not None:
            self._computed_probs[tree_id] = tree.computed_probability

        # Index all nodes
        self.index_tree(tree_id, tree)

        return tree

    def register_tree(self, tree_id: str, tree: SignalTree) -> None:
        """Register a tree directly (without loading from disk).

        Useful for trees that are being built and haven't been saved yet.

        Args:
            tree_id: Identifier for this tree
            tree: The SignalTree to register
        """
        self._trees[tree_id] = tree
        if tree.computed_probability is not None:
            self._computed_probs[tree_id] = tree.computed_probability
        self.index_tree(tree_id, tree)

    def index_tree(self, tree_id: str, tree: SignalTree) -> None:
        """Add all nodes from tree to search index.

        Args:
            tree_id: Identifier for this tree
            tree: The SignalTree to index
        """
        # Index root
        root_key = f"{tree_id}:{tree.target.id}"
        self._node_index[root_key] = (tree_id, tree.target)
        # Also index just tree_id as alias for root
        self._node_index[tree_id] = (tree_id, tree.target)

        # Index all signals
        for signal in tree.signals:
            key = f"{tree_id}:{signal.id}"
            self._node_index[key] = (tree_id, signal)

    def get_node(self, ref: str) -> SignalNode | None:
        """Resolve ref like 'house_2026:sig_abc' or 'house_2026' (root).

        Args:
            ref: Reference string in format 'tree_id:node_id' or 'tree_id'

        Returns:
            The referenced SignalNode, or None if not found
        """
        # Check if already indexed
        if ref in self._node_index:
            return self._node_index[ref][1]

        # Try to load tree if not indexed
        if ":" in ref:
            tree_id = ref.split(":")[0]
        else:
            tree_id = ref

        tree = self.load_tree(tree_id)
        if tree is None:
            return None

        # Try again after loading
        return self._node_index.get(ref, (None, None))[1]

    def get_probability(self, ref: str) -> float | None:
        """Get computed probability for referenced node.

        For root refs (tree_id or tree_id:target), returns the tree's
        computed_probability. For other nodes, returns the node's base_rate.

        Args:
            ref: Reference string

        Returns:
            Probability value, or None if not found/computed
        """
        node = self.get_node(ref)
        if node is None:
            return None

        # Determine tree_id
        if ":" in ref:
            tree_id = ref.split(":")[0]
        else:
            tree_id = ref

        # For root node, return computed_probability
        tree = self._trees.get(tree_id)
        if tree and node.id == tree.target.id:
            return tree.computed_probability

        # For other nodes, return base_rate
        return node.base_rate

    def search_similar(
        self, text: str, limit: int = 5
    ) -> list[tuple[str, SignalNode, float]]:
        """Find semantically similar existing signals using substring matching.

        This is a simple text-based search. For MVP, uses case-insensitive
        substring matching and word overlap. Can be upgraded to embeddings later.

        Args:
            text: Signal text to find matches for
            limit: Maximum number of results

        Returns:
            List of (tree_id, node, similarity_score) tuples, sorted by score desc
        """
        text_lower = text.lower()
        text_words = set(re.findall(r"\w+", text_lower))

        results: list[tuple[str, SignalNode, float]] = []

        for ref, (tree_id, node) in self._node_index.items():
            # Skip tree_id-only aliases (they duplicate tree_id:node_id entries)
            if ":" not in ref:
                continue

            node_text_lower = node.text.lower()
            node_words = set(re.findall(r"\w+", node_text_lower))

            # Compute simple similarity score
            score = 0.0

            # Substring match bonus
            if text_lower in node_text_lower or node_text_lower in text_lower:
                score += 0.5

            # Word overlap (Jaccard-like)
            if text_words and node_words:
                intersection = len(text_words & node_words)
                union = len(text_words | node_words)
                score += 0.5 * (intersection / union)

            if score > 0.1:  # Minimum threshold
                results.append((tree_id, node, score))

        # Sort by score descending, take top N
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def list_trees(self) -> list[str]:
        """List all available tree IDs in the results directory.

        Returns:
            List of tree_id strings (directory names that contain tree_*.json)
        """
        if not self.results_dir.exists():
            return []

        tree_ids = []
        for subdir in self.results_dir.iterdir():
            if subdir.is_dir():
                tree_files = list(subdir.glob("tree_*.json"))
                if tree_files:
                    tree_ids.append(subdir.name)
        return sorted(tree_ids)

    def get_tree(self, tree_id: str) -> SignalTree | None:
        """Get a tree by ID (load if not cached).

        Args:
            tree_id: The tree identifier

        Returns:
            The SignalTree, or None if not found
        """
        if tree_id in self._trees:
            return self._trees[tree_id]
        return self.load_tree(tree_id)


def parse_ref(ref: str) -> tuple[str, str | None]:
    """Parse a reference string into tree_id and node_id.

    Args:
        ref: Reference string like 'house_2026:sig_abc' or 'house_2026'

    Returns:
        Tuple of (tree_id, node_id) where node_id is None for root refs
    """
    if ":" in ref:
        parts = ref.split(":", 1)
        return parts[0], parts[1]
    return ref, None


def make_ref(tree_id: str, node_id: str | None = None) -> str:
    """Create a reference string from tree_id and optional node_id.

    Args:
        tree_id: The tree identifier
        node_id: Optional node identifier (omit for root)

    Returns:
        Reference string like 'house_2026:sig_abc' or 'house_2026'
    """
    if node_id:
        return f"{tree_id}:{node_id}"
    return tree_id
