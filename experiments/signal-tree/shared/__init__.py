"""Signal tree shared modules."""

from .decomposition import build_signal_tree, generate_signals_for_node
from .rollup import analyze_tree, compute_signal_contribution, rollup_tree
from .tree import SignalNode, SignalTree, TreeGenerationConfig

__all__ = [
    "SignalNode",
    "SignalTree",
    "TreeGenerationConfig",
    "build_signal_tree",
    "generate_signals_for_node",
    "rollup_tree",
    "analyze_tree",
    "compute_signal_contribution",
]
