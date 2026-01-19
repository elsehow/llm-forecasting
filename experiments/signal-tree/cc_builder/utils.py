"""Utility functions for CC-driven tree building."""

from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalNode, SignalTree
from shared.rollup import rollup_tree, analyze_tree


def create_signal(
    text: str,
    parent_id: str,
    resolution_date: date | str,
    base_rate: float,
    rho: float,
    rho_reasoning: str = "",
    depth: int = 1,
    is_leaf: bool = True,
) -> SignalNode:
    """Create a signal node with auto-generated ID."""
    if isinstance(resolution_date, str):
        resolution_date = date.fromisoformat(resolution_date)

    return SignalNode(
        id=f"sig_{uuid.uuid4().hex[:8]}",
        text=text,
        resolution_date=resolution_date,
        base_rate=base_rate,
        probability_source="manual",
        parent_id=parent_id,
        rho=rho,
        rho_reasoning=rho_reasoning,
        depth=depth,
        is_leaf=is_leaf,
    )


def create_target(
    question: str,
    target_id: str = "target",
) -> SignalNode:
    """Create a target (root) node."""
    return SignalNode(
        id=target_id,
        text=question,
        depth=0,
        is_leaf=False,
    )


def build_tree(
    target: SignalNode,
    signals: list[SignalNode],
) -> SignalTree:
    """Build a tree from target and flat signal list.

    Automatically:
    - Links children to parents
    - Computes max_depth and leaf_count
    """
    # Build parent -> children mapping
    children_map: dict[str, list[SignalNode]] = {}
    for signal in signals:
        if signal.parent_id not in children_map:
            children_map[signal.parent_id] = []
        children_map[signal.parent_id].append(signal)

    # Attach children to nodes
    target.children = children_map.get(target.id, [])
    for signal in signals:
        signal.children = children_map.get(signal.id, [])
        signal.is_leaf = len(signal.children) == 0

    # Compute metadata
    max_depth = max((s.depth for s in signals), default=0)
    leaf_count = sum(1 for s in signals if s.is_leaf)

    return SignalTree(
        target=target,
        signals=signals,
        max_depth=max_depth,
        leaf_count=leaf_count,
    )


def save_tree(
    tree: SignalTree,
    target_slug: str,
    suffix: str = "cc",
    results_dir: Path | None = None,
) -> Path:
    """Save tree to JSON with consistent naming."""
    results_dir = results_dir or Path(__file__).parent.parent / "results"
    output_dir = results_dir / target_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = date.today().strftime("%Y%m%d")
    output_file = output_dir / f"tree_{timestamp}_{suffix}.json"

    # Serialize with analysis
    tree_dict = tree.model_dump(mode="json")

    with open(output_file, "w") as f:
        json.dump(tree_dict, f, indent=2, default=str)

    return output_file


def print_tree_summary(tree: SignalTree, target_prior: float = 0.5) -> dict:
    """Analyze and print tree summary."""
    analysis = analyze_tree(tree, target_prior)

    print(f"\nTarget: {tree.target.text}")
    print(f"Total signals: {len(tree.signals)}")
    print(f"Leaves: {tree.leaf_count}")
    print(f"Max depth: {tree.max_depth}")
    print(f"\nComputed probability: {analysis['computed_probability']:.1%}")
    print(f"(Prior was {target_prior:.0%})")

    print(f"\nEvidence breakdown:")
    print(f"  Positive: {analysis['evidence_breakdown']['positive']:+.4f}")
    print(f"  Negative: {analysis['evidence_breakdown']['negative']:+.4f}")
    print(f"  Net: {analysis['evidence_breakdown']['net']:+.4f}")

    print(f"\nTop contributors:")
    for c in analysis["top_contributors"][:5]:
        sign = "+" if c["evidence"] >= 0 else ""
        print(f"  {sign}{c['evidence']:.4f} | rho={c['rho']:+.2f} | p={c['base_rate']:.0%} | {c['text'][:50]}...")

    return analysis
