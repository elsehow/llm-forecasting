"""Shared fixtures for signal-tree tests."""

from datetime import date, timedelta

import pytest

import sys
from pathlib import Path


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires LLM calls)",
    )

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalNode, SignalTree, TreeGenerationConfig


@pytest.fixture
def today() -> date:
    """Fixed date for reproducible tests."""
    return date(2026, 1, 19)


@pytest.fixture
def default_config(today: date) -> TreeGenerationConfig:
    """Default config for testing."""
    return TreeGenerationConfig(
        target_question="Will X happen?",
        target_id="target",
        minimum_resolution_days=7,
        max_signals=100,
        actionable_horizon_days=365,
        signals_per_node=5,
    )


@pytest.fixture
def leaf_node(today: date) -> SignalNode:
    """A simple leaf node."""
    return SignalNode(
        id="leaf_1",
        text="Will event A happen?",
        resolution_date=today + timedelta(days=30),
        base_rate=0.6,
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
        rho=0.4,
    )


@pytest.fixture
def short_horizon_signal(today: date) -> SignalNode:
    """A signal that resolves within minimum_resolution_days (should NOT decompose)."""
    return SignalNode(
        id="short_horizon",
        text="Will event B happen?",
        resolution_date=today + timedelta(days=3),  # 3 days < 7 days
        base_rate=0.7,
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
    )


@pytest.fixture
def long_horizon_signal(today: date) -> SignalNode:
    """A signal that resolves beyond minimum_resolution_days (should decompose)."""
    return SignalNode(
        id="long_horizon",
        text="Will event C happen?",
        resolution_date=today + timedelta(days=30),  # 30 days > 7 days
        base_rate=0.5,
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
    )


@pytest.fixture
def no_date_signal() -> SignalNode:
    """A signal with no resolution date (should decompose)."""
    return SignalNode(
        id="no_date",
        text="Will event D happen?",
        resolution_date=None,
        base_rate=0.5,
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
    )


@pytest.fixture
def sample_tree(today: date) -> SignalTree:
    """A simple tree for testing rollup."""
    root = SignalNode(
        id="target",
        text="Will X win the election?",
        depth=0,
        is_leaf=False,
        base_rate=0.5,
    )

    # Child 1: positive rho, p > 0.5 (should increase parent probability)
    child1 = SignalNode(
        id="child_1",
        text="Will X get endorsement from Y?",
        resolution_date=today + timedelta(days=5),
        base_rate=0.7,  # Likely to happen
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
        rho=0.5,  # Positive correlation
    )

    # Child 2: negative rho, p > 0.5 (should decrease parent probability)
    child2 = SignalNode(
        id="child_2",
        text="Will Z (competitor) win primary?",
        resolution_date=today + timedelta(days=5),
        base_rate=0.6,  # Likely
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
        rho=-0.4,  # Competition - negative correlation
    )

    # Child 3: zero rho (should not affect parent)
    child3 = SignalNode(
        id="child_3",
        text="Will it rain tomorrow?",
        resolution_date=today + timedelta(days=1),
        base_rate=0.8,
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
        rho=0.0,  # Independent
    )

    root.children = [child1, child2, child3]

    return SignalTree(
        target=root,
        signals=[child1, child2, child3],
        max_depth=1,
        leaf_count=3,
        actionable_horizon_days=365,
    )


@pytest.fixture
def deep_tree(today: date) -> SignalTree:
    """A deeper tree (depth 3) for testing recursive rollup."""
    root = SignalNode(
        id="target",
        text="Will the company succeed?",
        depth=0,
        is_leaf=False,
        base_rate=0.5,
    )

    # Level 1 node (internal)
    level1 = SignalNode(
        id="level1",
        text="Will they secure funding?",
        resolution_date=today + timedelta(days=60),
        base_rate=0.6,
        probability_source="llm",
        parent_id="target",
        is_leaf=False,
        depth=1,
        rho=0.6,
    )

    # Level 2 nodes (leaves)
    level2a = SignalNode(
        id="level2a",
        text="Will VC firm A show interest?",
        resolution_date=today + timedelta(days=5),
        base_rate=0.7,
        probability_source="llm",
        parent_id="level1",
        is_leaf=True,
        depth=2,
        rho=0.5,
    )

    level2b = SignalNode(
        id="level2b",
        text="Will VC firm B show interest?",
        resolution_date=today + timedelta(days=5),
        base_rate=0.4,
        probability_source="llm",
        parent_id="level1",
        is_leaf=True,
        depth=2,
        rho=0.4,
    )

    # Another level 1 node (leaf)
    level1b = SignalNode(
        id="level1b",
        text="Will product launch on time?",
        resolution_date=today + timedelta(days=30),
        base_rate=0.65,
        probability_source="llm",
        parent_id="target",
        is_leaf=True,
        depth=1,
        rho=0.3,
    )

    level1.children = [level2a, level2b]
    root.children = [level1, level1b]

    return SignalTree(
        target=root,
        signals=[level1, level2a, level2b, level1b],
        max_depth=2,
        leaf_count=3,
        actionable_horizon_days=365,
    )
