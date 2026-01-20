"""Signal tree data structures."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class SignalNode(BaseModel):
    """A node in the signal tree (either target or signal)."""

    id: str = Field(description="Unique identifier")
    text: str = Field(description="The question/signal text")
    resolution_date: date | None = Field(
        default=None, description="When this signal resolves"
    )

    # Probability data
    base_rate: float | None = Field(
        default=None, description="Current probability (from market or estimate)"
    )
    probability_source: Literal["polymarket", "metaculus", "llm", "manual", "market"] | None = (
        Field(default=None, description="Source of base_rate")
    )

    # Market data (for leaves with matched market prices)
    market_price: float | None = Field(
        default=None,
        description="Market probability (0-1) if matched to a prediction market",
    )
    market_url: str | None = Field(
        default=None,
        description="URL to the prediction market",
    )
    market_platform: str | None = Field(
        default=None,
        description="Platform name (polymarket, metaculus, etc.)",
    )
    market_match_confidence: float | None = Field(
        default=None,
        description="Confidence in the market match (0-1)",
    )

    # Tree structure
    parent_id: str | None = Field(
        default=None, description="ID of parent node (None for root)"
    )
    children: list[SignalNode] = Field(
        default_factory=list, description="Child signals"
    )

    # Relationship to parent
    relationship_type: Literal["correlation", "necessity", "sufficiency"] = Field(
        default="correlation",
        description=(
            "Type of relationship to parent: "
            "'correlation' (default) uses rho model; "
            "'necessity' means signal=NO implies parent=0 (e.g., must be nominated to win); "
            "'sufficiency' means signal=YES implies parent=1 (rare)"
        ),
    )
    rho: float | None = Field(
        default=None,
        description="Correlation with parent (-1 to +1). Positive = same direction. Used when relationship_type='correlation'.",
    )
    rho_reasoning: str | None = Field(
        default=None, description="Explanation of correlation"
    )

    # Computed fields (populated during rollup)
    p_parent_given_yes: float | None = Field(
        default=None, description="P(parent=YES | this=YES)"
    )
    p_parent_given_no: float | None = Field(
        default=None, description="P(parent=YES | this=NO)"
    )

    # Cross-tree references
    ref: str | None = Field(
        default=None,
        description="Reference to another tree's node: 'tree_id:node_id' or 'tree_id' (for root). When set, base_rate is pulled from referenced node at rollup time.",
    )

    # Metadata
    is_leaf: bool = Field(
        default=True, description="True if this is a leaf node (no children)"
    )
    depth: int = Field(default=0, description="Depth in tree (0 = root)")


class SignalTree(BaseModel):
    """A complete signal tree for a target question."""

    target: SignalNode = Field(description="Root node (the target question)")
    signals: list[SignalNode] = Field(
        default_factory=list, description="All signals (flat list for easy access)"
    )

    # Tree metadata
    max_depth: int = Field(default=0, description="Maximum depth of tree")
    leaf_count: int = Field(default=0, description="Number of leaf nodes")
    actionable_horizon_days: int = Field(
        default=365, description="Signals must resolve within this many days"
    )

    # Computed probability
    computed_probability: float | None = Field(
        default=None, description="P(target=YES) computed from leaf signals"
    )

    def get_leaves(self) -> list[SignalNode]:
        """Get all leaf nodes."""
        return [s for s in self.signals if s.is_leaf]

    def get_node(self, node_id: str) -> SignalNode | None:
        """Get a node by ID."""
        if self.target.id == node_id:
            return self.target
        for s in self.signals:
            if s.id == node_id:
                return s
        return None

    def get_children(self, node_id: str) -> list[SignalNode]:
        """Get children of a node."""
        return [s for s in self.signals if s.parent_id == node_id]

    def get_path_to_root(self, node_id: str) -> list[SignalNode]:
        """Get path from a node to the root."""
        path = []
        current = self.get_node(node_id)
        while current:
            path.append(current)
            if current.parent_id:
                current = self.get_node(current.parent_id)
            else:
                break
        return path


class TreeGenerationConfig(BaseModel):
    """Configuration for signal tree generation."""

    target_question: str = Field(description="The target question to decompose")
    target_id: str = Field(default="target", description="ID for target node")
    target_context: str | None = Field(
        default=None,
        description="Background context about the target (current status, market odds, etc.)",
    )

    # Decomposition termination settings
    # Primary rule: Stop when signal resolves within minimum_resolution_days
    minimum_resolution_days: int = Field(
        default=7,
        description="Stop decomposing when signal resolves within this many days",
    )
    # Safety valve: Stop if tree grows too large
    max_signals: int = Field(
        default=100,
        description="Maximum total signals in tree (safety valve)",
    )
    # Legacy: actionable_horizon_days used for signal generation prompt
    actionable_horizon_days: int = Field(
        default=365,
        description="Signals must resolve within this many days to be 'actionable'",
    )
    signals_per_node: int = Field(
        default=5, description="Number of signals to generate per parent"
    )

    # Model settings
    generation_model: str = Field(
        default="claude-sonnet-4-20250514", description="Model for signal generation"
    )
    rho_model: str = Field(
        default="anthropic/claude-3-haiku-20240307",
        description="Model for rho estimation",
    )

    # Market data
    include_market_signals: bool = Field(
        default=True, description="Include signals from prediction markets"
    )
    market_match_threshold: float = Field(
        default=0.6, description="Semantic similarity threshold for market matching"
    )
