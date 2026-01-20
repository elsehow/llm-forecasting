"""CC Builder - Dual-approach signal tree building.

Three-phase approach:
1. Structure: Generate logical decomposition (necessity, exclusivity, causal)
2. Markets: Discover related prediction markets via semantic search
3. Reconcile: Map markets onto structure, build signal tree

Main entry point is `build_tree_dual()`.
"""

from .dual_builder import (
    build_tree_dual,
    save_tree,
    print_tree_summary,
    validate_market_price,
)
from .structure import (
    generate_logical_structure,
    LogicalStructure,
    NecessityConstraint,
    ExclusivityConstraint,
    CausalPathway,
)
from .markets import (
    discover_markets,
    MarketSignal,
    MarketSemanticSearcher,
    get_market_searcher,
    check_market_price,
    refresh_market_data,
    get_market_data_stats,
    extract_entities,
    generate_competing_outcomes,
)
from .reconcile import (
    reconcile,
    identify_uncertain_signals,
    estimate_base_rate,
    estimate_rho,
)

__all__ = [
    # Main entry point
    "build_tree_dual",
    "save_tree",
    "print_tree_summary",
    "validate_market_price",
    # Structure (Phase 1)
    "generate_logical_structure",
    "LogicalStructure",
    "NecessityConstraint",
    "ExclusivityConstraint",
    "CausalPathway",
    # Markets (Phase 2)
    "discover_markets",
    "MarketSignal",
    "MarketSemanticSearcher",
    "get_market_searcher",
    "check_market_price",
    "refresh_market_data",
    "get_market_data_stats",
    "extract_entities",
    "generate_competing_outcomes",
    # Reconcile (Phase 3)
    "reconcile",
    "identify_uncertain_signals",
    "estimate_base_rate",
    "estimate_rho",
]
