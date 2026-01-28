"""Dual-approach signal tree builder.

Orchestrates the three phases:
1. Top-down: Generate logical structure (necessity, exclusivity, causal pathways)
2. Bottom-up: Discover market signals via semantic search
3. Reconciliation: Map markets onto structure, recurse on uncertain nodes
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalNode, SignalTree
from shared.rollup import rollup_tree, analyze_tree
from shared.registry import TreeRegistry

from .structure import generate_logical_structure, LogicalStructure, structure_to_dict
from .markets import discover_markets, MarketSignal, refresh_market_data, check_market_price, get_market_searcher
from .reconcile import reconcile, identify_uncertain_signals

logger = logging.getLogger(__name__)


async def build_tree_dual(
    target: str,
    target_id: str = "target",
    context: str | None = None,
    max_depth: int = 3,
    min_resolution_days: int = 14,
    target_prior: float = 0.5,
    db_path: str | Path = "forecastbench.db",
    registry: TreeRegistry | None = None,
    auto_refresh_markets: bool = True,
    refresh_platforms: list[str] | None = None,
    refresh_min_liquidity: float = 5000,
    refresh_limit: int = 2000,
) -> SignalTree:
    """Build signal tree using dual approach.

    Phase 1 (logical structure) and Phase 2 (market discovery) run in parallel.
    Phase 3 reconciles them into a signal tree.
    Recursively decomposes uncertain nodes up to max_depth.

    Args:
        target: The target question to decompose
        target_id: ID for the target node
        context: Optional background context
        max_depth: Maximum tree depth
        min_resolution_days: Stop recursion for signals resolving within this many days
        target_prior: Prior probability for the target
        db_path: Path to market database
        registry: Optional TreeRegistry for cross-tree refs
        auto_refresh_markets: Whether to refresh market data before building (default True)
        refresh_platforms: Platforms to refresh (default: ["polymarket", "metaculus"])
        refresh_min_liquidity: Minimum liquidity for market refresh (default: 5000)
        refresh_limit: Maximum markets to fetch per platform (default: 2000)

    Returns:
        SignalTree with computed probabilities
    """
    logger.info(f"Building tree for: {target[:60]}...")
    logger.info(f"Max depth: {max_depth}, Min resolution days: {min_resolution_days}")

    # Auto-refresh market data before building
    if auto_refresh_markets:
        platforms = refresh_platforms or ["polymarket", "metaculus"]
        logger.info(f"Refreshing market data from {platforms}...")
        counts = await refresh_market_data(
            platforms=platforms,
            min_liquidity=refresh_min_liquidity,
            limit=refresh_limit,
            db_path=db_path,
        )
        logger.info(f"Refreshed markets: {counts}")

        # Rebuild embedding cache after refresh
        searcher = get_market_searcher(db_path)
        logger.info("Rebuilding market embedding cache...")
        cache_count = searcher.build_cache()
        logger.info(f"Cached {cache_count} market embeddings")
    else:
        searcher = get_market_searcher(db_path)

    # Check if a direct market exists for the target question
    # If so, just return that market as the answer - no need to build a tree
    logger.info("Checking for direct market match on target...")
    from .reconcile import find_best_match_with_rerank

    target_candidates = searcher.search(target, top_k=5, search_type="direct")
    target_candidates = [c for c in target_candidates if c.current_probability is not None]

    if target_candidates:
        best_match, confidence = await find_best_match_with_rerank(
            target, target_candidates, min_similarity=0.3, min_confidence=0.7
        )

        if best_match and confidence >= 0.7:
            logger.info(f"Direct market match found: {best_match.title}")
            logger.info(f"Market price: {best_match.current_probability:.1%}")
            logger.info(f"Match confidence: {confidence:.0%}")
            logger.info("Returning market as leaf node - no tree needed")

            # Create a simple tree with just the market
            target_node = SignalNode(
                id=target_id,
                text=best_match.title,
                depth=0,
                is_leaf=True,
                base_rate=best_match.current_probability,
                probability_source="market",
                market_price=best_match.current_probability,
                market_url=best_match.url,
                market_platform=best_match.platform,
                market_match_confidence=confidence,
                base_rate_sources=[f"market:{best_match.platform}"],
            )

            return SignalTree(
                target=target_node,
                signals=[],
                max_depth=0,
                leaf_count=1,
                computed_probability=best_match.current_probability,
            )

    logger.info("No direct market match - building signal tree...")

    # Phase 1 & 2 run in parallel
    logger.info("Phase 1 & 2: Generating structure and discovering markets...")
    structure, markets = await asyncio.gather(
        generate_logical_structure(target, context),
        discover_markets(target, db_path=db_path),
    )

    logger.info(f"Found {len(structure.necessity_constraints)} necessity constraints")
    logger.info(f"Found {len(structure.exclusivity_constraints)} exclusivity constraints")
    logger.info(f"Found {len(structure.causal_pathways)} causal pathways")
    logger.info(f"Discovered {len(markets)} markets")

    # Phase 3: Reconcile
    logger.info("Phase 3: Reconciling structure with markets...")
    tree = await reconcile(
        structure,
        markets,
        target_id=target_id,
        parent_prior=target_prior,
        db_path=str(db_path),
    )

    logger.info(f"Initial tree has {len(tree.signals)} signals")

    # Recursive decomposition on uncertain nodes
    if max_depth > 1:
        uncertain = identify_uncertain_signals(
            tree,
            min_resolution_days=min_resolution_days,
        )
        logger.info(f"Found {len(uncertain)} uncertain signals for potential decomposition")

        for signal in uncertain:
            logger.info(f"Decomposing: {signal.text[:50]}...")

            # Build subtree for this signal
            # NOTE: auto_refresh_markets=False for recursive calls since parent already refreshed
            child_tree = await build_tree_dual(
                target=signal.text,
                target_id=signal.id,
                context=f"This signal is part of: {target}",
                max_depth=max_depth - 1,
                min_resolution_days=min_resolution_days,
                target_prior=signal.base_rate or 0.5,
                db_path=db_path,
                registry=registry,
                auto_refresh_markets=False,
            )

            # Integrate subtree children into main tree
            if child_tree.signals:
                # Update signal to be non-leaf
                signal.is_leaf = False
                signal.children = child_tree.target.children

                # Reparent child signals
                for child_signal in child_tree.signals:
                    child_signal.depth = signal.depth + child_signal.depth
                    # Update parent_id for direct children
                    if child_signal.parent_id == child_tree.target.id:
                        child_signal.parent_id = signal.id

                # Add to tree
                tree.signals.extend(child_tree.signals)

        # Update tree metadata
        tree.max_depth = max((s.depth for s in tree.signals), default=0)
        tree.leaf_count = sum(1 for s in tree.signals if s.is_leaf)

    # Compute probabilities
    logger.info("Computing rollup probabilities...")
    rollup_tree(tree, target_prior=target_prior, registry=registry)
    logger.info(f"Computed probability: {tree.computed_probability:.1%}")

    return tree


def save_tree(
    tree: SignalTree,
    target_slug: str,
    suffix: str = "dual",
    results_dir: Path | None = None,
    include_analysis: bool = True,
    target_prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> Path:
    """Save tree to JSON with consistent naming.

    Args:
        tree: The signal tree to save
        target_slug: Slug for the target (used for directory name)
        suffix: Suffix for the filename (default "dual")
        results_dir: Directory to save results (default: ../results)
        include_analysis: Whether to include full analysis in output
        target_prior: Prior probability for analysis
        registry: Optional TreeRegistry for analysis

    Returns:
        Path to the saved JSON file
    """
    results_dir = results_dir or Path(__file__).parent.parent / "results"
    output_dir = results_dir / target_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = date.today().strftime("%Y%m%d")
    output_file = output_dir / f"tree_{timestamp}_{suffix}.json"

    # Serialize tree
    tree_dict = tree.model_dump(mode="json")

    # Add analysis if requested
    if include_analysis:
        analysis = analyze_tree(tree, target_prior=target_prior, registry=registry)
        tree_dict["analysis"] = {
            "computed_probability": analysis["computed_probability"],
            "evidence_breakdown": analysis["evidence_breakdown"],
            "top_contributors": [
                {
                    "signal_id": c["signal_id"],
                    "text": c["text"],
                    "evidence": c["evidence"],
                    "base_rate": c["base_rate"],
                    "relationship_type": c["relationship_type"],
                }
                for c in analysis["top_contributors"][:5]
            ],
        }

    with open(output_file, "w") as f:
        json.dump(tree_dict, f, indent=2, default=str)

    logger.info(f"Saved tree to: {output_file}")
    return output_file


def print_tree_summary(
    tree: SignalTree,
    target_prior: float = 0.5,
    registry: TreeRegistry | None = None,
) -> dict:
    """Analyze and print tree summary.

    Args:
        tree: The signal tree to analyze
        target_prior: Prior probability for target
        registry: Optional TreeRegistry for resolving cross-tree refs

    Returns:
        Analysis dict
    """
    analysis = analyze_tree(tree, target_prior, registry)

    print(f"\nTarget: {tree.target.text}")
    print(f"Total signals: {len(tree.signals)}")
    print(f"Leaves: {tree.leaf_count}")
    print(f"Max depth: {tree.max_depth}")

    # Count by relationship type
    by_type: dict[str, int] = {}
    for signal in tree.signals:
        rt = signal.relationship_type
        by_type[rt] = by_type.get(rt, 0) + 1
    print(f"\nBy relationship type:")
    for rt, count in sorted(by_type.items()):
        print(f"  {rt}: {count}")

    # Count refs
    ref_count = sum(1 for s in tree.signals if s.ref)
    if ref_count:
        print(f"Cross-tree refs: {ref_count}")

    print(f"\nComputed probability: {analysis['computed_probability']:.1%}")
    print(f"(Prior was {target_prior:.0%})")

    print(f"\nEvidence breakdown:")
    print(f"  Positive: {analysis['evidence_breakdown']['positive']:+.4f}")
    print(f"  Negative: {analysis['evidence_breakdown']['negative']:+.4f}")
    print(f"  Net: {analysis['evidence_breakdown']['net']:+.4f}")

    print(f"\nTop contributors:")
    for c in analysis["top_contributors"][:5]:
        sign = "+" if c["evidence"] >= 0 else ""
        ref_marker = f" [ref:{c['ref']}]" if c.get("ref") else ""
        rt_marker = f" [{c['relationship_type']}]" if c["relationship_type"] != "correlation" else ""
        print(
            f"  {sign}{c['evidence']:.4f} | p={c['base_rate']:.0%} | "
            f"{c['text'][:40]}...{rt_marker}{ref_marker}"
        )

    return analysis


async def validate_market_price(
    tree: SignalTree,
    db_path: str | Path = "forecastbench.db",
) -> dict[str, Any] | None:
    """Validate computed probability against prediction market.

    Args:
        tree: The signal tree to validate
        db_path: Path to market database

    Returns:
        Validation dict or None if no market match
    """
    if tree.computed_probability is None:
        return None

    result = await check_market_price(tree.target.text, db_path=db_path)
    if not result:
        return None

    gap_pp = (tree.computed_probability - result["market_price"]) * 100

    if abs(gap_pp) <= 5:
        status = "OK"
    elif abs(gap_pp) <= 15:
        status = "WARNING - gap >5pp"
    else:
        status = "REVIEW - gap >15pp"

    return {
        "platform": result["platform"],
        "matched_question": result["matched_question"],
        "market_price": result["market_price"],
        "computed_probability": tree.computed_probability,
        "gap_pp": round(gap_pp, 1),
        "status": status,
        "url": result["url"],
        "match_confidence": result["match_confidence"],
    }
