#!/usr/bin/env python
"""Add market signals to an existing tree.

Usage:
    python scripts/add_market_signals.py results/my_target/tree.json
    python scripts/add_market_signals.py results/my_target/tree.json --refresh
    python scripts/add_market_signals.py results/my_target/tree.json --output tree_with_markets.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalTree, SignalNode
from shared.rollup import rollup_tree, analyze_tree, compute_node_gap
from cc_builder.utils import check_market_price, refresh_market_data, get_market_data_stats


def update_node_market(node: SignalNode, result: dict):
    """Update market fields on a node."""
    node.market_price = result["market_price"]
    node.market_platform = result["platform"]
    node.market_url = result["url"]
    node.market_match_confidence = result["match_confidence"]
    node.probability_source = "market"
    node.base_rate = result["market_price"]


def find_in_nested(parent: SignalNode, node_id: str) -> SignalNode | None:
    """Find a node in the nested children structure."""
    if parent.id == node_id:
        return parent
    for child in parent.children or []:
        found = find_in_nested(child, node_id)
        if found:
            return found
    return None


async def main():
    parser = argparse.ArgumentParser(description="Add market signals to a tree")
    parser.add_argument("tree_path", help="Path to tree JSON file")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh market data before processing",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path (default: replaces _cc with _cc_market in filename)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum match confidence (default: 0.5)",
    )
    parser.add_argument(
        "--target-prior",
        type=float,
        default=0.5,
        help="Target prior for rollup (default: 0.5)",
    )
    args = parser.parse_args()

    tree_path = Path(args.tree_path)
    if not tree_path.exists():
        print(f"Error: Tree file not found: {tree_path}")
        sys.exit(1)

    print("=" * 60)
    print("Adding Market Signals to Tree")
    print("=" * 60)

    # Step 0: Optionally refresh market data
    if args.refresh:
        print("\n0. Refreshing market data...")
        try:
            counts = await refresh_market_data(
                platforms=["polymarket"],
                min_liquidity=5000,
                limit=2000,
            )
            print(f"   Fetched {counts.get('polymarket', 0)} Polymarket markets")
        except Exception as e:
            print(f"   Warning: Failed to refresh: {e}")

    # Show current stats
    stats = await get_market_data_stats()
    print(f"\nMarket data: {stats['total_markets']} markets cached")
    for platform, count in stats["by_platform"].items():
        print(f"  - {platform}: {count}")

    # Load tree
    print(f"\nLoading tree from: {tree_path}")
    with open(tree_path) as f:
        tree_data = json.load(f)
    tree = SignalTree.model_validate(tree_data)
    print(f"  Target: {tree.target.text[:60]}...")
    print(f"  Signals: {len(tree.signals)}, Leaves: {tree.leaf_count}")

    # Check target market
    print("\n1. Checking market for target question...")
    target_market = await check_market_price(
        tree.target.text,
        min_confidence=args.min_confidence,
    )
    if target_market:
        print(f"   Found: {target_market['platform']} @ {target_market['market_price']:.0%}")
        print(f"   Confidence: {target_market['match_confidence']:.0%}")
        update_node_market(tree.target, target_market)
    else:
        print("   No market match found for target")

    # Check leaf markets
    print("\n2. Checking markets for leaf signals...")
    leaves = tree.get_leaves()
    market_matches = 0

    for leaf in leaves:
        result = await check_market_price(
            leaf.text,
            min_confidence=args.min_confidence,
        )
        if result:
            market_matches += 1
            print(f"   {leaf.text[:50]}...")
            print(f"      -> {result['platform']} @ {result['market_price']:.0%} (conf: {result['match_confidence']:.0%})")

            # Update in flat signals list
            update_node_market(leaf, result)

            # Also update in nested children structure
            nested_node = find_in_nested(tree.target, leaf.id)
            if nested_node and nested_node is not leaf:
                update_node_market(nested_node, result)

    print(f"\n   Found {market_matches} market matches out of {len(leaves)} leaves")

    # Rollup
    print("\n3. Running rollup...")
    computed_prob = rollup_tree(tree, target_prior=args.target_prior)
    print(f"   Computed probability: {computed_prob:.1%}")

    if tree.target.market_price is not None:
        gap = compute_node_gap(tree.target, computed_prob)
        print(f"   Market price: {tree.target.market_price:.1%}")
        print(f"   Gap: {gap:+.1f}pp")
        if abs(gap) <= 5:
            print("   Status: OK")
        elif abs(gap) <= 15:
            print("   Status: WARNING")
        else:
            print("   Status: REVIEW")

    # Analysis
    print("\n4. Top contributors:")
    analysis = analyze_tree(tree, target_prior=args.target_prior)
    for i, contrib in enumerate(analysis["top_contributors"][:5]):
        sign = "+" if contrib["evidence"] >= 0 else ""
        print(f"   {i+1}. {sign}{contrib['evidence']:.4f} | rho={contrib['rho']:+.2f} | {contrib['text'][:40]}...")

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: replace _cc with _cc_market or append _market
        stem = tree_path.stem
        if "_cc" in stem and "_market" not in stem:
            new_stem = stem.replace("_cc", "_cc_market")
        else:
            new_stem = f"{stem}_market"
        output_path = tree_path.parent / f"{new_stem}.json"

    print(f"\n5. Saving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(tree.model_dump(mode="json"), f, indent=2)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
