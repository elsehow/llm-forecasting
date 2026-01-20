#!/usr/bin/env python
"""Find markets for entities mentioned in a signal tree.

Usage:
    python scripts/find_entity_markets.py results/my_target/tree.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalTree
from shared.entity_markets import find_entity_markets, print_entity_report

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "forecastbench.db"


async def main():
    parser = argparse.ArgumentParser(
        description="Find markets for entities in a signal tree"
    )
    parser.add_argument("tree_path", help="Path to tree JSON file")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to market database",
    )
    args = parser.parse_args()

    tree_path = Path(args.tree_path)
    if not tree_path.exists():
        print(f"Error: Tree file not found: {tree_path}")
        sys.exit(1)

    # Load tree
    with open(tree_path) as f:
        tree = SignalTree.model_validate(json.load(f))

    print(f"Tree: {tree.target.text[:60]}...")

    # Find entity markets
    entities = await find_entity_markets(tree, args.db_path)

    # Report
    print_entity_report(entities, tree)


if __name__ == "__main__":
    asyncio.run(main())
