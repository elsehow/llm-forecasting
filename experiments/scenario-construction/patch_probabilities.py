#!/usr/bin/env python3
"""
Patch missing market probabilities into scenario results.

Reads a results JSON file, fetches current probabilities for signals
that are missing base_rate, and writes the updated file.

Usage:
    uv run python experiments/scenario-construction/patch_probabilities.py results/us_gdp_2029/dual_v7_*.json
"""

import asyncio
import json
import sys
from pathlib import Path

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent))
from shared.signals import fetch_missing_probabilities


async def patch_file(filepath: str) -> None:
    """Patch probabilities in a single results file."""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    print(f"\nPatching: {path.name}")
    print("=" * 60)

    # Load existing results
    with open(path) as f:
        data = json.load(f)

    signals = data.get("signals", [])
    if not signals:
        print("  No signals found in file")
        return

    # Count missing before
    missing_before = sum(1 for s in signals if s.get("base_rate") is None)
    print(f"  Signals missing base_rate: {missing_before}/{len(signals)}")

    if missing_before == 0:
        print("  Nothing to patch!")
        return

    # Fetch missing probabilities
    print("\n  Fetching from market APIs...")
    await fetch_missing_probabilities(signals, verbose=True)

    # Count missing after
    missing_after = sum(1 for s in signals if s.get("base_rate") is None)
    patched = missing_before - missing_after
    print(f"\n  Patched {patched} signals, {missing_after} still missing")

    if patched > 0:
        # Write updated file
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python patch_probabilities.py <results_file.json> [...]")
        print("\nExample:")
        print("  uv run python experiments/scenario-construction/patch_probabilities.py \\")
        print("    results/us_gdp_2029/dual_v7_20260116_131942.json")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        await patch_file(filepath)


if __name__ == "__main__":
    asyncio.run(main())
