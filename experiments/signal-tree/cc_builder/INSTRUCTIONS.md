# Dual-Approach Signal Tree Builder

This module builds signal trees using a three-phase dual approach:

1. **Top-down (Structure)**: Generate logical decomposition with necessity, exclusivity, and causal constraints
2. **Bottom-up (Markets)**: Discover related prediction markets via semantic search
3. **Reconciliation**: Map markets onto structure, recurse on uncertain nodes

## Quick Start

```python
from cc_builder import build_tree_dual, save_tree, print_tree_summary

import asyncio

async def main():
    # Build tree
    tree = await build_tree_dual(
        target="Will One Battle win Best Picture at the 2026 Oscars?",
        max_depth=2,
        min_resolution_days=14,
    )

    # Analyze
    print_tree_summary(tree)

    # Save
    save_tree(tree, "one_battle_best_picture")

asyncio.run(main())
```

## Three Phases

### Phase 1: Logical Structure (`structure.py`)

Uses LLM to identify three types of constraints:

1. **Necessity**: What MUST happen for target to be true
   - Example: "Must be nominated to win"
   - Creates `relationship_type="necessity"` signals

2. **Exclusivity**: Competitors that preclude target
   - Example: "If Sinners wins, One Battle loses"
   - Creates `relationship_type="exclusivity"` signals

3. **Causal Pathways**: Earlier events that inform outcome
   - Example: "Golden Globe win → momentum → Oscar win"
   - Creates `relationship_type="correlation"` signals

```python
from cc_builder import generate_logical_structure

structure = await generate_logical_structure(
    "Will X win?",
    context="X is currently leading"  # Optional
)

print(structure.necessity_constraints)
print(structure.exclusivity_constraints)
print(structure.causal_pathways)
```

### Phase 2: Market Discovery (`markets.py`)

Finds related prediction markets via semantic search:

1. **Direct search**: Search for target question
2. **Entity-based search**: Extract entities, search for each
3. **Competing outcome search**: Generate competitors, search for each

```python
from cc_builder import discover_markets, refresh_market_data

# Refresh market cache first
await refresh_market_data()

# Discover related markets
markets = await discover_markets("Will X win Best Picture?")
for m in markets:
    print(f"{m.title}: {m.current_probability:.0%} ({m.search_type})")
```

### Phase 3: Reconciliation (`reconcile.py`)

Maps markets onto structure to build signal tree:

- Matches markets to constraints by semantic similarity
- Estimates base_rate for unmatched signals via LLM
- Sets appropriate relationship types and parameters

```python
from cc_builder import reconcile

tree = await reconcile(structure, markets)
```

## Relationship Types

| Type | Description | Rollup Behavior |
|------|-------------|-----------------|
| `correlation` | Statistical relationship via rho | Evidence = (base_rate - 0.5) × spread |
| `necessity` | Must happen for parent | Caps parent at signal's base_rate |
| `sufficiency` | If true, parent is true | Floors parent at signal's base_rate |
| `exclusivity` | Competes with parent | Uses p_target_given_yes/no |

### Exclusivity Example

```python
from shared.tree import SignalNode

# Competitor signal: if Sinners wins, One Battle loses
competitor = SignalNode(
    id="sinners",
    text="Will Sinners win Best Picture?",
    base_rate=0.15,
    relationship_type="exclusivity",
    p_target_given_yes=0.01,   # If Sinners wins, target ~0%
    p_target_given_no=0.55,    # If Sinners doesn't win, slight boost
    parent_id="target",
    depth=1,
)
```

## Recursive Decomposition

The builder recursively decomposes uncertain signals:

```python
tree = await build_tree_dual(
    target="Will a Democrat win 2028?",
    max_depth=3,              # Max tree depth
    min_resolution_days=14,   # Stop for near-term signals
)
```

Decomposition criteria:
- Signal is a leaf node
- Base rate between 0.2 and 0.8 (uncertain)
- Resolution date beyond `min_resolution_days`

## Cross-Tree References

Use the registry for shared signals across trees:

```python
from shared.registry import TreeRegistry
from cc_builder import build_tree_dual

registry = TreeRegistry()
registry.load_tree("house_2026")
registry.load_tree("senate_2026")

tree = await build_tree_dual(
    "Will a Democrat win 2028?",
    registry=registry,  # Enables cross-tree refs
)
```

## Market Data Setup

Before building trees, refresh the market cache:

```bash
# Via script
python scripts/refresh_markets.py

# Or programmatically
from cc_builder import refresh_market_data
await refresh_market_data(platforms=["polymarket"])
```

## File Structure

```
cc_builder/
├── __init__.py         # Exports all public functions
├── dual_builder.py     # Main entry point
├── structure.py        # Phase 1: Logical structure
├── markets.py          # Phase 2: Market discovery
├── reconcile.py        # Phase 3: Reconciliation
└── INSTRUCTIONS.md     # This file
```

## Testing

```bash
# Run all tests
uv run pytest experiments/signal-tree/tests/ -v

# Run specific module tests
uv run pytest experiments/signal-tree/tests/test_structure.py -v
uv run pytest experiments/signal-tree/tests/test_reconcile.py -v
```

## Manual Tree Building

For CC-driven tree building (not using the automated pipeline):

```python
from shared.tree import SignalNode, SignalTree
from shared.rollup import rollup_tree, analyze_tree
from cc_builder import check_market_price

# Create target
target = SignalNode(
    id="target",
    text="Will X win Best Picture?",
    depth=0,
    is_leaf=False,
)

# Create signals manually
signals = []

# Necessity signal
nomination = SignalNode(
    id="nom",
    text="Will X be nominated?",
    base_rate=0.98,
    relationship_type="necessity",
    parent_id="target",
    depth=1,
    is_leaf=True,
)
signals.append(nomination)

# Exclusivity signal
competitor = SignalNode(
    id="comp",
    text="Will Y win?",
    base_rate=0.20,
    relationship_type="exclusivity",
    p_target_given_yes=0.01,
    p_target_given_no=0.55,
    parent_id="target",
    depth=1,
    is_leaf=True,
)
signals.append(competitor)

# Correlation signal with market
result = await check_market_price("Will X win Golden Globe?")
golden_globe = SignalNode(
    id="gg",
    text="Will X win Golden Globe?",
    base_rate=result["market_price"] if result else 0.5,
    market_price=result["market_price"] if result else None,
    market_url=result["url"] if result else None,
    rho=0.6,
    relationship_type="correlation",
    parent_id="target",
    depth=1,
    is_leaf=True,
)
signals.append(golden_globe)

# Build tree
target.children = signals
tree = SignalTree(
    target=target,
    signals=signals,
    max_depth=1,
    leaf_count=len(signals),
)

# Compute probabilities
rollup_tree(tree, target_prior=0.5)
print(f"Computed: {tree.computed_probability:.1%}")

# Analyze
analysis = analyze_tree(tree)
for c in analysis["top_contributors"][:5]:
    print(f"{c['evidence']:+.4f} | {c['text'][:40]}...")
```

## Design Principles

1. **Validate before trusting**: Structure generation and market matching need verification
2. **Explicit relationship types**: Makes reasoning transparent and debuggable
3. **Market prices as ground truth**: Use markets when available, estimate otherwise
4. **Recursive decomposition**: Drill down on uncertain signals
5. **Cross-tree composability**: Share signals across related forecasting questions
