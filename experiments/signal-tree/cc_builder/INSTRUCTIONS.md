# Signal Tree Builder Instructions

You (Claude Code) are building a signal tree to decompose a forecasting question into actionable signals.

## Key Insight

The main advantage of CC-driven tree building over the automated pipeline is **validation**:
- Web search to verify facts before generating signals
- Catch domain constraint violations (e.g., wrong eligibility years)
- Recursive context gathering for each node

## Decomposition Rule

**A signal should be decomposed into child signals if ALL of these are true:**

1. **Far resolution**: Signal resolves more than 7 days from today
2. **Uncertain**: Signal has base_rate between 0.2 and 0.8 (not already near-certain)
3. **Decomposable**: There exist earlier-resolving events that would inform this signal

**A signal should remain a leaf if ANY of these are true:**

1. **Near-term**: Resolves within 7 days (not enough time for sub-signals to help)
2. **Near-certain**: base_rate < 0.1 or > 0.9 (already confident, decomposition adds noise)
3. **Atomic**: No meaningful earlier-resolving events exist (e.g., coin flip, announcement)

**When decomposing a parent signal:**

1. Gather context specifically for that signal (web search if needed)
2. Generate 3-5 child signals that resolve BEFORE the parent
3. Validate each child signal (no constraint violations)
4. Apply this rule recursively to each child

## Example Session

See `results/one_battle_best_picture/tree_20260119_cc_validated.json` for a tree built
with this process. Key improvements over automated pipeline:
- Validated that competitors (Sinners, Hamnet, etc.) are actually in same Oscar cycle
- Used real precursor dates (DGA Feb 7, PGA Feb 28, etc.)
- Applied correct rho signs based on entity identification

## Goal

Given a target question, recursively decompose it into signals that:
1. Resolve BEFORE the target (provide early information)
2. Are INFORMATIVE (knowing their outcome updates belief about target)
3. Are VALID (don't violate domain constraints)

## Process

### Phase 1: Context Gathering

Before generating signals, gather context about the target:

1. **Web search** for current status, key players, timeline
2. **Identify** the domain and its constraints (eligibility rules, competition structure, etc.)
3. **Document** what you learn in a context block

### Phase 2: Signal Generation

For each node that needs decomposition:

1. Generate 3-5 candidate signals that would inform the parent
2. For each signal, consider:
   - Does it resolve before the parent?
   - Is it actually informative (non-trivial correlation)?
   - Does it violate any domain constraints?

### Phase 3: Validation

For each candidate signal:

1. **Check domain constraints** via web search if needed
   - Are the entities eligible for the same competition?
   - Is the timeline correct?
   - Are there factual errors?

2. **Estimate rho** using the entity identification framework:
   - Identify entity in target, entity in signal
   - Same entity? → momentum/necessity (positive)
   - Different entity, same prize? → competition (negative)
   - Different entity, helps target? → indirect help (positive)
   - No connection? → independent (zero)

3. **Estimate base_rate** (probability signal resolves YES)

### Phase 4: Recursion

For signals that:
- Resolve more than 7 days from now
- Would benefit from further decomposition

Repeat Phases 1-3 with the signal as the new target.

### Phase 5: Rollup

Use the utility functions to compute:
- Evidence contribution from each leaf
- Aggregate probability for target
- Contribution breakdown (positive vs negative evidence)

## Utilities Available

```python
from shared.tree import SignalNode, SignalTree, TreeGenerationConfig
from shared.rollup import rollup_tree, analyze_tree, compute_signal_contribution
```

## Output Format

Save the tree as JSON with consistent naming:
```
results/{target_slug}/tree_{YYYYMMDD}_cc.json
```

## Key Principles

1. **Validate before trusting** - Don't assume generated signals are factually correct
2. **Context is recursive** - Each node may need its own context gathering
3. **Competition is negative** - Different entities competing for same prize → negative rho
4. **Same entity is positive** - Momentum, prerequisites, quality signals → positive rho
5. **When in doubt, search** - Use web search to verify claims

## Cross-Tree Signal References

When building multiple related trees (e.g., House 2026, Senate 2026, President 2028),
signals often overlap. Rather than duplicating signals, use **references** to share them.

### Reference Format

References use the format `tree_id:node_id` or just `tree_id` for the root:
- `house_2026:sig_economy` - References a specific signal in another tree
- `house_2026` - References another tree's root (computed probability)

### When to Use References

Use a reference signal when:
1. The same underlying question appears in multiple trees (e.g., "Will there be a recession?")
2. One tree's outcome is an input to another (e.g., House 2026 result → President 2028)
3. You want consistent probability updates when shared signals change

### Creating Reference Signals

```python
from shared.registry import TreeRegistry
from cc_builder.utils import create_ref_signal

# Load existing trees
registry = TreeRegistry()
registry.load_tree("house_2026")

# Create a reference signal
ref_signal = create_ref_signal(
    ref="house_2026:sig_economy",  # Reference to existing signal
    parent_id="target",             # Parent in THIS tree
    rho=-0.35,                      # Correlation in THIS tree's context
    rho_reasoning="Economic conditions affect Senate races similarly to House",
    registry=registry,
    depth=1,
)
```

### Reference Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  House 2026     │     │  Senate 2026    │
│  ├─ sig_economy │◄────┼─ [REF] economy  │
│  ├─ sig_approval│◄────┼─ [REF] approval │
│  └─ house-only  │     │  └─ senate-only │
└─────────────────┘     └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │  President 2028 │
           │  ├─ [REF] house_2026 (root)
           │  ├─ [REF] senate_2026 (root)
           │  └─ candidate-specific
           └─────────────────┘
```

### How References Work

When `rollup_tree()` encounters a reference signal:
1. It looks up the referenced node in the `TreeRegistry`
2. Uses the referenced node's probability as the `base_rate`
3. Applies THIS tree's `rho` to compute evidence contribution

This means:
- Reference signals are **always leaves** (not decomposed further)
- The `base_rate` is pulled from the referenced tree at rollup time
- Each tree can have its own `rho` for the same shared signal

### Using the Registry

```python
from shared.registry import TreeRegistry
from shared.rollup import rollup_tree, analyze_tree

# Create registry and load trees
registry = TreeRegistry()
registry.load_tree("house_2026")
registry.load_tree("senate_2026")

# Build a tree with references
# ... (tree building code) ...

# Rollup with registry to resolve refs
prob = rollup_tree(tree, target_prior=0.5, registry=registry)
analysis = analyze_tree(tree, target_prior=0.5, registry=registry)
```

### Finding Similar Existing Signals

Before creating a new signal, check if a similar one already exists:

```python
# Search for similar signals across all loaded trees
candidates = registry.search_similar("economic recession 2026", limit=5)
for tree_id, node, score in candidates:
    print(f"{score:.2f} [{tree_id}:{node.id}] {node.text}")
```

### Utilities Available (Updated)

```python
from shared.tree import SignalNode, SignalTree, TreeGenerationConfig
from shared.registry import TreeRegistry, parse_ref, make_ref
from shared.rollup import rollup_tree, analyze_tree, compute_signal_contribution, compute_node_gap
from cc_builder.utils import (
    create_signal,
    create_signal_with_market,  # NEW: async, auto-fetches market prices
    create_ref_signal,
    create_target,
    build_tree,
    check_market_price,
)
```

## Market Signals

### Design Principle

Signals can be backed by market prices for ground truth:

| Node Type | Primary Probability | Market Role |
|-----------|---------------------|-------------|
| **Leaf** | `market_price` if available, else `base_rate` | Source of truth |
| **Parent** | `computed_probability` from children | Comparison (show gap) |
| **Target** | `computed_probability` from tree | Comparison (show gap) |

### Creating Market-Backed Signals

Use `create_signal_with_market()` to automatically check prediction markets:

```python
# Async function that checks markets
signal = await create_signal_with_market(
    text="Will One Battle win Best Picture at 2026 Oscars?",
    parent_id="target",
    resolution_date="2026-03-03",
    rho=0.9,
    rho_reasoning="Same entity, same outcome",
)

# If market found, these fields are populated:
print(f"Market price: {signal.market_price}")      # 0.81
print(f"Platform: {signal.market_platform}")        # "polymarket"
print(f"URL: {signal.market_url}")                  # "https://..."
print(f"Match confidence: {signal.market_match_confidence}")  # 0.92
```

### How Market Prices Affect Rollup

1. **Leaves**: If `market_price` is set, rollup uses it instead of `base_rate`
2. **Parents/Target**: Still computed from children, but gap to market shows model accuracy

### Viewing Market Gaps in UI

The Signal Tree Explorer shows:
- Market indicator (📊) on nodes with market data
- Market comparison section in detail panel with:
  - Market price
  - Platform link
  - Gap in percentage points (computed - market)
  - Match confidence

### Gap Status Interpretation

| Status | Gap Range | Meaning |
|--------|-----------|---------|
| OK | ≤5pp | Tree estimate aligns with market |
| WARNING | 5-15pp | Notable difference, review assumptions |
| REVIEW | >15pp | Large gap, investigate rho signs and base rates |

## Market Data Setup

### Refreshing Market Data

**IMPORTANT**: Before building trees with market signals, refresh the market data cache:

```bash
# Refresh Polymarket data (recommended before each session)
python scripts/refresh_markets.py

# With options
python scripts/refresh_markets.py --platforms polymarket metaculus --min-liquidity 5000 --limit 2000
```

Or programmatically:

```python
from cc_builder.utils import refresh_market_data, get_market_data_stats

# Refresh markets
counts = await refresh_market_data(platforms=["polymarket"], min_liquidity=5000)
print(f"Fetched {counts['polymarket']} markets")

# Check stats
stats = await get_market_data_stats()
print(f"Total markets: {stats['total_markets']}")
```

### Why Refresh?

The market matcher searches a local SQLite cache. If data is stale:
- New markets won't be found
- Prices will be outdated
- Matching may fail silently

**Recommended**: Refresh at the start of each tree-building session.

## Market Validation

### Automatic Validation on Save

When you save a tree with `save_tree()`, it automatically compares the computed probability
against prediction market prices. This catches discrepancies like:
- Computed probability: 57.6%
- Market price: 81%
- Gap: -23.4pp → Status: **REVIEW - gap >15pp**

The validation is printed to console and saved in the JSON output:

```json
{
  "target": { ... },
  "computed_probability": 0.576,
  "market_validation": {
    "platform": "polymarket",
    "matched_question": "One Battle After Another to win Best Picture?",
    "market_price": 0.81,
    "gap_pp": -23.4,
    "status": "REVIEW - gap >15pp",
    "url": "https://polymarket.com/...",
    "match_confidence": 0.92
  }
}
```

### Interpreting Gap Status

| Status | Gap Range | Meaning |
|--------|-----------|---------|
| `OK` | ≤5pp | Tree estimate aligns with market |
| `WARNING - gap >5pp` | 5-15pp | Notable difference, review assumptions |
| `REVIEW - gap >15pp` | >15pp | Large gap, investigate rho signs and base rates |

### Checking Market Prices During Build

Use `check_market_price()` to verify base rates as you build:

```python
from cc_builder.utils import check_market_price

# Check market for a signal you're about to add
result = await check_market_price("Will One Battle win Best Picture at the 2026 Oscars?")
if result:
    print(f"Market: {result['platform']} @ {result['market_price']:.0%}")
    print(f"Match confidence: {result['match_confidence']:.0%}")
```

This is useful for:
- Setting accurate base rates for signals that have market prices
- Validating assumptions before finalizing the tree
- Catching errors early (e.g., using wrong entity or wrong year)

### When No Market Match is Found

If `check_market_price()` returns None or validation shows no match:
1. The question may not have a market (novel/niche questions)
2. Keywords may not match market titles (try different phrasing)
3. Market data cache may be stale (re-fetch from providers)

### Disabling Validation

To skip market validation (e.g., for questions with no market):

```python
save_tree(tree, target_slug, validate_market=False)
```
