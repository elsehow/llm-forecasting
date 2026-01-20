# Recursive Dual-Approach Tree Builder

## Motivation

The current tree building process conflates different types of relationships:
- **Correlation** (rho): Statistical relationship, both outcomes possible
- **Logical exclusivity**: If A happens, B cannot happen (not just unlikely)
- **Necessity**: A must happen for B to happen (B implies A)

Example: "Will JD Vance win 2028?" vs "Will a Democrat win 2028?"
- These aren't just negatively correlated (rho=-0.8)
- They're **mutually exclusive** for the same prize
- P(Dem | Vance wins) ≈ 0, not just low

## Design Overview

### Two Parallel Phases (can run concurrently)

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│     PHASE 1: TOP-DOWN           │    │     PHASE 2: BOTTOM-UP          │
│     (Logical Structure)         │    │     (Market Discovery)          │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ 1. Identify target entity       │    │ 1. Semantic search for markets  │
│ 2. Find necessity constraints   │    │    matching target              │
│ 3. Find competing entities      │    │ 2. Expand search to related     │
│ 4. Identify causal pathways     │    │    markets                      │
│ 5. Build logical skeleton       │    │ 3. Rank by relevance            │
└─────────────────────────────────┘    └─────────────────────────────────┘
                 │                                      │
                 └──────────────┬───────────────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │   PHASE 3: RECONCILIATION   │
                 ├─────────────────────────────┤
                 │ 1. Map markets → structure  │
                 │ 2. Fill probabilities       │
                 │ 3. Identify gaps            │
                 │ 4. Recurse on uncertain     │
                 └─────────────────────────────┘
```

### Phase 1: Top-Down (Logical Structure)

Generate a decomposition that captures **logical relationships**:

#### 1.1 Necessity Constraints
What MUST happen for target to be true?

```python
class NecessityConstraint:
    """If prerequisite=NO, then target=NO with certainty."""
    prerequisite: str  # "Democrat wins nomination"
    target: str        # "Democrat wins presidency"
    # P(target | prerequisite=NO) = 0
```

Example for "Democrat wins 2028":
- Must win Democratic nomination (necessity)
- Must win enough electoral votes (necessity)

#### 1.2 Exclusivity Constraints
What outcomes are mutually exclusive?

```python
class ExclusivityConstraint:
    """If competitor wins prize, target cannot."""
    target_entity: str      # "Democratic candidate"
    competitor_entity: str  # "JD Vance"
    prize: str              # "2028 US Presidency"
    # P(target | competitor wins) ≈ 0
```

Example for "Democrat wins 2028":
- If Vance wins → Democrat loses (exclusivity)
- If any Republican wins → Democrat loses (exclusivity)

#### 1.3 Causal Pathways
What leads to what? (Time-ordered, informational)

```python
class CausalPathway:
    """Earlier event informs later event."""
    upstream: str     # "Strong 2026 midterms"
    downstream: str   # "Good 2028 chances"
    mechanism: str    # "Political momentum, party strength"
    rho: float        # Correlation strength and direction
```

Example for "Democrat wins 2028":
- 2026 midterms → momentum → 2028
- Economic conditions → incumbent blame → challenger advantage

### Phase 2: Bottom-Up (Market Discovery)

Search for markets that could inform the tree:

```python
async def discover_markets(target: str, db_path: str) -> list[MarketSignal]:
    """Find semantically related markets."""

    # 1. Direct target search
    direct = await semantic_search(target)

    # 2. Entity-based search (search for entities mentioned)
    entities = extract_entities(target)  # ["Democrat", "2028", "President"]
    entity_markets = []
    for entity in entities:
        entity_markets.extend(await semantic_search(f"{entity} 2028"))

    # 3. Inverse search (competing outcomes)
    inverse_markets = await semantic_search("Republican wins 2028 presidency")

    return deduplicate(direct + entity_markets + inverse_markets)
```

### Phase 3: Reconciliation

Map markets onto logical structure:

```python
def reconcile(structure: LogicalStructure, markets: list[MarketSignal]) -> SignalTree:
    """Map markets to structure, identify gaps."""

    # 1. Match markets to logical nodes
    for node in structure.all_nodes():
        best_match = find_best_market_match(node, markets)
        if best_match and best_match.confidence > 0.7:
            node.base_rate = best_match.market_price
            node.market_url = best_match.url
            node.market_platform = best_match.platform
        else:
            # No market - estimate or flag for decomposition
            node.base_rate = estimate_base_rate(node)
            node.needs_decomposition = True

    # 2. Handle exclusivity constraints
    for exclusivity in structure.exclusivity_constraints:
        # Create inverse signal with P(target | competitor) ≈ 0
        competitor_signal = create_exclusivity_signal(
            competitor=exclusivity.competitor_entity,
            prize=exclusivity.prize,
            base_rate=markets.find(exclusivity.competitor_entity).price,
            # NOT rho - actual conditional probability
            p_target_given_yes=0.01,  # If competitor wins, target loses
        )

    # 3. Recurse on uncertain nodes
    for node in structure.all_nodes():
        if should_decompose(node):
            child_structure = await generate_logical_structure(node.text)
            child_markets = await discover_markets(node.text)
            child_tree = reconcile(child_structure, child_markets)
            node.children = child_tree.children

    return build_tree(structure)
```

### Relationship Types (Updated)

| Type | Meaning | Math |
|------|---------|------|
| `correlation` | Statistical relationship | Use rho → posteriors |
| `necessity` | A required for B | P(B\|A=NO) = 0 |
| `sufficiency` | A guarantees B | P(B\|A=YES) = 1 |
| `exclusivity` | A and B can't both happen | P(A\|B=YES) ≈ 0 |

### Example: Democrat President 2028

#### Top-Down Structure
```
Target: Democrat wins 2028 presidency

NECESSITY:
├── Must: Democrat wins nomination
├── Must: Win electoral college

EXCLUSIVITY:
├── Competes: JD Vance wins presidency
├── Competes: Any Republican wins presidency

CAUSAL PATHWAYS:
├── 2026 midterms → momentum
├── Economy 2028 → incumbent blame
└── Candidate quality → electability
```

#### Bottom-Up Markets
```
Direct matches:
- (none found for "Democrat wins 2028")

Entity-based:
- JD Vance wins 2028 nomination (52%)
- JD Vance wins 2028 presidency (26%)
- Republican controls House after 2026 (22%)

Related:
- Various candidate-specific markets
```

#### Reconciled Tree
```
Target: Democrat wins 2028 presidency
├── [EXCLUSIVITY] JD Vance wins 2028 presidency
│   base_rate: 26% (polymarket)
│   relationship: exclusivity (P(Dem|Vance wins) ≈ 0)
│
├── [CORRELATION] Democrats win House 2026
│   base_rate: 73.5% (computed from child tree)
│   ref: democrat_house_2026
│   rho: +0.35
│
├── [CORRELATION] Economy perception 2028
│   base_rate: 40% (manual)
│   rho: -0.45 (good economy helps incumbent R)
│
└── [NECESSITY] Strong Dem nominee emerges
    base_rate: 70% (manual)
    relationship: necessity (weak nominee → lose)
```

## Implementation Plan

### Step 1: Add relationship types to SignalNode
```python
class SignalNode(BaseModel):
    relationship_type: Literal["correlation", "necessity", "sufficiency", "exclusivity"]

    # For exclusivity
    competes_with_target: bool = False
    p_target_given_yes: float | None  # Direct conditional instead of rho
```

### Step 2: Update rollup to handle exclusivity
```python
def compute_exclusivity_evidence(signal: SignalNode) -> float:
    """Evidence from exclusivity constraint."""
    if not signal.competes_with_target:
        return 0.0

    # If competitor likely to win, strong negative evidence for target
    base_rate = signal.market_price or signal.base_rate
    # P(target) ≈ P(target | competitor loses) * P(competitor loses)
    #           = some_prob * (1 - base_rate)
    return -base_rate * 2  # Strong negative evidence
```

### Step 3: Create dual-approach builder
```python
async def build_tree_dual(
    target: str,
    registry: TreeRegistry | None = None,
    max_depth: int = 3,
) -> SignalTree:
    """Build tree using dual approach."""

    # Phase 1 & 2 can run in parallel
    structure_task = generate_logical_structure(target)
    markets_task = discover_markets(target)

    structure, markets = await asyncio.gather(structure_task, markets_task)

    # Phase 3: Reconcile
    tree = reconcile(structure, markets)

    # Recursive decomposition
    for node in tree.uncertain_nodes():
        if should_decompose(node):
            child_tree = await build_tree_dual(
                node.text,
                registry=registry,
                max_depth=max_depth - 1,
            )
            node.children = child_tree.target.children

    return tree
```

## Open Questions

1. **How to estimate P(target | exclusivity signal = YES)?**
   - Currently hardcoding 0.01, but should consider edge cases
   - What if multiple competitors? (Sum of their probs can exceed 1)

2. **How to handle necessity chains?**
   - A requires B requires C
   - P(target | any necessity fails) = 0

3. **When to stop recursion?**
   - Current: 7 days, base_rate outside 0.2-0.8
   - Should we add: "market available" as a stopping condition?

4. **How to validate logical structure?**
   - LLM might generate incorrect necessity/exclusivity claims
   - Need validation step or human review
