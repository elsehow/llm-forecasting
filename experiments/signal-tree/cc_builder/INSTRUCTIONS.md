# Signal Tree Builder Instructions

You (Claude Code) are building a signal tree to decompose a forecasting question into actionable signals.

## Key Insight

The main advantage of CC-driven tree building over the automated pipeline is **validation**:
- Web search to verify facts before generating signals
- Catch domain constraint violations (e.g., wrong eligibility years)
- Recursive context gathering for each node

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
