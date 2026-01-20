# Signal Tree Experiment

Recursive signal decomposition as an alternative to scenario-based forecasting.

## Motivation

Scenarios feel artificial for short-horizon questions (e.g., Oscar Best Picture). Signals are directly informative — they don't need a "scenario" intermediary.

**Key insight:** Scenarios are a flat approximation of what could be a recursive signal tree.

## The Approach

```
Target: P(Democrat wins 2028)
├── Signal: "Strong economy in 2028" (long-horizon → decompose)
│   ├── Sub-signal: "Recession in 2025?" ✓
│   ├── Sub-signal: "Unemployment > 5% by Dec 2026?" ✓
│   └── Sub-signal: "Inflation < 3% by Dec 2026?" ✓
├── Signal: "Trump is GOP nominee" (long-horizon → decompose)
│   ├── Sub-signal: "Trump endorses Vance by Sept 2026?" ✓
│   └── Sub-signal: "Trump faces major legal setback by 2026?" ✓
└── Signal: "Strong Dem candidate emerges"
    └── Sub-signal: "Dem under 55 announces exploratory by Oct 2026?" ✓
```

### Algorithm

1. Generate signals for target (any horizon)
2. For each signal with horizon > threshold (e.g., 1 year):
   - Recursively generate sub-signals that inform IT
3. Continue until all leaves are "actionable" (resolve within horizon)
4. Each edge has a **ρ** (correlation coefficient via two-step estimation)
5. Roll up probabilities from leaves → root mathematically

## Structure

```
signal-tree/
├── README.md              # This file
├── generate.py            # Main entry point
├── shared/
│   ├── decomposition.py   # Recursive signal decomposition
│   ├── tree.py            # Signal tree data structures
│   └── rollup.py          # Probability computation (leaf → root)
├── results/               # Output JSON files
└── web/                   # Visualization (tree explorer)
```

## Usage

```bash
# Generate signal tree for a target
uv run python experiments/signal-tree/generate.py --target "one_battle_best_picture"

# With recursive decomposition for long-horizon
uv run python experiments/signal-tree/generate.py --target "democrat_whitehouse_2028" --max-depth 3
```

## Comparison to Scenario Construction

| Aspect | Scenarios | Signal Tree |
|--------|-----------|-------------|
| Structure | Flat (4-5 scenarios) | Recursive (depth varies by horizon) |
| Probability source | LLM holistic estimate | Math from leaf signals |
| Best for | Explanatory narratives | Grounded probability computation |
| Short-horizon | Feels artificial | Natural fit |
| Long-horizon | Works but arbitrary | Natural decomposition |

## Market Validation

When saving a tree, `save_tree()` automatically validates the computed probability against prediction market prices:

```python
from cc_builder.utils import save_tree

path = save_tree(tree, "one_battle_best_picture")
# Prints:
# === Market Validation ===
# Market: Will One Battle After Another win Best Picture?
# Market price: 81.0%
# Computed: 57.6%
# Gap: -23.4pp
# Status: REVIEW - gap >15pp
```

The validation is saved in the output JSON:

```json
{
  "market_validation": {
    "platform": "polymarket",
    "matched_question": "...",
    "market_price": 0.81,
    "gap_pp": -23.4,
    "status": "REVIEW - gap >15pp",
    "match_confidence": 0.92
  }
}
```

| Status | Gap | Action |
|--------|-----|--------|
| `OK` | ≤5pp | Aligned with market |
| `WARNING` | 5-15pp | Review assumptions |
| `REVIEW` | >15pp | Investigate rho signs, base rates |

To check market prices during tree building:

```python
from cc_builder.utils import check_market_price

result = await check_market_price("Will One Battle win Best Picture?")
if result:
    print(f"{result['platform']}: {result['market_price']:.0%}")
```

## Dependencies

- `llm_forecasting.voi` — Two-step ρ estimation, VOI calculations
- `llm_forecasting.market_data` — Market matching, validation, price fetching
- `scenario-construction/shared/signals.py` — Signal generation (partially reused)

## Related

- Vault: `sources/2026-01-19-atomic-signals-experiment.md`
- Project: `projects/Tree of Life.md`
