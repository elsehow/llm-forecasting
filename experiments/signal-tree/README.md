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

## Dependencies

- `llm_forecasting.voi` — Two-step ρ estimation, VOI calculations
- `llm_forecasting.market_data` — Market signal fetching
- `scenario-construction/shared/signals.py` — Signal generation (partially reused)

## Related

- Vault: `sources/2026-01-19-atomic-signals-experiment.md`
- Project: `projects/Tree of Life.md`
