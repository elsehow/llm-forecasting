# FB Conditional Experiment

Validation experiment: Can LLM agents do conditional forecasting better than the independence baseline?

## The Idea

Permute resolved ForecastBench questions into conditional pairs:

```
Original questions (resolved):
- Q1: "Will X happen by Y?" → resolved YES
- Q2: "Will A happen by B?" → resolved NO

Conditional query:
- "If X happens by Y, what's P(A happens by B)?"
- Compare agent forecast to: P(A) × P(B) [independence assumption]
```

**Success metric:** Agent beats independence baseline on positive controls, shows no spurious sensitivity on negative controls.

## Experimental Design

100 question pairs stratified into:
- 30 strong positive controls (obvious causal/logical link)
- 40 weak positive controls (same domain, unclear relationship)
- 30 negative controls (random pairs, unrelated domains)

## Usage

```bash
# 1. Ensure data is migrated
uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py --db data/forecastbench.db

# 2. Generate candidate pairs
uv run python experiments/fb-conditional/generate_pairs.py

# 3. Run experiment
uv run python experiments/fb-conditional/run_experiment.py

# 4. Analyze results
uv run python experiments/fb-conditional/analyze.py
```

## Data

Uses `data/forecastbench.db` - shared SQLite database with:
- Questions from Manifold, Metaculus, Polymarket, etc.
- Market forecasts (freeze values)
- Resolutions

**Important:** Only use prediction market sources for this experiment:
- `manifold`, `metaculus`, `polymarket`, `infer`

Skip data sources (`acled`, `fred`, `yfinance`) — they have templated question text
with `{resolution_date}` placeholders, not suitable for conditional reasoning.

## References

- Design doc: `projects/FB Conditional Experiment.md` (Obsidian vault)
- Parent project: `projects/Conditional Forecasting Trees.md`
