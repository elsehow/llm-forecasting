# FB Conditional Experiment

**Question:** Can LLM agents reason conditionally, or do they treat forecasting questions as independent?

**Answer:** See [FINDINGS.md](FINDINGS.md) for results and methodology.

## Quick Start

```bash
# 1. Ensure data is migrated
uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py --db data/forecastbench.db

# 2. Generate candidate pairs (LLM-assisted)
uv run python experiments/fb-conditional/generate_pairs.py

# 3. Run experiment
uv run python experiments/fb-conditional/run_experiment.py --model claude-opus-4-5-20251101

# With other models:
uv run python experiments/fb-conditional/run_experiment.py --model claude-sonnet-4-20250514
uv run python experiments/fb-conditional/run_experiment.py --model gpt-4o
```

Results are auto-saved to `results/{model}_{timestamp}.json`.

## Data

Uses `data/forecastbench.db` - shared SQLite database with:
- Questions from Manifold, Metaculus, Polymarket, etc.
- Market forecasts (freeze values)
- Resolutions

**Note:** Only prediction market sources are used (`manifold`, `metaculus`, `polymarket`, `infer`). Data sources (`acled`, `fred`, `yfinance`) have templated question text with placeholders, not suitable for this experiment.

## Files

| File | Purpose |
|------|---------|
| `FINDINGS.md` | Results and methodology |
| `generate_pairs.py` | LLM-assisted pair generation |
| `run_experiment.py` | Main experiment runner |
| `pairs.json` | Generated question pairs |
| `results/` | Experiment output by model |

### Longitudinal experiment (Jan 2026)

| File | Purpose |
|------|---------|
| `fetch_pending_questions.py` | Generate stock questions with baseline prices |
| `resolve_and_run.py` | Resolve questions and run experiment |
| `pending_pairs_2026-01-14.json` | 40 pairs awaiting resolution |

**TODO: Run on January 14, 2026:**
```bash
uv run python experiments/fb-conditional/resolve_and_run.py
```

## References

- Design doc: `projects/FB Conditional Experiment.md` (Obsidian vault)
- Parent project: `projects/Conditional Forecasting Trees.md`
