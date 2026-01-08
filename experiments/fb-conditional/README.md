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
| `run_experiment.py` | Basic elicitation: P(A), P(A\|B=1), P(A\|B=0) |
| `bayesian_check.py` | Full Bayesian check: adds P(B), P(B\|A=1), P(B\|A=0) |
| `analyze_direction.py` | Resolution-independent analysis |
| `run_full_eval.py` | **Full pipeline runner** |
| `pairs.json` | Generated question pairs |
| `results/` | Experiment output by model |

## Full Evaluation Pipeline

Run all models with all checks:

```bash
# Full run (all models, all checks)
uv run python experiments/fb-conditional/run_full_eval.py

# Quick test (1 model, 3 pairs)
uv run python experiments/fb-conditional/run_full_eval.py --test

# Specific models
uv run python experiments/fb-conditional/run_full_eval.py --models claude-sonnet-4-20250514:thinking gpt-5.2

# With new pairs
uv run python experiments/fb-conditional/run_full_eval.py --pairs pairs_new.json

# Just analyze existing results
uv run python experiments/fb-conditional/run_full_eval.py --analyze-only
```

### What the pipeline does:

1. **Basic elicitation** (`run_experiment.py`): P(A), P(A|B=1), P(A|B=0)
   - Computes Brier improvement (resolution-dependent)

2. **Bayesian check** (`bayesian_check.py`): P(B), P(B|A=1), P(B|A=0)
   - Tests if P(A|B) × P(B) = P(B|A) × P(A)

3. **Analysis** (`analyze_direction.py`):
   - Direction of update: Is P(A|B=1) > P(A) > P(A|B=0)?
   - LOTP consistency: Does a valid P(B) exist?
   - Bayes consistency: Do both directions agree?

### Longitudinal experiment (Jan 2026)

| File | Purpose |
|------|---------|
| `fetch_pending_questions.py` | Generate stock questions with baseline prices |
| `resolve_and_run.py` | Resolve questions and run experiment |
| `pending_pairs_2026-01-14.json` | 40 pairs awaiting resolution |

**TODO: Run on January 14, 2026:**
```bash
# Resolve questions and run basic experiment
uv run python experiments/fb-conditional/resolve_and_run.py

# Then run full evaluation on resolved pairs
uv run python experiments/fb-conditional/run_full_eval.py --pairs pairs_resolved.json
```

## References

- Design doc: `projects/FB Conditional Experiment.md` (Obsidian vault)
- Parent project: `projects/Conditional Forecasting Trees.md`
