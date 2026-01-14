# Experiment 2: Coherence Scaffolding

**Finding:** Two-Stage elicitation solves direction (→85-100%), false positives (→0%), impossible updates (→0%). But coherence ≠ accuracy — forcing Bayesian consistency doesn't improve forecasts.

## Approaches Tested

| Approach | Hypothesis | Outcome |
|----------|------------|---------|
| `two-stage/` | Separate classification from elicitation | **Winner**: 0% FP, 100% direction |
| `bracket-elicitation/` | Force direction commitment first | 0% impossible updates, 85% direction |
| `joint-probability/` | Elicit joint tables for coherence | Coherent but not more accurate |
| `mechanized-bayes/` | Elicit marginals separately | No improvement |
| `causal-dag-first/` | Elicit causal structure first | Sometimes worse (-0.056 Brier) |
| `consistency-check-loop/` | Show models their inconsistency | Self-correction doesn't help |
| `adaptive-scaffolding/` | Route to different scaffolds | Two-Stage-Uniform wins |

## Key Insight

Coherence is achievable but orthogonal to accuracy. Forcing structure can make things worse by interfering with the model's natural reasoning.

## Running

Each approach has its own `run_experiment.py`:

```bash
uv run python experiments/conditional-forecasting/exp2-scaffolding/two-stage/run_experiment.py
uv run python experiments/conditional-forecasting/exp2-scaffolding/bracket-elicitation/run_experiment.py
```

## Baseline

Comparison to Experiment 1 baseline (Sonnet 4 + thinking):
- Direction correct (strong): 57% → 85-100%
- False positives (none): 7% → 0%
- Impossible updates: 14% → 0%

## Source

See `sources/2026-01-12-coherence-scaffolding-experiments.md` in the vault.
