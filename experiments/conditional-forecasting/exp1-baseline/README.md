# Experiment 1: Baseline Validation

**Finding:** Vanilla LLMs can't do conditional forecasting reliably.

| Problem | Rate |
|---------|------|
| Wrong direction | 43% |
| False positives (hallucinated correlations) | 7-50% |
| Impossible updates (both conditionals shift same way) | 14% |
| Magnitude miscalibration | Pervasive |

## What We Tested

Can LLM agents reason conditionally, or do they treat forecasting questions as independent?

Elicited P(A), P(A|B=YES), P(A|B=NO) from various models on ForecastBench question pairs.

## Key Scripts

| File | Purpose |
|------|---------|
| `run_experiment.py` | Basic elicitation |
| `analyze_direction.py` | Direction accuracy analysis |
| `bayesian_check.py` | Test Bayes rule compliance |
| `run_full_eval.py` | Full pipeline |
| `FINDINGS.md` | Detailed results |

## Running

```bash
# Basic experiment
uv run python experiments/conditional-forecasting/exp1-baseline/run_experiment.py --model claude-opus-4-5-20251101

# Full evaluation pipeline
uv run python experiments/conditional-forecasting/exp1-baseline/run_full_eval.py
```

Results in `results/`.

## Models Tested

- GPT-5.2
- Claude Opus 4.5 (thinking)
- Claude Sonnet 4 (thinking)
- Gemini 3 Pro
- Grok 4

GPT-5.2 showed statistically significant improvement (p<0.01). Sonnet 4 + thinking showed positive mean but 95% CI crossed zero.

## Source

See `sources/2026-01-06-fb-conditional-experiment.md` in the vault.
