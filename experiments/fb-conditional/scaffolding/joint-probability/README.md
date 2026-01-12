# Joint Probability Table Elicitation

Tests whether eliciting joint probability tables improves coherence and accuracy over direct conditional elicitation.

## Hypothesis

Forcing the model to fill out a 2x2 joint probability table (which must sum to 1.0) guarantees Bayesian consistency and may improve reasoning.

## Method

Instead of asking for P(A), P(A|B=1), P(A|B=0) separately, we ask:

```
        B=YES    B=NO
A=YES   [  ]     [  ]
A=NO    [  ]     [  ]
```

Conditionals are derived mathematically from the table.

## Run

```bash
uv run python experiments/fb-conditional/scaffolding/joint-probability/run_experiment.py
```

## Results (2026-01-12)

| Metric                     | Baseline | Joint Table | Delta   |
|----------------------------|----------|-------------|---------|
| Format compliance          | —        | **100%**    | —       |
| Brier (strong)             | +0.007   | **+0.015**  | +0.008  |
| Brier (weak)               | -0.002   | **+0.021**  | +0.023  |
| Brier (none)               | -0.015   | **-0.001**  | +0.014  |
| Direction correct (strong) | 57%      | **71%**     | +14%    |
| Inconsistent (strong)      | 14%      | **0%**      | -14%    |
| False positives (none)     | 7%       | 20%         | **+13%**|
| Bayes consistent           | 50%      | **100%**    | +50%    |

## Key Finding

Joint table elicitation:
- **Guarantees** Bayesian consistency (100%)
- **Eliminates** mathematically impossible answers (0% inconsistent)
- **Improves** direction accuracy on correlated pairs (+14%)
- **But increases** false positives on unrelated pairs (+13%)

The format may bias the model toward finding correlations where none exist.

## Files

- `run_experiment.py` — Main experiment script
- `results_*.json` — Raw results
