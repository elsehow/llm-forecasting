# Scaffolding Experiments

Experiments testing whether different elicitation methods improve conditional reasoning.

## Experiments

| Experiment | Hypothesis | Status |
|------------|-----------|--------|
| `joint-probability/` | Eliciting joint tables forces coherence | In progress |
| `bracket-elicitation/` | Forcing direction commitment first reduces impossible updates | In progress |
| `mechanized-bayes/` | Eliciting marginals/joints separately, then computing conditionals, improves accuracy | Ready |
| `consistency-check-loop/` | Showing models their inconsistency enables self-correction | Ready |

## Baseline

All experiments compare to the baseline from the main FB Conditional Experiment:
- **Model**: Sonnet 4 + extended thinking (2000 token budget)
- **Prompt**: Skeptical prompt (prime toward independence)
- **Metrics**: Brier improvement, direction accuracy, Bayes consistency

Baseline results (Sonnet 4 + thinking):
- Brier (strong): +0.007
- Brier (weak): -0.002
- Brier (none): -0.015
- Direction correct (strong): 57%
- False positives (none): 7%
- Bayes consistent (strong): 50%
