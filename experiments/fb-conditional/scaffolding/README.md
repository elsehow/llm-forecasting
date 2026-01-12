# Scaffolding Experiments

Experiments testing whether different elicitation methods improve conditional reasoning.

## Experiments

| Experiment | Hypothesis | Status |
|------------|-----------|--------|
| `joint-probability/` | Eliciting joint tables forces coherence | Complete |
| `bracket-elicitation/` | Forcing direction commitment first reduces impossible updates | Complete |
| `two-stage/` | Separating classification from elicitation reduces false positives | Complete |
| `mechanized-bayes/` | Eliciting marginals/joints separately improves accuracy | Complete |
| `consistency-check-loop/` | Showing models their inconsistency enables self-correction | Complete |
| `adaptive-scaffolding/` | Routing to different scaffolds based on context improves performance | Complete |

## Winner: Two-Stage

The [[Adaptive Scaffolding Experiment]] found that **Two-Stage-Uniform** outperforms all other approaches:
- Best overall Brier (+0.008)
- Zero false positives (0%)
- 100% direction accuracy on bracket-routed pairs

Use Two-Stage unless pairs are pre-filtered as correlated (then use Bracket).

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
