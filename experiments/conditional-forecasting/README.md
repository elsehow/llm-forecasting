# Conditional Forecasting Experiments

Getting LLMs to produce reliable conditional probability estimates. Required for VOI calculations, tree propagation, and question evaluation.

**Where we started:** LLMs can't do conditional forecasting reliably. 57% direction accuracy, frequent hallucinations, incoherent probability models.

**Where we are now:** LLMs can do conditional forecasting at ~85% accuracy with a simple single-prompt approach. When ρ estimation is good, 92%.

## Experiments

| Folder | Experiment | Key Finding |
|--------|------------|-------------|
| `exp1-baseline/` | Baseline validation | Vanilla LLMs are unreliable: 43% wrong direction, 7-50% false positives, 14% impossible updates |
| `exp2-scaffolding/` | Coherence scaffolding | Two-Stage wins: 85-100% direction, 0% false positives, 0% impossible. But coherence ≠ accuracy |
| `exp3-rho-magnitude/` | ρ and magnitude | Providing ρ achieves 87% direction, halves MAE. LLM ρ estimation 90% accurate |
| `exp4-e2e/` | End-to-end composition | E2E 1-call achieves 85% direction, 92% when ρ estimation good. Adding constraints hurts |

## Production Recommendation

**Use E2E 1-call** (single prompt, no staging, no constraints):

```
Step 1: Estimate the correlation coefficient (ρ) between these questions.
Step 2: Estimate P(A) and P(A|B=YES), using your ρ estimate to calibrate
        the magnitude of update.
```

## Data

Shared data files in `data/`:

| File | Contents |
|------|----------|
| `pairs.json` | Generated pairs from ForecastBench |
| `curated_pairs.json` | 20 curated pairs for ρ experiments |
| `price_history/` | Polymarket candlestick data |

## Results

All experiment results in `results/`. Key files:

| File | From |
|------|------|
| `evaluation_e2e_*.json` | Experiment 4 |
| `evaluation.json` | Experiment 3 |
| `rho_estimation.json` | Experiment 3 (ρ estimation) |

## Documentation

See `projects/Conditional Forecasting.md` in the Obsidian vault for:
- Full analysis and learnings
- The Market-Logic Gap problem
- Open questions
- Implementation notes
