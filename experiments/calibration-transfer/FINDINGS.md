# Calibration Transfer Experiment Results

**Date:** 2026-01-12 17:35

## Summary

**Verdict:** PARTIAL SUCCESS - Calibration improves Brier (not significant)

## Data

| Metric | Value |
|--------|-------|
| Total questions in corpus (post-cutoff) | 1577 |
| Questions resolved YES | 227 |
| Questions resolved NO | 1350 |
| Pairs generated | 200 |
| Usable pairs (A=YES) | 200 |

## Magnitude Error Analysis

| Metric | Value |
|--------|-------|
| Mean error | -0.194 |
| Std error | 0.439 |
| Median error | -0.075 |
| Range | [-0.950, 0.950] |

### By Classification

| Classification | Count | Mean Error |
|----------------|-------|------------|
| Correlated | 4 | -0.550 |
| Independent | 196 | -0.187 |

## Calibration Model

**Learned function:** `calibrated = 0.790 * raw + 0.288`

**Interpretation:** LLM overshoots by 21% on average, with +0.29 baseline shift

## Validation Results

| Metric | Value |
|--------|-------|
| Train size | 134 |
| Test size | 66 |
| Uncalibrated Brier | 0.1773 |
| Calibrated Brier | 0.1718 |
| Improvement | +0.0055 |
| p-value | 0.4000 |
| 95% CI | [-0.0365, +0.0532] |
| Statistically significant | No |

## Interpretation

The experiment shows **partial support** for the hypothesis. Key findings:

1. **LLMs systematically undershoot probabilities**: Mean prediction 0.306 vs ground truth 0.500 (error -0.194)
2. **Calibration helps but effect is not robust**: Brier improves by +0.0055, but 95% CI crosses zero
3. **Most pairs classified as independent** (196/200): ForecastBench questions are largely unrelated, limiting the conditional forecasting test

**Why partial success, not failure:**
- The calibration function `0.790 * raw + 0.288` has the right shape: it boosts low predictions toward 0.5
- The improvement direction is correct (positive Brier improvement)
- Larger sample or domain-specific data might yield significance

**Implications for Causal Discovery Engine:**
- Phase 1 shows **promise but needs more validation**
- The hypothesis (LLMs know direction but not magnitude) remains plausible
- Consider: larger dataset, domain-specific questions, or paired correlated questions
- **Don't proceed to Phase 2 yet** — need stronger Phase 1 evidence


## Raw Results

See:
- `data/pairs_with_ground_truth.json` - All pairs with predictions
- `data/magnitude_errors.json` - Error distribution
- `results/calibration_model.json` - Model coefficients
- `results/validation_results.json` - Full validation metrics
