# Experiment 4: End-to-End Composition

**Finding:** E2E 1-call achieves 85% direction accuracy (+10pp over baseline). When ρ estimation is good, 92% (beats oracle). Adding constraints hurts — simpler is better.

## What We Tested

Can the model estimate ρ *and* use it in a single prompt? Yes.

| Condition | Direction Accuracy | Notes |
|-----------|-------------------|-------|
| Baseline | 75% | No ρ information |
| Oracle | 90% | Perfect ρ provided |
| E2E 2-call | 75% | Fresh context loses reasoning chain |
| **E2E 1-call** | **85%** | Same-context reasoning wins |
| E2E 1-call (good ρ) | 92% | Beats oracle! |

## Key Insight

Same-context reasoning outperforms staged pipelines. When the model estimates ρ and uses it in the same prompt, it achieves better results than when those steps are separated.

## Adding Constraints Hurts (Exp 4b)

We tested combining E2E with Bracket constraints from Experiment 2:

| Condition | Direction Accuracy |
|-----------|-------------------|
| E2E 1-call | 85% |
| E2E + Bracket | 75% |
| E2E + Skeptical | 80% |

The structured prompts interfered with ρ estimation. **Simpler is better.**

## Key Scripts

| File | Purpose |
|------|---------|
| `evaluate_e2e.py` | Core E2E experiment (4 conditions) |
| `evaluate_e2e_enhanced.py` | E2E + constraints experiment |

## Running

```bash
# Core E2E experiment
uv run python experiments/conditional-forecasting/exp4-e2e/evaluate_e2e.py

# E2E + constraints experiment
uv run python experiments/conditional-forecasting/exp4-e2e/evaluate_e2e_enhanced.py
```

**Note:** These scripts make LLM API calls. Without `--dry-run`, expect ~80 API calls (20 pairs × 4 conditions) and ~$5-10 cost.

## Data

Uses `../data/curated_pairs.json` — same 20 pairs as Experiment 3.

## Production Recommendation

Use **E2E 1-call** for general use:

```
Step 1: Estimate the correlation coefficient (ρ) between these questions.
Step 2: Estimate P(A) and P(A|B=YES), using your ρ estimate to calibrate
        the magnitude of update.
```

## Sources

- `sources/2026-01-14-e2e-composition-experiment.md` — Core E2E results
- `sources/2026-01-14-e2e-enhanced-experiment.md` — Constraints experiment
- `sources/2026-01-14-fed-chair-pair-failure.md` — Canonical failure case
