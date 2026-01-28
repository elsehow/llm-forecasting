# Experiments

Research experiments testing LLM forecasting capabilities.

## Directories

| Experiment | Question | Status |
|------------|----------|--------|
| `conditional-forecasting/` | Can LLMs reason conditionally about related questions? | Complete — E2E 1-call at 85% accuracy |
| `question-generation/` | Can we validate VOI as a metric for question quality? | Complete — VOI directionally validated |
| `scenario-construction/` | Which approach produces best outcome-specific scenarios? | **Active** — comparing hybrid/top-down/bottom-up |
| `calibration-transfer/` | Can we learn calibration curves from conditional updates? | Partial success (see FINDINGS.md) |
| `magnitude/` | How to elicit accurate update magnitudes? | Superseded by ρ-based approach |

## fb-conditional

Tests whether LLMs treat forecasting questions as independent or can reason about dependencies. Elicits P(A), P(A|B=YES), P(A|B=NO) and checks for:
- Correct update direction
- Law of total probability consistency
- Bayes rule consistency

**Key finding:** LLMs often get direction right but magnitude wrong.

## calibration-transfer

Tests whether a learned calibration function from P(A|B) - P(A) errors can improve Brier scores. Uses ForecastBench resolved questions.

**Key finding:** Calibration helps (+0.0055 Brier) but not statistically significant. Most question pairs are independent, limiting the test.

## magnitude

Three sub-experiments exploring the magnitude problem:

| Sub-experiment | Approach |
|----------------|----------|
| `anchor-adjust/` | Anchor-and-adjust prompting strategy |
| `linear-voi/` | Linear VOI vs entropy-based VOI |
| `strength-elicitation/` | Different prompts for eliciting update strength |

## Running Experiments

```bash
# fb-conditional full pipeline
uv run python experiments/fb-conditional/run_full_eval.py

# calibration-transfer
uv run python experiments/calibration-transfer/calibration_v4_nonsceptical.py

# magnitude sub-experiments
uv run python experiments/magnitude/anchor-adjust/run_experiment.py
uv run python experiments/magnitude/linear-voi/run_experiment.py
uv run python experiments/magnitude/strength-elicitation/run_experiment.py
```

## Data

Most experiments use `data/forecastbench.db` - run migration first:
```bash
uv run python packages/llm-forecasting/scripts/migrate_forecastbench.py --db data/forecastbench.db
```
