# Experiment 3: ρ and Magnitude

**Finding:** ρ is the key ingredient. Providing the correlation coefficient achieves 87% direction accuracy and halves MAE. LLM ρ estimation is 90% accurate on "obvious" cases.

## What We Learned

1. **Providing ρ works:** 87% direction accuracy, 48% MAE reduction
2. **Models already know how to use ρ:** No formula teaching needed
3. **LLM ρ estimation is good:** 90% accurate on clear relationships
4. **Bottleneck shifts:** From "how to update" to "estimating ρ correctly"

## Failure Modes

LLM ρ estimation fails on **spurious correlations** — where market sentiment diverges from logical structure. The Fed chair pair (Bullard vs Laffer) is the canonical example: logically ρ = -0.95 (mutually exclusive), but market ρ = +0.69 (shared "unconventional pick" sentiment).

## Key Scripts

| File | Purpose |
|------|---------|
| `compute_correlations.py` | Calculate ρ from price comovement |
| `evaluate.py` | Test baseline vs oracle (with ρ) |
| `evaluate_v2.py` | Extended evaluation |
| `evaluate_rho_estimation.py` | Test LLM ρ estimation accuracy |
| `explicit_formula.txt` | Formula variants tested |

## Running

```bash
# Compute correlations for pairs
uv run python experiments/conditional-forecasting/exp3-rho-magnitude/compute_correlations.py

# Evaluate oracle condition (providing true ρ)
uv run python experiments/conditional-forecasting/exp3-rho-magnitude/evaluate.py

# Test LLM ρ estimation
uv run python experiments/conditional-forecasting/exp3-rho-magnitude/evaluate_rho_estimation.py
```

## Data

Uses `../data/curated_pairs.json` — 20 pairs with computed correlations from Polymarket price history.

## Source

See `sources/2026-01-13-market-comovement-experiment.md` in the vault.
