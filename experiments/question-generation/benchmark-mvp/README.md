# MVP Benchmark: Question Generation (Phase 1)

**Goal:** Test end-to-end question generation pipeline with VOI scoring.

## Summary

| Metric | Value |
|--------|-------|
| Ultimates tested | 20 |
| Cruxes generated | 200 (10 per ultimate) |
| Mean VOI | 0.157 |
| Max VOI | 0.260 |
| VOI vs LLM ranking | ρ = 0.11 (low agreement) |

**Key finding:** VOI provides a different signal than naive LLM ranking. Generated cruxes are domain-relevant, unlike spurious market-derived pairs.

## Pipeline

1. **Select ultimates:** High-volume ($1M+) Polymarket questions, excluding sports matches and nomination questions
2. **Generate cruxes:** LLM (Sonnet) brainstorms 10 cruxes per ultimate
3. **Score by VOI:**
   - Estimate ρ (Haiku) between ultimate and crux
   - Elicit conditionals (Haiku) with ρ in prompt
   - Compute Linear VOI
4. **Baseline:** Naive LLM ranking ("rank these by usefulness")

## Results

### Generated Cruxes are Sensible

Examples:
- **Ultimate:** "No change in Fed interest rates after January 2026?"
  - **Top crux:** "Will the Federal Reserve cut interest rates at least once before January 2026?" (ρ=-0.50, VOI=0.25)

- **Ultimate:** "Russia x Ukraine ceasefire by January 31, 2026?"
  - **Top crux:** "Will Russia control less territory in Ukraine by January 15, 2026?" (ρ=0.70, VOI=0.20)

- **Ultimate:** "Will Tim Walz win the 2028 US Presidential Election?"
  - **Top crux:** "Will Tim Walz announce his candidacy for the 2028 Democratic presidential nomination?" (ρ=0.70, VOI=0.26)

### VOI vs LLM Ranking: Low Correlation

Mean ρ = 0.11 between VOI rankings and naive LLM rankings.

Interpretation: VOI provides a **different signal** than intuitive "usefulness." This could mean:
1. VOI captures something LLMs miss (information-theoretic value)
2. VOI rankings need validation against ground truth

### Cross-Validation: Market Pairs are Spurious

Market-derived high-ρ pairs are mostly noise:
- "Memphis Grizzlies NBA Finals" ↔ "Trump nominate Malpass as Fed Chair" (ρ=-0.89)
- "No change in Fed rates" ↔ "Fogo FDV above $300M" (ρ=+0.69)
- "Barron Trump as Fed Chair" ↔ "Mark Cuban Democratic nomination" (ρ=-1.00)

Meanwhile, LLM-generated cruxes are domain-relevant. This confirms Phase 0's finding that data-mined ρ is spurious noise.

**Implication:** LLM ρ estimation is more meaningful than market-derived ρ for question generation.

## Top Cruxes by VOI

| VOI | ρ_est | Ultimate | Crux |
|-----|-------|----------|------|
| 0.260 | 0.70 | Tim Walz 2028 | Will Tim Walz announce candidacy? |
| 0.260 | 0.80 | Tim Walz 2028 | Will Tim Walz finish top 3 in Iowa? |
| 0.260 | 0.70 | Chicago Bulls NBA | Will Bulls acquire All-NBA player? |
| 0.260 | 0.80 | Greg Abbott 2028 | Will Greg Abbott announce campaign? |
| 0.250 | -0.50 | Fed no change | Will Fed cut rates before Jan 2026? |

## Files

- `run_benchmark.py` - Main benchmark script
- `cross_validate.py` - Cross-validation against known pairs
- `results/benchmark_results.json` - Full results
- `results/cross_validation.json` - Cross-validation results

## Usage

```bash
uv run python experiments/question-generation/benchmark-mvp/run_benchmark.py
```

## Next Steps

1. **Validate VOI rankings:** Use short-horizon questions where ground truth becomes available
2. **Test generation quality:** Do high-VOI cruxes actually predict market movement?
3. **Compare to Nadja's pipeline:** A/B test naive generation vs structured approach
4. **Add rubric filtering:** Parsimony, resolvability, readability thresholds

## Cost

~$1-2 total (using Haiku for bulk operations, Sonnet for generation).
