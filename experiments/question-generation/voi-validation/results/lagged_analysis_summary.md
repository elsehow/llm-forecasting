# Lagged Correlation Analysis Results

**Date:** 2026-01-23 16:06
**N pairs (full 30-day lag):** 8
**N pairs (short 7-day lag):** 25

## Methodology

Tests whether Market A's price movements *predict* Market B's movements at future time points.

- **Contemporaneous:** corr(returns_A[t], returns_B[t])
- **Lagged:** corr(returns_A[t], returns_B[t+lag])

**Note:** The original VOI validation (r=0.653) tests whether pairs with higher price co-movement (ρ)
show larger belief shifts when one market resolves. This lagged analysis tests whether *daily returns*
in one market predict daily returns in another. These are complementary but different tests.

**Bonferroni correction:** α = 0.05 / 10 = 0.0050

## Full Lag Analysis (up to 30 days)

Limited to pairs with 35+ overlapping trading days.

| Lag | Mean r | Std | N | Sig (α=0.05) | Sig (Bonferroni) |
|-----|--------|-----|---|--------------|------------------|
| Contemporaneous | -0.0245 | 0.1995 | 8 | 1/8 (12%) | 1/8 (12%) |
| A→B (1 day) | -0.0087 | 0.1941 | 8 | 1/8 (12%) | 1/8 (12%) |
| A→B (7 days) | +0.0560 | 0.1425 | 8 | 1/8 (12%) | 0/8 (0%) |
| A→B (14 days) | -0.0539 | 0.2064 | 8 | 1/8 (12%) | 0/8 (0%) |
| A→B (30 days) | +0.0312 | 0.1508 | 8 | 0/8 (0%) | 0/8 (0%) |
| B→A (1 day) | +0.0282 | 0.1668 | 8 | 0/8 (0%) | 0/8 (0%) |
| B→A (7 days) | -0.1084 | 0.1010 | 8 | 0/8 (0%) | 0/8 (0%) |
| B→A (14 days) | -0.0247 | 0.1531 | 8 | 0/8 (0%) | 0/8 (0%) |
| B→A (30 days) | -0.0414 | 0.2334 | 7 | 0/7 (0%) | 0/7 (0%) |

## Directional Asymmetry

Tests whether A→B correlation differs from B→A (paired t-test on |r|).

| Lag | Mean |r| A→B | Mean |r| B→A | Dominant | p-value |
|-----|---------------|---------------|----------|---------|
| 1 days | 0.1447 | 0.1510 | B→A | 0.9278 |
| 7 days | 0.1138 | 0.1214 | B→A | 0.9053 |
| 14 days | 0.1754 | 0.1263 | A→B | 0.4280 |
| 30 days | 0.1032 | 0.1821 | B→A | 0.2896 |

## Short Lag Analysis (up to 7 days)

Includes pairs with 12+ overlapping trading days.

| Lag | Mean r | N | p-value |
|-----|--------|---|---------|
| Contemporaneous | +0.0540 | 25 | 0.3118 |
| A→B (1 day) | +0.0566 | 25 | 0.2777 |
| A→B (7 days) | +0.0777 | 25 | 0.1699 |
| B→A (1 day) | -0.0305 | 25 | 0.4601 |
| B→A (7 days) | -0.0116 | 25 | 0.8305 |

## Interpretation

### Key Findings

1. **Contemporaneous return correlation** (n=25): r = +0.054
   - p-value = 0.312 (testing if mean r ≠ 0)

2. **Lagged return correlations** show slightly higher values:
   - 1-day lag: r = +0.057
   - 7-day lag: r = +0.078

### Relationship to Original VOI Validation

The original VOI validation (r=0.653, p<0.0001) tested a different hypothesis:

- **Original:** Cross-sectional test across pairs — do pairs with higher price-level ρ
  show larger belief shifts after resolution?
- **This analysis:** Time-series test within pairs — do daily returns in A predict
  daily returns in B?

**Why the difference matters:**

- Price-level co-movement (original ρ) captures long-term relationships
- Return correlation captures short-term predictability
- Even without short-term predictability, the original validation stands:
  markets with related outcomes (high ρ) DO show correlated belief updates

### Conclusion

❌ **No significant lagged correlations** — return correlations are near zero.

However, this does NOT invalidate the original VOI validation because:
1. The original test was cross-sectional (across pairs), not time-series (within pairs)
2. Price-level co-movement (ρ=0.5-0.7) can exist without short-term return predictability
3. VOI measures information flow at resolution, not daily price dynamics

**Bottom line:** The lagged analysis shows no strong evidence of short-term predictability,
but the contemporaneous correlation (r=0.653) at the *market level* remains valid evidence
that VOI captures meaningful information relationships.