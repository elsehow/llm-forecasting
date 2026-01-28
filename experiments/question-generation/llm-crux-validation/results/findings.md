# LLM Crux Validation: Findings

## Summary

**Match rate: 0%** — LLM-generated cruxes do not exist as Polymarket markets.

This is itself an important finding about the nature of LLM vs human question generation.

## Key Observation

### What LLMs generate (causal drivers):
- "Will Khamenei's health significantly deteriorate?"
- "Will Iran experience sustained nationwide protests?"
- "Does Iran acquire or test a nuclear weapon?"
- "Will a major faction of Iran's Revolutionary Guard Corps publicly break with the Supreme Leader?"

### What Polymarket has (timeline variants):
- "Khamenei out by January 31?"
- "Khamenei out by March 31?"
- "Khamenei out by June 30?"
- "US strikes Iran by January 14?"
- "US strikes Iran by January 15?"

## Interpretation

LLMs perform **causal reasoning**: "What upstream factors would make this happen?"

Polymarket traders create **timeline arbitrage**: "Will the same thing happen by different deadlines?"

These are fundamentally different question types:
- LLM cruxes = **mechanistic** (why/how)
- Market questions = **temporal** (when)

## Implications

1. **Cannot validate LLM cruxes on Polymarket** — the equivalent markets don't exist
2. **Russell 2000 approach is necessary** — we can control the crux markets (signals) rather than hoping they exist
3. **LLM cruxes may be more useful than market questions** for understanding information flow, but we can't test this yet

## What This Tells Us

The 0% match rate suggests one of:
- **LLMs generate better cruxes** — causal drivers are more informative than timeline variants
- **LLMs generate different cruxes** — neither better nor worse, just different question types
- **LLMs miss what traders care about** — timeline specificity matters for trading

We cannot distinguish between these without external validation.

## Next Steps

1. **Accept the null result** — LLM cruxes can't be validated on Polymarket
2. **Focus on Russell 2000** — domain-specific signals we can control and track
3. **Consider hybrid approach** — use LLM cruxes to FIND relevant timeline markets

## Statistics

- Ultimates selected: 32
- Cruxes generated: 160
- Cruxes matched: 0 (0.0%)
- Topic distribution: politics_intl (31%), other (25%), crypto (16%), politics_us (13%), sports (13%), finance (3%)

Generated: 2026-01-23
