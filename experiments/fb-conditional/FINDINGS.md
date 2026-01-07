**Question**: Can LLM agents reason conditionally, or do they treat forecasting questions as independent?

**Answer**: Most models cannot reliably reason conditionally on novel questions. Sonnet 4 with extended thinking and GPT-5.2 show promising results, but sample sizes are small and effect sizes are modest.

## My takeaways

- We need to keep an eye on this:
	- For question generation: You mentioned something like, “every time we generate a question, VOI depends on conditional forecasts.” Our question generation pipeline will inherit this problem. If conditional estimates are noisy, VOI rankings are noisy.
	- For conditional trees: maintaining a ‘tree of life’ assumes LLMs can propagate conditional updates reliably. If they can't, we’d need:
		- Sonnet 4 + thinking everywhere (doable)
		- human-in-the-loop for conditionals (doesn't scale)
		- wait for better models (and keep testing)
- Why I’m not a doomer:
	- Sonnet 4 + thinking has a 79% win rate on strong pairs.
	- Prompt engineering matters—there may be room for improvement there.
	- This may be a capability gap (models improve) rather than a fundamental limit of LLMs. We can keep testing.
	- Even small effects might be enough for directional guidance ("this goes up, not down") even if magnitudes are unreliable.

## Results

| Model | Thinking | Strong | Weak | None | Verdict |
|-------|----------|--------|------|------|---------|
| **Sonnet 4** | **Yes** | **+0.007** | -0.002 | **-0.015** | **Best** |
| **GPT-5.2** | No | **+0.032** | +0.002 | **-0.011** | **Good** |
| Opus 4.5 | Yes | -0.003 | +0.035 | -0.005 | Mixed |
| Opus 4.5 | No | -0.027 | +0.023 | +0.007 ⚠️ | Wrong direction |
| Sonnet 4 | No | -0.094 | -0.008 | -0.013 | Wrong direction |
| GPT-4o | No | -0.017 | -0.055 | -0.049 | All negative |
| Gemini 3 Pro | No | +0.096 | -0.125 | +0.066 ⚠️ | High false positive |

*Strong/Weak/None values are mean Brier improvement over independence baseline. Positive = conditioning helps; negative = conditioning hurts. ⚠️ = improvement on None pairs is bad; means the model is hallucinating correlations on unrelated pairs.*

### Findings

1. **Sonnet 4 + extended thinking** is the only model that beats baseline on strong pairs (+0.007) while showing no spurious correlations on none pairs (-0.015). Win rate on strong pairs: 79%.

2. **GPT-5.2** also performs well (+0.032 strong, -0.011 none) with 86% win rate on strong pairs.

3. **Most models update in the wrong direction.** Sonnet 4 (no thinking), GPT-4o, and Opus 4.5 (no thinking) all show negative improvement on strong pairs — they detect that something should change but update the wrong way.

4. **Gemini 3 Pro hallucinates correlations.** It shows +0.066 improvement on unrelated pairs, meaning it finds patterns where none exist.

5. **Extended thinking helps Sonnet 4 dramatically** — from -0.094 to +0.007 on strong pairs. It doesn't help Opus 4.5 as much.

### Statistical Significance

| Model | Category | Mean | 95% CI | p-value |
|-------|----------|------|--------|---------|
| **GPT-5.2** | **Strong** | **+0.032** | **[+0.010, +0.053]** | **~0.007** |
| Sonnet 4 (thinking) | Strong | +0.007 | [-0.023, +0.037] | n.s. |

**Only GPT-5.2's strong pair improvement is statistically significant** (p<0.01). Sonnet 4 + thinking has a positive mean but the 95% CI crosses zero—we can't rule out that the true effect is zero. With n=14 strong pairs, we're underpowered to detect small effects.

### Caveats

- Sample size is small (45 pairs, 14 strong); most results are not statistically significant
- Effect sizes are modest even for best performers
- The "skeptical" prompt was used for all results (see Appendix)
- Ground truth for pair categorization was LLM-generated, not human-verified


## Methods

We take pairs of *already-resolved* ForecastBench questions and ask agents:

- P(A) — unconditional probability of question A
- P(A|B=YES) — probability of A given B resolved YES
- P(A|B=NO) — probability of A given B resolved NO

**Independence baseline:** P(A|B) = P(A). If the agent ignores the conditioning, it matches this baseline.

**Success metric:** Agent beats independence baseline on correlated pairs, shows no spurious sensitivity on unrelated pairs.

### Question sampling & stratification

1. I sampled 150 resolved binary questions from ForecastBench (prediction market sources only).
	- To ameliorate data contamination, I filtered for questions that resolved after the release date of the most recently released model (2025-10-01 - GPT-5.2 came out 2025-12-11; knowledge cutoff was likely prior to that).
	- This is not a fantastic method; we should run this result in the future in an FB sort of way.
2. I passed all questions to Sonnet 4 with a prompt that asked it to find pairs and classify them:
    - strong: "Obvious causal or logical link (e.g., 'Russia invades Ukraine' + 'NATO membership expands')"
    - weak: "Same domain but unclear if related (e.g., two AI questions that might be independent)"
    - none: "Clearly unrelated domains (e.g., 'Bitcoin price' + 'Lakers championship')"
3. From 150 questions, we were able to find 45 question pairs:
	- 14 strong positive controls (obvious causal/logical link)
	- 16 weak positive controls (same domain, unclear relationship)
	- 15 negative controls (unrelated domains)

### The Core Test

For each pair (A, B):

```
Agent provides:     P(A), P(A|B=1), P(A|B=0)
We know:            A_actual ∈ {0,1}, B_actual ∈ {0,1}

Independence baseline:  P(A|B) = P(A)
Agent's conditional:    P(A|B=B_actual)

Brier_independence = (P(A) - A_actual)²
Brier_conditional  = (P(A|B=B_actual) - A_actual)²

improvement = Brier_independence - Brier_conditional
```

**Agent wins if:** improvement > 0 (conditional forecast is more accurate)

### Metrics

| Metric | What it tests |
|--------|---------------|
| Mean improvement by category | Overall conditional reasoning ability |
| Mean sensitivity by category | Does agent react to conditioning? |
| Conditional win rate | How often does conditioning help? |


## Data

- **Source:** ForecastBench public datasets (downloaded from [github.com/forecastingresearch/forecastbench](https://github.com/forecastingresearch/forecastbench))
- **Database:** `data/forecastbench.db`
- **Questions used:** Prediction market sources only (manifold, metaculus, polymarket, infer)
- **Pairs generated:** 45 pairs from 150 sampled resolved binary questions

Questions were randomly sampled from resolved binary questions in ForecastBench, filtered to prediction market sources only (excluding data sources like ACLED, FRED, and yfinance which have templated question text).

## Run Date

2026-01-07

## Future Work

0. **Run in real time**. Like ForecastBench does—the only sure way to prevent data contamination!

1. **More models** — DeepSeek R1, Grok, Llama 4. Do other reasoning models (R1, o1) have the same spurious correlation problem?

2. **Different thinking budgets** — We used 2000 tokens. Is there a sweet spot (500? 5000?) that gets reasoning benefits without hallucinated correlations?

3. **Larger sample size** — 45 pairs is small. More pairs would enable bootstrap confidence intervals and significance testing.

4. **Asymmetry test** — Check if P(A|B) updates are consistent with P(B|A). Bayesian consistency check.

5. **Tool use** — Does giving models web search help or confound the reasoning test?

6. **Error analysis** — Which specific pairs do models fail on? Are there patterns (topic, question length, ambiguity)?

7. **Human-verified pair labels** — Current strong/weak/none labels are LLM-generated. Human verification would strengthen validity.


# Appendix

## Prompt Engineering Experiments

I tested whether prompt modifications could improve conditional reasoning on a subset (15 pairs).

### Variations

- **Baseline** — Simple "give a probability" prompt (default)

- **Justify First** — Asks model to state whether there's a relationship and its direction before giving probability:
 
 > "Before giving your probability, first answer: 1) Is there a causal or logical relationship? 2) If yes, what direction?"

This prompt *decreased* performance. I suspected that the prompt might be priming models to find connections — even spurious ones. Inspired by this failure, I introduced a new prompt:

- **Skeptical** — Prime toward independence:

> Before giving your probability, first answer:
>
> 1. Why might these questions be INDEPENDENT? (What would need to be true for them to be unrelated?)
> 2. Is there a DIRECT causal mechanism that overrides this independence?

> If no direct causal link, keep your estimate similar to what you'd give without the context.

This prompt reduced false positives on "none" pairs while preserving (or improving) performance on "strong" pairs. **All main results in this document use the skeptical prompt.**

## Win Rates

How often does conditioning improve the forecast (improvement > 0)?

| Model | Thinking | Strong | Weak | None |
|-------|----------|--------|------|------|
| **Sonnet 4** | **Yes** | **79%** | 38% | 20% |
| **GPT-5.2** | No | **86%** | 31% | 27% |
| Opus 4.5 | Yes | 57% | 38% | 13% |
| Opus 4.5 | No | 50% | 56% | 20% |
| Sonnet 4 | No | 36% | 12% | 13% |
| GPT-4o | No | 36% | 19% | 7% |
| Gemini 3 Pro | No | 33% | 20% | 27% |

*For Strong pairs, higher is better. For None pairs, lower is better (high win rate on None means hallucinated correlations).*