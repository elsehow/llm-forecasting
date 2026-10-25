"""
Recompute max entropy two ways from saved logprob data files.

Problem: the saved single_q_*_logprobs.json files only store mean_token_entropy
per quantile, not the raw per-token entropies. We need to re-run the API calls
to get the raw token-level data.

BUT: the Together AI files (full_logprobs.json, h1_h3_logprobs.json) from the
batched runs DO have per-record data. And the single_question scripts stored
enough info to recompute.

Actually — we need to re-query the APIs to get the raw token sequences.
Instead, let's check if we can approximate from what we have:
- mean_token_entropy per quantile + n_tokens gives us a rough signal
- For single-token quantile values (n_tokens=1), mean IS the per-token entropy

For multi-token values, mean underestimates the max token entropy within that value.
But for the "max across quantiles" measure, the max of means is a lower bound on
the max of all individual token entropies.

For the "percentiles block only" method: max(mean_token_entropy) across 5 quantiles
— this is what we already computed. It's an approximation of max(map(entropy, percentile_tokens)).

For the "all tokens" method: we genuinely need the raw data. Let's re-run a subset
to compare the two methods.
"""

# This script won't work without re-running API calls.
# See single_question_rawtokens.py for the actual re-run.
print("This script documents the issue. See single_question_rawtokens.py for the re-run.")
