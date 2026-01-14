# Market Comovement Experiment

Test whether LLMs can learn correlation strength from prediction market price comovement, improving magnitude calibration for conditional forecasts.

## Background

The magnitude problem: LLMs know *that* things correlate but not *how much* to update. Markets encode correlation strength through price comovement. If we can teach models to map question pairs → correlation strength → update magnitude, we solve the bottleneck.

See `projects/Causal Discovery Engine.md` in the vault for full context.

## Setup

1. Get a free API key from [DomeAPI.io](https://domeapi.io)

2. Install dependencies:
   ```bash
   pip install dome-api-sdk litellm numpy scipy
   ```

3. Set environment variables:
   ```bash
   export DOME_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"  # for LiteLLM
   ```

## Pipeline

Run in order:

```bash
# 1. Fetch active Polymarket markets
python fetch_markets.py
# -> data/markets.json

# 2. Get price history for each market (rate-limited, takes ~1hr for 100 markets)
python fetch_history.py
# -> data/price_history/*.json

# 3. Compute pairwise correlations
python compute_correlations.py
# -> data/pairs.json

# 4. Generate worked examples for training
python generate_examples.py
# -> data/training_examples.json

# 5. Evaluate few-shot vs baseline
python evaluate.py
# -> results/evaluation.json
```

## Correlation Buckets

| Bucket | |ρ| Range | Meaning |
|--------|----------|---------|
| independent | < 0.1 | No meaningful relationship |
| weak | 0.1 - 0.3 | Slight tendency to move together |
| moderate | 0.3 - 0.6 | Clear relationship |
| strong | ≥ 0.6 | Strongly linked outcomes |

## Success Criteria

| Phase | Success |
|-------|---------|
| Data collection | 500+ pairs with reasonable bucket distribution |
| Example generation | Worked examples look sensible across all buckets |
| Evaluation | Mean magnitude error decreases >20% with few-shot |

## Files

```
market-comovement/
├── fetch_markets.py       # Get Polymarket markets via Dome API
├── fetch_history.py       # Get price history per market
├── compute_correlations.py # Pairwise correlation analysis
├── generate_examples.py   # Create worked examples
├── evaluate.py            # Test few-shot vs baseline
├── data/
│   ├── markets.json       # Market metadata
│   ├── price_history/     # Per-market candlesticks
│   ├── pairs.json         # Correlation-labeled pairs
│   └── training_examples.json
└── results/
    └── evaluation.json    # Final results
```

## Notes

- Uses daily returns (not price levels) for correlation to avoid spurious trends
- Requires 10+ overlapping data points for a pair to be included
- Free Dome API tier: 1 query/sec (sufficient for prototyping)
