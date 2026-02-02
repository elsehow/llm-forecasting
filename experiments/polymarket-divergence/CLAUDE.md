# Polymarket Divergence Experiment

Find divergences between LLM forecasts and prediction market prices.

## Data Architecture

| File | Mutability | Purpose |
|------|------------|---------|
| `forecasts.jsonl` | **Append-only, immutable** | Historical record with price AT FORECAST TIME |
| `markets.jsonl` | Append-only | Price snapshots for backtesting |
| Live signals | **Computed on-the-fly** | show_signals.py fetches live prices (not persisted) |

**Critical:** `market_price_at_forecast` is NEVER updated. This preserves the historical record needed for calibration ("what did the market say when we forecasted?").

## Workflow

### 1. Fetch Markets

```bash
cd /Users/elsehow/Projects/llm-forecasting
uv run python experiments/polymarket-divergence/fetch_markets.py --limit 10
```

Fetches active Polymarket markets with >$10k liquidity.
Appends to `results/markets.jsonl`.

### 2. Run Forecasts in Parallel

Use the **Task tool** with `subagent_type=general-purpose` to run forecasts concurrently.

**Prompt template for each Task agent:**

```
Run a structured forecast for this question. Do web searches for base rates,
prediction markets, and recent news. Then synthesize into a probability estimate.

**Question:** [market title]
**Resolution criteria:** [market description/resolution rules]
**Resolution date:** [close_date]

Output format:
- Point estimate (probability)
- Confidence interval (25th-75th percentile)
- 2-3 sentence rationale
- Key sources consulted
```

### 3. Record Forecast (IMMUTABLE)

```bash
uv run python experiments/polymarket-divergence/record_forecast.py \
  --question "..." \
  --url "https://polymarket.com/..." \
  --close-date "2026-02-28" \
  --market-price 0.55 \
  --forecast 0.40 \
  --ci-low 0.35 \
  --ci-high 0.45 \
  --category "geopolitics" \
  --rationale "..."
```

**Note:** `--market-price` is the price AT THE TIME of forecast. This is never updated.

### 4. View Live Signals

```bash
uv run python experiments/polymarket-divergence/show_signals.py
```

This script:
- Reads forecasts from `forecasts.jsonl`
- Fetches **live** prices from Polymarket API
- Computes divergence and signal for each
- Prints signal table
- **Does NOT modify forecasts.jsonl**

Options:
- `--json` — Output JSON to stdout
- `--all` — Show all forecasts, not just those with signals
- `--include-resolved` — Include resolved forecasts

### 5. Update Resolution

```bash
uv run python experiments/polymarket-divergence/update_resolution.py --id f_20260130_001 --resolution YES
```

List pending forecasts:
```bash
uv run python experiments/polymarket-divergence/update_resolution.py --list
```

## Signal Logic

```
divergence = forecast - live_market_price
```

- `BUY_YES` if `divergence > +0.10`
- `BUY_NO` if `divergence < -0.10`
- No signal if within threshold

## File Structure

```
polymarket-divergence/
├── CLAUDE.md              # This file
├── README.md              # Full documentation
├── fetch_markets.py       # Fetch markets from Polymarket API
├── record_forecast.py     # Record new forecast (append-only)
├── show_signals.py        # Compute & display live signals (read-only)
├── check_resolutions.py   # List forecasts needing resolution (read-only)
├── update_resolution.py   # Update resolution when market closes
├── migrate_forecasts.py   # One-time migration script
└── results/
    ├── markets.jsonl      # Raw market snapshots (append-only)
    └── forecasts.jsonl    # All forecasts (append-only, immutable)
```

## Tips

- **Diverse markets**: Mix categories (geopolitics, sports, crypto, elections)
- **Skip near-resolved**: Markets at >95% or <5% rarely have actionable divergences
- **Check liquidity**: Low liquidity markets may have stale prices
- **Wide CIs are honest**: Geopolitical events often warrant 30%+ confidence intervals
