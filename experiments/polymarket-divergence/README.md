# Polymarket Divergence Experiment

Find tradeable divergences between LLM forecasts and prediction markets.

## Quick Start

```bash
cd /Users/elsehow/Projects/llm-forecasting

# 1. Fetch markets
uv run python experiments/polymarket-divergence/fetch_markets.py --limit 10

# 2. Record a forecast (after researching)
uv run python experiments/polymarket-divergence/record_forecast.py \
  --question "Will X happen by Y?" \
  --url "https://polymarket.com/..." \
  --close-date "2026-02-28" \
  --market-price 0.55 \
  --forecast 0.40 \
  --ci-low 0.35 --ci-high 0.45 \
  --category "geopolitics" \
  --rationale "Market overweights..."

# 3. View live signals (fetches current prices)
uv run python experiments/polymarket-divergence/show_signals.py

# 4. After market resolves
uv run python experiments/polymarket-divergence/update_resolution.py --id f_20260130_001 --resolution YES
```

## Data Files

### `results/forecasts.jsonl`

**Append-only, immutable.** Each line is a forecast record:

```json
{
  "id": "f_20260130_001",
  "timestamp": "2026-01-30T11:30:00",
  "question": "US strikes Iran by February 28, 2026?",
  "url": "https://polymarket.com/event/...",
  "market_id": "us-strikes-iran-by-february-28-2026-227-967",
  "close_date": "2026-02-28",
  "market_price_at_forecast": 0.675,
  "forecast": 0.55,
  "ci": [0.4, 0.7],
  "category": "geopolitics",
  "rationale": "Market overweights recent escalation...",
  "status": "active",
  "resolution": null
}
```

**Key fields:**
- `market_price_at_forecast` — Price when forecast was made (NEVER updated)
- `forecast` — Our probability estimate
- `status` — `active`, `resolved`, or `effectively_resolved`
- `resolution` — `YES`, `NO`, or `null` if unresolved

**Not stored (computed live):**
- `divergence` — Computed by `show_signals.py`
- `signal` — Computed by `show_signals.py`

### `results/markets.jsonl`

Append-only market snapshots from `fetch_markets.py`:

```json
{
  "fetch_timestamp": "2026-01-30T10:00:00",
  "market_id": "us-strikes-iran-...",
  "question": "...",
  "resolution_rules": "...",
  "market_price": 0.675,
  "close_date": "2026-02-28",
  "liquidity": 125000,
  "url": "https://polymarket.com/..."
}
```

## Scripts

| Script | Purpose | Modifies files? |
|--------|---------|-----------------|
| `fetch_markets.py` | Fetch markets from Polymarket API | Appends to `markets.jsonl` |
| `record_forecast.py` | Record a new forecast | Appends to `forecasts.jsonl` |
| `show_signals.py` | Display live signals | **No** (read-only) |
| `check_resolutions.py` | List forecasts needing resolution | **No** (read-only) |
| `update_resolution.py` | Mark forecast as resolved | Updates `forecasts.jsonl` (status/resolution only) |
| `migrate_forecasts.py` | One-time schema migration | Rewrites `forecasts.jsonl` |

## Workflow

```
1. fetch_markets.py     → populates markets.jsonl
2. Research & forecast  → Task agents or manual
3. record_forecast.py   → appends to forecasts.jsonl (IMMUTABLE after this)
4. show_signals.py      → fetches live prices, computes signals (NO writes)
5. check_resolutions.py → lists what's past close_date (run periodically)
6. update_resolution.py → updates status/resolution when market closes
```

## Ongoing: Resolution Tracking

Check in periodically to resolve closed markets:

```bash
cd /Users/elsehow/Projects/llm-forecasting

# 1. See what needs attention
uv run python experiments/polymarket-divergence/check_resolutions.py

# 2. For each forecast past close_date, check the Polymarket URL
#    Then record the resolution:
uv run python experiments/polymarket-divergence/update_resolution.py --id f_20260130_111 --resolution YES

# 3. View current live signals on active forecasts
uv run python experiments/polymarket-divergence/show_signals.py
```

**Goal:** Accumulate ~20+ resolved forecasts to answer the key calibration question:

> **When we disagreed with the market by >10%, who was right?**

This tells us whether LLM forecasts add value beyond market prices.

## Calibration

The key insight: **we need to preserve the market price at forecast time** to measure calibration.

**Bad (old design):** `update_prices.py` overwrote `market_price` with live prices.
- Lost historical record
- Can't answer: "When we forecasted 55%, what did the market say?"

**Good (new design):** `market_price_at_forecast` is immutable.
- Calibration: Compare `forecast` vs `resolution`, bucketed by `market_price_at_forecast`
- Signal evaluation: Compare `forecast - market_price_at_forecast` vs `resolution`

### Calibration Analysis

```python
# Example: Are we calibrated when disagreeing with markets?
import json

forecasts = [json.loads(l) for l in open("forecasts.jsonl")]
resolved = [f for f in forecasts if f["resolution"]]

for f in resolved:
    mkt = f["market_price_at_forecast"]
    fcst = f["forecast"]
    outcome = 1 if f["resolution"] == "YES" else 0
    divergence = fcst - mkt
    print(f"mkt:{mkt:.0%} fcst:{fcst:.0%} div:{divergence:+.0%} → {f['resolution']}")
```

## Signal Logic

```python
divergence = forecast - live_market_price

if divergence > 0.10:
    signal = "BUY_YES"   # We think market underprices YES
elif divergence < -0.10:
    signal = "BUY_NO"    # We think market overprices YES
else:
    signal = None        # Within threshold, no trade
```

## P&L Calculation

When a market resolves:

| Signal | Resolution | P&L |
|--------|------------|-----|
| BUY_YES | YES | `+(1 - entry_price_yes)` |
| BUY_YES | NO | `-entry_price_yes` |
| BUY_NO | NO | `+(1 - entry_price_no)` |
| BUY_NO | YES | `-entry_price_no` |

Where:
- `entry_price_yes = market_price_at_forecast`
- `entry_price_no = 1 - market_price_at_forecast`
