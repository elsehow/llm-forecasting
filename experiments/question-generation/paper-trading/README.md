# Paper Trading: VOI Edge Validation

**Hypothesis:** If VOI accurately predicts information flow, we can profit by:
1. Identifying high-VOI cruxes for ultimate questions
2. Forecasting crux outcomes better than consensus
3. Trading ultimates before cruxes resolve

## Strategy

```
For each (ultimate, crux) pair with high VOI:
1. Record current prices: P(ultimate), P(crux)
2. Forecast crux outcome: P_forecast(crux = YES)
3. Compute expected ultimate price after crux resolves:
   - If crux YES: P(ultimate | crux=YES)
   - If crux NO: P(ultimate | crux=NO)
4. If expected move > threshold, "trade":
   - Buy ultimate if expecting price to rise
   - Sell ultimate if expecting price to fall
5. When crux resolves, record actual ultimate price change
6. Compute P&L
```

## Metrics

- **Hit rate:** % of times ultimate moved in predicted direction
- **MAE:** Mean absolute error of predicted vs actual shift
- **Simulated P&L:** If we had bet $100 on each trade

## Data Sources

1. **Phase 1 generated cruxes:** 200 cruxes for 20 Polymarket ultimates
2. **Validated pairs:** 34 pairs with known real relationships
3. **New pairs:** Refresh market data to find more opportunities

## Files

- `setup_trades.py` - Initialize paper trades from generated cruxes
- `track_prices.py` - Fetch current prices and record changes
- `evaluate.py` - Compute P&L and metrics
- `trades.json` - Active and completed trades
