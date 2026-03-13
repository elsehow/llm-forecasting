# Prophet Arena API

Prophet Arena (prophetarena.co) is a live AI forecasting benchmark from UChicago that evaluates LLM predictions against real-world outcomes. They use Kalshi as their primary data source.

## API Base URL

```
https://api.prophetarena.co
```

## Documentation

- **Swagger UI:** https://api.prophetarena.co/docs
- **OpenAPI JSON:** https://api.prophetarena.co/openapi.json

## Key Endpoints

### Get Paginated Events

```bash
curl "https://api.prophetarena.co/api/events/paginated?limit=50&include_predictions=true&resolved_type=open"
```

Parameters:
- `limit`: Number of events (default: 30)
- `include_predictions`: Include AI model forecasts (default: true)
- `resolved_type`: `open`, `resolved`, or `all` (default: open)
- `topic`: Filter by category (e.g., "Politics", "Economics", "Sports")
- `sort_by`: `close_time`, `volume`, `liquidity`, `updated_at`
- `order`: `ASC` or `DESC`
- `search`: Text search across titles

### Response Structure

```json
{
  "message": "Successfully retrieved events",
  "data": [
    {
      "event_ticker": "KXKHAMENEIOUT-AKHA",
      "title": "Ali Khamenei out as Supreme Leader?",
      "category": "Politics",
      "markets": "[\"Before September 1, 2026\", \"Before July 1, 2026\"]",
      "close_time": "2026-09-01T14:00:00+00:00",
      "volume": 12345,
      "liquidity": 1234567.0,
      "top_markets": [
        {
          "market": "Before September 1, 2026",
          "avg_probability": 0.63,
          "predictors": [
            {"predictor_name": "x-ai/grok-4", "probability": 0.65},
            {"predictor_name": "google/gemini-2.5-pro", "probability": 0.61}
          ]
        }
      ]
    }
  ]
}
```

### Get Event Details

```bash
curl "https://api.prophetarena.co/api/events/{event_ticker}"
```

### Get Event Predictions

```bash
curl "https://api.prophetarena.co/api/events/{event_ticker}/predictions"
```

## Available AI Models

- `google/gemini-2.5-pro`
- `x-ai/grok-4`
- `anthropic/claude-opus-4.1`
- `openai/gpt-5-high`
- `google/gemini-2.5-flash`
- `x-ai/grok-3-mini`

## Categories

- Sports (majority of events)
- Politics
- Economics
- Entertainment
- Companies
- Science and Technology
- Elections
- Climate and Weather
- Crypto
- World

## Use Cases

1. **Cross-reference with Kalshi:** Find where AI consensus diverges from prediction markets
2. **Model calibration:** Compare individual model forecasts
3. **Research:** Analyze AI forecasting performance across domains

---

# Kalshi API

Prophet Arena events use Kalshi event tickers (prefixed with `KX`). You can query Kalshi directly for market prices.

## API Base URL

```
https://api.elections.kalshi.com/trade-api/v2
```

(Note: Old endpoint `trading-api.kalshi.com` redirects here)

## Key Endpoints

### Get All Markets (paginated)

```bash
curl "https://api.elections.kalshi.com/trade-api/v2/markets"
```

### Get Event with All Markets

```bash
curl "https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
```

Example:
```bash
curl "https://api.elections.kalshi.com/trade-api/v2/events/KXKHAMENEIOUT-AKHA"
```

### Market Price Fields

- `yes_bid` / `yes_ask`: Current bid/ask in cents (divide by 100 for probability)
- `no_bid` / `no_ask`: Complement prices
- `last_price`: Last traded price
- `volume`: Total volume
- `liquidity`: Order book depth

---

# Finding Divergences: Prophet Arena AI vs Kalshi Market

## Methodology

1. **Fetch Prophet Arena events** with AI predictions:
   ```python
   resp = requests.get("https://api.prophetarena.co/api/events/paginated",
                       params={"include_predictions": True, "resolved_type": "open"})
   events = resp.json()["data"]
   ```

2. **For each event, query Kalshi** using the same ticker:
   ```python
   kalshi = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/events/{ticker}")
   ```

3. **Match markets by name**: Prophet Arena's `top_markets[].market` ≈ Kalshi's `yes_sub_title`

4. **Compare AI forecast vs market midpoint**:
   ```python
   ai_prob = prophet_market["avg_probability"]
   kalshi_mid = (kalshi_market["yes_bid"] + kalshi_market["yes_ask"]) / 200  # cents to prob
   divergence = ai_prob - kalshi_mid
   ```

5. **Signal**: If divergence > 10%, consider:
   - `BUY_YES` if AI > market (AIs think more likely)
   - `BUY_NO` if AI < market (AIs think less likely)

## Example Script

```python
#!/usr/bin/env python3
import requests

def scan_divergences(min_divergence=0.10):
    # Get Prophet Arena events
    pa_resp = requests.get("https://api.prophetarena.co/api/events/paginated",
                           params={"limit": 200, "include_predictions": True, "resolved_type": "open"})
    events = pa_resp.json().get("data", [])

    for event in events:
        ticker = event.get("event_ticker")

        # Get Kalshi prices
        k_resp = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/events/{ticker}")
        if k_resp.status_code != 200:
            continue
        kalshi = k_resp.json()

        # Build lookup of Kalshi markets
        k_markets = {m.get("yes_sub_title", ""): m for m in kalshi.get("markets", [])
                     if m.get("status") == "active"}

        for pa_market in event.get("top_markets", []):
            market_name = pa_market.get("market")
            ai_prob = pa_market.get("avg_probability", 0)

            # Find matching Kalshi market
            for k_name, k_market in k_markets.items():
                if market_name.lower() in k_name.lower():
                    k_mid = (k_market["yes_bid"] + k_market["yes_ask"]) / 200
                    div = ai_prob - k_mid
                    if abs(div) >= min_divergence:
                        direction = "BUY_YES" if div > 0 else "BUY_NO"
                        print(f"{direction} Δ{abs(div)*100:.0f}%: {event['title']} - {market_name}")
                        print(f"  AI: {ai_prob*100:.0f}% | Kalshi: {k_mid*100:.0f}%")

if __name__ == "__main__":
    scan_divergences()
```

## Caveats

- **Market matching is imperfect**: Names don't always align perfectly
- **Wide spreads**: Some Kalshi markets have wide bid/ask spreads; use caution
- **Liquidity**: Low-liquidity markets may not be tradeable at displayed prices
- **AI calibration unknown**: We don't know if Prophet Arena's AI models are well-calibrated
