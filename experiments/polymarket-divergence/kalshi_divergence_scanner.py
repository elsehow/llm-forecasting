#!/usr/bin/env python3
"""Scan for divergences between Prophet Arena AI forecasts and Kalshi market prices."""
import json
import requests
from dataclasses import dataclass

PROPHET_API = "https://api.prophetarena.co/api/events/paginated"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2/events"

@dataclass
class Divergence:
    event: str
    market: str
    ai_forecast: float
    kalshi_mid: float
    kalshi_bid: float
    kalshi_ask: float
    divergence: float
    direction: str
    
def get_prophet_events():
    resp = requests.get(PROPHET_API, params={
        "limit": 200, "include_predictions": True, "resolved_type": "open"
    })
    return resp.json().get("data", [])

def get_kalshi_event(ticker):
    try:
        resp = requests.get(f"{KALSHI_API}/{ticker}")
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def main():
    print("Scanning Prophet Arena vs Kalshi for divergences...\n")
    
    events = get_prophet_events()
    non_sports = [e for e in events if e.get("category") not in ["Sports"]]
    
    divergences = []
    
    for event in non_sports:
        ticker = event.get("event_ticker", "")
        title = event.get("title", "")
        top_markets = event.get("top_markets", [])
        
        if not top_markets:
            continue
            
        # Get Kalshi data
        kalshi = get_kalshi_event(ticker)
        if not kalshi or "markets" not in kalshi:
            continue
        
        kalshi_markets = {m.get("yes_sub_title", m.get("subtitle", "")): m for m in kalshi.get("markets", []) 
                         if m.get("status") == "active"}
        
        for pa_market in top_markets:
            market_name = pa_market.get("market", "")
            ai_prob = pa_market.get("avg_probability", 0)
            
            # Find matching Kalshi market
            kalshi_match = None
            for k_name, k_market in kalshi_markets.items():
                if market_name.lower() in k_name.lower() or k_name.lower() in market_name.lower():
                    kalshi_match = k_market
                    break
            
            if not kalshi_match:
                continue
            
            # Kalshi prices are in cents
            k_bid = kalshi_match.get("yes_bid", 0) / 100
            k_ask = kalshi_match.get("yes_ask", 0) / 100
            k_mid = (k_bid + k_ask) / 2
            
            if k_mid == 0:
                continue
            
            div = ai_prob - k_mid
            if abs(div) >= 0.10:  # 10%+ divergence
                direction = "BUY_YES" if div > 0 else "BUY_NO"
                divergences.append(Divergence(
                    event=title,
                    market=market_name,
                    ai_forecast=ai_prob,
                    kalshi_mid=k_mid,
                    kalshi_bid=k_bid,
                    kalshi_ask=k_ask,
                    divergence=div,
                    direction=direction
                ))
    
    # Sort by divergence magnitude
    divergences.sort(key=lambda x: abs(x.divergence), reverse=True)
    
    print(f"Found {len(divergences)} divergences >= 10%\n")
    print("="*80)
    
    for d in divergences[:20]:
        print(f"\n{d.direction} | Δ{abs(d.divergence)*100:.0f}%")
        print(f"Event: {d.event}")
        print(f"Market: {d.market}")
        print(f"  AI forecast: {d.ai_forecast*100:.0f}%")
        print(f"  Kalshi: {d.kalshi_bid*100:.0f}¢/{d.kalshi_ask*100:.0f}¢ (mid: {d.kalshi_mid*100:.0f}%)")

if __name__ == "__main__":
    main()
