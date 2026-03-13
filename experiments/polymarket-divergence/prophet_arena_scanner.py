#!/usr/bin/env python3
"""Scan Prophet Arena forecasts and compare with Polymarket for divergences."""
import json
import requests
from datetime import datetime

PROPHET_API = "https://api.prophetarena.co/api/events/paginated"
POLYMARKET_API = "https://gamma-api.polymarket.com/markets"

def get_prophet_events():
    """Fetch all open events from Prophet Arena."""
    resp = requests.get(PROPHET_API, params={
        "limit": 200,
        "include_predictions": True,
        "resolved_type": "open"
    })
    return resp.json().get("data", [])

def get_prophet_predictions(event_ticker):
    """Get detailed predictions for a specific event."""
    url = f"https://api.prophetarena.co/api/events/{event_ticker}/predictions"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()
    return None

def search_polymarket(query):
    """Search Polymarket for matching markets."""
    resp = requests.get(POLYMARKET_API, params={"_q": query, "limit": 5})
    return resp.json() if resp.status_code == 200 else []

def main():
    print("Fetching Prophet Arena events...")
    events = get_prophet_events()
    
    # Focus on non-sports events
    interesting_events = [e for e in events if e.get("category") not in ["Sports"]]
    
    print(f"\nFound {len(interesting_events)} non-sports events\n")
    print("=" * 80)
    
    for event in interesting_events:
        title = event.get("title", "")
        ticker = event.get("event_ticker", "")
        category = event.get("category", "")
        close_time = event.get("close_time", "")
        top_markets = event.get("top_markets", [])
        
        print(f"\n[{category}] {title}")
        print(f"  Ticker: {ticker}")
        print(f"  Closes: {close_time}")
        
        if top_markets:
            print("  Prophet Arena predictions:")
            for m in top_markets[:3]:
                market = m.get("market", "")
                avg_prob = m.get("avg_probability", 0)
                predictors = m.get("predictors", [])
                pred_str = ", ".join([f"{p['predictor_name'].split('/')[-1]}:{p['probability']*100:.0f}%" for p in predictors])
                print(f"    {market}: {avg_prob*100:.0f}% [{pred_str}]")
        
        print()

if __name__ == "__main__":
    main()
