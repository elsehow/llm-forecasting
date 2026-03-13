#!/usr/bin/env python3
"""Find divergences between Prophet Arena and Polymarket."""
import json
import requests

def get_prophet_events():
    """Fetch from Prophet Arena."""
    resp = requests.get("https://api.prophetarena.co/api/events/paginated", params={
        "limit": 200, "include_predictions": True, "resolved_type": "open"
    })
    return resp.json().get("data", [])

def search_polymarket(query):
    """Search Polymarket."""
    resp = requests.get("https://gamma-api.polymarket.com/markets", params={"_q": query, "limit": 10})
    return resp.json() if resp.status_code == 200 else []

# Key mappings: Prophet Arena event -> Polymarket search terms
MAPPINGS = {
    "KXGOVSHUTLENGTH-26FEB28": ["government shutdown", "shutdown last"],
    "KXKHAMENEIOUT-AKHA": ["khamenei", "iran supreme leader"],  
    "KXBTCMAX150-25": ["bitcoin 150000", "btc 150k"],
    "LEAVEPOWELL-25": ["powell fed chair", "powell resign"],
    "KXGTA6ONTIME": ["gta 6 release", "gta vi"],
    "KX14AMENDCASE-26": ["birthright citizenship"],
    "KXRETIREPELOSI-26": ["pelosi retire"],
}

print("Scanning for divergences between Prophet Arena and Polymarket...\n")

events = get_prophet_events()
events_by_ticker = {e["event_ticker"]: e for e in events}

for ticker, search_terms in MAPPINGS.items():
    if ticker not in events_by_ticker:
        print(f"⚠️  {ticker} not found in Prophet Arena")
        continue
    
    event = events_by_ticker[ticker]
    print(f"\n{'='*70}")
    print(f"PROPHET ARENA: {event['title']}")
    print(f"Closes: {event['close_time']}")
    
    if event.get("top_markets"):
        print("PA Predictions:")
        for m in event.get("top_markets", [])[:3]:
            avg = m.get("avg_probability", 0)
            preds = m.get("predictors", [])
            pred_str = ", ".join([f"{p['predictor_name'].split('/')[-1]}:{p['probability']*100:.0f}%" for p in preds])
            print(f"  {m.get('market')}: {avg*100:.0f}% [{pred_str}]")
    
    # Search Polymarket
    for term in search_terms:
        markets = search_polymarket(term)
        if markets:
            print(f"\nPOLYMARKET matches for '{term}':")
            for mkt in markets[:3]:
                q = mkt.get("question", "")[:60]
                prices = mkt.get("outcomePrices", "[]")
                closed = mkt.get("closed", False)
                slug = mkt.get("slug", "")
                if not closed:
                    try:
                        p = json.loads(prices)
                        yes_price = float(p[0]) if p else 0
                        print(f"  • {q}...")
                        print(f"    YES: {yes_price*100:.0f}% | slug: {slug}")
                    except:
                        pass
