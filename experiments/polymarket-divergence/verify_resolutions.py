#!/usr/bin/env python3
"""Verify forecast resolutions against Polymarket API."""
import json
import sys
import requests

def get_market_status(slug_or_id):
    """Query Polymarket API for market status."""
    # Try by slug first
    url = f"https://gamma-api.polymarket.com/markets?slug={slug_or_id}"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if data:
        return data[0]
    return None

def parse_resolution(market_data):
    """Parse resolution from market data."""
    if not market_data:
        return None, "not_found"
    
    prices = json.loads(market_data.get("outcomePrices", "[]"))
    closed = market_data.get("closed", False)
    uma_status = market_data.get("umaResolutionStatus", "")
    
    if not closed:
        return None, "active"
    
    if uma_status != "resolved":
        return None, f"closed_but_{uma_status}"
    
    # prices[0] = YES, prices[1] = NO
    # If YES=1, NO=0 -> resolved YES
    # If YES=0, NO=1 -> resolved NO
    if len(prices) >= 2:
        if float(prices[0]) == 1:
            return "YES", "resolved"
        elif float(prices[1]) == 1:
            return "NO", "resolved"
    
    return None, "unknown"

def main():
    # Load forecasts
    with open("results/forecasts.jsonl") as f:
        forecasts = [json.loads(line) for line in f if line.strip()]
    
    # Check each forecast marked as resolved
    resolved = [f for f in forecasts if f.get("resolution") in ["YES", "NO"]]
    
    print("Verifying resolved forecasts against Polymarket API...\n")
    
    mismatches = []
    for f in resolved:
        market_id = f.get("market_id", "")
        our_resolution = f["resolution"]
        
        market = get_market_status(market_id)
        api_resolution, status = parse_resolution(market)
        
        match = "✓" if our_resolution == api_resolution else "✗"
        if our_resolution != api_resolution:
            mismatches.append((f["id"], our_resolution, api_resolution, status))
        
        print(f"{match} {f['id']}: ours={our_resolution}, api={api_resolution} ({status})")
        print(f"   {f['question'][:60]}")
    
    if mismatches:
        print(f"\n⚠️  {len(mismatches)} MISMATCHES FOUND:")
        for fid, ours, api, status in mismatches:
            print(f"   {fid}: recorded={ours}, actual={api} ({status})")
    else:
        print("\n✓ All resolutions verified correct")

if __name__ == "__main__":
    main()
