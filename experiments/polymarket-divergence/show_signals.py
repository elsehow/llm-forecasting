"""Show live signals by computing divergence from current market prices.

This script reads forecasts and fetches LIVE prices from Polymarket,
computing signals on-the-fly without modifying the forecasts file.

The forecasts.jsonl file remains immutable - market_price_at_forecast
preserves what the market said when we made our forecast (for calibration).
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).parent / "results"
FORECASTS_FILE = RESULTS_DIR / "forecasts.jsonl"
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Signal thresholds
DIVERGENCE_THRESHOLD = 0.10


async def fetch_price_by_slug(client: httpx.AsyncClient, slug: str) -> float | None:
    """Fetch current YES price for a market by its slug."""
    try:
        response = await client.get(
            f"{GAMMA_API_URL}/markets",
            params={"slug": slug},
            timeout=10.0,
        )
        response.raise_for_status()
        markets = response.json()

        if not markets:
            return None

        market = markets[0]
        outcome_prices = market.get("outcomePrices")
        if outcome_prices:
            prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
            return float(prices[0])  # YES is always index 0
    except Exception as e:
        print(f"  Error fetching {slug}: {e}", file=sys.stderr)
    return None


def extract_slug(url: str) -> str | None:
    """Extract market slug from Polymarket URL."""
    patterns = [
        r"polymarket\.com/market/([^/?]+)",
        r"polymarket\.com/event/[^/]+/([^/?]+)",
        r"polymarket\.com/event/([^/?]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def compute_signal(forecast: float, market_price: float) -> str | None:
    """Compute trading signal based on divergence."""
    divergence = forecast - market_price
    if divergence > DIVERGENCE_THRESHOLD:
        return "BUY_YES"
    elif divergence < -DIVERGENCE_THRESHOLD:
        return "BUY_NO"
    return None


async def main(json_output: bool = False, active_only: bool = True, show_all: bool = False):
    """Display live signals for all forecasts.

    Args:
        json_output: Output JSON to stdout instead of table
        active_only: Only show active (unresolved) forecasts
        show_all: Show all forecasts, not just those with signals
    """
    if not FORECASTS_FILE.exists():
        print(f"No forecasts file found at {FORECASTS_FILE}", file=sys.stderr)
        return

    # Read all forecasts
    forecasts = []
    with open(FORECASTS_FILE) as f:
        for line in f:
            if line.strip():
                forecasts.append(json.loads(line))

    # Filter to active only if requested
    if active_only:
        forecasts = [fc for fc in forecasts if fc.get("status") == "active"]

    if not forecasts:
        print("No active forecasts found", file=sys.stderr)
        return

    if not json_output:
        print(f"Fetching live prices for {len(forecasts)} forecasts...\n", file=sys.stderr)

    # Fetch live prices and compute signals
    results = []
    async with httpx.AsyncClient() as client:
        for fc in forecasts:
            url = fc.get("url", "")
            slug = extract_slug(url)

            if not slug:
                continue

            live_price = await fetch_price_by_slug(client, slug)

            if live_price is None:
                continue

            forecast_prob = fc.get("forecast")
            if forecast_prob is None:
                continue

            # Get price at forecast time for comparison
            price_at_forecast = fc.get("market_price_at_forecast") or fc.get("market_price")

            divergence = forecast_prob - live_price
            signal = compute_signal(forecast_prob, live_price)

            result = {
                "id": fc.get("id"),
                "question": fc.get("question"),
                "url": url,
                "close_date": fc.get("close_date"),
                "market_price_at_forecast": price_at_forecast,
                "live_price": live_price,
                "forecast": forecast_prob,
                "divergence": round(divergence, 4),
                "signal": signal,
                "rationale": fc.get("rationale", ""),
            }
            results.append(result)

            # Rate limit
            await asyncio.sleep(0.1)

    # Filter to signals only unless show_all
    if not show_all:
        results = [r for r in results if r["signal"]]

    # Sort by absolute divergence
    results.sort(key=lambda x: abs(x["divergence"]), reverse=True)

    if json_output:
        print(json.dumps(results, indent=2))
        return

    # Print table
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== LIVE SIGNALS as of {now} ===\n")

    if not results:
        print("No signals (all forecasts within ±10% of market)")
        return

    for r in results:
        signal = r["signal"] or "—"
        q = r["question"][:55] if r["question"] else "?"
        live = r["live_price"]
        fcst = r["forecast"]
        div = r["divergence"]
        orig = r["market_price_at_forecast"]
        close = r["close_date"][:10] if r["close_date"] else "N/A"

        print(f"{signal:7} | mkt:{live:.0%} fcst:{fcst:.0%} div:{div:+.0%} | {q}")
        print(f"         orig:{orig:.0%} | close:{close} | {r['id']}")
        print()

    print(f"Total: {len(results)} signal{'s' if len(results) != 1 else ''}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Show live trading signals (reads forecasts, fetches live prices)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON to stdout"
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all forecasts, not just signals"
    )
    parser.add_argument(
        "--include-resolved", action="store_true", help="Include resolved forecasts"
    )

    args = parser.parse_args()
    asyncio.run(main(
        json_output=args.json,
        active_only=not args.include_resolved,
        show_all=args.all,
    ))
