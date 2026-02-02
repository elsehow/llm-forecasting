"""Append a forecast to forecasts.jsonl.

IMPORTANT: forecasts.jsonl is append-only and immutable.
- `market_price_at_forecast` preserves what the market said when we forecasted
- Signals are computed live by show_signals.py, not stored here
- This enables proper calibration analysis
"""
import json
import re
from datetime import datetime
from pathlib import Path

FORECASTS_FILE = Path(__file__).parent / "results" / "forecasts.jsonl"


def extract_market_id(url: str) -> str | None:
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


def record_forecast(
    question: str,
    url: str,
    close_date: str,
    market_price_at_forecast: float,
    forecast: float,
    ci: tuple[float, float],
    category: str = "",
    rationale: str = "",
    status: str = "active",
    market_id: str | None = None,
) -> dict:
    """Record a forecast and append to forecasts.jsonl.

    Args:
        question: The market question
        url: Polymarket URL
        close_date: ISO date string for market close
        market_price_at_forecast: Market probability AT THE TIME of forecast (immutable)
        forecast: Our probability estimate
        ci: Confidence interval as (low, high)
        category: Optional category (politics, geopolitics, sports, etc.)
        rationale: Optional reasoning for the forecast
        status: active, resolved, or effectively_resolved
        market_id: Optional market slug (extracted from URL if not provided)

    Returns:
        The recorded forecast dict
    """
    # Extract market_id from URL if not provided
    if market_id is None:
        market_id = extract_market_id(url)

    record = {
        "id": f"f_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "url": url,
        "market_id": market_id,
        "close_date": close_date,
        "market_price_at_forecast": market_price_at_forecast,
        "forecast": forecast,
        "ci": list(ci),
        "category": category,
        "rationale": rationale,
        "status": status,
        "resolution": None,
    }

    FORECASTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FORECASTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record a forecast")
    parser.add_argument("--question", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--close-date", required=True)
    parser.add_argument("--market-price", type=float, required=True,
                        help="Market price AT THE TIME of forecast")
    parser.add_argument("--forecast", type=float, required=True)
    parser.add_argument("--ci-low", type=float, required=True)
    parser.add_argument("--ci-high", type=float, required=True)
    parser.add_argument("--category", default="")
    parser.add_argument("--rationale", default="")
    parser.add_argument("--market-id", default=None,
                        help="Market slug (auto-extracted from URL if not provided)")

    args = parser.parse_args()

    result = record_forecast(
        question=args.question,
        url=args.url,
        close_date=args.close_date,
        market_price_at_forecast=args.market_price,
        forecast=args.forecast,
        ci=(args.ci_low, args.ci_high),
        category=args.category,
        rationale=args.rationale,
        market_id=args.market_id,
    )

    print(f"Recorded forecast: {result['id']}")
    print(f"Market at forecast: {result['market_price_at_forecast']:.1%}")
    print(f"Our forecast: {result['forecast']:.1%}")
    print(f"Run 'show_signals.py' to see live divergence and signals")
