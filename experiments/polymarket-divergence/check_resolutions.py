"""Check resolution status for forecasts from Polymarket API.

Fetches ground truth from Polymarket to verify/update resolutions.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx

GAMMA_API_URL = "https://gamma-api.polymarket.com"
FORECASTS_FILE = Path(__file__).parent / "results" / "forecasts.jsonl"
THRESHOLD = 0.10  # Divergence threshold for signal


async def fetch_market_by_slug(client: httpx.AsyncClient, slug: str) -> dict | None:
    """Fetch market data from Polymarket by slug."""
    try:
        response = await client.get(f"{GAMMA_API_URL}/markets", params={"slug": slug})
        response.raise_for_status()
        markets = response.json()
        return markets[0] if markets else None
    except Exception as e:
        print(f"    Error fetching {slug}: {e}")
        return None


def parse_resolution(raw: dict) -> tuple[str | None, float | None, str]:
    """Parse resolution status from market data.

    Returns:
        (resolution, resolved_price, status_detail)
        resolution is 'YES', 'NO', or None if not resolved
    """
    uma_status = raw.get("umaResolutionStatus", "unknown")
    closed = raw.get("closed", False)
    active = raw.get("active", True)

    status_detail = f"uma={uma_status}, closed={closed}, active={active}"

    if uma_status != "resolved":
        return None, None, status_detail

    try:
        outcomes = json.loads(raw.get("outcomes", "[]"))
        prices = json.loads(raw.get("outcomePrices", "[]"))

        yes_idx = 0 if outcomes[0].lower() == "yes" else 1
        yes_price = float(prices[yes_idx])

        resolution = "YES" if yes_price > 0.5 else "NO"
        return resolution, yes_price, status_detail
    except (json.JSONDecodeError, TypeError, IndexError, ValueError) as e:
        return None, None, f"parse error: {e}"


def compute_signal(llm_prob: float, market_prob: float) -> str:
    """Compute trading signal from divergence."""
    divergence = llm_prob - market_prob
    if divergence > THRESHOLD:
        return "BUY_YES"
    elif divergence < -THRESHOLD:
        return "BUY_NO"
    return "NO_TRADE"


async def main(update: bool = False):
    """Check resolution status for all forecasts.

    Args:
        update: If True, update the forecasts file with resolutions
    """
    if not FORECASTS_FILE.exists():
        print(f"No forecasts file at {FORECASTS_FILE}")
        return

    forecasts = []
    with open(FORECASTS_FILE) as f:
        for line in f:
            if line.strip():
                forecasts.append(json.loads(line))

    print(f"Loaded {len(forecasts)} forecasts\n")

    today = datetime.now().strftime("%Y-%m-%d")

    # Find forecasts past close_date that need checking
    needs_check = []
    for fc in forecasts:
        # Skip already resolved (unless PENDING which means needs recheck)
        res = fc.get("resolution")
        if res and res not in ["PENDING", None]:
            continue
        if fc.get("status") == "effectively_resolved":
            continue

        close_date = fc.get("close_date", "9999")
        if close_date <= today:
            needs_check.append(fc)

    print(f"Checking {len(needs_check)} overdue forecasts...\n")

    if not needs_check:
        print("No overdue forecasts to check.")
        _print_summary(forecasts)
        return

    updated = []
    async with httpx.AsyncClient(timeout=30) as client:
        for fc in needs_check:
            slug = fc.get("market_id")
            if not slug:
                print(f"  {fc['id']}: No market_id, skipping")
                continue

            print(f"{fc['id']}: {fc['question'][:55]}...")

            raw = await fetch_market_by_slug(client, slug)
            if not raw:
                print(f"    Could not fetch market\n")
                continue

            resolution, resolved_price, status_detail = parse_resolution(raw)

            if resolution:
                print(f"    ✓ RESOLVED: {resolution} (yes_price={resolved_price:.2f})")
                fc["resolution"] = resolution
                fc["status"] = "resolved"
                fc["resolved_price"] = resolved_price
                updated.append(fc)
            else:
                print(f"    ○ Not resolved ({status_detail})")
            print()

    # Update file if requested
    if update and updated:
        print(f"Updating {len(updated)} forecasts...")
        fc_dict = {f["id"]: f for f in forecasts}
        for f in updated:
            fc_dict[f["id"]] = f

        with open(FORECASTS_FILE, "w") as out:
            for f in fc_dict.values():
                out.write(json.dumps(f) + "\n")
        print("Forecasts file updated.\n")

    _print_summary(forecasts if not update else list(fc_dict.values()))


def _print_summary(forecasts: list[dict]):
    """Print summary stats and signal performance."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Compute stats for resolved signals
    resolved_signals = []
    for fc in forecasts:
        res = fc.get("resolution")
        if not res or res not in ["YES", "NO"]:
            continue

        llm = fc.get("forecast", fc.get("llm_probability"))
        mkt = fc.get("market_price_at_forecast", fc.get("market_probability"))
        if llm is None or mkt is None:
            continue

        signal = compute_signal(llm, mkt)
        if signal == "NO_TRADE":
            continue

        resolved_signals.append({
            "id": fc["id"],
            "question": fc["question"][:45],
            "signal": signal,
            "resolution": res,
            "llm": llm,
            "market": mkt,
        })

    if resolved_signals:
        wins = 0
        print(f"\nResolved signals ({len(resolved_signals)}):\n")
        for s in resolved_signals:
            won = (s["signal"] == "BUY_YES" and s["resolution"] == "YES") or \
                  (s["signal"] == "BUY_NO" and s["resolution"] == "NO")
            if won:
                wins += 1
            status = "✅ WIN " if won else "❌ LOSS"
            print(f"  {status} {s['id']}: {s['signal']} → {s['resolution']}")
            print(f"         LLM: {s['llm']:.0%} vs Mkt: {s['market']:.0%} | {s['question']}...")
        print(f"\n  Hit rate: {wins}/{len(resolved_signals)} = {wins/len(resolved_signals):.0%}")
    else:
        print("\nNo signals have resolved yet.")

    # Overall stats
    total = len(forecasts)
    resolved = len([f for f in forecasts if f.get("resolution") in ["YES", "NO"]])
    pending = total - resolved
    print(f"\n  Total: {total} | Resolved: {resolved} | Pending: {pending}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check Polymarket resolution status")
    parser.add_argument("--update", action="store_true", help="Update forecasts file with resolutions")
    args = parser.parse_args()
    asyncio.run(main(update=args.update))
