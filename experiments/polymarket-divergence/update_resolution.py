"""Update resolution for a forecast by ID."""
import json
from pathlib import Path

FORECASTS_FILE = Path(__file__).parent / "results" / "forecasts.jsonl"


def update_resolution(forecast_id: str, resolution: str, status: str = "resolved") -> bool:
    """Update the resolution field for a forecast.

    Args:
        forecast_id: The forecast ID (e.g., "f_20260130_001")
        resolution: The resolution value (YES, NO, or other outcome)
        status: New status (default: resolved)

    Returns:
        True if forecast was found and updated, False otherwise
    """
    if not FORECASTS_FILE.exists():
        print(f"Error: {FORECASTS_FILE} not found")
        return False

    lines = FORECASTS_FILE.read_text().strip().split("\n")
    updated = []
    found = False

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if record["id"] == forecast_id:
            record["resolution"] = resolution
            record["status"] = status
            found = True
        updated.append(json.dumps(record))

    if found:
        FORECASTS_FILE.write_text("\n".join(updated) + "\n")
        print(f"Updated {forecast_id}: resolution={resolution}, status={status}")
    else:
        print(f"Forecast {forecast_id} not found")

    return found


def list_pending() -> list[dict]:
    """List all forecasts without a resolution."""
    if not FORECASTS_FILE.exists():
        return []

    pending = []
    for line in FORECASTS_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("resolution") is None and record.get("status") == "active":
            pending.append(record)
    return pending


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update forecast resolution")
    parser.add_argument("--id", help="Forecast ID to update")
    parser.add_argument("--resolution", help="Resolution value (YES/NO)")
    parser.add_argument("--list", action="store_true", help="List pending forecasts")

    args = parser.parse_args()

    if args.list:
        pending = list_pending()
        if not pending:
            print("No pending forecasts")
        else:
            print(f"\n{len(pending)} pending forecasts:\n")
            for p in pending:
                signal_str = f" [{p['signal']}]" if p.get("signal") else ""
                print(f"  {p['id']}: {p['question'][:60]}...{signal_str}")
                print(f"    Close: {p['close_date']} | Market: {p['market_price']:.1%} | Forecast: {p['forecast']:.1%}")
    elif args.id and args.resolution:
        update_resolution(args.id, args.resolution)
    else:
        parser.print_help()
