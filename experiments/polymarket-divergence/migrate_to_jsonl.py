"""One-time migration: Convert forecasts JSON to JSONL format."""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SOURCE_FILE = RESULTS_DIR / "forecasts_20260130_round2.json"
TARGET_FILE = RESULTS_DIR / "forecasts.jsonl"


def migrate():
    if not SOURCE_FILE.exists():
        print(f"Source file not found: {SOURCE_FILE}")
        return

    data = json.load(open(SOURCE_FILE))
    timestamp = data["timestamp"]

    with open(TARGET_FILE, "w") as f:
        for i, rec in enumerate(data["forecasts"]):
            # Transform to new schema
            new_rec = {
                "id": f"f_20260130_{i:03d}",
                "timestamp": timestamp,
                "question": rec["question"],
                "url": rec["url"],
                "close_date": rec["close_date"],
                "market_price": rec["market_price"],
                "forecast": rec["forecast"],
                "ci": rec.get("confidence_interval", [rec["forecast"], rec["forecast"]]),
                "divergence": rec["divergence"],
                "signal": rec.get("signal"),
                "category": rec.get("category", ""),
                "rationale": rec.get("rationale", ""),
                "status": rec.get("status", "active"),
                "resolution": rec.get("resolution"),
            }
            f.write(json.dumps(new_rec) + "\n")

    print(f"Migrated {len(data['forecasts'])} forecasts to {TARGET_FILE}")


if __name__ == "__main__":
    migrate()
