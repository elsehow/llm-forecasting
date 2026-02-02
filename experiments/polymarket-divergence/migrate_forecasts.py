"""One-time migration to fix forecast data schema.

Migrates forecasts.jsonl:
- Renames `market_price` → `market_price_at_forecast` (immutable record)
- Removes mutable fields: `divergence`, `signal`, `price_updated_at`
- Adds `market_id` field if extractable from URL

After migration, forecasts.jsonl becomes append-only and immutable.
Live signals are computed on-the-fly by show_signals.py.
"""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FORECASTS_FILE = RESULTS_DIR / "forecasts.jsonl"


def extract_market_id_from_url(url: str) -> str | None:
    """Extract market slug from Polymarket URL to use as market_id."""
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


def migrate_forecast(record: dict) -> dict:
    """Migrate a single forecast record to new schema."""
    migrated = {}

    # Copy fields we want to keep
    keep_fields = [
        "id", "timestamp", "question", "url", "close_date",
        "forecast", "ci", "category", "rationale", "status", "resolution"
    ]
    for field in keep_fields:
        if field in record:
            migrated[field] = record[field]

    # Rename market_price → market_price_at_forecast
    if "market_price" in record:
        migrated["market_price_at_forecast"] = record["market_price"]
    elif "market_price_at_forecast" in record:
        migrated["market_price_at_forecast"] = record["market_price_at_forecast"]

    # Add market_id from URL
    url = record.get("url", "")
    market_id = extract_market_id_from_url(url)
    if market_id:
        migrated["market_id"] = market_id

    # Fields explicitly NOT copied (they're computed at display time):
    # - divergence
    # - signal
    # - price_updated_at

    return migrated


def main(dry_run: bool = False):
    """Run the migration."""
    if not FORECASTS_FILE.exists():
        print(f"No forecasts file at {FORECASTS_FILE}")
        return

    # Read existing forecasts
    original_records = []
    with open(FORECASTS_FILE) as f:
        for line in f:
            if line.strip():
                original_records.append(json.loads(line))

    print(f"Read {len(original_records)} forecasts")

    # Migrate each record
    migrated_records = [migrate_forecast(r) for r in original_records]

    # Show sample migration
    if original_records:
        print("\n--- Sample migration (first record) ---")
        print("BEFORE:")
        sample_before = original_records[0]
        for k in ["market_price", "divergence", "signal", "price_updated_at"]:
            if k in sample_before:
                print(f"  {k}: {sample_before[k]}")

        print("AFTER:")
        sample_after = migrated_records[0]
        for k in ["market_price_at_forecast", "market_id"]:
            if k in sample_after:
                print(f"  {k}: {sample_after[k]}")

        # Check removed fields
        removed = [k for k in ["divergence", "signal", "price_updated_at"] if k in sample_before]
        if removed:
            print(f"  (removed: {', '.join(removed)})")
        print("---\n")

    if dry_run:
        print("[DRY RUN] Would write migrated forecasts")
        return migrated_records

    # Create backup
    backup_path = FORECASTS_FILE.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    shutil.copy(FORECASTS_FILE, backup_path)
    print(f"Created backup: {backup_path}")

    # Write migrated forecasts
    with open(FORECASTS_FILE, "w") as f:
        for record in migrated_records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(migrated_records)} migrated forecasts to {FORECASTS_FILE}")

    # Verify
    print("\nVerification:")
    with open(FORECASTS_FILE) as f:
        first_line = f.readline()
        record = json.loads(first_line)
        assert "market_price_at_forecast" in record, "Missing market_price_at_forecast"
        assert "divergence" not in record, "Still has divergence field"
        assert "signal" not in record, "Still has signal field"
        assert "price_updated_at" not in record, "Still has price_updated_at field"
    print("✓ Schema migration verified")

    return migrated_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate forecasts.jsonl to new schema")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")

    args = parser.parse_args()
    main(dry_run=args.dry_run)
