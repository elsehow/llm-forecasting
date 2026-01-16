"""Shared datetime parsing utilities."""

from datetime import datetime, timezone


def parse_iso_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO format datetime string to datetime object.

    Handles common variations:
    - "2025-01-15T10:30:00Z" -> datetime with UTC timezone
    - "2025-01-15T10:30:00+00:00" -> datetime with timezone
    - "N/A" or None -> None

    Args:
        dt_str: ISO format datetime string, or None.

    Returns:
        Parsed datetime with timezone, or None if parsing fails.
    """
    if not dt_str or dt_str == "N/A":
        return None
    try:
        # Handle Z suffix (Zulu time = UTC)
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(dt_str)
        # Ensure timezone is set
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def epoch_ms_to_datetime(epoch_ms: int) -> datetime:
    """Convert Unix epoch milliseconds to datetime.

    Args:
        epoch_ms: Unix timestamp in milliseconds.

    Returns:
        Datetime with UTC timezone.
    """
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)


def parse_date_string(date_str: str, fmt: str = "%Y-%m-%d") -> datetime | None:
    """Parse a date string with a specific format.

    Args:
        date_str: Date string to parse.
        fmt: strptime format string.

    Returns:
        Parsed datetime, or None if parsing fails.
    """
    try:
        return datetime.strptime(date_str, fmt)
    except (ValueError, TypeError):
        return None
