"""Shared resolution conversion utilities."""


def binary_resolution_to_float(
    resolution: str | None,
    mkt_probability: float | None = None,
) -> float | None:
    """Convert a binary resolution string to a float probability.

    Handles common resolution formats from prediction markets:
    - "YES" / "yes" -> 1.0
    - "NO" / "no" -> 0.0
    - "MKT" / "mkt" -> use mkt_probability value
    - "CANCEL" / "cancel" / "N/A" -> None (ambiguous)

    Args:
        resolution: Resolution string from the market.
        mkt_probability: Probability to use for MKT resolution.

    Returns:
        Float probability (0.0, 1.0, or mkt_probability), or None if ambiguous.
    """
    if resolution is None:
        return None

    res_lower = resolution.lower().strip()

    if res_lower == "yes":
        return 1.0
    elif res_lower == "no":
        return 0.0
    elif res_lower == "mkt":
        return mkt_probability
    else:
        # CANCEL, N/A, or unknown -> ambiguous
        return None


def kalshi_resolution_to_float(result: str | None) -> float | None:
    """Convert Kalshi-style resolution to float.

    Kalshi uses lowercase "yes"/"no" for results. This is a thin wrapper
    around binary_resolution_to_float for API compatibility.

    Args:
        result: Result string from Kalshi API.

    Returns:
        1.0 for yes, 0.0 for no, None otherwise.
    """
    return binary_resolution_to_float(result)
