"""Unit inference and normalization utilities for forecast tree questions.

Values are stored in their DISPLAY units so the UI can format with a simple append:
    `${value}${unit.short_label}` → "58$T" for GDP, "9.4B" for population

Canonical storage units:
    usd_trillions       → Trillions (170T GDP → 170)
    usd                 → Raw dollars ($500 → 500)
    population_billions → Billions (9.4B → 9.4)
    population          → Millions (340M → 340)
    percent             → 0-100 (15% → 15)
    years               → Years (76 → 76)
    rate                → Per 1,000 (34 per 1k → 34)
    count               → Raw count (UI handles K/M/B scaling)
"""

import re

from llm_forecasting.models import Unit


# Expected ranges for validation (min, max) - values outside this likely need normalization
EXPECTED_RANGES = {
    "usd_trillions": (0.1, 1000),  # $0.1T to $1000T
    "usd": (0, 1e12),  # Up to $1T in raw dollars
    "population_billions": (0.01, 50),  # 10M to 50B people
    "population": (0.1, 10000),  # 100K to 10B in millions
    "percent": (0, 100),
    "years": (0, 200),
    "rate": (0, 1000),  # per 1,000
    "count": (0, 1e15),  # no upper bound really
    "ratio": (0, 1000),
}

# Smart normalization: detect value scale and apply appropriate conversion
# Each unit type can have multiple possible source scales
SCALE_DETECTORS = {
    "usd_trillions": [
        # (value_range, divisor, description)
        ((1e11, 1e15), 1e12, "raw dollars → trillions"),
        ((1e8, 1e11), 1e9, "billions of dollars → trillions"),
        ((1e5, 1e8), 1e6, "millions of dollars → trillions"),
    ],
    "population_billions": [
        ((1e8, 1e12), 1e9, "raw people → billions"),
        ((1e5, 1e8), 1e6, "millions → billions (value in millions)"),
        ((1e3, 1e5), 1e3, "thousands of millions → billions"),
    ],
    "population": [
        ((1e6, 1e12), 1e6, "raw people → millions"),
        ((1e3, 1e6), 1e3, "thousands → millions"),
    ],
}


def normalize_value(value: float, unit_type: str) -> float:
    """Normalize a raw value to its display unit using smart scale detection.

    Automatically detects what scale the value is in based on its magnitude
    and applies the appropriate conversion.

    Args:
        value: The raw value (possibly in various scales)
        unit_type: The unit type string (e.g., "usd_trillions")

    Returns:
        Value in display units

    Examples:
        normalize_value(170_000_000_000_000, "usd_trillions") → 170.0  (raw dollars)
        normalize_value(9_700, "population_billions") → 9.7  (value was in millions)
    """
    # Check if already in expected range
    expected_min, expected_max = EXPECTED_RANGES.get(unit_type, (None, None))
    if expected_min is not None and expected_max is not None:
        if expected_min <= value <= expected_max:
            return value

    # Try scale detection
    detectors = SCALE_DETECTORS.get(unit_type, [])
    for (low, high), divisor, _ in detectors:
        if low <= value <= high:
            return value / divisor

    # Fallback: no transformation if we can't detect the scale
    return value


def needs_normalization(value: float, unit_type: str) -> bool:
    """Check if a value likely needs normalization based on expected ranges.

    Returns True if the value is outside the expected range for its unit type.
    """
    if unit_type not in EXPECTED_RANGES:
        return False

    expected_min, expected_max = EXPECTED_RANGES[unit_type]
    return value < expected_min or value > expected_max


def validate_normalized_value(value: float, unit_type: str) -> tuple[bool, str]:
    """Validate that a value is within expected range for its unit type.

    Returns (is_valid, message).
    """
    if unit_type not in EXPECTED_RANGES:
        return True, "No range defined for unit type"

    expected_min, expected_max = EXPECTED_RANGES[unit_type]
    if value < expected_min:
        return False, f"Value {value} below expected min {expected_min} for {unit_type}"
    if value > expected_max:
        return False, f"Value {value} above expected max {expected_max} for {unit_type}"
    return True, "OK"


def infer_unit_from_question(question_id: str, question_text: str) -> Unit | None:
    """Infer the appropriate unit from question ID and text.

    Uses pattern matching on common question patterns to determine units.
    Returns None if unit cannot be confidently inferred.
    """
    text_lower = question_text.lower()

    # Probability/percentage patterns
    probability_patterns = [
        r"\bprobability\b",
        r"\bproportion\b",
        r"\bpercentage\b",
        r"\bshare\b",
        r"\b%\b",
        r"\bparticipation rate\b",
    ]
    for pattern in probability_patterns:
        if re.search(pattern, text_lower):
            return Unit.from_type("percent")

    # Mortality/birth rates (per 1,000)
    if re.search(r"\bmortality\b", text_lower) or re.search(r"\bbirth rate\b", text_lower):
        return Unit.from_type("rate")

    # Life expectancy (years)
    if re.search(r"\blife expectancy\b", text_lower):
        return Unit.from_type("years")

    # GDP patterns
    if re.search(r"\bgdp\b", text_lower):
        return Unit.from_type("usd_trillions")

    # Population patterns
    if re.search(r"\bpopulation\b", text_lower):
        if "world" in text_lower or "global" in text_lower:
            return Unit.from_type("population_billions")
        return Unit.from_type("population")

    # Cost/price patterns (USD)
    if re.search(r"\bcost\b", text_lower) or re.search(r"\bprice\b", text_lower):
        return Unit.from_type("usd")

    # Death counts
    if re.search(r"\bkilled\b", text_lower) or re.search(r"\bdeaths\b", text_lower):
        if "number" in text_lower:
            return Unit.from_type("count")

    # Wealth patterns
    if re.search(r"\bwealth\b", text_lower):
        if "proportion" in text_lower or "share" in text_lower or "controlled" in text_lower:
            return Unit.from_type("percent")
        return Unit.from_type("usd_trillions")

    return None


def parse_unit_from_dict(unit_dict: dict) -> Unit:
    """Parse a Unit from a dictionary representation.

    Expected format:
    {
        "type": "percent",
        "label": "percent",  # optional, defaults from type
        "short_label": "%"   # optional, defaults from type
    }
    """
    unit_type = unit_dict["type"]
    default_unit = Unit.from_type(unit_type)
    return Unit(
        type=unit_type,
        label=unit_dict.get("label", default_unit.label),
        short_label=unit_dict.get("short_label", default_unit.short_label),
    )
