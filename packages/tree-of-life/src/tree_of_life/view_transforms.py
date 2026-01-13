"""View transformations for forecast tree output.

Transforms the raw pipeline output into a view-ready format that requires
minimal logic to render - just loops and string interpolation.
"""

import re


def generate_question_display_name(text: str) -> str:
    """Generate a concise display name from verbose question text.

    Removes filler words and question syntax while preserving
    distinguishing information.

    Examples:
        "What will global real GDP be in 2050?" → "Global Real GDP in 2050"
        "What percentage of the world's population lives in extreme poverty by 2050?"
            → "Global Extreme Poverty Rate by 2050"
        "Probability of a catastrophe (10%+ world population death) caused by AI by 2050"
            → "AI Catastrophe Probability by 2050"
    """
    result = text

    # Extract year suffix early
    year_match = re.search(r"(?:by |in )(\d{4})", text)
    year_suffix = f" {year_match.group(0)}" if year_match else ""

    # Handle "Probability of X caused by Y" → "Y X Probability"
    caused_by_match = re.search(
        r"probability of (?:a )?(.+?) caused by (.+?)(?:\s+by\s+\d{4})?$",
        result,
        re.IGNORECASE,
    )
    if caused_by_match:
        event, cause = caused_by_match.groups()
        event = re.sub(r"\([^)]+\)", "", event).strip()
        result = f"{cause} {event} Probability{year_suffix}"

    # Handle "Probability a/that catastrophe leads to..." → "Catastrophe Probability"
    elif re.match(r"probability (?:that |a )", result, re.IGNORECASE):
        result = re.sub(r"^probability (?:that |a )", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\([^)]+\)", "", result)
        result = re.sub(r"leads to.*", "", result, flags=re.IGNORECASE)
        # Remove year from middle if present, we'll add it at end
        result = re.sub(r"\s*by \d{4}\s*", " ", result)
        result = result.strip()
        if not result.lower().endswith("probability"):
            result = result + " Probability"
        result = result + year_suffix

    # Handle "What percentage/proportion of the world's population..."
    elif re.match(r"what (?:percentage|proportion) of", result, re.IGNORECASE):
        # Extract what they live in / are / do
        lives_match = re.search(r"lives? in (.+?)(?:\s+by\s+\d{4}|\s*\(|$)", result, re.IGNORECASE)
        if lives_match:
            condition = lives_match.group(1).strip()
            condition = re.sub(r"\([^)]+\)", "", condition).strip()
            result = f"Global {condition} Rate{year_suffix}"
        else:
            # Fallback: just clean up
            result = re.sub(r"^what (?:percentage|proportion) of [^?]+\??\s*", "", result, flags=re.IGNORECASE)
            result = result.strip() or text

    # Handle "Proportion of X" → "X Share"
    elif re.match(r"proportion of", result, re.IGNORECASE):
        result = re.sub(r"^proportion of (the )?", "", result, flags=re.IGNORECASE)
        # Keep as-is but clean up

    else:
        # Standard cleanup
        patterns = [
            (r"^What will\s+", ""),
            (r"^What is the\s+", ""),
            (r"^How many\s+", ""),
            (r"^How much\s+", ""),
            (r"\s+be\s+in\s+", " in "),
            (r"\?$", ""),
        ]
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean up parentheticals (all of them for cleaner display names)
    result = re.sub(r"\s*\([^)]+\)", "", result)
    result = " ".join(result.split())

    # Title case with acronym preservation
    return _title_case_with_acronyms(result)


def _title_case_with_acronyms(text: str) -> str:
    """Title case text while preserving acronyms like GDP, AI, US, U.S."""
    words = text.split()
    title_words = []
    acronyms = {"gdp", "ai", "us", "un", "who", "imf", "bea", "bls", "u.s.", "u.s"}
    small_words = {"in", "of", "the", "by", "to", "for", "and", "or", "at", "a", "an"}

    for i, word in enumerate(words):
        word_clean = word.lower().rstrip(".,;:?!")
        if word_clean in acronyms or word_clean.replace(".", "") in {"us"}:
            # Handle U.S. specially
            if "." in word:
                title_words.append(word.upper())
            else:
                title_words.append(word.upper())
        elif i > 0 and word_clean in small_words:
            title_words.append(word.lower())
        else:
            title_words.append(word.capitalize())

    return " ".join(title_words)


def generate_signal_display_name(text: str, max_length: int = 50) -> str:
    """Generate a concise display name from verbose signal text.

    Extracts the key action/event from the signal description.

    Examples:
        "A major AI lab (OpenAI, Anthropic, Google DeepMind, or Meta) announces..."
            → "Major AI Lab Announces Breakthrough"
    """
    # Remove parentheticals (often contain examples/lists)
    result = re.sub(r"\([^)]+\)", "", text)

    # Clean up extra whitespace
    result = " ".join(result.split())

    # Truncate at natural breakpoints if too long
    if len(result) > max_length:
        # Try to break at common conjunctions/punctuation
        breakpoints = [" that ", " which ", " where ", ", ", " - "]
        for bp in breakpoints:
            if bp in result[:max_length]:
                idx = result.index(bp)
                if idx > 20:  # Don't break too early
                    result = result[:idx]
                    break

        # Final truncation if still too long
        if len(result) > max_length:
            result = result[: max_length - 3].rsplit(" ", 1)[0] + "..."

    # Title case
    words = result.split()
    title_words = []
    acronyms = {"ai", "us", "un", "gdp", "phd"}
    for i, word in enumerate(words):
        if word.lower() in acronyms:
            title_words.append(word.upper())
        elif i == 0 or word.lower() not in {"a", "an", "the", "in", "of", "to", "for", "and", "or"}:
            title_words.append(word.capitalize())
        else:
            title_words.append(word.lower())

    return " ".join(title_words)


def transform_signal_direction(direction: str) -> str:
    """Convert signal direction to view-ready value."""
    return "up" if direction == "increases" else "down"


def transform_signal_magnitude(magnitude: str) -> str:
    """Convert signal magnitude to view-ready value."""
    mapping = {
        "large": "strong",
        "medium": "moderate",
        "small": "weak",
    }
    return mapping.get(magnitude, magnitude)


def compute_question_aggregates(
    question_id: str, scenarios: list[dict], conditionals: list[dict]
) -> dict:
    """Compute expected value, min, and max for a question across scenarios.

    Returns dict with 'expected', 'min', 'max' keys.
    """
    # Get all values for this question across scenarios
    values_by_scenario = {}
    for c in conditionals:
        if c.get("question_id") == question_id:
            scenario_id = c.get("scenario_id")
            # Use median for continuous, probability for binary
            value = c.get("median") or c.get("probability")
            if value is not None:
                values_by_scenario[scenario_id] = value

    if not values_by_scenario:
        return {"expected": None, "min": None, "max": None}

    # Get scenario probabilities
    scenario_probs = {s["id"]: s.get("probability", 0) for s in scenarios}

    # Compute expected value (probability-weighted)
    expected = 0.0
    total_prob = 0.0
    values = []

    for scenario_id, value in values_by_scenario.items():
        prob = scenario_probs.get(scenario_id, 0)
        expected += value * prob
        total_prob += prob
        values.append(value)

    # Normalize if probabilities don't sum to 1
    if total_prob > 0 and total_prob != 1.0:
        expected = expected / total_prob

    return {
        "expected": round(expected, 2) if expected else None,
        "min": round(min(values), 2) if values else None,
        "max": round(max(values), 2) if values else None,
    }


def build_scenario_outcomes(scenario_id: str, conditionals: list[dict]) -> dict:
    """Build outcomes dict for a scenario: question_id -> value."""
    outcomes = {}
    for c in conditionals:
        if c.get("scenario_id") == scenario_id:
            qid = c.get("question_id")
            # Use median for continuous, probability for binary, probabilities for categorical
            value = c.get("median") or c.get("probability") or c.get("probabilities")
            if value is not None:
                outcomes[qid] = value
    return outcomes
