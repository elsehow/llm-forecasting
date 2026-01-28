"""Pairwise comparison prompts for crux evaluation."""

# Framing 1: Shift - which causes larger probability update
SHIFT_PROMPT = """For predicting: "{ultimate_question}"

Which question's resolution would SHIFT YOUR FORECAST more?

A: "{crux_a}"
B: "{crux_b}"

If you learned A resolved vs B resolved, which causes a larger probability update?

Reply with just "A" or "B" (single letter only)."""

# Framing 2: Informative - which is more informative about outcome
INFORMATIVE_PROMPT = """For predicting: "{ultimate_question}"

Which question is MORE INFORMATIVE about the outcome?

A: "{crux_a}"
B: "{crux_b}"

Reply with just "A" or "B" (single letter only)."""

# Framing 3: Decision-relevant - which changes betting behavior more
DECISION_PROMPT = """For predicting: "{ultimate_question}"

Which question's answer would CHANGE WHAT YOU'D BET more?

A: "{crux_a}"
B: "{crux_b}"

Reply with just "A" or "B" (single letter only)."""

FRAMINGS = {
    "shift": SHIFT_PROMPT,
    "informative": INFORMATIVE_PROMPT,
    "decision": DECISION_PROMPT,
}


def format_prompt(framing: str, ultimate_question: str, crux_a: str, crux_b: str) -> str:
    """Format a pairwise comparison prompt.

    Args:
        framing: One of "shift", "informative", "decision"
        ultimate_question: The prediction question
        crux_a: First crux question
        crux_b: Second crux question

    Returns:
        Formatted prompt string
    """
    template = FRAMINGS[framing]
    return template.format(
        ultimate_question=ultimate_question,
        crux_a=crux_a,
        crux_b=crux_b,
    )
