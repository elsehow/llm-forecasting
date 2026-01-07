"""Phase 0: Identify and fetch base rates for anchoring forecasts.

Uses Claude's web search tool to find current reference values
based on the input questions. Uses structured outputs to guarantee
valid JSON with properly formatted dates.
"""

import json
import logging
from datetime import date

import anthropic

from conditional_trees.config import MODEL_BASE_RATES, STRUCTURED_OUTPUTS_BETA
from ..models import Question
from ..schemas import BaseRatesResponse, make_strict_schema

logger = logging.getLogger(__name__)

BASE_RATES_SYSTEM = """You identify and fetch current reference data needed to anchor forecasts.

Given a set of forecasting questions, you will:
1. Determine what current base rates would help calibrate predictions
2. Use web search to find authoritative current values
3. Return structured data with values, sources, and dates

Focus on quantitative data that serves as anchoring points:
- Current economic indicators (GDP, yields, employment)
- Demographic data (population, growth rates)
- Index values (Freedom House scores, climate indices)
- Recent statistics from authoritative sources

Be selective - only fetch what's directly relevant to the questions.
3-8 base rates is typical."""

BASE_RATES_USER = """For these forecasting questions, find current reference values to anchor predictions:

Questions:
{questions}

Search for authoritative current data for each relevant base rate.

IMPORTANT for the as_of field:
- Use the date the DATA refers to, NOT the date you fetched it
- For quarterly data like GDP: use quarter end date (e.g., "2025-09-30" for Q3 2025)
- For annual data: use year end (e.g., "2024-12-31")
- For point-in-time data like yields: use the date of the reading
- Format: YYYY-MM-DD

Only include base rates where you found concrete, recent data."""


async def fetch_base_rates(
    questions: list[Question],
    max_searches: int = 8,
    model: str = MODEL_BASE_RATES,
    verbose: bool = True,
) -> dict[str, dict]:
    """Identify and fetch base rates using web search.

    Args:
        questions: List of forecasting questions
        max_searches: Maximum number of web searches allowed
        model: Model to use
        verbose: Print progress messages

    Returns:
        Dict of base_rate_name -> {value, unit, as_of, source, description}
    """
    if verbose:
        print("Phase 0: Identifying base rates...")

    client = anthropic.Anthropic()

    # Format questions for prompt
    questions_text = "\n".join(f"- {q.text}" for q in questions)
    user_prompt = BASE_RATES_USER.format(questions=questions_text)

    if verbose:
        print(f"  Searching for base rates (max {max_searches} searches)...")

    try:
        # Build schema for structured outputs
        schema = make_strict_schema(BaseRatesResponse)

        response = client.messages.create(
            model=model.replace("anthropic/", ""),
            max_tokens=4096,
            system=BASE_RATES_SYSTEM,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_searches,
            }],
            messages=[{"role": "user", "content": user_prompt}],
            extra_headers={"anthropic-beta": STRUCTURED_OUTPUTS_BETA},
            extra_body={
                "output_format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        )

        # Extract text content - structured outputs guarantee valid JSON
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        # Parse JSON - structured outputs guarantee valid format
        result = json.loads(text_content) if text_content else {"base_rates": []}
        base_rates_list = result.get("base_rates", [])

        # Convert to dict format
        base_rates = {}
        for br in base_rates_list:
            name = br.get("name", "unknown")
            base_rates[name] = {
                "value": br.get("value"),
                "unit": br.get("unit", "unknown"),
                "as_of": br.get("as_of", date.today().isoformat()),
                "source": br.get("source", "web search"),
                "description": br.get("description", ""),
            }
            if verbose:
                print(f"  Found {name}: {br.get('value')} {br.get('unit', '')}")

        if verbose:
            print(f"  Fetched {len(base_rates)} base rates")

        return base_rates

    except anthropic.APIError as e:
        logger.error(f"API error fetching base rates: {e}")
        if verbose:
            print(f"  Warning: Could not fetch base rates: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching base rates: {e}")
        if verbose:
            print(f"  Warning: Could not fetch base rates: {e}")
        return {}


def format_base_rates_context(base_rates: dict[str, dict] | None) -> str:
    """Format base rates for injection into prompts.

    Args:
        base_rates: Dict of base_rate_name -> {value, unit, as_of, source}

    Returns:
        Formatted string for prompt injection
    """
    if not base_rates:
        return ""

    lines = ["Current reference values (for anchoring predictions):"]
    for name, data in base_rates.items():
        value = data.get("value", "?")
        unit = data.get("unit", "").replace("_", " ")
        as_of = data.get("as_of", "unknown date")
        source = data.get("source", "")

        # Truncate long source URLs
        if len(source) > 50:
            source = source[:47] + "..."

        lines.append(f"- {name.replace('_', ' ').title()}: {value} {unit} (as of {as_of}, {source})")

    return "\n".join(lines)


