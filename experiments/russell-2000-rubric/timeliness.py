"""Timeliness evaluator for crux filtering.

A crux is TIMELY if it could plausibly resolve on or around the given date.
This filters out "moonshot cruxes" - questions about events that are unlikely
to occur on any given day but would be high-impact IF they happened.
"""

import asyncio
from typing import Optional

import litellm

TIMELINESS_MODEL = "anthropic/claude-3-haiku-20240307"

TIMELINESS_PROMPT = """You are evaluating whether a forecasting question is TIMELY for a specific date.

A question is TIMELY if:
- It's about something that could plausibly resolve on or around the given date
- There's a scheduled event, announcement, or decision that would answer it
- The base rate of the event happening is non-trivial (>10%)

A question is NOT TIMELY if:
- It's about a hypothetical event with no scheduled occurrence
- It's a "moonshot" - something that would be big news IF it happened, but is unlikely on this specific date
- P(event occurs on this specific date) is very low

Examples:
- TIMELY: "Will AAON's Q4 earnings beat expectations?" on earnings day
  → YES: Earnings are scheduled, will definitely resolve
- NOT TIMELY: "Will FDA approve PTCT's drug?" on a random date
  → NO: FDA decisions are rare events, unlikely on any specific date
- NOT TIMELY: "Will a major manufacturer announce orders?" on a random date
  → NO: Major announcements are rare, low base rate
- TIMELY: "Will the Fed announce a rate cut?" on FOMC meeting day
  → YES: Fed decision is scheduled
- NOT TIMELY: "Will CEO announce resignation?" on a random date
  → NO: CEO departures are rare, low base rate

Question: {crux}
Date: {date}
Context: {context}

Is this question TIMELY for this date? Answer YES or NO, then explain briefly in one sentence."""


async def evaluate_timeliness(
    crux: str,
    date: str,
    context: str = "",
) -> dict:
    """Evaluate whether a crux is timely for a given date.

    Args:
        crux: The crux question
        date: Date string (YYYY-MM-DD)
        context: Additional context (e.g., "This is AAON's earnings day")

    Returns:
        {
            "timely": bool,
            "reason": str,
            "raw_response": str
        }
    """
    prompt = TIMELINESS_PROMPT.format(
        crux=crux,
        date=date,
        context=context if context else "No special context for this date.",
    )

    try:
        response = await litellm.acompletion(
            model=TIMELINESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,  # Deterministic for reproducibility
        )

        raw = response.choices[0].message.content.strip()

        # Parse YES/NO from start of response
        first_word = raw.split()[0].upper().rstrip(".,:")
        timely = first_word == "YES"

        # Extract reason (everything after YES/NO)
        reason = raw.split(maxsplit=1)[1] if len(raw.split()) > 1 else ""
        reason = reason.lstrip(".,: ")

        return {
            "timely": timely,
            "reason": reason,
            "raw_response": raw,
        }

    except Exception as e:
        return {
            "timely": False,
            "reason": f"Error: {e}",
            "raw_response": "",
        }


async def evaluate_timeliness_batch(
    cruxes: list[dict],
    batch_size: int = 20,
) -> list[dict]:
    """Evaluate timeliness for a batch of cruxes.

    Args:
        cruxes: List of dicts with 'crux', 'date', and optional 'context' keys
        batch_size: Number of concurrent requests

    Returns:
        List of crux dicts with timeliness results added
    """
    results = []
    total = len(cruxes)

    for i in range(0, total, batch_size):
        batch = cruxes[i:i+batch_size]
        print(f"  Evaluating timeliness {i+1}-{min(i+batch_size, total)} of {total}...")

        tasks = [
            evaluate_timeliness(
                c["crux"],
                c["date"],
                c.get("context", ""),
            )
            for c in batch
        ]

        timeliness_results = await asyncio.gather(*tasks)

        for crux_data, tl in zip(batch, timeliness_results):
            results.append({
                **crux_data,
                "timely": tl["timely"],
                "timeliness_reason": tl["reason"],
            })

        # Small delay to avoid rate limits
        await asyncio.sleep(0.3)

    return results


# Quick test
if __name__ == "__main__":
    async def test():
        # Test cases
        tests = [
            {
                "crux": "Will AAON's Q4 earnings beat analyst expectations?",
                "date": "2025-11-06",
                "context": "This is AAON's earnings release day.",
                "expected": True,
            },
            {
                "crux": "Will FDA approve PTCT's new drug application?",
                "date": "2025-12-12",
                "context": "",
                "expected": False,
            },
            {
                "crux": "Will a major manufacturer announce significant orders?",
                "date": "2025-11-15",
                "context": "",
                "expected": False,
            },
            {
                "crux": "Will the CEO announce a major acquisition?",
                "date": "2025-11-10",
                "context": "",
                "expected": False,
            },
        ]

        print("Testing timeliness evaluator:")
        for t in tests:
            result = await evaluate_timeliness(t["crux"], t["date"], t["context"])
            match = "✓" if result["timely"] == t["expected"] else "✗"
            print(f"\n{match} Crux: {t['crux'][:60]}...")
            print(f"  Date: {t['date']}, Context: {t['context'][:40] or '(none)'}")
            print(f"  Expected: {t['expected']}, Got: {result['timely']}")
            print(f"  Reason: {result['reason'][:80]}")

    asyncio.run(test())
