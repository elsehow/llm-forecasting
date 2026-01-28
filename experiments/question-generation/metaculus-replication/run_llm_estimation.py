#!/usr/bin/env python3
"""
Run LLM conditional estimation on Metaculus pairs.

For each (Q, X) pair, ask the LLM to estimate:
- P(X resolves YES | Q resolves YES)
- P(X resolves YES | Q resolves NO)

Then compute estimated_shift = |P(X|Q=Y) - P(X|Q=N)|

This tests whether LLMs can estimate logical conditional relationships.
"""

import asyncio
import json
from pathlib import Path
import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL = "anthropic/claude-opus-4-5-20251101"
RATE_LIMIT_DELAY = 0.5  # seconds between calls
DATA_PATH = Path(__file__).parent / "data" / "metaculus_pairs_v2.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "llm_estimations.json"

ESTIMATION_PROMPT = """You are estimating conditional probabilities between two forecasting questions.

Question Q: "{q_title}"
Question X: "{x_title}"

Assume Q has not yet resolved. Estimate:
1. P(X resolves YES | Q resolves YES)
2. P(X resolves YES | Q resolves NO)

Think about:
- Are these questions about related events?
- Would Q's outcome logically affect X's likelihood?
- What's the direction and magnitude of the relationship?

Respond with JSON only:
{{"p_x_given_q_yes": <float 0-1>, "p_x_given_q_no": <float 0-1>, "reasoning": "<brief explanation>"}}"""


async def estimate_conditionals(q_title: str, x_title: str) -> dict:
    prompt = ESTIMATION_PROMPT.format(q_title=q_title, x_title=x_title)

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return {
            "p_x_given_q_yes": float(result.get("p_x_given_q_yes", 0.5)),
            "p_x_given_q_no": float(result.get("p_x_given_q_no", 0.5)),
            "reasoning": result.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "p_x_given_q_yes": 0.5,
            "p_x_given_q_no": 0.5,
            "reasoning": "",
            "error": str(e),
        }


async def main():
    # Load pairs
    with open(DATA_PATH) as f:
        data = json.load(f)
    pairs = data["pairs"]

    print(f"Running LLM estimation on {len(pairs)} pairs...")
    print(f"Model: {MODEL}")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY}s")

    results = []
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  [{i+1}/{len(pairs)}]...")

        est = await estimate_conditionals(pair["q_title"], pair["x_title"])
        await asyncio.sleep(RATE_LIMIT_DELAY)

        # Compute estimated shift
        estimated_shift = abs(est["p_x_given_q_yes"] - est["p_x_given_q_no"])

        results.append({
            **pair,
            "p_x_given_q_yes": est["p_x_given_q_yes"],
            "p_x_given_q_no": est["p_x_given_q_no"],
            "estimated_shift": estimated_shift,
            "llm_reasoning": est["reasoning"],
            "llm_error": est["error"],
        })

    # Save
    output = {
        "metadata": {
            "model": MODEL,
            "n_pairs": len(results),
            "n_errors": sum(1 for r in results if r["llm_error"]),
        },
        "results": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Errors: {output['metadata']['n_errors']}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
