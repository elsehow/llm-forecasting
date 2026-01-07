#!/usr/bin/env python3
"""Generate candidate question pairs for the conditional forecasting experiment.

Uses LLM to find pairs of resolved questions that may be correlated, classifying into:
- strong: Obvious causal/logical link
- weak: Same domain, unclear relationship
- none: Unrelated (negative control)

Usage:
    uv run python experiments/fb-conditional/generate_pairs.py
    uv run python experiments/fb-conditional/generate_pairs.py --num-pairs 50
"""

import argparse
import asyncio
import json
import random
from datetime import date, datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import litellm

from llm_forecasting.storage.sqlite import SQLiteStorage

# Only use prediction market sources - data sources have templated question text
VALID_SOURCES = {"manifold", "metaculus", "polymarket", "infer"}

PAIR_FINDING_PROMPT = """You are helping design a forecasting experiment. Given a list of resolved prediction market questions, find pairs that might be correlated.

For each pair, classify the expected relationship:
- "strong": Obvious causal or logical link (e.g., "Russia invades Ukraine" + "NATO membership expands")
- "weak": Same domain but unclear if related (e.g., two AI questions that might be independent)
- "none": Clearly unrelated domains (e.g., "Bitcoin price" + "Lakers championship")

Return a JSON array of pairs. Each object should have:
- "id_a": first question ID
- "id_b": second question ID
- "category": one of "strong", "weak", or "none"
- "reason": brief explanation

Find approximately:
- 15 strong pairs (obvious correlation)
- 20 weak pairs (same domain, unclear)
- 15 none pairs (negative controls - deliberately unrelated)

QUESTIONS:
{questions}

Return ONLY the JSON array, no other text."""


async def get_resolved_binary_questions(
    storage: SQLiteStorage,
    min_resolution_date: date | None = None,
) -> list[dict]:
    """Get all resolved binary questions with their resolutions.

    Filters to prediction market sources only (manifold, metaculus, polymarket, infer).
    Skips data sources (acled, fred, yfinance) which have templated question text.

    Args:
        storage: SQLite storage instance
        min_resolution_date: If provided, only include questions that resolved on or after this date
    """
    questions = await storage.get_questions()

    resolved = []
    for q in questions:
        # Skip non-prediction-market sources
        if q.source not in VALID_SOURCES:
            continue

        if q.question_type.value != "binary":
            continue

        resolution = await storage.get_resolution(q.source, q.id)
        if resolution and resolution.value in (0.0, 1.0):
            # Filter by resolution date if specified
            if min_resolution_date and resolution.date < min_resolution_date:
                continue

            resolved.append({
                "id": q.id,
                "source": q.source,
                "text": q.text,
                "resolution": resolution.value,
                "resolution_date": resolution.date.isoformat(),
            })

    return resolved


async def find_pairs_with_llm(questions: list[dict], model: str = "claude-sonnet-4-20250514") -> list[dict]:
    """Use LLM to find correlated question pairs."""
    # Format questions for the prompt
    q_lines = []
    for q in questions:
        res = "YES" if q["resolution"] == 1.0 else "NO"
        text = q["text"][:200]
        q_lines.append(f"- [{q['id']}] ({q['source']}) {text}... → resolved {res}")
    q_list = "\n".join(q_lines)

    prompt = PAIR_FINDING_PROMPT.format(questions=q_list)

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content

    # Extract JSON from response
    try:
        # Try to find JSON array in response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            pairs = json.loads(content[start:end])
            return pairs
    except json.JSONDecodeError:
        pass

    return []


def add_source_info(pairs: list[dict], questions_by_id: dict) -> list[dict]:
    """Add source information to pairs."""
    enriched = []
    for p in pairs:
        q_a = questions_by_id.get(p["id_a"])
        q_b = questions_by_id.get(p["id_b"])
        if q_a and q_b:
            enriched.append({
                **p,
                "source_a": q_a["source"],
                "source_b": q_b["source"],
                "text_a": q_a["text"],
                "text_b": q_b["text"],
                "resolution_a": q_a["resolution"],
                "resolution_b": q_b["resolution"],
            })
    return enriched


async def main():
    parser = argparse.ArgumentParser(description="Generate question pairs for experiment")
    parser.add_argument(
        "--db",
        type=str,
        default="data/forecastbench.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/fb-conditional/pairs.json",
        help="Output file for pairs",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to consider for pairing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for pair finding",
    )
    parser.add_argument(
        "--min-resolution-date",
        type=str,
        default=None,
        help="Only include questions that resolved on or after this date (YYYY-MM-DD). "
             "Use to avoid memorization confound by filtering to dates after model knowledge cutoffs.",
    )
    args = parser.parse_args()

    # Parse min resolution date if provided
    min_resolution_date = None
    if args.min_resolution_date:
        min_resolution_date = date.fromisoformat(args.min_resolution_date)

    storage = SQLiteStorage(args.db)

    print("Loading resolved binary questions...")
    if min_resolution_date:
        print(f"Filtering to questions resolved on or after {min_resolution_date}")
    questions = await get_resolved_binary_questions(storage, min_resolution_date)
    print(f"Found {len(questions)} resolved binary questions from prediction markets")

    if len(questions) < 10:
        print("Not enough resolved questions. Run migration first.")
        await storage.close()
        return

    # Sample questions for pairing
    if len(questions) > args.num_questions:
        questions = random.sample(questions, args.num_questions)
        print(f"Sampled {args.num_questions} questions for pairing")

    # Build lookup
    questions_by_id = {q["id"]: q for q in questions}

    print(f"\nFinding pairs using {args.model}...")
    pairs = await find_pairs_with_llm(questions, args.model)
    print(f"Found {len(pairs)} candidate pairs")

    # Enrich with source info
    pairs = add_source_info(pairs, questions_by_id)
    print(f"Enriched {len(pairs)} pairs with metadata")

    # Count by category
    by_category = {}
    for p in pairs:
        cat = p.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    print(f"By category: {by_category}")

    # Save output
    output = {
        "pairs": pairs,
        "metadata": {
            "generated_at": date.today().isoformat(),
            "model": args.model,
            "num_questions_considered": len(questions),
            "num_pairs": len(pairs),
            "by_category": by_category,
            "min_resolution_date": min_resolution_date.isoformat() if min_resolution_date else None,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(pairs)} pairs to {output_path}")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
