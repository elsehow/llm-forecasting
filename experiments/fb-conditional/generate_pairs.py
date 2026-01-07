#!/usr/bin/env python3
"""Generate candidate question pairs for the conditional forecasting experiment.

Finds pairs of resolved questions that may be correlated, using LLM-assisted
classification into:
- Strong positive controls (obvious causal/logical link)
- Weak positive controls (same domain, unclear relationship)
- Negative controls (random pairs, unrelated domains)

Usage:
    uv run python experiments/fb-conditional/generate_pairs.py
    uv run python experiments/fb-conditional/generate_pairs.py --db data/forecastbench.db
    uv run python experiments/fb-conditional/generate_pairs.py --output pairs.json
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from llm_forecasting.storage.sqlite import SQLiteStorage

# Only use prediction market sources - data sources have templated question text
VALID_SOURCES = {"manifold", "metaculus", "polymarket", "infer"}


async def get_resolved_binary_questions(storage: SQLiteStorage) -> list[dict]:
    """Get all resolved binary questions with their resolutions.

    Filters to prediction market sources only (manifold, metaculus, polymarket, infer).
    Skips data sources (acled, fred, yfinance) which have templated question text.
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
            resolved.append({
                "id": q.id,
                "source": q.source,
                "text": q.text,
                "resolution": resolution.value,
                "resolution_date": resolution.date.isoformat(),
            })

    return resolved


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
        default="experiments/fb-conditional/candidates.json",
        help="Output file for candidate pairs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max questions to consider for pairing",
    )
    args = parser.parse_args()

    storage = SQLiteStorage(args.db)

    print("Loading resolved binary questions...")
    questions = await get_resolved_binary_questions(storage)
    print(f"Found {len(questions)} resolved binary questions")

    if len(questions) < 2:
        print("Not enough resolved questions for pairing. Run migration first.")
        return

    # Limit for initial experimentation
    questions = questions[:args.limit]

    # TODO: LLM-assisted pair generation
    # For now, output the questions for manual review
    output = {
        "questions": questions,
        "pairs": [],  # To be populated by LLM or manual curation
        "metadata": {
            "generated_at": date.today().isoformat(),
            "question_count": len(questions),
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(questions)} questions to {output_path}")
    print("Next: Add LLM-assisted pair generation or manually curate pairs")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
