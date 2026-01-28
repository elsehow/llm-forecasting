#!/usr/bin/env python3
"""
Step 2: Generate cruxes via LLM for each ultimate market.

For each ultimate, prompts the LLM for 3-5 binary questions that would
most change the forecast if resolved.

Usage:
    uv run python experiments/question-generation/llm-crux-validation/generate_cruxes.py
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import litellm

load_dotenv()

# Paths
RESULTS_DIR = Path(__file__).parent / "results"

# Model config
MODEL = "claude-sonnet-4-20250514"

CRUX_GENERATION_PROMPT = """You are forecasting: "{question}"
Current probability: {current_price:.0%}
End date: {end_date}

What 3-5 binary questions, if resolved, would most change your forecast?

Focus on questions that:
- Could resolve before {end_date}
- Have clear YES/NO resolution criteria
- Aren't just restating the ultimate question
- Would cause a meaningful probability shift (>5%) if resolved

Think about:
- Key prerequisites or blockers
- Causal drivers of the outcome
- Information that would dramatically update the probability
- Near-term events that signal the direction

Return JSON only:
[
  {{"crux": "<binary question text>", "direction": "+/-", "magnitude": "high/medium/low", "rationale": "<why this matters>"}},
  ...
]

Where:
- "crux" is a clear YES/NO question
- "direction" is "+" if YES increases the ultimate's probability, "-" if it decreases
- "magnitude" is the expected impact: "high" (>20% shift), "medium" (10-20%), "low" (5-10%)
"""


async def generate_cruxes_for_ultimate(ultimate: dict) -> dict:
    """Generate cruxes for a single ultimate market."""
    prompt = CRUX_GENERATION_PROMPT.format(
        question=ultimate["question"],
        current_price=ultimate["current_price"],
        end_date=ultimate["end_date"][:10],  # Just the date part
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,  # Slight temperature for diversity
        )
        text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        cruxes = json.loads(text)

        return {
            "ultimate_id": ultimate["condition_id"],
            "ultimate_question": ultimate["question"],
            "cruxes": cruxes,
            "error": None,
        }

    except Exception as e:
        return {
            "ultimate_id": ultimate["condition_id"],
            "ultimate_question": ultimate["question"],
            "cruxes": [],
            "error": str(e),
        }


async def main():
    print("=" * 70)
    print("GENERATE CRUXES VIA LLM")
    print("=" * 70)

    # Load ultimates
    ultimates_path = RESULTS_DIR / "ultimates.json"
    if not ultimates_path.exists():
        print(f"\n❌ {ultimates_path} not found. Run select_ultimates.py first.")
        return

    with open(ultimates_path) as f:
        data = json.load(f)
    ultimates = data["ultimates"]
    print(f"\nLoaded {len(ultimates)} ultimates")

    # Generate cruxes
    print(f"\nGenerating cruxes using {MODEL}...")
    results = []
    batch_size = 5

    for i in range(0, len(ultimates), batch_size):
        batch = ultimates[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            generate_cruxes_for_ultimate(ult)
            for ult in batch
        ])
        results.extend(batch_results)
        print(f"  Processed {min(i+batch_size, len(ultimates))}/{len(ultimates)}")

    # Stats
    n_success = sum(1 for r in results if r["cruxes"])
    n_error = sum(1 for r in results if r["error"])
    total_cruxes = sum(len(r["cruxes"]) for r in results)

    print(f"\n✅ Generated cruxes for {n_success}/{len(ultimates)} ultimates")
    print(f"   Total cruxes: {total_cruxes}")
    if n_error:
        print(f"   ⚠️ Errors: {n_error}")

    # Magnitude distribution
    magnitudes = []
    for r in results:
        for c in r["cruxes"]:
            magnitudes.append(c.get("magnitude", "unknown"))

    from collections import Counter
    mag_counts = Counter(magnitudes)
    print(f"\nMagnitude distribution:")
    for mag, count in mag_counts.most_common():
        print(f"  {mag}: {count}")

    # Direction distribution
    directions = []
    for r in results:
        for c in r["cruxes"]:
            directions.append(c.get("direction", "?"))

    dir_counts = Counter(directions)
    print(f"\nDirection distribution:")
    for d, count in dir_counts.most_common():
        print(f"  {d}: {count}")

    # Show samples
    print(f"\nSample cruxes:")
    for i, r in enumerate(results[:3]):
        if r["cruxes"]:
            print(f"\n{i+1}. {r['ultimate_question'][:60]}...")
            for j, c in enumerate(r["cruxes"][:3]):
                print(f"   {j+1}. [{c.get('direction', '?')}{c.get('magnitude', '?')}] {c['crux'][:55]}...")

    # Save
    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "n_ultimates": len(ultimates),
            "n_successful": n_success,
            "total_cruxes": total_cruxes,
            "magnitude_distribution": dict(mag_counts),
            "direction_distribution": dict(dir_counts),
        },
        "cruxes": results,
    }

    output_path = RESULTS_DIR / "cruxes.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
