#!/usr/bin/env python3
"""
Generate worked examples for few-shot training.
Uses LLM to explain why pairs are/aren't correlated.
"""

import json
import os
from pathlib import Path

import litellm

DATA_DIR = Path(__file__).parent / "data"

# Model for generating explanations
MODEL = os.environ.get("MODEL", "claude-sonnet-4-20250514")

EXAMPLE_TEMPLATE = """Question A: "{q_a}"
Question B: "{q_b}"

Market correlation: ρ = {rho:.2f} ({bucket})

Analysis: {analysis}

Implication for conditional forecasting:
{implication}"""


def generate_analysis(q_a: str, q_b: str, rho: float, bucket: str) -> tuple[str, str]:
    """Use LLM to generate analysis and implication."""
    prompt = f"""You are helping create training examples for teaching LLMs about correlation strength in forecasting.

Given these two prediction market questions:
- Question A: "{q_a}"
- Question B: "{q_b}"

Their market prices have correlation ρ = {rho:.2f}, which is "{bucket}".

Write two short sections:

1. ANALYSIS (2-3 sentences): Explain why these questions have this correlation level. What causal or informational link exists (or doesn't exist)?

2. IMPLICATION (3-4 bullet points): What does this mean for conditional forecasting? Include:
   - Whether P(A|B=YES) differs from P(A)
   - Expected update magnitude in percentage points
   - A concrete example (e.g., "If P(A) = 0.30 and B=YES, expect P(A|B=YES) ≈ ...")

Be concise and specific. Use the actual question content to ground your explanation."""

    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    text = response.choices[0].message.content

    # Parse sections
    if "ANALYSIS:" in text and "IMPLICATION:" in text:
        parts = text.split("IMPLICATION:")
        analysis = parts[0].replace("ANALYSIS:", "").strip()
        implication = parts[1].strip()
    else:
        # Fallback: split in half
        lines = text.strip().split("\n")
        mid = len(lines) // 2
        analysis = " ".join(lines[:mid])
        implication = "\n".join(lines[mid:])

    return analysis, implication


def sample_pairs_per_bucket(pairs: list[dict], n_per_bucket: int = 5) -> list[dict]:
    """Sample n pairs from each bucket, prioritizing clear examples."""
    buckets = {"independent": [], "weak": [], "moderate": [], "strong": []}

    for p in pairs:
        bucket = p.get("bucket")
        if bucket in buckets:
            buckets[bucket].append(p)

    sampled = []
    for bucket, bucket_pairs in buckets.items():
        # For independent: pick pairs with lowest |ρ|
        # For strong: pick pairs with highest |ρ|
        if bucket == "independent":
            sorted_pairs = sorted(bucket_pairs, key=lambda x: abs(x["rho"]))
        else:
            sorted_pairs = sorted(bucket_pairs, key=lambda x: abs(x["rho"]), reverse=True)

        sampled.extend(sorted_pairs[:n_per_bucket])

    return sampled


def main():
    # Load pairs
    pairs_path = DATA_DIR / "pairs.json"
    if not pairs_path.exists():
        print("Run compute_correlations.py first")
        return

    with open(pairs_path) as f:
        pairs = json.load(f)

    # Sample pairs
    sampled = sample_pairs_per_bucket(pairs, n_per_bucket=5)
    print(f"Generating examples for {len(sampled)} pairs...")

    examples = []
    for i, pair in enumerate(sampled):
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        rho = pair["rho"]
        bucket = pair["bucket"]

        if not q_a or not q_b:
            continue

        print(f"[{i+1}/{len(sampled)}] {bucket}: {q_a[:30]}... ↔ {q_b[:30]}...")

        try:
            analysis, implication = generate_analysis(q_a, q_b, rho, bucket)

            example = EXAMPLE_TEMPLATE.format(
                q_a=q_a,
                q_b=q_b,
                rho=rho,
                bucket=bucket,
                analysis=analysis,
                implication=implication,
            )

            examples.append({
                "pair": pair,
                "example_text": example,
            })
        except Exception as e:
            print(f"  Error: {e}")

    # Save examples
    output_path = DATA_DIR / "training_examples.json"
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\nSaved {len(examples)} examples to {output_path}")

    # Also save as plain text for easy inspection
    text_path = DATA_DIR / "training_examples.txt"
    with open(text_path, "w") as f:
        for ex in examples:
            f.write("=" * 80 + "\n")
            f.write(ex["example_text"] + "\n\n")

    print(f"Saved plain text to {text_path}")


if __name__ == "__main__":
    main()
