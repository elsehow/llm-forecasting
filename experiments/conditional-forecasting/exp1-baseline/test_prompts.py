#!/usr/bin/env python3
"""Test different prompt engineering approaches on a subset of pairs.

Tests 3 prompt variations against baseline:
1. "Justify first" - explicitly state relationship before probability
2. "Conservative prior" - note that most pairs are unrelated
3. "Direction check" - explicit guidance on update direction

Usage:
    uv run python experiments/fb-conditional/test_prompts.py
    uv run python experiments/fb-conditional/test_prompts.py --model claude-opus-4-5-20251101
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm

# Baseline prompt (same as run_experiment.py)
BASELINE_PROMPT = """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Respond with ONLY a number between 0 and 1 representing your probability estimate.
For example: 0.75

Your estimate:"""

# Variation 1: Justify first
JUSTIFY_FIRST_PROMPT = """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Before giving your probability, first answer briefly:
1. Is there a causal or logical relationship between the context and this question? (yes/no)
2. If yes, what direction? (makes question more likely / less likely / unclear)

Then on a new line, give ONLY a number between 0 and 1.

Your answer:"""

# Variation 2: Conservative prior
CONSERVATIVE_PROMPT = """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

IMPORTANT: Most question pairs have no meaningful causal relationship. Only adjust your probability from what you'd estimate without the context if there's a clear, direct causal or logical link.

Respond with ONLY a number between 0 and 1 representing your probability estimate.
For example: 0.75

Your estimate:"""

# Variation 3: Direction check
DIRECTION_CHECK_PROMPT = """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Reasoning check:
- If the context makes this question MORE likely to resolve YES, your estimate should be HIGHER than without context.
- If the context makes this question LESS likely to resolve YES, your estimate should be LOWER than without context.
- If the context is unrelated, your estimate should be similar to without context.

Respond with ONLY a number between 0 and 1 representing your probability estimate.
For example: 0.75

Your estimate:"""

PROMPTS = {
    "baseline": BASELINE_PROMPT,
    "justify_first": JUSTIFY_FIRST_PROMPT,
    "conservative": CONSERVATIVE_PROMPT,
    "direction_check": DIRECTION_CHECK_PROMPT,
}


async def elicit_probability(
    question: str,
    condition: str | None,
    prompt_template: str,
    model: str,
) -> float | None:
    """Elicit a probability using a specific prompt template."""
    if condition:
        condition_text = f"IMPORTANT CONTEXT: Assume the following is TRUE: {condition}\n\nGiven this assumption,"
    else:
        condition_text = ""

    prompt = prompt_template.format(condition=condition_text, question=question)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,  # More tokens for justify_first
        "temperature": 0.2,
    }

    response = await litellm.acompletion(**kwargs)
    content = response.choices[0].message.content
    if content is None:
        return None
    content = content.strip()

    # Extract probability - look for last decimal number (in case of justification text)
    try:
        matches = re.findall(r"0?\.\d+|1\.0|(?<!\d)[01](?!\d)", content)
        if matches:
            prob = float(matches[-1])  # Take last match
            return max(0.0, min(1.0, prob))
    except (ValueError, AttributeError):
        pass

    return None


async def run_pair_with_prompt(pair: dict, prompt_name: str, prompt_template: str, model: str) -> dict:
    """Run a single pair with a specific prompt."""
    text_a = pair["text_a"]
    text_b = pair["text_b"]
    resolution_a = pair["resolution_a"]
    resolution_b = pair["resolution_b"]

    p_a, p_a_given_b1, p_a_given_b0 = await asyncio.gather(
        elicit_probability(text_a, None, prompt_template, model),
        elicit_probability(text_a, f'"{text_b}" resolved YES.', prompt_template, model),
        elicit_probability(text_a, f'"{text_b}" resolved NO.', prompt_template, model),
    )

    if p_a is None or p_a_given_b1 is None or p_a_given_b0 is None:
        return {"prompt": prompt_name, "error": "Failed to elicit"}

    p_a_given_actual_b = p_a_given_b1 if resolution_b == 1.0 else p_a_given_b0
    brier_independence = (p_a - resolution_a) ** 2
    brier_conditional = (p_a_given_actual_b - resolution_a) ** 2
    improvement = brier_independence - brier_conditional
    sensitivity = abs(p_a_given_b1 - p_a_given_b0)

    return {
        "prompt": prompt_name,
        "category": pair.get("category"),
        "p_a": p_a,
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
        "improvement": improvement,
        "sensitivity": sensitivity,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--limit", type=int, default=15, help="Number of pairs to test")
    args = parser.parse_args()

    # Load pairs
    pairs_path = Path("experiments/fb-conditional/pairs.json")
    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])

    # Get a balanced subset: 5 strong, 5 weak, 5 none
    strong = [p for p in pairs if p.get("category") == "strong"][:5]
    weak = [p for p in pairs if p.get("category") == "weak"][:5]
    none = [p for p in pairs if p.get("category") == "none"][:5]
    test_pairs = strong + weak + none

    print(f"Testing {len(test_pairs)} pairs with {args.model}")
    print(f"Categories: {len(strong)} strong, {len(weak)} weak, {len(none)} none")
    print()

    results = {name: [] for name in PROMPTS.keys()}

    for i, pair in enumerate(test_pairs):
        cat = pair.get("category", "?")
        print(f"[{i+1}/{len(test_pairs)}] {cat}: {pair['id_a'][:8]}...")

        # Run all prompts in parallel for this pair
        tasks = [
            run_pair_with_prompt(pair, name, template, args.model)
            for name, template in PROMPTS.items()
        ]
        pair_results = await asyncio.gather(*tasks)

        for r in pair_results:
            if "error" not in r:
                results[r["prompt"]].append(r)

        # Print comparison for this pair
        for r in pair_results:
            if "error" not in r:
                print(f"    {r['prompt']:15} P(A)={r['p_a']:.2f} imp={r['improvement']:+.3f} sens={r['sensitivity']:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY PROMPT TYPE")
    print("=" * 70)

    for prompt_name in PROMPTS.keys():
        prompt_results = results[prompt_name]
        if not prompt_results:
            continue

        print(f"\n{prompt_name.upper()}:")

        for cat in ["strong", "weak", "none"]:
            cat_results = [r for r in prompt_results if r["category"] == cat]
            if not cat_results:
                continue
            n = len(cat_results)
            mean_imp = sum(r["improvement"] for r in cat_results) / n
            mean_sens = sum(r["sensitivity"] for r in cat_results) / n
            wins = sum(1 for r in cat_results if r["improvement"] > 0)
            print(f"  {cat:6} (n={n}): imp={mean_imp:+.4f}  sens={mean_sens:.3f}  wins={wins}/{n}")

    # Save results
    output = {
        "model": args.model,
        "num_pairs": len(test_pairs),
        "results": results,
    }
    output_path = Path(f"experiments/fb-conditional/results/prompt_test_{args.model.replace('/', '_')}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
