#!/usr/bin/env python3
"""Run the conditional forecasting experiment.

For each question pair, elicits:
- P(A) - unconditional probability of A
- P(A|B=1) - probability of A given B resolved YES
- P(A|B=0) - probability of A given B resolved NO

Then scores against actual resolutions.

Usage:
    uv run python experiments/fb-conditional/run_experiment.py
    uv run python experiments/fb-conditional/run_experiment.py --pairs pairs.json
    uv run python experiments/fb-conditional/run_experiment.py --model claude-sonnet-4-20250514
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import litellm

PROMPTS = {
    "baseline": """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Respond with ONLY a number between 0 and 1 representing your probability estimate.
For example: 0.75

Your estimate:""",

    "justify_first": """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Before giving your probability, first answer briefly:
1. Is there a causal or logical relationship between the context and this question? (yes/no)
2. If yes, what direction? (makes question more likely / less likely / unclear)

Then on a new line, give ONLY a number between 0 and 1.

Your answer:""",

    "skeptical": """You are an expert forecaster. Give a probability estimate for the following question.

{condition}

Question: {question}

Before giving your probability, briefly answer:
1. Why might these questions be INDEPENDENT? (1-2 sentences)
2. Is there a DIRECT causal mechanism that overrides this? (1-2 sentences)

If no direct causal link, keep your estimate similar to what you'd give without the context.

End your response with EXACTLY this format on its own line:
PROBABILITY: [your number between 0 and 1]

Your answer:""",
}


async def elicit_probability(
    question: str,
    condition: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = False,
    prompt_style: str = "baseline",
) -> float | None:
    """Elicit a probability forecast from an LLM.

    Args:
        question: The forecasting question
        condition: Optional conditioning statement
        model: Model to use
        thinking: Enable extended thinking (Claude models only)
        prompt_style: Which prompt template to use

    Returns:
        Probability estimate (0-1) or None if parsing failed
    """
    if condition:
        condition_text = f"IMPORTANT CONTEXT: Assume the following is TRUE: {condition}\n\nGiven this assumption,"
    else:
        condition_text = ""

    prompt_template = PROMPTS.get(prompt_style, PROMPTS["baseline"])
    prompt = prompt_template.format(condition=condition_text, question=question)

    # Models with reasoning need more tokens (reasoning tokens + response tokens)
    # justify_first and skeptical prompts also need more tokens for the justification text
    if "gpt-5" in model or "gemini-3" in model:
        max_tokens = 500
    elif prompt_style == "skeptical":
        max_tokens = 500  # skeptical prompt elicits verbose reasoning
    elif prompt_style == "justify_first":
        max_tokens = 300
    else:
        max_tokens = 50

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    # Temperature settings vary by model
    if "gpt-5" in model:
        pass  # GPT-5 doesn't support temperature
    elif "gemini-3" in model:
        kwargs["temperature"] = 1.0  # Gemini 3 requires temperature=1.0
    else:
        kwargs["temperature"] = 0.2

    # Enable extended thinking for Claude models
    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["max_tokens"] = 4000  # Need more tokens to accommodate thinking + response
        del kwargs["temperature"]  # Temperature not supported with thinking

    # Retry with exponential backoff on transient errors
    for attempt in range(3):
        try:
            response = await litellm.acompletion(**kwargs)
            break
        except Exception as e:
            if attempt == 2 or "overload" not in str(e).lower():
                raise
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s

    # Extract content - try main content first, then reasoning_content for thinking models
    content = response.choices[0].message.content
    if content is None:
        # For Gemini 3 thinking mode, content might be in reasoning_content
        content = getattr(response.choices[0].message, 'reasoning_content', None)
    if content is None:
        return None
    content = content.strip()

    # Extract probability from response
    # First try to find PROBABILITY: format (used by skeptical prompt)
    prob_match = re.search(r"PROBABILITY:\s*(0?\.\d+|1\.0|[01](?:\.\d+)?)", content, re.IGNORECASE)
    if prob_match:
        try:
            prob = float(prob_match.group(1))
            return max(0.0, min(1.0, prob))
        except ValueError:
            pass

    # Fallback: take the last number (for justify_first and other prompts)
    try:
        matches = re.findall(r"0?\.\d+|1\.0|(?<!\d)[01](?!\d)", content)
        if matches:
            prob = float(matches[-1])  # Take last match
            return max(0.0, min(1.0, prob))
    except (ValueError, AttributeError):
        pass

    return None


async def run_pair(pair: dict, model: str, thinking: bool = False, prompt_style: str = "baseline") -> dict:
    """Run experiment on a single question pair.

    Elicits P(A), P(A|B=1), P(A|B=0) and computes scores.
    """
    text_a = pair["text_a"]
    text_b = pair["text_b"]
    resolution_a = pair["resolution_a"]
    resolution_b = pair["resolution_b"]

    # Elicit forecasts in parallel
    p_a, p_a_given_b1, p_a_given_b0 = await asyncio.gather(
        elicit_probability(text_a, condition=None, model=model, thinking=thinking, prompt_style=prompt_style),
        elicit_probability(text_a, condition=f'"{text_b}" resolved YES.', model=model, thinking=thinking, prompt_style=prompt_style),
        elicit_probability(text_a, condition=f'"{text_b}" resolved NO.', model=model, thinking=thinking, prompt_style=prompt_style),
    )

    if p_a is None or p_a_given_b1 is None or p_a_given_b0 is None:
        return {
            "pair_id": f"{pair['id_a']}_{pair['id_b']}",
            "error": "Failed to elicit probabilities",
        }

    # Select conditional forecast based on actual B resolution
    p_a_given_actual_b = p_a_given_b1 if resolution_b == 1.0 else p_a_given_b0

    # Compute scores
    brier_independence = (p_a - resolution_a) ** 2
    brier_conditional = (p_a_given_actual_b - resolution_a) ** 2
    improvement = brier_independence - brier_conditional
    sensitivity = abs(p_a_given_b1 - p_a_given_b0)

    return {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        # Elicited probabilities
        "p_a": p_a,
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
        "p_a_given_actual_b": p_a_given_actual_b,
        # Actual resolutions
        "resolution_a": resolution_a,
        "resolution_b": resolution_b,
        # Scores
        "brier_independence": brier_independence,
        "brier_conditional": brier_conditional,
        "improvement": improvement,  # positive = conditional wins
        "sensitivity": sensitivity,
        # Question text (for debugging)
        "text_a": text_a[:100],
        "text_b": text_b[:100],
    }


async def main():
    parser = argparse.ArgumentParser(description="Run conditional forecasting experiment")
    parser.add_argument(
        "--pairs",
        type=str,
        default="experiments/fb-conditional/pairs.json",
        help="Path to pairs file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,  # Auto-generate based on model and timestamp
        help="Output file for results (auto-generated if not specified)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for forecasting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs to run (for testing)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable extended thinking (Claude models only)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="baseline",
        choices=list(PROMPTS.keys()),
        help="Prompt style to use",
    )
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        print("Run generate_pairs.py first.")
        return

    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])
    if not pairs:
        print("No pairs found in file.")
        return

    if args.limit:
        pairs = pairs[:args.limit]

    # Auto-generate output filename if not specified
    if args.output is None:
        model_short = args.model.replace("/", "_").replace("anthropic_", "")
        thinking_suffix = "_thinking" if args.thinking else ""
        prompt_suffix = f"_{args.prompt}" if args.prompt != "baseline" else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"experiments/fb-conditional/results/{model_short}{thinking_suffix}{prompt_suffix}_{timestamp}.json"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running experiment on {len(pairs)} pairs using {args.model}")
    if args.thinking:
        print("Extended thinking: ENABLED")
    if args.prompt != "baseline":
        print(f"Prompt style: {args.prompt}")
    print(f"Results will be saved to {args.output}")
    print()

    # Run pairs in parallel with concurrency limit
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent pairs

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair['id_a'][:10]}... x {pair['id_b'][:10]}...")
            result = await run_pair(pair, args.model, thinking=args.thinking, prompt_style=args.prompt)
            if "error" not in result:
                print(f"    P(A)={result['p_a']:.2f}, P(A|B=1)={result['p_a_given_b1']:.2f}, P(A|B=0)={result['p_a_given_b0']:.2f}")
                print(f"    Improvement: {result['improvement']:+.4f}, Sensitivity: {result['sensitivity']:.2f}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Compute summary statistics by category
    print("\n" + "=" * 60)
    print("SUMMARY BY CATEGORY")
    print("=" * 60)

    categories = {}
    for r in results:
        if "error" in r:
            continue
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cat_results in sorted(categories.items()):
        n = len(cat_results)
        mean_improvement = sum(r["improvement"] for r in cat_results) / n
        mean_sensitivity = sum(r["sensitivity"] for r in cat_results) / n
        wins = sum(1 for r in cat_results if r["improvement"] > 0)

        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Mean improvement:  {mean_improvement:+.4f}")
        print(f"  Mean sensitivity:  {mean_sensitivity:.3f}")
        print(f"  Conditional wins:  {wins}/{n} ({100*wins/n:.0f}%)")

    # Save results
    output = {
        "results": results,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "model": args.model,
            "thinking": args.thinking,
            "prompt_style": args.prompt,
            "num_pairs": len(pairs),
            "num_successful": len([r for r in results if "error" not in r]),
        },
        "summary": {
            cat: {
                "n": len(cat_results),
                "mean_improvement": sum(r["improvement"] for r in cat_results) / len(cat_results),
                "mean_sensitivity": sum(r["sensitivity"] for r in cat_results) / len(cat_results),
                "conditional_win_rate": sum(1 for r in cat_results if r["improvement"] > 0) / len(cat_results),
            }
            for cat, cat_results in categories.items()
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
