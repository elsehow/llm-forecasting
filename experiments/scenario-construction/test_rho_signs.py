#!/usr/bin/env python3
"""Test rho sign accuracy with different approaches.

Compares:
1. Current rho estimation (from results file)
2. Simple direction check: "If B=YES, is A more or less likely?"
3. Improved prompt with examples

Usage:
    uv run python experiments/scenario-construction/test_rho_signs.py
"""

import asyncio
import json
from pathlib import Path

import litellm
from pydantic import BaseModel


# Test file
RESULTS_FILE = Path(__file__).parent / "results/democrat_whitehouse_2028/dual_v7_20260116_172017.json"
TARGET = "Will a Democrat win the White House in 2028?"

MODEL = "claude-sonnet-4-20250514"


class DirectionCheck(BaseModel):
    """Simple direction check response."""
    direction: str  # "more_likely", "less_likely", or "no_effect"
    reasoning: str


class RhoEstimate(BaseModel):
    """Rho estimation response."""
    rho: float
    reasoning: str


# Prompt 1: Simple direction check
DIRECTION_PROMPT = """Question A: "{target}"
Question B: "{signal}"

If Question B resolves YES, does that make Question A MORE likely or LESS likely to also be YES?

Think step by step:
1. What does B=YES mean concretely?
2. How would that affect the probability of A=YES?

Respond with JSON:
{{"direction": "more_likely" or "less_likely" or "no_effect", "reasoning": "<brief explanation>"}}"""


# Prompt 2: Current rho prompt (from voi.py)
CURRENT_RHO_PROMPT = """You are estimating the correlation between two prediction market questions.

Question A: "{target}"
Question B: "{signal}"

Estimate the correlation coefficient (ρ) between these two questions. This measures how much knowing the outcome of one question tells you about the other:
- ρ = +1: Perfect positive correlation (if A is YES, B is definitely YES)
- ρ = 0: Independent (knowing A tells you nothing about B)
- ρ = -1: Perfect negative correlation (if A is YES, B is definitely NO)

Think about:
- Are these questions about related events?
- Would one outcome make the other more or less likely?
- Are they measuring the same underlying phenomenon?

Respond with JSON only:
{{"rho": <float from -1 to +1>, "reasoning": "<brief explanation>"}}"""


# Prompt 3: Improved prompt with examples
IMPROVED_RHO_PROMPT = """You are estimating the correlation between two prediction market questions.

Question A: "{target}"
Question B: "{signal}"

Estimate the correlation coefficient (ρ) between these two questions:
- ρ = +1: Perfect positive correlation (B=YES makes A=YES more likely)
- ρ = 0: Independent (B outcome doesn't affect A probability)
- ρ = -1: Perfect negative correlation (B=YES makes A=YES less likely)

EXAMPLES to calibrate your thinking:
- "Republican wins 2024" vs "Democrat wins 2024": ρ ≈ -0.95 (mutually exclusive)
- "Strong economy in 2024" vs "Incumbent party wins 2024": ρ ≈ +0.5 (both go up together)
- "AI breakthrough in 2025" vs "Tech stocks rise 2025": ρ ≈ +0.4 (positive relationship)
- "Trade war escalates" vs "Global GDP growth": ρ ≈ -0.3 (one up, other down)

CRITICAL: If B=YES makes A LESS likely, ρ must be NEGATIVE.

Think step by step:
1. If B resolves YES, does A become more likely (+) or less likely (-)?
2. How strong is this relationship? (weak: 0.1-0.3, moderate: 0.3-0.6, strong: 0.6-0.9)

Respond with JSON only:
{{"rho": <float from -1 to +1>, "reasoning": "<brief explanation>"}}"""


async def check_direction(signal: str, target: str) -> tuple[str, str]:
    """Ask simple direction question."""
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": DIRECTION_PROMPT.format(target=target, signal=signal)}],
        response_format=DirectionCheck,
    )
    result = DirectionCheck.model_validate_json(response.choices[0].message.content)
    return result.direction, result.reasoning


async def estimate_rho_current(signal: str, target: str) -> tuple[float, str]:
    """Estimate rho with current prompt."""
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": CURRENT_RHO_PROMPT.format(target=target, signal=signal)}],
        response_format=RhoEstimate,
    )
    result = RhoEstimate.model_validate_json(response.choices[0].message.content)
    return result.rho, result.reasoning


async def estimate_rho_improved(signal: str, target: str) -> tuple[float, str]:
    """Estimate rho with improved prompt."""
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": IMPROVED_RHO_PROMPT.format(target=target, signal=signal)}],
        response_format=RhoEstimate,
    )
    result = RhoEstimate.model_validate_json(response.choices[0].message.content)
    return result.rho, result.reasoning


def direction_to_sign(direction: str) -> int:
    """Convert direction to expected sign."""
    if direction == "more_likely":
        return 1
    elif direction == "less_likely":
        return -1
    return 0


def rho_to_sign(rho: float) -> int:
    """Convert rho to sign (with small tolerance)."""
    if rho > 0.05:
        return 1
    elif rho < -0.05:
        return -1
    return 0


async def test_single_signal(signal_data: dict, target: str) -> dict:
    """Test all approaches on a single signal."""
    signal_text = signal_data["text"]
    original_rho = signal_data["rho"]

    # Run all checks in parallel
    direction_result, current_result, improved_result = await asyncio.gather(
        check_direction(signal_text, target),
        estimate_rho_current(signal_text, target),
        estimate_rho_improved(signal_text, target),
    )

    direction, direction_reasoning = direction_result
    current_rho, current_reasoning = current_result
    improved_rho, improved_reasoning = improved_result

    # Compare signs
    expected_sign = direction_to_sign(direction)
    original_sign = rho_to_sign(original_rho)
    current_sign = rho_to_sign(current_rho)
    improved_sign = rho_to_sign(improved_rho)

    return {
        "signal": signal_text[:60] + "..." if len(signal_text) > 60 else signal_text,
        "direction": direction,
        "direction_reasoning": direction_reasoning,
        "expected_sign": expected_sign,
        "original_rho": original_rho,
        "original_sign": original_sign,
        "original_match": original_sign == expected_sign or expected_sign == 0,
        "current_rho": current_rho,
        "current_sign": current_sign,
        "current_match": current_sign == expected_sign or expected_sign == 0,
        "improved_rho": improved_rho,
        "improved_sign": improved_sign,
        "improved_match": improved_sign == expected_sign or expected_sign == 0,
    }


async def main():
    # Load signals
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    signals = [s for s in data["signals"] if s.get("rho") is not None]

    # Test first 10 signals (to save API costs)
    test_signals = signals[:10]

    print(f"Testing {len(test_signals)} signals...")
    print("=" * 80)

    results = []
    for i, signal in enumerate(test_signals):
        print(f"\n[{i+1}/{len(test_signals)}] {signal['text'][:50]}...")
        result = await test_single_signal(signal, TARGET)
        results.append(result)

        # Print quick summary
        dir_emoji = "↑" if result["expected_sign"] > 0 else "↓" if result["expected_sign"] < 0 else "–"
        orig_ok = "✓" if result["original_match"] else "✗"
        curr_ok = "✓" if result["current_match"] else "✗"
        impr_ok = "✓" if result["improved_match"] else "✗"

        print(f"  Direction: {result['direction']} ({dir_emoji})")
        print(f"  Original rho={result['original_rho']:+.2f} {orig_ok}")
        print(f"  Current rho={result['current_rho']:+.2f} {curr_ok}")
        print(f"  Improved rho={result['improved_rho']:+.2f} {impr_ok}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = len(results)
    original_correct = sum(1 for r in results if r["original_match"])
    current_correct = sum(1 for r in results if r["current_match"])
    improved_correct = sum(1 for r in results if r["improved_match"])

    print(f"Original rho (from file):  {original_correct}/{n} correct ({100*original_correct/n:.0f}%)")
    print(f"Current prompt:            {current_correct}/{n} correct ({100*current_correct/n:.0f}%)")
    print(f"Improved prompt:           {improved_correct}/{n} correct ({100*improved_correct/n:.0f}%)")

    # Show disagreements
    print("\n" + "-" * 80)
    print("SIGN DISAGREEMENTS (direction vs rho):")
    print("-" * 80)

    for r in results:
        if not r["original_match"] or not r["current_match"] or not r["improved_match"]:
            print(f"\n{r['signal']}")
            print(f"  Direction: {r['direction']} → expected sign: {r['expected_sign']}")
            print(f"  Reasoning: {r['direction_reasoning'][:100]}...")
            if not r["original_match"]:
                print(f"  ✗ Original: {r['original_rho']:+.2f}")
            if not r["current_match"]:
                print(f"  ✗ Current:  {r['current_rho']:+.2f}")
            if not r["improved_match"]:
                print(f"  ✗ Improved: {r['improved_rho']:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())
