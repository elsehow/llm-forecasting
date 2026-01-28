#!/usr/bin/env python3
"""
Validate LLM conditional estimation against observed Metaculus Δp.

Primary metric: r(|estimated shift|, |observed ΔP|) > 0?

Success criteria:
- r > 0.15: LLMs can estimate conditionals for logical relationships
- Direction accuracy > 55%: LLMs get relationship sign right
- Improvement over Polymarket (r=0.07-0.16, direction 17-29%)
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "llm_estimations.json"


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r["llm_error"] is None]
    print(f"Valid results: {len(results)}")
    print(f"Errors excluded: {data['metadata']['n_errors']}")

    # Extract arrays
    estimated_shifts = np.array([r["estimated_shift"] for r in results])
    observed_dps = np.array([abs(r["x_delta_p"]) for r in results])

    # Primary metric: correlation between |estimated shift| and |observed ΔP|
    r, p_value = stats.spearmanr(estimated_shifts, observed_dps)
    print(f"\n{'='*60}")
    print("PRIMARY METRIC: Magnitude Correlation")
    print(f"{'='*60}")
    print(f"  r(|estimated shift|, |observed ΔP|) = {r:.3f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  n = {len(results)}")

    # Direction accuracy
    # If LLM says positive relationship (p_yes > p_no), ΔP should be positive when Q=YES
    correct = 0
    total = 0
    for result in results:
        # LLM's predicted direction: does Q=YES increase or decrease P(X)?
        llm_direction = 1 if result["p_x_given_q_yes"] > result["p_x_given_q_no"] else -1

        # Q resolved to 1 (YES) or 0 (NO)
        q_resolution = result.get("q_resolution", 1)

        # Expected ΔP sign based on LLM prediction and Q's resolution
        # If LLM says Q=YES increases X, and Q resolved YES, expect positive ΔP
        # If LLM says Q=YES increases X, and Q resolved NO, expect negative ΔP
        expected_dp_sign = llm_direction if q_resolution == 1 else -llm_direction
        actual_dp_sign = 1 if result["x_delta_p"] > 0 else -1

        if abs(result["x_delta_p"]) > 0.01:  # Only count if there was meaningful movement
            total += 1
            if expected_dp_sign == actual_dp_sign:
                correct += 1

    direction_accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print("SECONDARY METRIC: Direction Accuracy")
    print(f"{'='*60}")
    print(f"  Correct: {correct}/{total} = {direction_accuracy:.1%}")
    print(f"  (Only pairs with |ΔP| > 1%)")

    # Distribution stats
    print(f"\n{'='*60}")
    print("DISTRIBUTION STATS")
    print(f"{'='*60}")
    print(f"\n  Estimated shifts:")
    print(f"    min: {estimated_shifts.min():.3f}")
    print(f"    max: {estimated_shifts.max():.3f}")
    print(f"    mean: {estimated_shifts.mean():.3f}")
    print(f"    median: {np.median(estimated_shifts):.3f}")

    print(f"\n  Observed |ΔP|:")
    print(f"    min: {observed_dps.min():.3f}")
    print(f"    max: {observed_dps.max():.3f}")
    print(f"    mean: {observed_dps.mean():.3f}")
    print(f"    median: {np.median(observed_dps):.3f}")

    # Compare to Polymarket baseline
    print(f"\n{'='*60}")
    print("COMPARISON TO POLYMARKET BASELINE")
    print(f"{'='*60}")
    print(f"  Polymarket r: 0.07-0.16")
    print(f"  Polymarket direction: 17-29%")
    print(f"  Metaculus r: {r:.3f}")
    print(f"  Metaculus direction: {direction_accuracy:.1%}")

    # Assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT")
    print(f"{'='*60}")

    if r > 0.15 and direction_accuracy > 0.55:
        print("✓ SUCCESS: LLMs can estimate conditionals for logical relationships")
        print("  - Magnitude correlation exceeds threshold (r > 0.15)")
        print("  - Direction accuracy exceeds threshold (> 55%)")
        print("  - Validates hypothesis: LLMs estimate logical ρ, Metaculus moves on logical ρ")
    elif r > 0.10 or direction_accuracy > 0.50:
        print("~ MARGINAL: Some signal, but not strong")
        print("  - May need larger sample or cleaner pairs")
        print("  - Consider CivBench for cleaner causal ground truth")
    else:
        print("✗ FAILURE: LLMs don't predict Metaculus ΔP either")
        print("  - Either LLMs can't estimate conditionals")
        print("  - Or Metaculus community behavior is also noisy")
        print("  - Recommend CivBench world forking for cleaner ground truth")

    # Sample pairs for inspection
    print(f"\n{'='*60}")
    print("SAMPLE PAIRS (Highest estimated shift)")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda x: -x["estimated_shift"])[:5]
    for i, r in enumerate(sorted_results):
        print(f"\n  [{i+1}] Q: {r['q_title'][:60]}...")
        print(f"      X: {r['x_title'][:60]}...")
        print(f"      LLM: P(X|Q=Y)={r['p_x_given_q_yes']:.2f}, P(X|Q=N)={r['p_x_given_q_no']:.2f}")
        print(f"      Est. shift: {r['estimated_shift']:.2f}, Obs. ΔP: {r['x_delta_p']:+.2f}")
        print(f"      Reasoning: {r['llm_reasoning'][:80]}...")


if __name__ == "__main__":
    main()
