#!/usr/bin/env python3
"""Analyze conditional reasoning without depending on resolution luck.

Two resolution-independent metrics:
1. Direction consistency: Does P(A|B=1) > P(A) > P(A|B=0) or the reverse?
   (Monotonic update in a consistent direction)
2. Sensitivity: Does the model update at all? (already computed)

For Bayesian consistency, we'd need P(B) and P(B|A) which requires
a symmetric experiment design. This script flags that as future work.
"""

import json
import sys
from pathlib import Path


def analyze_direction(results_path: str):
    """Analyze direction of conditional updates."""
    with open(results_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if "error" not in r]

    print("=" * 70)
    print("DIRECTION OF UPDATE ANALYSIS")
    print("=" * 70)
    print()
    print("Key insight: This metric doesn't depend on how A resolved.")
    print("We check if the model updates P(A) monotonically when conditioning on B.")
    print()

    by_category = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n = len(cat_results)

        # Metrics
        monotonic_up = 0      # P(A|B=1) > P(A) > P(A|B=0)
        monotonic_down = 0    # P(A|B=1) < P(A) < P(A|B=0)
        partial_update = 0    # Updates one direction but not monotonic
        no_update = 0         # P(A|B=1) ≈ P(A) ≈ P(A|B=0)
        inconsistent = 0      # Updates in contradictory ways

        details = []

        for r in cat_results:
            p_a = r["p_a"]
            p_b1 = r["p_a_given_b1"]
            p_b0 = r["p_a_given_b0"]

            # Tolerance for "no change"
            eps = 0.02

            delta_b1 = p_b1 - p_a  # Change when B=1
            delta_b0 = p_b0 - p_a  # Change when B=0

            if abs(delta_b1) < eps and abs(delta_b0) < eps:
                status = "no_update"
                no_update += 1
            elif delta_b1 > eps and delta_b0 < -eps:
                # B=1 increases P(A), B=0 decreases P(A) → positive correlation
                if p_b1 > p_a > p_b0:
                    status = "monotonic_up"
                    monotonic_up += 1
                else:
                    status = "partial"
                    partial_update += 1
            elif delta_b1 < -eps and delta_b0 > eps:
                # B=1 decreases P(A), B=0 increases P(A) → negative correlation
                if p_b1 < p_a < p_b0:
                    status = "monotonic_down"
                    monotonic_down += 1
                else:
                    status = "partial"
                    partial_update += 1
            elif (delta_b1 > eps and delta_b0 > eps) or (delta_b1 < -eps and delta_b0 < -eps):
                # Both conditions move P(A) in same direction - inconsistent!
                status = "inconsistent"
                inconsistent += 1
            else:
                # One side updates, other doesn't
                status = "partial"
                partial_update += 1

            details.append({
                "text_a": r["text_a"][:50],
                "text_b": r["text_b"][:50],
                "p_a": p_a,
                "p_b1": p_b1,
                "p_b0": p_b0,
                "delta_b1": delta_b1,
                "delta_b0": delta_b0,
                "status": status,
            })

        consistent = monotonic_up + monotonic_down
        print(f"\n{cat.upper()} (n={n}):")
        print(f"  Monotonic (correct direction):  {consistent}/{n} ({100*consistent/n:.0f}%)")
        print(f"    - Positive correlation:       {monotonic_up}")
        print(f"    - Negative correlation:       {monotonic_down}")
        print(f"  Partial update:                 {partial_update}/{n} ({100*partial_update/n:.0f}%)")
        print(f"  No update (treats as indep):    {no_update}/{n} ({100*no_update/n:.0f}%)")
        print(f"  Inconsistent (same direction):  {inconsistent}/{n} ({100*inconsistent/n:.0f}%)")

        # For "none" pairs, no_update is GOOD
        if cat == "none":
            correct_for_none = no_update
            print(f"  → Correct behavior (no update): {correct_for_none}/{n} ({100*correct_for_none/n:.0f}%)")

        # Show examples
        print(f"\n  Examples:")
        for d in details[:3]:
            print(f"    [{d['status']}] P(A)={d['p_a']:.2f}, P(A|B=1)={d['p_b1']:.2f}, P(A|B=0)={d['p_b0']:.2f}")
            print(f"         {d['text_a']}...")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
For STRONG pairs:
  - High "monotonic" rate = model correctly identifies correlation direction
  - "Inconsistent" = model hallucinates weird relationships
  - "No update" = model fails to recognize obvious correlations

For NONE pairs:
  - High "no update" rate = model correctly ignores unrelated questions
  - High "monotonic" rate = model hallucinates correlations (bad!)
  - "Inconsistent" = really confused

This metric is RESOLUTION-INDEPENDENT. It measures whether the model
understands the causal/correlational structure, not whether it got lucky
on this particular realization.
""")


def analyze_consistency(results_path: str):
    """Check law of total probability: P(A) ≈ P(A|B=1)*P(B) + P(A|B=0)*(1-P(B))

    We don't have P(B), but we can check if there EXISTS a P(B) that makes this work.
    Solving: P(A) = P(A|B=1)*x + P(A|B=0)*(1-x) for x

    x = (P(A) - P(A|B=0)) / (P(A|B=1) - P(A|B=0))

    If x ∈ [0,1], the estimates are consistent. If x < 0 or x > 1, impossible.
    """
    with open(results_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if "error" not in r]

    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK (Law of Total Probability)")
    print("=" * 70)
    print()
    print("For each pair, we solve for implied P(B) that would make")
    print("P(A) = P(A|B=1)*P(B) + P(A|B=0)*(1-P(B))")
    print("If implied P(B) is outside [0,1], the conditionals are inconsistent.")
    print()

    by_category = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n = len(cat_results)

        consistent = 0
        impossible = 0
        trivial = 0  # P(A|B=1) = P(A|B=0), so any P(B) works (or none if P(A) differs)

        implied_pbs = []

        for r in cat_results:
            p_a = r["p_a"]
            p_b1 = r["p_a_given_b1"]
            p_b0 = r["p_a_given_b0"]

            denom = p_b1 - p_b0
            if abs(denom) < 0.001:
                # P(A|B=1) ≈ P(A|B=0) - trivial case
                if abs(p_a - p_b1) < 0.05:
                    trivial += 1
                    consistent += 1
                else:
                    impossible += 1
                continue

            implied_pb = (p_a - p_b0) / denom

            if -0.05 <= implied_pb <= 1.05:  # Allow small tolerance
                consistent += 1
                implied_pbs.append(max(0, min(1, implied_pb)))
            else:
                impossible += 1

        print(f"{cat.upper()} (n={n}):")
        print(f"  Consistent (implied P(B) ∈ [0,1]):  {consistent}/{n} ({100*consistent/n:.0f}%)")
        print(f"  Impossible (implied P(B) outside):  {impossible}/{n} ({100*impossible/n:.0f}%)")
        if implied_pbs:
            print(f"  Mean implied P(B): {sum(implied_pbs)/len(implied_pbs):.2f}")


def main():
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Default to most recent results
        results_dir = Path("experiments/fb-conditional/results")
        if results_dir.exists():
            files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                results_path = str(files[0])
            else:
                results_path = "experiments/fb-conditional/results.json"
        else:
            results_path = "experiments/fb-conditional/results.json"

    print(f"Analyzing: {results_path}\n")
    analyze_direction(results_path)
    analyze_consistency(results_path)


if __name__ == "__main__":
    main()
