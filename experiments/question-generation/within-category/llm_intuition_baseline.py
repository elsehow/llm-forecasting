#!/usr/bin/env python3
"""
LLM Intuition Baseline vs VOI.

Tests whether a simple "how useful is this question?" prompt predicts
actual market shifts as well as the VOI framework.

If LLM intuition ≈ VOI → VOI framework doesn't add value
If VOI > LLM intuition → mathematical framework contributes signal
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from scipy import stats
import numpy as np
from dotenv import load_dotenv
import litellm

load_dotenv()

# Paths
INPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Config
MODEL = "anthropic/claude-3-haiku-20240307"
CONCURRENCY = 10  # Parallel requests

INTUITION_PROMPT = """Given these two prediction market questions:
- Question A (resolved): "{resolved_question}"
- Question B (still open): "{other_question}"

How useful would knowing the answer to Question A be for updating your belief about Question B?

Rate from 0 to 10:
- 0 = completely independent, knowing A tells you nothing about B
- 5 = moderately useful, would update B somewhat
- 10 = extremely useful, would dramatically change your B prediction

Respond with just a number."""


async def get_intuition_score(resolved_q: str, other_q: str, semaphore: asyncio.Semaphore) -> float | None:
    """Get LLM intuition score for a pair."""
    async with semaphore:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": INTUITION_PROMPT.format(
                        resolved_question=resolved_q,
                        other_question=other_q
                    )
                }],
                max_tokens=10,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            # Parse number
            score = float(text.split()[0].replace(",", ""))
            return min(max(score, 0), 10)  # Clamp to 0-10
        except Exception as e:
            print(f"  Error: {e}")
            return None


async def main():
    print("=" * 70)
    print("LLM INTUITION BASELINE VS VOI")
    print("=" * 70)

    # Load validated pairs
    print("\n[1/3] Loading validated pairs...")
    input_path = INPUT_DIR / "within_vs_cross_validation.json"
    with open(input_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    print(f"      Loaded {len(pairs)} pairs")

    # Get intuition scores
    print(f"\n[2/3] Getting LLM intuition scores (model: {MODEL})...")
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def process_pair(i: int, pair: dict) -> dict:
        score = await get_intuition_score(
            pair["resolved_question"],
            pair["other_question"],
            semaphore
        )
        if (i + 1) % 20 == 0:
            print(f"      Processed {i + 1}/{len(pairs)}")
        return {**pair, "llm_intuition": score}

    tasks = [process_pair(i, p) for i, p in enumerate(pairs)]
    results = await asyncio.gather(*tasks)

    # Filter out failures
    valid_results = [r for r in results if r["llm_intuition"] is not None]
    print(f"      Valid scores: {len(valid_results)}/{len(pairs)}")

    # Analyze
    print("\n[3/3] Analyzing results...")

    # Split by within/cross category
    within = [r for r in valid_results if r["is_within_category"]]
    cross = [r for r in valid_results if not r["is_within_category"]]

    def analyze_group(name: str, group: list) -> dict | None:
        if len(group) < 3:
            return None

        # Normalize intuition to [0, 1]
        intuition = np.array([r["llm_intuition"] / 10.0 for r in group])
        voi = np.array([r["linear_voi"] for r in group])
        shifts = np.array([r["actual_shift"] for r in group])

        r_intuition, p_intuition = stats.pearsonr(intuition, shifts)
        r_voi, p_voi = stats.pearsonr(voi, shifts)

        # Also check correlation between intuition and VOI
        r_intuition_voi, p_intuition_voi = stats.pearsonr(intuition, voi)

        return {
            "n": len(group),
            "intuition_vs_shift": {"r": float(r_intuition), "p": float(p_intuition)},
            "voi_vs_shift": {"r": float(r_voi), "p": float(p_voi)},
            "intuition_vs_voi": {"r": float(r_intuition_voi), "p": float(p_intuition_voi)},
            "mean_intuition": float(np.mean(intuition)),
            "mean_voi": float(np.mean(voi)),
            "mean_shift": float(np.mean(shifts)),
        }

    all_result = analyze_group("ALL", valid_results)
    cross_result = analyze_group("CROSS", cross)
    within_result = analyze_group("WITHIN", within)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def print_group(name: str, result: dict | None):
        if not result:
            print(f"\n{name}: Too few pairs")
            return

        print(f"\n{name} (n={result['n']}):")
        print(f"  LLM Intuition vs Shift: r={result['intuition_vs_shift']['r']:.3f}, p={result['intuition_vs_shift']['p']:.4f}")
        print(f"  VOI vs Shift:           r={result['voi_vs_shift']['r']:.3f}, p={result['voi_vs_shift']['p']:.4f}")
        print(f"  Intuition vs VOI:       r={result['intuition_vs_voi']['r']:.3f}")

    print_group("ALL PAIRS", all_result)
    print_group("CROSS-CATEGORY", cross_result)
    print_group("WITHIN-CATEGORY", within_result)

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: VOI vs LLM Intuition")
    print("=" * 70)

    print("\n| Group | VOI r | LLM Intuition r | Winner |")
    print("|-------|-------|-----------------|--------|")

    for name, result in [("All", all_result), ("Cross-cat", cross_result), ("Within-cat", within_result)]:
        if result:
            voi_r = result["voi_vs_shift"]["r"]
            int_r = result["intuition_vs_shift"]["r"]
            winner = "VOI" if voi_r > int_r else ("LLM" if int_r > voi_r else "Tie")
            print(f"| {name:9} | {voi_r:5.2f} | {int_r:15.2f} | {winner:6} |")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if cross_result:
        voi_r = cross_result["voi_vs_shift"]["r"]
        int_r = cross_result["intuition_vs_shift"]["r"]
        int_voi_r = cross_result["intuition_vs_voi"]["r"]

        if abs(voi_r - int_r) < 0.10:
            print(f"\n≈ EQUIVALENT: VOI and LLM intuition perform similarly (Δr={abs(voi_r-int_r):.2f})")
            print("  → VOI framework may not add value over simple prompting")
        elif voi_r > int_r:
            print(f"\n✓ VOI WINS: VOI outperforms LLM intuition by Δr={voi_r-int_r:.2f}")
            print("  → Mathematical framework contributes signal beyond naive rating")
        else:
            print(f"\n✗ LLM WINS: LLM intuition outperforms VOI by Δr={int_r-voi_r:.2f}")
            print("  → Simpler baseline is better; VOI may overcomplicate")

        print(f"\nCorrelation between intuition and VOI: r={int_voi_r:.2f}")
        if int_voi_r > 0.7:
            print("  → High overlap: LLM intuition largely recapitulates VOI")
        elif int_voi_r > 0.4:
            print("  → Moderate overlap: some shared signal, some unique")
        else:
            print("  → Low overlap: measuring different things")

    # Save results
    output = {
        "metadata": {
            "experiment": "llm_intuition_baseline",
            "model": MODEL,
            "n_pairs": len(valid_results),
            "n_cross": len(cross),
            "n_within": len(within),
            "run_at": datetime.now().isoformat(),
        },
        "results": {
            "all": all_result,
            "cross_category": cross_result,
            "within_category": within_result,
        },
        "pairs": valid_results,
    }

    output_path = OUTPUT_DIR / "llm_intuition_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
