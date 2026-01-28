#!/usr/bin/env python3
"""
Cross-validate generated cruxes against known Polymarket pairs.

For ultimates in our benchmark that have known high-ρ pairs:
1. Did we generate a crux similar to the known paired question?
2. If so, did our estimated ρ match the market-derived ρ?
3. Do high-|ρ| pairs appear as high-VOI cruxes?

Usage:
    uv run python experiments/question-generation/benchmark-mvp/cross_validate.py
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import asyncio
from dotenv import load_dotenv
import litellm

load_dotenv()

BENCHMARK_DIR = Path(__file__).parent
CONDITIONAL_DIR = BENCHMARK_DIR.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
VOI_DIR = BENCHMARK_DIR.parent / "voi-validation"

MODEL = "anthropic/claude-sonnet-4-20250514"


def load_data():
    """Load benchmark results and known pairs."""
    with open(BENCHMARK_DIR / "results" / "benchmark_results.json") as f:
        benchmark = json.load(f)

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)

    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)

    # Build lookups
    cond_to_question = {m["condition_id"]: m["question"] for m in markets}

    # Find high-ρ pairs for each condition_id
    high_rho_pairs = defaultdict(list)
    for p in pairs:
        rho = p["rho"]
        if np.isnan(rho) or abs(rho) < 0.3:
            continue

        cond_a = p["market_a"]["condition_id"]
        cond_b = p["market_b"]["condition_id"]
        q_a = p["market_a"]["question"]
        q_b = p["market_b"]["question"]

        high_rho_pairs[cond_a].append({
            "paired_cond": cond_b,
            "paired_question": q_b,
            "rho": rho,
        })
        high_rho_pairs[cond_b].append({
            "paired_cond": cond_a,
            "paired_question": q_a,
            "rho": rho,
        })

    return benchmark, high_rho_pairs, cond_to_question


SIMILARITY_PROMPT = """Rate the semantic similarity between these two questions on a scale of 0-1.

Question A: {q_a}
Question B: {q_b}

0 = Completely unrelated topics
0.3 = Same broad topic but different focus
0.5 = Related and overlapping concerns
0.7 = Very similar, asking about related outcomes
1.0 = Essentially the same question

Respond with JSON only: {{"similarity": <0-1>, "reasoning": "<brief explanation>"}}"""


async def compute_similarity(q_a: str, q_b: str) -> tuple[float, str]:
    """Compute semantic similarity between two questions."""
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": SIMILARITY_PROMPT.format(q_a=q_a, q_b=q_b)
            }],
            max_tokens=150,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return result["similarity"], result.get("reasoning", "")
    except Exception as e:
        return 0.0, str(e)


async def main():
    print("=" * 70)
    print("CROSS-VALIDATION: Generated Cruxes vs Known Pairs")
    print("=" * 70)

    benchmark, high_rho_pairs, cond_to_question = load_data()

    print(f"\nBenchmark: {len(benchmark['results'])} ultimates")
    print(f"High-ρ pairs available: {sum(len(v) for v in high_rho_pairs.values())} relationships")

    # Find ultimates that have known high-ρ pairs
    matches = []

    for result in benchmark["results"]:
        if "error" in result:
            continue

        cond_id = result["condition_id"]
        ultimate = result["ultimate"]

        if cond_id not in high_rho_pairs:
            continue

        known_pairs = high_rho_pairs[cond_id]
        cruxes = result["crux_scores"]

        print(f"\n{'='*60}")
        print(f"Ultimate: {ultimate[:60]}...")
        print(f"  Has {len(known_pairs)} known high-ρ pairs")

        # For each known pair, find best matching generated crux
        for kp in known_pairs[:5]:  # Limit to top 5 to avoid too many API calls
            paired_q = kp["paired_question"]
            known_rho = kp["rho"]

            print(f"\n  Known pair (ρ={known_rho:+.2f}): {paired_q[:50]}...")

            best_match = None
            best_sim = 0

            # Check similarity to each generated crux
            for cs in cruxes:
                sim, reason = await compute_similarity(cs["crux"], paired_q)
                if sim > best_sim:
                    best_sim = sim
                    best_match = {
                        "crux": cs["crux"],
                        "similarity": sim,
                        "reason": reason,
                        "estimated_rho": cs["rho_estimated"],
                        "voi": cs["voi"],
                        "known_rho": known_rho,
                    }

            if best_match and best_sim >= 0.3:
                print(f"    Best match (sim={best_sim:.2f}): {best_match['crux'][:50]}...")
                print(f"      Estimated ρ: {best_match['estimated_rho']:.2f}, Known ρ: {known_rho:.2f}")
                print(f"      VOI: {best_match['voi']:.3f}")
                matches.append(best_match)
            else:
                print(f"    No good match found (best sim={best_sim:.2f})")

    # Analyze matches
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if not matches:
        print("\nNo matches found with similarity >= 0.3")
        return

    print(f"\nFound {len(matches)} matches with similarity >= 0.3")

    # ρ estimation accuracy
    rho_errors = []
    for m in matches:
        if m["known_rho"] is not None:
            error = abs(m["estimated_rho"] - m["known_rho"])
            rho_errors.append(error)
            # Check direction
            if m["estimated_rho"] * m["known_rho"] > 0:
                direction = "correct"
            elif m["estimated_rho"] == 0 or m["known_rho"] == 0:
                direction = "neutral"
            else:
                direction = "wrong"
            print(f"\n  Crux: {m['crux'][:50]}...")
            print(f"    Similarity: {m['similarity']:.2f}")
            print(f"    ρ estimated: {m['estimated_rho']:.2f}, ρ known: {m['known_rho']:.2f} ({direction})")
            print(f"    VOI: {m['voi']:.3f}")

    if rho_errors:
        print(f"\nρ Estimation Accuracy:")
        print(f"  Mean absolute error: {np.mean(rho_errors):.2f}")
        print(f"  Median absolute error: {np.median(rho_errors):.2f}")

        # Direction accuracy
        direction_correct = sum(1 for m in matches
                               if m["estimated_rho"] * m["known_rho"] > 0)
        print(f"  Direction accuracy: {direction_correct}/{len(matches)} ({100*direction_correct/len(matches):.0f}%)")

    # VOI correlation with |ρ|
    if len(matches) >= 3:
        vois = [m["voi"] for m in matches]
        abs_rhos = [abs(m["known_rho"]) for m in matches]
        from scipy import stats
        r, p = stats.pearsonr(vois, abs_rhos)
        print(f"\nVOI vs |ρ| correlation:")
        print(f"  r = {r:.2f}, p = {p:.3f}")

    # Save results
    output = {
        "n_matches": len(matches),
        "matches": matches,
        "rho_mae": float(np.mean(rho_errors)) if rho_errors else None,
    }

    output_path = BENCHMARK_DIR / "results" / "cross_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
