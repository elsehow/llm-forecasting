#!/usr/bin/env python3
"""
Estimate P(Q resolves YES) for Metaculus pairs using LLM.

This tests whether using LLM-estimated P(Q) improves entropy VOI
correlation with observed |ΔP|.

Background:
- Current entropy VOI uses P(Q)=0.5 assumption
- This assumption may be wrong, hurting entropy VOI correlation
- If we estimate P(Q) properly, entropy VOI might improve

Phase 1: Pilot on first 100 pairs (--pilot flag)
Phase 2: Full run on all 629 pairs (default)

Usage:
    uv run python estimate_p_q.py --pilot     # Run on 100 pairs
    uv run python estimate_p_q.py             # Run on all pairs
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import litellm
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# Configuration
MODEL = "anthropic/claude-opus-4-20250514"  # High quality for better calibration
RATE_LIMIT_DELAY = 0.2  # seconds between calls
DATA_PATH = Path(__file__).parent / "data" / "llm_estimations.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "llm_estimations_with_p_q.json"
OUTPUT_PATH_PILOT = Path(__file__).parent / "data" / "llm_estimations_with_p_q_pilot.json"

P_Q_ESTIMATION_PROMPT = """You are estimating the probability that a forecasting question resolves YES.

Question: "{q_title}"

Estimate the probability this question resolves YES, considering:
- Base rates for similar events
- Historical context and trends
- Logical constraints

IMPORTANT: Provide your estimate as a probability between 0 and 1.
Be calibrated - don't anchor too strongly on 0.5.

Respond with JSON only:
{{"p_q": <float 0.0-1.0>, "reasoning": "<brief explanation of key factors>"}}"""


async def estimate_p_q(q_title: str, semaphore: asyncio.Semaphore) -> dict:
    """Estimate P(Q resolves YES) for a single question."""
    async with semaphore:
        prompt = P_Q_ESTIMATION_PROMPT.format(q_title=q_title)

        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()

            # Parse JSON (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            p_q = float(result.get("p_q", 0.5))
            # Clamp to valid range
            p_q = max(0.01, min(0.99, p_q))

            await asyncio.sleep(RATE_LIMIT_DELAY)

            return {
                "p_q_llm": p_q,
                "p_q_reasoning": result.get("reasoning", ""),
                "p_q_error": None,
            }
        except Exception as e:
            return {
                "p_q_llm": 0.5,
                "p_q_reasoning": "",
                "p_q_error": str(e),
            }


async def main():
    parser = argparse.ArgumentParser(description="Estimate P(Q) for Metaculus pairs")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot on first 100 pairs only",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API calls (default: 5)",
    )
    args = parser.parse_args()

    # Load existing estimations
    with open(DATA_PATH) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r["llm_error"] is None]
    print(f"Loaded {len(results)} valid pairs")

    # Filter for pilot if requested
    if args.pilot:
        results = results[:100]
        output_path = OUTPUT_PATH_PILOT
        print(f"Running pilot on {len(results)} pairs")
    else:
        output_path = OUTPUT_PATH
        print(f"Running full estimation on {len(results)} pairs")

    print(f"Model: {MODEL}")
    print(f"Concurrency: {args.concurrency}")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    # Run estimations
    tasks = []
    for r in results:
        task = estimate_p_q(r["q_title"], semaphore)
        tasks.append(task)

    p_q_results = await tqdm_asyncio.gather(*tasks, desc="Estimating P(Q)")

    # Merge results
    enriched_results = []
    n_errors = 0
    for original, p_q_est in zip(results, p_q_results):
        enriched = {**original, **p_q_est}
        enriched_results.append(enriched)
        if p_q_est["p_q_error"]:
            n_errors += 1

    # Compute calibration stats
    p_q_values = [r["p_q_llm"] for r in enriched_results if r["p_q_error"] is None]
    q_resolutions = [r["q_resolution"] for r in enriched_results if r["p_q_error"] is None]

    # Basic calibration: correlation between P(Q) and actual resolution
    from scipy import stats
    if len(p_q_values) > 10:
        r_calib, p_val = stats.spearmanr(p_q_values, q_resolutions)
    else:
        r_calib, p_val = None, None

    # Distribution stats
    import numpy as np
    p_q_array = np.array(p_q_values)

    # Save output
    output = {
        "metadata": {
            "model": MODEL,
            "n_pairs": len(enriched_results),
            "n_errors": n_errors,
            "pilot": args.pilot,
            "timestamp": datetime.now().isoformat(),
            "calibration": {
                "r_p_q_vs_resolution": float(r_calib) if r_calib is not None else None,
                "p_value": float(p_val) if p_val is not None else None,
                "p_q_mean": float(np.mean(p_q_array)),
                "p_q_median": float(np.median(p_q_array)),
                "p_q_std": float(np.std(p_q_array)),
                "resolution_rate": float(np.mean(q_resolutions)),
            },
        },
        "results": enriched_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(f"Errors: {n_errors}/{len(enriched_results)}")

    # Print calibration summary
    print(f"\n{'='*60}")
    print("P(Q) CALIBRATION SUMMARY")
    print(f"{'='*60}")
    calib = output["metadata"]["calibration"]
    print(f"  P(Q) mean: {calib['p_q_mean']:.3f}")
    print(f"  P(Q) median: {calib['p_q_median']:.3f}")
    print(f"  P(Q) std: {calib['p_q_std']:.3f}")
    print(f"  Actual resolution rate: {calib['resolution_rate']:.3f}")
    if calib["r_p_q_vs_resolution"] is not None:
        print(f"  r(P(Q), resolution): {calib['r_p_q_vs_resolution']:.3f} (p={calib['p_value']:.4f})")
        if calib["r_p_q_vs_resolution"] > 0.3:
            print("  -> P(Q) estimates correlate with actual resolutions")
        elif calib["r_p_q_vs_resolution"] > 0.1:
            print("  -> Weak correlation with actual resolutions")
        else:
            print("  -> Little correlation with actual resolutions (concerning)")


if __name__ == "__main__":
    asyncio.run(main())
