"""Run H1 + Confidence Gating on full Metaculus dataset.

Combines the H1 hybrid approach with confidence gating (require >=80% confidence).

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/run_h1_confidence_full.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILE = SCRIPT_DIR / "results" / "h1_confidence_full_results.json"
H1_RESULTS = SCRIPT_DIR / "results" / "h1_full_scale_results.json"

MODEL = "claude-opus-4-5-20251101"

# H1 + Confidence Gating Prompt
PROMPT_H1_CONFIDENCE = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

PHASE 1 - INDEPENDENCE TEST (be skeptical):
Most prediction market question pairs are UNRELATED. Default assumption: INDEPENDENT.

To claim a relationship, you need a SPECIFIC mechanism - not just topical similarity.
Ask: Would a rational bettor change their X position by >5% after learning Q's outcome?

PHASE 2 - CONFIDENCE CHECK:
Rate your confidence (0-100) that a genuine causal/logical relationship exists:
- 0-50: Probably unrelated, any connection is speculative
- 51-79: Possible connection but uncertain
- 80-100: Strong evidence of direct relationship

RULE: Only claim "related" if confidence >= 80.

PHASE 3 - IF RELATED (confidence >= 80), estimate magnitude:
- How much would knowing Q shift your belief about X? (0.05 to 0.5)
- Does Q=YES make X more likely (positive) or less likely (negative)?

Respond with JSON only:
{{"confidence": <int 0-100>, "is_related": true | false, "mechanism": "<specific mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float, 0 if confidence < 80>, "direction": "positive" | "negative" | "none"}}"""


def parse_response(text: str) -> tuple[float, float, float, dict]:
    """Parse response. Returns (p_yes, p_no, shift, raw_data)."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        confidence = int(data.get("confidence", 0))
        base = float(data.get("base_p_x", 0.5))
        shift = float(data.get("shift_magnitude", 0))
        direction = data.get("direction", "none")
        is_related = data.get("is_related", False)

        # Enforce confidence threshold
        if confidence < 80 or not is_related:
            return base, base, 0, data

        if direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


async def run_pair(semaphore, litellm, pair, pair_idx):
    """Run a single pair with semaphore for rate limiting."""
    async with semaphore:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": PROMPT_H1_CONFIDENCE.format(
                        question_q=pair["question_q"],
                        question_x=pair["question_x"],
                    )
                }],
                max_tokens=600,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no, shift, raw = parse_response(text)
        except Exception as e:
            text = str(e)
            p_yes, p_no, shift, raw = 0.5, 0.5, 0, {"error": str(e)}

        p_conditional = p_yes if pair["q_outcome"] else p_no
        x_actual = 1.0 if pair["x_outcome"] else 0.0
        brier = (p_conditional - x_actual) ** 2
        baseline = pair.get("x_prob_before", 0.5)
        brier_baseline = (baseline - x_actual) ** 2

        # Determine if marked as related (confidence >= 80)
        confidence = raw.get("confidence", 0)
        is_related = raw.get("is_related", False) and confidence >= 80

        return {
            "q_id": pair.get("q_id"),
            "x_id": pair.get("x_id"),
            "question_q": pair["question_q"][:80],
            "question_x": pair["question_x"][:80],
            "p_x_given_q_yes": p_yes,
            "p_x_given_q_no": p_no,
            "spread": abs(p_yes - p_no),
            "confidence": confidence,
            "is_related": is_related,
            "mechanism": raw.get("mechanism", ""),
            "p_conditional": p_conditional,
            "q_outcome": pair["q_outcome"],
            "x_outcome": pair["x_outcome"],
            "brier": brier,
            "brier_baseline": brier_baseline,
            "improvement": brier_baseline - brier,
            "rho": pair.get("rho"),
            "x_prob_before": baseline,
        }


async def main():
    import litellm

    print("=" * 80)
    print("H1 + CONFIDENCE GATING FULL SCALE (n=598)")
    print("=" * 80)

    # Load pairs from H1 results (already have ground truth)
    print("\nLoading pairs from H1 results...")
    with open(H1_RESULTS) as f:
        h1_data = json.load(f)

    # Extract pairs with their ground truth
    all_pairs = []
    for r in h1_data["results"]:
        all_pairs.append({
            "q_id": r.get("q_id"),
            "x_id": r.get("x_id"),
            "question_q": r["question_q"],
            "question_x": r["question_x"],
            "q_outcome": r["q_outcome"],
            "x_outcome": r["x_outcome"],
            "x_prob_before": r["x_prob_before"],
            "rho": r.get("rho"),
        })

    print(f"Total pairs: {len(all_pairs)}")
    print(f"Model: {MODEL}")

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(10)

    print(f"\nRunning H1 + Confidence on all {len(all_pairs)} pairs...")
    print("(This may take a few minutes)")

    tasks = [
        run_pair(semaphore, litellm, pair, i)
        for i, pair in enumerate(all_pairs)
    ]

    results = []
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"  Processed {min(i+batch_size, len(tasks))}/{len(tasks)}...")

    # Compute statistics
    print("\n" + "=" * 80)
    print(f"RESULTS (n={len(results)})")
    print("=" * 80)

    briers = [r["brier"] for r in results]
    baselines = [r["brier_baseline"] for r in results]
    improvements = [r["improvement"] for r in results]
    spreads = [r["spread"] for r in results]
    confidences = [r["confidence"] for r in results]

    print("\n--- OVERALL ---")
    print(f"Mean Brier (H1+Conf):  {np.mean(briers):.4f}")
    print(f"Mean Brier (baseline): {np.mean(baselines):.4f}")
    print(f"Mean improvement:      {np.mean(improvements):+.4f}")
    print(f"% improved:            {100*sum(1 for i in improvements if i > 0)/len(improvements):.1f}%")
    print(f"Mean spread:           {np.mean(spreads):.3f}")
    print(f"Mean confidence:       {np.mean(confidences):.1f}")
    print(f"% detected related:    {100*sum(1 for r in results if r['is_related'])/len(results):.1f}%")

    t, p = stats.ttest_1samp(improvements, 0)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"t-test (improvement>0): t={t:.2f}, p={p:.6f} {sig}")

    # By rho
    high_rho = [r for r in results if abs(r.get("rho", 0) or 0) > 0.5]
    low_rho = [r for r in results if abs(r.get("rho", 0) or 0) <= 0.5]

    print(f"\n--- BY RELATIONSHIP STRENGTH ---")
    print(f"\nHigh |rho| > 0.5 (n={len(high_rho)}):")
    if high_rho:
        print(f"  Mean improvement: {np.mean([r['improvement'] for r in high_rho]):+.4f}")
        print(f"  % improved:       {100*sum(1 for r in high_rho if r['improvement'] > 0)/len(high_rho):.1f}%")
        print(f"  % detected:       {100*sum(1 for r in high_rho if r['is_related'])/len(high_rho):.1f}%")
        print(f"  Mean confidence:  {np.mean([r['confidence'] for r in high_rho]):.1f}")

    print(f"\nLow |rho| <= 0.5 (n={len(low_rho)}):")
    if low_rho:
        print(f"  Mean improvement: {np.mean([r['improvement'] for r in low_rho]):+.4f}")
        print(f"  % improved:       {100*sum(1 for r in low_rho if r['improvement'] > 0)/len(low_rho):.1f}%")
        print(f"  % independent:    {100*sum(1 for r in low_rho if not r['is_related'])/len(low_rho):.1f}%")
        print(f"  Mean confidence:  {np.mean([r['confidence'] for r in low_rho]):.1f}")

    # Compare to H1 without confidence
    print("\n--- COMPARISON TO H1 (no confidence gating) ---")
    h1_summary = h1_data["summary"]
    print(f"H1:           Overall {h1_summary['overall']['mean_improvement']:+.4f}, ", end="")
    print(f"High-rho {h1_summary['high_rho']['mean_improvement']:+.4f}, ", end="")
    print(f"Low-rho {h1_summary['low_rho']['mean_improvement']:+.4f}")

    print(f"H1+Conf:      Overall {np.mean(improvements):+.4f}, ", end="")
    if high_rho:
        print(f"High-rho {np.mean([r['improvement'] for r in high_rho]):+.4f}, ", end="")
    if low_rho:
        print(f"Low-rho {np.mean([r['improvement'] for r in low_rho]):+.4f}")

    # False positive analysis
    print("\n--- FALSE POSITIVE ANALYSIS ---")
    # In low-rho, how many still detected as related?
    low_rho_related = [r for r in low_rho if r["is_related"]]
    print(f"Low-rho detected as related: {len(low_rho_related)}/{len(low_rho)} ({100*len(low_rho_related)/len(low_rho):.1f}%)")
    print(f"  vs H1: {h1_summary['low_rho']['n'] - int(h1_summary['low_rho']['pct_independent'] * h1_summary['low_rho']['n'])}/{h1_summary['low_rho']['n']} ({100*(1-h1_summary['low_rho']['pct_independent']):.1f}%)")

    if low_rho_related:
        print(f"  Mean improvement (false positives): {np.mean([r['improvement'] for r in low_rho_related]):+.4f}")

    # Save results
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "prompt": "H1_confidence_gating",
            "confidence_threshold": 80,
            "n": len(results),
        },
        "summary": {
            "overall": {
                "n": len(results),
                "mean_brier": float(np.mean(briers)),
                "mean_baseline": float(np.mean(baselines)),
                "mean_improvement": float(np.mean(improvements)),
                "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements),
                "mean_spread": float(np.mean(spreads)),
                "mean_confidence": float(np.mean(confidences)),
                "pct_related": sum(1 for r in results if r["is_related"]) / len(results),
                "t_stat": float(t),
                "p_value": float(p),
            },
            "high_rho": {
                "n": len(high_rho),
                "mean_improvement": float(np.mean([r["improvement"] for r in high_rho])) if high_rho else None,
                "pct_improved": sum(1 for r in high_rho if r["improvement"] > 0) / len(high_rho) if high_rho else None,
                "pct_detected": sum(1 for r in high_rho if r["is_related"]) / len(high_rho) if high_rho else None,
                "mean_confidence": float(np.mean([r["confidence"] for r in high_rho])) if high_rho else None,
            },
            "low_rho": {
                "n": len(low_rho),
                "mean_improvement": float(np.mean([r["improvement"] for r in low_rho])) if low_rho else None,
                "pct_improved": sum(1 for r in low_rho if r["improvement"] > 0) / len(low_rho) if low_rho else None,
                "pct_independent": sum(1 for r in low_rho if not r["is_related"]) / len(low_rho) if low_rho else None,
                "mean_confidence": float(np.mean([r["confidence"] for r in low_rho])) if low_rho else None,
            },
        },
        "results": convert(results),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
