"""Run H1 hybrid prompt on ALL Metaculus pairs with ground truth.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/run_h1_full_scale.py
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILE = SCRIPT_DIR / "results" / "h1_full_scale_results.json"
CACHED_RESULTS = SCRIPT_DIR / "results" / "closed_conditional_metaculus_results.json"
COMOVEMENT_FILE = SCRIPT_DIR.parent / "metaculus-replication" / "data" / "metaculus_comovement_pairs.json"

MODEL = "claude-opus-4-5-20251101"
METACULUS_API_URL = "https://www.metaculus.com/api2/questions"
METACULUS_API_KEY = os.environ.get("METACULUS_API_KEY")

# H1 Hybrid Prompt (best performer)
PROMPT_H1 = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

PHASE 1 - INDEPENDENCE TEST (be skeptical):
Most prediction market question pairs are UNRELATED. Default assumption: INDEPENDENT.

To claim a relationship, you need a SPECIFIC mechanism - not just topical similarity.
Ask: Would a rational bettor change their X position by >5% after learning Q's outcome?

PHASE 2 - IF RELATED, estimate magnitude:
Only if Phase 1 found a real relationship:
- How much would knowing Q shift your belief about X? (0.05 to 0.5)
- Does Q=YES make X more likely (positive) or less likely (negative)?

Respond with JSON only:
{{"is_related": true | false, "mechanism": "<specific mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float, 0 if independent>, "direction": "positive" | "negative" | "none"}}"""


def parse_h1(text: str) -> tuple[float, float, float, dict]:
    """Parse H1 response."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        base = float(data.get("base_p_x", 0.5))
        shift = float(data.get("shift_magnitude", 0))
        direction = data.get("direction", "none")
        is_related = data.get("is_related", False)

        if not is_related or direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


async def run_pair(semaphore, litellm, pair, pair_idx, total):
    """Run a single pair with semaphore for rate limiting."""
    async with semaphore:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": PROMPT_H1.format(
                        question_q=pair["question_q"],
                        question_x=pair["question_x"],
                    )
                }],
                max_tokens=800,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no, shift, raw = parse_h1(text)
        except Exception as e:
            text = str(e)
            p_yes, p_no, shift, raw = 0.5, 0.5, 0, {"error": str(e)}

        p_conditional = p_yes if pair["q_outcome"] else p_no
        x_actual = 1.0 if pair["x_outcome"] else 0.0
        brier = (p_conditional - x_actual) ** 2
        baseline = pair.get("x_prob_before", 0.5)
        brier_baseline = (baseline - x_actual) ** 2

        return {
            "q_id": pair.get("q_id"),
            "x_id": pair.get("x_id"),
            "question_q": pair["question_q"][:80],
            "question_x": pair["question_x"][:80],
            "p_x_given_q_yes": p_yes,
            "p_x_given_q_no": p_no,
            "spread": abs(p_yes - p_no),
            "is_related": raw.get("is_related", False),
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
    print("H1 FULL SCALE EXPERIMENT (n=598)")
    print("=" * 80)

    # Step 1: Load cached X resolutions
    print("Loading cached X resolutions...")
    x_resolutions = {}
    if CACHED_RESULTS.exists():
        with open(CACHED_RESULTS) as f:
            cached = json.load(f)
        for k, v in cached.get("x_resolutions", {}).items():
            x_resolutions[int(k)] = v
    print(f"  Cached: {len(x_resolutions)} X resolutions")

    # Step 2: Load all pairs
    print(f"Loading pairs from {COMOVEMENT_FILE}...")
    with open(COMOVEMENT_FILE) as f:
        comovement_data = json.load(f)
    raw_pairs = comovement_data["pairs"]
    print(f"  Total pairs: {len(raw_pairs)}")

    # Step 3: Find missing X questions
    all_x_ids = set(p["x_id"] for p in raw_pairs)
    missing_x_ids = all_x_ids - set(x_resolutions.keys())
    print(f"  Missing X resolutions: {len(missing_x_ids)}")

    # Step 4: Fetch missing X resolutions from API
    if missing_x_ids:
        print(f"\nFetching {len(missing_x_ids)} missing X resolutions from Metaculus API...")
        headers = {}
        if METACULUS_API_KEY:
            headers["Authorization"] = f"Token {METACULUS_API_KEY}"

        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            for i, x_id in enumerate(sorted(missing_x_ids)):
                try:
                    resp = await client.get(f"{METACULUS_API_URL}/{x_id}/")
                    resp.raise_for_status()
                    d = resp.json()

                    is_resolved = d.get("resolved", False)
                    if is_resolved:
                        question = d.get("question", {})
                        res = question.get("resolution", "").lower()
                        if res == "yes":
                            x_resolutions[x_id] = 1.0
                        elif res == "no":
                            x_resolutions[x_id] = 0.0
                        else:
                            x_resolutions[x_id] = None
                        print(f"  {x_id}: {res}")
                    else:
                        print(f"  {x_id}: not resolved")

                    await asyncio.sleep(2.0)  # Rate limiting
                except Exception as e:
                    print(f"  {x_id}: error - {e}")

    # Step 5: Filter to fully resolved pairs
    all_pairs = []
    for p in raw_pairs:
        q_res = p.get("q_resolution")
        x_id = p.get("x_id")
        x_res = x_resolutions.get(x_id)

        # Need clear YES/NO resolutions for both
        if q_res is None or x_res is None:
            continue

        all_pairs.append({
            "q_id": p.get("q_id"),
            "x_id": x_id,
            "question_q": p.get("q_title"),
            "question_x": p.get("x_title"),
            "q_outcome": q_res == 1.0,
            "x_outcome": x_res == 1.0,
            "x_prob_before": p.get("x_prob_before"),
            "rho": p.get("rho"),
        })

    print(f"\nTotal pairs with ground truth: {len(all_pairs)}")
    print(f"Model: {MODEL}")

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

    print(f"\nRunning H1 on all {len(all_pairs)} pairs...")
    print("(This may take a few minutes)")

    tasks = [
        run_pair(semaphore, litellm, pair, i, len(all_pairs))
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

    print("\n--- OVERALL ---")
    print(f"Mean Brier (H1):       {np.mean(briers):.4f}")
    print(f"Mean Brier (baseline): {np.mean(baselines):.4f}")
    print(f"Mean improvement:      {np.mean(improvements):+.4f}")
    print(f"% improved:            {100*sum(1 for i in improvements if i > 0)/len(improvements):.1f}%")
    print(f"Mean spread:           {np.mean(spreads):.3f}")
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

    print(f"\nLow |rho| <= 0.5 (n={len(low_rho)}):")
    if low_rho:
        print(f"  Mean improvement: {np.mean([r['improvement'] for r in low_rho]):+.4f}")
        print(f"  % improved:       {100*sum(1 for r in low_rho if r['improvement'] > 0)/len(low_rho):.1f}%")
        print(f"  % independent:    {100*sum(1 for r in low_rho if not r['is_related'])/len(low_rho):.1f}%")

    # Comparison to n=50 results
    print("\n--- COMPARISON TO n=50 PILOT ---")
    print("n=50:  Overall +0.064, High-rho +0.083, Low-rho +0.044")
    print(f"n={len(results)}: Overall {np.mean(improvements):+.4f}, ", end="")
    if high_rho:
        print(f"High-rho {np.mean([r['improvement'] for r in high_rho]):+.4f}, ", end="")
    if low_rho:
        print(f"Low-rho {np.mean([r['improvement'] for r in low_rho]):+.4f}")

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
            "prompt": "H1_gate_then_magnitude",
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
                "pct_related": sum(1 for r in results if r["is_related"]) / len(results),
                "t_stat": float(t),
                "p_value": float(p),
            },
            "high_rho": {
                "n": len(high_rho),
                "mean_improvement": float(np.mean([r["improvement"] for r in high_rho])) if high_rho else None,
                "pct_improved": sum(1 for r in high_rho if r["improvement"] > 0) / len(high_rho) if high_rho else None,
                "pct_detected": sum(1 for r in high_rho if r["is_related"]) / len(high_rho) if high_rho else None,
            },
            "low_rho": {
                "n": len(low_rho),
                "mean_improvement": float(np.mean([r["improvement"] for r in low_rho])) if low_rho else None,
                "pct_improved": sum(1 for r in low_rho if r["improvement"] > 0) / len(low_rho) if low_rho else None,
                "pct_independent": sum(1 for r in low_rho if not r["is_related"]) / len(low_rho) if low_rho else None,
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
