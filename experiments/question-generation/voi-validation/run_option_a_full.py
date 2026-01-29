"""Run Option A (spread-first) prompt on all 50 Metaculus pairs."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

PROMPT_A = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

Step 1: How much would knowing Q's outcome change your belief about X?
- If Q and X are unrelated, the answer is 0 (independent)
- If related, estimate the shift in percentage points (0.0 to 0.5)

Step 2: What is your base probability P(X) without knowing Q? (0.0 to 1.0)

Step 3: Only if shift ≠ 0, which direction?
- Does Q=YES make X more likely (positive) or less likely (negative)?

Respond with JSON only:
{{"base_p_x": <float>, "shift_magnitude": <float 0-0.5>, "direction": "positive" | "negative" | "none"}}"""

MODEL = "claude-opus-4-5-20251101"
SCRIPT_DIR = Path(__file__).parent


def parse_response(text):
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

        if direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


async def run_experiment():
    import litellm

    # Load pairs
    results_file = SCRIPT_DIR / "results" / "closed_conditional_metaculus_results.json"
    with open(results_file) as f:
        data = json.load(f)

    pairs = data["results"]  # All 50 pairs
    print(f"Running Option A on {len(pairs)} pairs...")
    print(f"Model: {MODEL}")

    results = []

    for i, pair in enumerate(pairs):
        if i > 0 and i % 10 == 0:
            print(f"  Processed {i}/{len(pairs)}...")

        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT_A.format(
                            question_q=pair["question_q"],
                            question_x=pair["question_x"],
                        ),
                    }
                ],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no, shift, raw = parse_response(text)
        except Exception as e:
            text = str(e)
            p_yes, p_no, shift, raw = 0.5, 0.5, 0, {}

        q_outcome = pair["q_outcome"]
        x_outcome = pair["x_outcome"]
        p_conditional = p_yes if q_outcome else p_no

        x_actual = 1.0 if x_outcome else 0.0
        brier = (p_conditional - x_actual) ** 2

        baseline = pair.get("x_prob_before", 0.5)
        brier_baseline = (baseline - x_actual) ** 2

        results.append(
            {
                "q_id": pair.get("q_id"),
                "x_id": pair.get("x_id"),
                "question_q": pair["question_q"][:60],
                "question_x": pair["question_x"][:60],
                "p_x_given_q_yes": p_yes,
                "p_x_given_q_no": p_no,
                "spread": abs(p_yes - p_no),
                "shift_magnitude": shift,
                "base_p_x": raw.get("base_p_x", 0.5),
                "direction": raw.get("direction", "none"),
                "p_conditional": p_conditional,
                "q_outcome": q_outcome,
                "x_outcome": x_outcome,
                "brier": brier,
                "brier_baseline": brier_baseline,
                "improvement": brier_baseline - brier,
                "x_prob_before": baseline,
                "rho": pair.get("rho"),
            }
        )

    # Compute summary
    briers = [r["brier"] for r in results]
    baselines = [r["brier_baseline"] for r in results]
    improvements = [r["improvement"] for r in results]
    spreads = [r["spread"] for r in results]

    print("\n" + "=" * 70)
    print("OPTION A - FULL RESULTS (n=50)")
    print("=" * 70)

    print("\n--- Overall ---")
    print(f"Mean Brier (Option A):  {np.mean(briers):.4f}")
    print(f"Mean Brier (baseline):  {np.mean(baselines):.4f}")
    print(f"Mean improvement:       {np.mean(improvements):+.4f}")
    print(
        f"% improved:             {100*sum(1 for i in improvements if i > 0)/len(improvements):.1f}%"
    )

    t, p = stats.ttest_1samp(improvements, 0)
    sig = "*" if p < 0.05 else ""
    print(f"t-test (improvement>0): t={t:.2f}, p={p:.4f}{sig}")

    print("\n--- Spread Analysis ---")
    print(f"Mean spread: {np.mean(spreads):.3f}")
    print(
        f"Spread > 0.1: {sum(1 for s in spreads if s > 0.1)}/50 ({100*sum(1 for s in spreads if s > 0.1)/50:.0f}%)"
    )
    print(f"Spread > 0.2: {sum(1 for s in spreads if s > 0.2)}/50")
    print(f"Spread = 0 (independent): {sum(1 for s in spreads if s == 0)}/50")

    # Compare to original
    print("\n--- vs Original Prompt ---")
    print("Original: Brier=0.215, improvement=-0.035, spread=0.04")
    print(
        f"Option A: Brier={np.mean(briers):.3f}, improvement={np.mean(improvements):+.3f}, spread={np.mean(spreads):.3f}"
    )

    # By relationship detection
    print("\n--- By Relationship Detection ---")
    detected = [r for r in results if r["spread"] > 0.1]
    not_detected = [r for r in results if r["spread"] <= 0.1]

    print(f"Detected relationship (spread>0.1): n={len(detected)}")
    if detected:
        print(
            f"  Mean improvement: {np.mean([r['improvement'] for r in detected]):+.4f}"
        )
        print(
            f"  % improved: {100*sum(1 for r in detected if r['improvement'] > 0)/len(detected):.0f}%"
        )

    print(f"No relationship (spread<=0.1): n={len(not_detected)}")
    if not_detected:
        print(
            f"  Mean improvement: {np.mean([r['improvement'] for r in not_detected]):+.4f}"
        )
        print(
            f"  % improved: {100*sum(1 for r in not_detected if r['improvement'] > 0)/len(not_detected):.0f}%"
        )

    # By actual rho
    print("\n--- By Actual Relationship (rho) ---")
    high_rho = [r for r in results if abs(r.get("rho", 0) or 0) > 0.5]
    low_rho = [r for r in results if abs(r.get("rho", 0) or 0) <= 0.5]

    print(f"High |rho| > 0.5: n={len(high_rho)}")
    if high_rho:
        print(
            f"  Mean improvement: {np.mean([r['improvement'] for r in high_rho]):+.4f}"
        )
        print(f"  Mean spread: {np.mean([r['spread'] for r in high_rho]):.3f}")

    print(f"Low |rho| <= 0.5: n={len(low_rho)}")
    if low_rho:
        print(
            f"  Mean improvement: {np.mean([r['improvement'] for r in low_rho]):+.4f}"
        )
        print(f"  Mean spread: {np.mean([r['spread'] for r in low_rho]):.3f}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "prompt": "A_spread_first",
            "n": len(results),
        },
        "summary": {
            "mean_brier": float(np.mean(briers)),
            "mean_baseline": float(np.mean(baselines)),
            "mean_improvement": float(np.mean(improvements)),
            "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements),
            "mean_spread": float(np.mean(spreads)),
            "t_stat": float(t),
            "p_value": float(p),
        },
        "results": convert(results),
    }

    output_file = SCRIPT_DIR / "results" / "option_a_full_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
