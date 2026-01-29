"""Test hybrid prompt combining F's skeptical gate with A's magnitude estimation.

Goal: Get F's independence detection (80% on low-rho) + A's relationship detection (77% on high-rho)

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/test_hybrid_prompt.py
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
METACULUS_RESULTS = SCRIPT_DIR / "results" / "closed_conditional_metaculus_results.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "hybrid_prompt_results.json"

MODEL = "claude-opus-4-5-20251101"

# =============================================================================
# HYBRID PROMPTS
# =============================================================================

# Hybrid 1: F's gate + A's magnitude (two-phase in one prompt)
PROMPT_HYBRID_1 = """QUESTION Q: "{question_q}"
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


# Hybrid 2: Explicit two-question format
PROMPT_HYBRID_2 = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

Answer TWO questions:

QUESTION 1: Is there a REAL relationship between Q and X?
- "Real" means: a specific causal pathway, not just topical similarity
- Most pairs are UNRELATED - lean toward "no" unless you can articulate a concrete mechanism
- Answer: yes or no

QUESTION 2: What are the probabilities?
- If unrelated: P(X|Q=yes) = P(X|Q=no) = P(X) (your base estimate)
- If related: P(X|Q=yes) and P(X|Q=no) should differ based on the mechanism

Respond with JSON only:
{{"is_related": true | false, "mechanism": "<one sentence or 'none'>", "p_x_given_q_yes": <float>, "p_x_given_q_no": <float>}}"""


# Hybrid 3: Concrete mechanism requirement with calibrated shift
PROMPT_HYBRID_3 = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

STEP 1: Can you complete this sentence with a SPECIFIC mechanism?
"Learning Q's outcome affects X because _______________"

If you cannot fill in a concrete, direct mechanism (not vague like "economic conditions"):
→ These questions are INDEPENDENT

STEP 2: Estimate probabilities
- Base P(X): Your estimate ignoring Q (use prediction market base rates: most events ~20-40%)
- If independent: shift = 0
- If related: shift = how much P(X) changes when you learn Q (typically 0.05-0.20)

Respond with JSON only:
{{"mechanism": "<specific mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float>, "direction": "positive" | "negative" | "none"}}"""


# Hybrid 4: Betting framing with skeptical default
PROMPT_HYBRID_4 = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

You're a prediction market trader deciding whether Q's outcome should affect your X position.

REALITY CHECK: In most markets, questions are UNRELATED. Your default should be:
"Q's outcome doesn't change my X position."

ONLY update your X position if you can answer YES to BOTH:
1. Is there a direct causal/logical link? (not just "both about tech" or "both in 2025")
2. Would you actually trade differently? (Would you bet at least $100 on the conditional?)

ESTIMATE:
- Your base P(X) estimate
- If you'd update: the shift magnitude (0.05-0.30) and direction
- If you wouldn't update: shift = 0

Respond with JSON only:
{{"would_update_position": true | false, "reasoning": "<one sentence>", "base_p_x": <float>, "shift_magnitude": <float>, "direction": "positive" | "negative" | "none"}}"""


# =============================================================================
# PARSING
# =============================================================================

def parse_hybrid_1(text: str) -> tuple[float, float, float, dict]:
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


def parse_hybrid_2(text: str) -> tuple[float, float, float, dict]:
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        p_yes = float(data.get("p_x_given_q_yes", 0.5))
        p_no = float(data.get("p_x_given_q_no", 0.5))
        is_related = data.get("is_related", False)

        # If not related, force equal
        if not is_related:
            avg = (p_yes + p_no) / 2
            return avg, avg, 0, data

        spread = abs(p_yes - p_no)
        return max(0.01, min(0.99, p_yes)), max(0.01, min(0.99, p_no)), spread, data
    except Exception:
        return 0.5, 0.5, 0, {}


def parse_hybrid_3(text: str) -> tuple[float, float, float, dict]:
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
        mechanism = data.get("mechanism", "independent")

        is_independent = mechanism.lower() in ["independent", "none", ""]

        if is_independent or direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


def parse_hybrid_4(text: str) -> tuple[float, float, float, dict]:
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
        would_update = data.get("would_update_position", False)

        if not would_update or direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


PROMPTS = {
    "H1_gate_then_magnitude": (PROMPT_HYBRID_1, parse_hybrid_1),
    "H2_two_questions": (PROMPT_HYBRID_2, parse_hybrid_2),
    "H3_mechanism_required": (PROMPT_HYBRID_3, parse_hybrid_3),
    "H4_betting_framing": (PROMPT_HYBRID_4, parse_hybrid_4),
}


# =============================================================================
# EXPERIMENT
# =============================================================================

async def test_prompt(prompt_name, prompt_template, parse_fn, pairs):
    import litellm

    results = []
    for pair in pairs:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(
                        question_q=pair["question_q"],
                        question_x=pair["question_x"],
                    )
                }],
                max_tokens=800,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no, shift, raw = parse_fn(text)
        except Exception as e:
            text = str(e)
            p_yes, p_no, shift, raw = 0.5, 0.5, 0, {}

        p_conditional = p_yes if pair["q_outcome"] else p_no
        x_actual = 1.0 if pair["x_outcome"] else 0.0
        brier = (p_conditional - x_actual) ** 2
        baseline = pair.get("x_prob_before", 0.5)
        brier_baseline = (baseline - x_actual) ** 2

        results.append({
            "q_id": pair.get("q_id"),
            "x_id": pair.get("x_id"),
            "question_q": pair["question_q"][:60],
            "question_x": pair["question_x"][:60],
            "prompt": prompt_name,
            "p_x_given_q_yes": p_yes,
            "p_x_given_q_no": p_no,
            "spread": abs(p_yes - p_no),
            "p_conditional": p_conditional,
            "brier": brier,
            "brier_baseline": brier_baseline,
            "improvement": brier_baseline - brier,
            "rho": pair.get("rho"),
            "raw_data": raw,
        })

    return results


async def main():
    print("=" * 80)
    print("HYBRID PROMPT TEST")
    print("Goal: F's independence detection + A's relationship detection")
    print("=" * 80)

    with open(METACULUS_RESULTS) as f:
        data = json.load(f)

    all_pairs = data["results"]

    # Split by rho
    high_rho = [p for p in all_pairs if abs(p.get("rho", 0) or 0) > 0.5]
    low_rho = [p for p in all_pairs if abs(p.get("rho", 0) or 0) <= 0.5]

    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"High |rho| > 0.5: {len(high_rho)}")
    print(f"Low |rho| <= 0.5: {len(low_rho)}")
    print(f"Model: {MODEL}")

    # Test on all pairs
    print("\nTesting hybrid prompts on ALL pairs...")

    all_results = {}
    for name, (template, parser) in PROMPTS.items():
        print(f"  Testing {name}...")
        results = await test_prompt(name, template, parser, all_pairs)
        all_results[name] = results

    # Compute summaries by rho category
    print("\n" + "=" * 80)
    print("RESULTS BY RHO CATEGORY")
    print("=" * 80)

    summaries = {}

    for name, results in all_results.items():
        high_results = [r for r in results if abs(r.get("rho", 0) or 0) > 0.5]
        low_results = [r for r in results if abs(r.get("rho", 0) or 0) <= 0.5]

        summaries[name] = {
            "overall": {
                "n": len(results),
                "mean_brier": float(np.mean([r["brier"] for r in results])),
                "mean_improvement": float(np.mean([r["improvement"] for r in results])),
                "pct_improved": sum(1 for r in results if r["improvement"] > 0) / len(results),
                "mean_spread": float(np.mean([r["spread"] for r in results])),
                "pct_detected": sum(1 for r in results if r["spread"] > 0) / len(results),
            },
            "high_rho": {
                "n": len(high_results),
                "mean_improvement": float(np.mean([r["improvement"] for r in high_results])) if high_results else 0,
                "pct_detected": sum(1 for r in high_results if r["spread"] > 0) / len(high_results) if high_results else 0,
            },
            "low_rho": {
                "n": len(low_results),
                "mean_improvement": float(np.mean([r["improvement"] for r in low_results])) if low_results else 0,
                "pct_independent": sum(1 for r in low_results if r["spread"] == 0) / len(low_results) if low_results else 0,
            },
        }

    # Print overall results
    print("\n--- OVERALL (n=50) ---")
    print("{:<25} {:>10} {:>12} {:>10} {:>10}".format(
        "Prompt", "Brier", "Improvement", "% Better", "% Detected"
    ))
    print("-" * 70)

    for name in sorted(summaries.keys(), key=lambda x: -summaries[x]["overall"]["mean_improvement"]):
        s = summaries[name]["overall"]
        print("{:<25} {:>10.4f} {:>+12.4f} {:>10.1%} {:>10.1%}".format(
            name, s["mean_brier"], s["mean_improvement"], s["pct_improved"], s["pct_detected"]
        ))

    # Print by rho
    print("\n--- HIGH-RHO (n=26) - want high detection ---")
    print("{:<25} {:>12} {:>15}".format("Prompt", "Improvement", "% Detected"))
    print("-" * 55)

    for name in sorted(summaries.keys(), key=lambda x: -summaries[x]["high_rho"]["mean_improvement"]):
        s = summaries[name]["high_rho"]
        print("{:<25} {:>+12.4f} {:>15.1%}".format(
            name, s["mean_improvement"], s["pct_detected"]
        ))

    print("\n--- LOW-RHO (n=24) - want high independence ---")
    print("{:<25} {:>12} {:>15}".format("Prompt", "Improvement", "% Independent"))
    print("-" * 55)

    for name in sorted(summaries.keys(), key=lambda x: -summaries[x]["low_rho"]["mean_improvement"]):
        s = summaries[name]["low_rho"]
        print("{:<25} {:>+12.4f} {:>15.1%}".format(
            name, s["mean_improvement"], s["pct_independent"]
        ))

    # Baselines for comparison
    print("\n--- BASELINES FOR COMPARISON ---")
    print("Option A: High-rho +0.054, Low-rho -0.066")
    print("Option F: High-rho +0.044, Low-rho +0.029")

    # Find best hybrid
    print("\n--- BEST HYBRID ---")
    best_overall = max(summaries.items(), key=lambda x: x[1]["overall"]["mean_improvement"])
    print(f"Best overall: {best_overall[0]}")
    print(f"  Overall improvement: {best_overall[1]['overall']['mean_improvement']:+.4f}")
    print(f"  High-rho: {best_overall[1]['high_rho']['mean_improvement']:+.4f}")
    print(f"  Low-rho: {best_overall[1]['low_rho']['mean_improvement']:+.4f}")

    # Save
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
            "n_pairs": len(all_pairs),
        },
        "summary": convert(summaries),
        "results": convert(all_results),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
