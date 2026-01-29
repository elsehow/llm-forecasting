"""Test Option F (skeptical prior) on HIGH-rho pairs to verify it doesn't break.

We need a prompt that works on BOTH:
- Low-rho: correctly identifies independence (F excels: +0.029)
- High-rho: correctly identifies relationships (A excels: +0.054)

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/test_option_f_high_rho.py
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
OUTPUT_FILE = SCRIPT_DIR / "results" / "option_f_high_rho_test.json"

MODEL = "claude-opus-4-5-20251101"

# Option A (baseline) and Option F for comparison
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


PROMPT_F = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

DEFAULT ASSUMPTION: These questions are INDEPENDENT until proven otherwise.

To claim a relationship exists, you must meet ALL THREE criteria:
1. State a specific causal pathway (not just topical similarity)
2. The pathway must be direct (not "butterfly effect" chains)
3. You would confidently bet money on the conditional differing from the marginal

INDEPENDENCE TEST: Ask yourself - if Q resolved YES vs NO, would a rational bettor
change their position on X by more than 5%? If not, treat as independent.

Step 1: Do these questions pass the independence test? (yes = related, no = independent)
Step 2: If independent, what is P(X)? If related, what is the shift?

Respond with JSON only:
{{"passes_independence_test": true | false, "reasoning": "<one sentence>", "base_p_x": <float>, "shift_magnitude": <float, 0 if independent>, "direction": "positive" | "negative" | "none"}}"""


def parse_a(text: str) -> tuple[float, float, float, dict]:
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


def parse_f(text: str) -> tuple[float, float, float, dict]:
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
        passes_test = data.get("passes_independence_test", False)

        if not passes_test:
            return base, base, 0, data

        if direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


PROMPTS = {
    "A_spread_first": (PROMPT_A, parse_a),
    "F_skeptical_prior": (PROMPT_F, parse_f),
}


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
    print("Testing Option F vs A on HIGH-RHO pairs")
    print("=" * 80)

    with open(METACULUS_RESULTS) as f:
        data = json.load(f)

    all_pairs = data["results"]

    # Filter to HIGH-rho pairs only
    high_rho_pairs = [p for p in all_pairs if abs(p.get("rho", 0) or 0) > 0.5]
    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"High |rho| > 0.5: {len(high_rho_pairs)}")
    print(f"Model: {MODEL}")

    # Test both prompts on all high-rho pairs
    all_results = {}
    for name, (template, parser) in PROMPTS.items():
        print(f"\nTesting {name}...")
        results = await test_prompt(name, template, parser, high_rho_pairs)
        all_results[name] = results

    # Results
    print("\n" + "=" * 80)
    print(f"RESULTS ON HIGH-RHO PAIRS (n={len(high_rho_pairs)})")
    print("=" * 80)

    print("\n{:<25} {:>10} {:>12} {:>10} {:>12} {:>10}".format(
        "Prompt", "Brier", "Improvement", "% Better", "Spread", "% Detected"
    ))
    print("-" * 80)

    summaries = {}
    for name, results in all_results.items():
        briers = [r["brier"] for r in results]
        improvements = [r["improvement"] for r in results]
        spreads = [r["spread"] for r in results]
        n_detected = sum(1 for s in spreads if s > 0)

        summaries[name] = {
            "mean_brier": float(np.mean(briers)),
            "mean_improvement": float(np.mean(improvements)),
            "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements),
            "mean_spread": float(np.mean(spreads)),
            "pct_relationship_detected": n_detected / len(results),
        }

        print("{:<25} {:>10.4f} {:>+12.4f} {:>10.1%} {:>12.3f} {:>10.1%}".format(
            name,
            summaries[name]["mean_brier"],
            summaries[name]["mean_improvement"],
            summaries[name]["pct_improved"],
            summaries[name]["mean_spread"],
            summaries[name]["pct_relationship_detected"],
        ))

    # Analysis
    print("\n" + "-" * 80)
    print("COMPARISON:")
    a_imp = summaries["A_spread_first"]["mean_improvement"]
    f_imp = summaries["F_skeptical_prior"]["mean_improvement"]
    print(f"  Option A improvement: {a_imp:+.4f}")
    print(f"  Option F improvement: {f_imp:+.4f}")
    print(f"  Difference (F - A):   {f_imp - a_imp:+.4f}")

    if f_imp >= a_imp * 0.8:  # Within 20% of A's performance
        print("\n  ✓ Option F maintains good performance on high-rho pairs!")
    else:
        print("\n  ✗ Option F loses too much on high-rho pairs")

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
            "n_pairs": len(high_rho_pairs),
            "rho_filter": "high (|rho| > 0.5)",
        },
        "summary": convert(summaries),
        "results": convert(all_results),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
