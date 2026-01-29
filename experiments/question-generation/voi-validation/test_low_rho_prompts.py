"""Test prompt strategies specifically for low-rho (unrelated) pairs.

Goal: Find a prompt that correctly identifies independence while still
working on related pairs.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/test_low_rho_prompts.py
"""

from __future__ import annotations

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
OUTPUT_FILE = SCRIPT_DIR / "results" / "low_rho_prompt_comparison.json"

MODEL = "claude-opus-4-5-20251101"

# =============================================================================
# PROMPT OPTIONS FOR LOW-RHO PAIRS
# =============================================================================

# Option A (baseline) - spread-first from previous experiment
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


# Option E: Mechanism-required gate
# Must articulate specific mechanism or default to independent
PROMPT_E = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

CRITICAL: Most prediction market question pairs have NO real relationship.

Step 1: Is there a SPECIFIC causal or logical mechanism connecting Q and X?
- Not "they're both about politics" - that's not a mechanism
- Not "they could both be affected by the economy" - too vague
- A real mechanism: "Q directly causes X" or "Q and X share a specific common cause Y"

If you cannot state a SPECIFIC mechanism in one sentence, they are INDEPENDENT.

Step 2: If independent, estimate P(X) ignoring Q entirely (0.0 to 1.0)

Step 3: If connected, state the mechanism and estimate shift magnitude (0.0 to 0.5)

Respond with JSON only:
{{"mechanism": "<specific mechanism OR 'independent'>>", "base_p_x": <float>, "shift_magnitude": <float, 0 if independent>, "direction": "positive" | "negative" | "none"}}"""


# Option F: Skeptical prior with evidence requirement
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


# Option G: Counterfactual portfolio decision
PROMPT_G = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

You manage a prediction market portfolio. You currently hold a position on X.

DECISION SCENARIO:
- Scenario A: You learn Q resolved YES
- Scenario B: You learn Q resolved NO

In which scenario would you CHANGE your position on X? By how much?

REALITY CHECK: In most cases, learning about one market shouldn't affect your
position in another. Only answer "change position" if you'd actually trade differently.

Step 1: Would you change your X position based on Q's outcome? (yes/no)
Step 2: If yes, how much would you shift? (in probability points, max 0.5)
Step 3: What is your base estimate for P(X)?

Respond with JSON only:
{{"would_change_position": true | false, "base_p_x": <float>, "shift_magnitude": <float, 0 if no change>, "direction": "positive" | "negative" | "none"}}"""


# Option H: Explicit shared cause decomposition
PROMPT_H = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

CAUSAL ANALYSIS:

Step 1: List ALL specific variables that could causally affect BOTH Q and X.
- Must be specific (not "the economy" or "world events")
- Must plausibly affect both outcomes
- Empty list = independent questions

Step 2: For each shared cause, how strong is its effect on both?
- If no strong shared causes, questions are INDEPENDENT

Step 3: Based on shared causes (or lack thereof), estimate:
- Base P(X) without knowing Q
- Shift magnitude (0 if independent)

Respond with JSON only:
{{"shared_causes": ["<cause1>", "<cause2>", ...] or [], "independent": true | false, "base_p_x": <float>, "shift_magnitude": <float>, "direction": "positive" | "negative" | "none"}}"""


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_standard(text: str) -> tuple[float, float, float, dict]:
    """Parse responses that have base_p_x, shift_magnitude, direction."""
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
        else:  # negative
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


def parse_option_f(text: str) -> tuple[float, float, float, dict]:
    """Parse Option F (skeptical prior)."""
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

        # If doesn't pass test, force independent
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


def parse_option_g(text: str) -> tuple[float, float, float, dict]:
    """Parse Option G (portfolio decision)."""
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
        would_change = data.get("would_change_position", False)

        # If wouldn't change position, force independent
        if not would_change:
            return base, base, 0, data

        if direction == "none" or shift == 0:
            return base, base, 0, data
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift), shift, data
        else:
            return max(0.01, base - shift), min(0.99, base + shift), shift, data
    except Exception:
        return 0.5, 0.5, 0, {}


def parse_option_h(text: str) -> tuple[float, float, float, dict]:
    """Parse Option H (shared cause decomposition)."""
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
        independent = data.get("independent", True)
        shared_causes = data.get("shared_causes", [])

        # If independent or no shared causes, force independent
        if independent or len(shared_causes) == 0:
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
    "A_spread_first": (PROMPT_A, parse_standard),
    "E_mechanism_gate": (PROMPT_E, parse_standard),
    "F_skeptical_prior": (PROMPT_F, parse_option_f),
    "G_portfolio_decision": (PROMPT_G, parse_option_g),
    "H_shared_causes": (PROMPT_H, parse_option_h),
}


# =============================================================================
# EXPERIMENT
# =============================================================================

async def test_prompt(
    prompt_name: str,
    prompt_template: str,
    parse_fn,
    pairs: list[dict],
) -> list[dict]:
    """Test a single prompt option on pairs."""
    import litellm

    results = []

    for pair in pairs:
        question_q = pair["question_q"]
        question_x = pair["question_x"]
        q_outcome = pair["q_outcome"]
        x_outcome = pair["x_outcome"]

        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(
                        question_q=question_q,
                        question_x=question_x,
                    )
                }],
                max_tokens=800,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no, shift, raw = parse_fn(text)

        except Exception as e:
            text = f"Error: {e}"
            p_yes, p_no, shift, raw = 0.5, 0.5, 0, {}

        # Select conditional based on Q outcome
        p_conditional = p_yes if q_outcome else p_no

        # Compute Brier score
        x_actual = 1.0 if x_outcome else 0.0
        brier = (p_conditional - x_actual) ** 2

        # Baseline comparison
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
            "shift_detected": shift,
            "p_conditional": p_conditional,
            "q_outcome": q_outcome,
            "x_outcome": x_outcome,
            "brier": brier,
            "brier_baseline": brier_baseline,
            "improvement": brier_baseline - brier,
            "rho": pair.get("rho"),
            "raw_data": raw,
        })

    return results


async def run_experiment():
    """Run all prompts on low-rho pairs."""
    print("=" * 80)
    print("Low-Rho Prompt Comparison Experiment")
    print("=" * 80)

    # Load data
    print("\nLoading Metaculus results...")
    with open(METACULUS_RESULTS) as f:
        data = json.load(f)

    all_pairs = data["results"]

    # Filter to low-rho pairs only
    low_rho_pairs = [p for p in all_pairs if abs(p.get("rho", 0) or 0) <= 0.5]
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Low |rho| <= 0.5: {len(low_rho_pairs)}")

    # Use subset for testing
    test_pairs = low_rho_pairs[:15]
    print(f"  Testing on: {len(test_pairs)} pairs")
    print(f"\nModel: {MODEL}")

    # Run all prompts
    print("\nRunning prompts...")
    all_results = {}

    for name, (template, parser) in PROMPTS.items():
        print(f"  Testing {name}...")
        results = await test_prompt(name, template, parser, test_pairs)
        all_results[name] = results

    # Compute summaries
    print("\n" + "=" * 80)
    print("RESULTS ON LOW-RHO PAIRS (n={})".format(len(test_pairs)))
    print("=" * 80)

    print("\n{:<25} {:>10} {:>12} {:>10} {:>12} {:>10}".format(
        "Prompt", "Brier", "Improvement", "% Better", "Spread", "% Indep"
    ))
    print("-" * 80)

    summaries = {}
    for name, results in sorted(all_results.items(), key=lambda x: -np.mean([r["improvement"] for r in x[1]])):
        briers = [r["brier"] for r in results]
        improvements = [r["improvement"] for r in results]
        spreads = [r["spread"] for r in results]
        n_independent = sum(1 for s in spreads if s == 0)

        summaries[name] = {
            "mean_brier": float(np.mean(briers)),
            "mean_improvement": float(np.mean(improvements)),
            "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements),
            "mean_spread": float(np.mean(spreads)),
            "pct_independent": n_independent / len(results),
            "n": len(results),
        }

        print("{:<25} {:>10.4f} {:>+12.4f} {:>10.1%} {:>12.3f} {:>10.1%}".format(
            name,
            summaries[name]["mean_brier"],
            summaries[name]["mean_improvement"],
            summaries[name]["pct_improved"],
            summaries[name]["mean_spread"],
            summaries[name]["pct_independent"],
        ))

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # What's the ideal for low-rho pairs?
    print("\nIdeal for low-rho pairs:")
    print("  - High % independent (correctly detecting no relationship)")
    print("  - Low spread (not hallucinating relationships)")
    print("  - Positive improvement (beating market baseline)")

    best = max(summaries.items(), key=lambda x: x[1]["mean_improvement"])
    print(f"\nBest prompt: {best[0]}")
    print(f"  Improvement: {best[1]['mean_improvement']:+.4f}")
    print(f"  % Independent: {best[1]['pct_independent']:.1%}")

    # Compare to Option A baseline
    if "A_spread_first" in summaries:
        a_stats = summaries["A_spread_first"]
        print(f"\nOption A baseline:")
        print(f"  Improvement: {a_stats['mean_improvement']:+.4f}")
        print(f"  % Independent: {a_stats['pct_independent']:.1%}")

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
            "n_pairs": len(test_pairs),
            "rho_filter": "low (|rho| <= 0.5)",
        },
        "summary": convert(summaries),
        "results": convert(all_results),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
