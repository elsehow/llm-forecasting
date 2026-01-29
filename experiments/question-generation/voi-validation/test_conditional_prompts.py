"""Test different prompt strategies for conditional probability estimation.

Tests 4 prompt options on Metaculus pairs to find which best elicits
conditional reasoning vs independence detection.

Options:
A) Force spread estimation first
B) Explicit independence gate
C) Contrastive "why different?"
D) Calibrated base rate + shift

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/test_conditional_prompts.py
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

# Paths
SCRIPT_DIR = Path(__file__).parent
METACULUS_RESULTS = SCRIPT_DIR / "results" / "closed_conditional_metaculus_results.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "prompt_comparison_results.json"

MODEL = "claude-opus-4-5-20251101"

# =============================================================================
# PROMPT OPTIONS
# =============================================================================

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


PROMPT_B = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

STEP 1 - INDEPENDENCE CHECK:
Is there ANY causal, logical, or informational connection between Q and X?
- Would a superforecaster update their P(X) after learning Q's outcome?
- Or are these essentially independent events?

If INDEPENDENT: respond {{"independent": true, "p_x": <your base estimate for X>}}

If CONNECTED: proceed to Step 2.

STEP 2 - CONDITIONAL ESTIMATION (only if connected):
Explain the mechanism: How does Q's outcome affect X?
Then estimate:
- P(X | Q=YES) = ?
- P(X | Q=NO) = ?

Respond with JSON only:
{{"independent": false, "mechanism": "<explanation>", "p_x_given_q_yes": <float>, "p_x_given_q_no": <float>}}"""


PROMPT_C = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

I need you to estimate P(X=YES | Q=YES) and P(X=YES | Q=NO).

CRITICAL: These two probabilities should ONLY differ if there's a real connection between Q and X.

Before estimating, answer this question:
"Why would P(X|Q=yes) differ from P(X|Q=no)?"

If you cannot articulate a concrete reason why they would differ, then they should be EQUAL: P(X|Q=yes) = P(X|Q=no) = your base rate for X.

Respond with JSON only:
{{"reason_for_difference": "<concrete explanation OR 'none - these are independent'>", "p_x_given_q_yes": <float>, "p_x_given_q_no": <float>}}"""


PROMPT_D = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

The base rate for prediction market questions resolving YES is typically 20-40%.

Step 1: Estimate P(X) - your probability X resolves YES, ignoring Q entirely.
(Anchor near 20-40% unless you have strong domain-specific reason to deviate)

Step 2: Would learning Q's outcome change this probability?
- MOST question pairs are UNRELATED - the shift should be 0
- Only estimate a non-zero shift if there's a clear causal or logical mechanism
- Shifts are typically small (0.05-0.15) even when questions are related

Respond with JSON only:
{{"p_x_base": <float>, "shift_if_q_yes": <float, usually 0>, "shift_if_q_no": <float, usually 0>, "mechanism": "<explanation or 'independent'>"}}"""


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_option_a(text: str) -> tuple[float, float]:
    """Parse Option A response into P(X|Q=yes), P(X|Q=no)."""
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
            return base, base
        elif direction == "positive":
            return min(0.99, base + shift), max(0.01, base - shift)
        else:  # negative
            return max(0.01, base - shift), min(0.99, base + shift)
    except Exception as e:
        return 0.5, 0.5


def parse_option_b(text: str) -> tuple[float, float]:
    """Parse Option B response into P(X|Q=yes), P(X|Q=no)."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)

        if data.get("independent", False):
            p_x = float(data.get("p_x", 0.5))
            return p_x, p_x
        else:
            p_yes = float(data.get("p_x_given_q_yes", 0.5))
            p_no = float(data.get("p_x_given_q_no", 0.5))
            return max(0.01, min(0.99, p_yes)), max(0.01, min(0.99, p_no))
    except Exception as e:
        return 0.5, 0.5


def parse_option_c(text: str) -> tuple[float, float]:
    """Parse Option C response into P(X|Q=yes), P(X|Q=no)."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        p_yes = float(data.get("p_x_given_q_yes", 0.5))
        p_no = float(data.get("p_x_given_q_no", 0.5))
        return max(0.01, min(0.99, p_yes)), max(0.01, min(0.99, p_no))
    except Exception as e:
        return 0.5, 0.5


def parse_option_d(text: str) -> tuple[float, float]:
    """Parse Option D response into P(X|Q=yes), P(X|Q=no)."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        base = float(data.get("p_x_base", 0.5))
        shift_yes = float(data.get("shift_if_q_yes", 0))
        shift_no = float(data.get("shift_if_q_no", 0))

        p_yes = max(0.01, min(0.99, base + shift_yes))
        p_no = max(0.01, min(0.99, base + shift_no))
        return p_yes, p_no
    except Exception as e:
        return 0.5, 0.5


PROMPTS = {
    "A_spread_first": (PROMPT_A, parse_option_a),
    "B_independence_gate": (PROMPT_B, parse_option_b),
    "C_contrastive": (PROMPT_C, parse_option_c),
    "D_calibrated_base": (PROMPT_D, parse_option_d),
}


# =============================================================================
# EXPERIMENT
# =============================================================================

async def test_prompt(
    prompt_name: str,
    prompt_template: str,
    parse_fn,
    pairs: list[dict],
    model: str = MODEL,
) -> list[dict]:
    """Test a single prompt option on all pairs."""
    import litellm

    results = []

    for pair in pairs:
        question_q = pair["question_q"]
        question_x = pair["question_x"]
        q_outcome = pair["q_outcome"]
        x_outcome = pair["x_outcome"]

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(
                        question_q=question_q,
                        question_x=question_x,
                    )
                }],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            p_yes, p_no = parse_fn(text)

        except Exception as e:
            text = f"Error: {e}"
            p_yes, p_no = 0.5, 0.5

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
            "prompt": prompt_name,
            "p_x_given_q_yes": p_yes,
            "p_x_given_q_no": p_no,
            "spread": abs(p_yes - p_no),
            "p_conditional": p_conditional,
            "q_outcome": q_outcome,
            "x_outcome": x_outcome,
            "brier": brier,
            "brier_baseline": brier_baseline,
            "improvement": brier_baseline - brier,
            "raw_response": text[:500],
        })

    return results


async def run_all_prompts(pairs: list[dict]) -> dict[str, list[dict]]:
    """Run all prompt options in parallel."""
    print(f"\nTesting {len(PROMPTS)} prompt options on {len(pairs)} pairs...")
    print(f"Model: {MODEL}")

    tasks = []
    for name, (template, parser) in PROMPTS.items():
        print(f"  Launching {name}...")
        task = test_prompt(name, template, parser, pairs)
        tasks.append((name, task))

    # Run all in parallel
    results = {}
    for name, task in tasks:
        results[name] = await task
        print(f"  Completed {name}")

    return results


def compute_summary(results: dict[str, list[dict]]) -> dict:
    """Compute summary statistics for each prompt option."""
    summary = {}

    for prompt_name, prompt_results in results.items():
        briers = [r["brier"] for r in prompt_results]
        baselines = [r["brier_baseline"] for r in prompt_results]
        improvements = [r["improvement"] for r in prompt_results]
        spreads = [r["spread"] for r in prompt_results]

        # Count how many detected relationships (spread > 0.1)
        n_relationships = sum(1 for s in spreads if s > 0.1)

        summary[prompt_name] = {
            "n": len(prompt_results),
            "mean_brier": float(np.mean(briers)),
            "mean_baseline": float(np.mean(baselines)),
            "mean_improvement": float(np.mean(improvements)),
            "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements),
            "mean_spread": float(np.mean(spreads)),
            "n_relationships_detected": n_relationships,
            "pct_relationships": n_relationships / len(prompt_results),
        }

        # Statistical test
        if len(improvements) >= 3:
            t, p = stats.ttest_1samp(improvements, 0)
            summary[prompt_name]["t_stat"] = float(t)
            summary[prompt_name]["p_value"] = float(p)

    return summary


def print_comparison(summary: dict):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("PROMPT COMPARISON RESULTS")
    print("=" * 80)

    print("\n{:<25} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
        "Prompt", "Brier", "Baseline", "Improvement", "% Better", "Spread"
    ))
    print("-" * 80)

    for name, stats in sorted(summary.items(), key=lambda x: -x[1]["mean_improvement"]):
        print("{:<25} {:>10.4f} {:>10.4f} {:>+12.4f} {:>10.1%} {:>12.3f}".format(
            name,
            stats["mean_brier"],
            stats["mean_baseline"],
            stats["mean_improvement"],
            stats["pct_improved"],
            stats["mean_spread"],
        ))

    print("\n{:<25} {:>15} {:>15}".format(
        "Prompt", "Relationships", "p-value"
    ))
    print("-" * 55)

    for name, stats in sorted(summary.items(), key=lambda x: -x[1]["mean_improvement"]):
        p_val = stats.get("p_value", float("nan"))
        print("{:<25} {:>10} ({:>3.0%}) {:>15.4f}".format(
            name,
            stats["n_relationships_detected"],
            stats["pct_relationships"],
            p_val,
        ))


async def main():
    """Run the prompt comparison experiment."""
    print("=" * 80)
    print("Conditional Probability Prompt Comparison")
    print("=" * 80)

    # Load previous results to get the pairs
    print("\nLoading Metaculus results...")
    with open(METACULUS_RESULTS) as f:
        data = json.load(f)

    # Use the pairs from previous experiment (already have outcomes)
    pairs = data["results"]
    print(f"  Loaded {len(pairs)} pairs with outcomes")

    # Use subset for testing (first 15 pairs)
    subset = pairs[:15]
    print(f"  Using subset of {len(subset)} pairs for comparison")

    # Run all prompts
    results = await run_all_prompts(subset)

    # Compute summary
    summary = compute_summary(results)

    # Print comparison
    print_comparison(summary)

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Best and worst performing prompt
    best = max(summary.items(), key=lambda x: x[1]["mean_improvement"])
    worst = min(summary.items(), key=lambda x: x[1]["mean_improvement"])

    print(f"\nBest prompt: {best[0]} (improvement: {best[1]['mean_improvement']:+.4f})")
    print(f"Worst prompt: {worst[0]} (improvement: {worst[1]['mean_improvement']:+.4f})")

    # Check spread distribution
    print("\n--- Spread Analysis ---")
    for name, prompt_results in results.items():
        spreads = [r["spread"] for r in prompt_results]
        print(f"{name}:")
        print(f"  Mean spread: {np.mean(spreads):.3f}")
        print(f"  Spread > 0.1: {sum(1 for s in spreads if s > 0.1)}/{len(spreads)}")
        print(f"  Spread > 0.2: {sum(1 for s in spreads if s > 0.2)}/{len(spreads)}")

    # Save results
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "n_pairs": len(subset),
        },
        "summary": summary,
        "results": {k: v for k, v in results.items()},
    }

    # Convert numpy types
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

    output = convert(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
