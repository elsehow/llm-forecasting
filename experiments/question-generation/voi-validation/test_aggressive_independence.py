"""Test aggressive independence thresholds on false positive pairs.

Tests 5 approaches:
2. Confidence gating (require >80% confidence)
4. Adversarial self-check (two-call)
5. Bidirectional test
6. Prior anchoring (>90% should be independent)
7. Specificity requirement (name ONE shared variable)

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/test_aggressive_independence.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import numpy as np

SCRIPT_DIR = Path(__file__).parent
H1_RESULTS = SCRIPT_DIR / "results" / "h1_full_scale_results.json"
OUTPUT_FILE = SCRIPT_DIR / "results" / "aggressive_independence_results.json"

MODEL = "claude-opus-4-5-20251101"

# =============================================================================
# PROMPT VARIANTS
# =============================================================================

# Option 2: Confidence Gating
PROMPT_2_CONFIDENCE = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

STEP 1: Is there a real relationship between Q and X?
Most question pairs are UNRELATED. Only claim a relationship if you have HIGH CONFIDENCE.

STEP 2: Rate your confidence (0-100) that a genuine causal/logical relationship exists.
- 0-50: Probably unrelated, any connection is speculative
- 51-79: Possible connection but uncertain
- 80-100: Strong evidence of direct relationship

RULE: Only claim "related" if confidence >= 80.

STEP 3: If related (confidence >= 80), estimate the shift magnitude.

Respond with JSON only:
{{"confidence": <int 0-100>, "is_related": true | false, "mechanism": "<specific mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float, 0 if independent>, "direction": "positive" | "negative" | "none"}}"""


# Option 4: Adversarial Self-Check (first call - detect)
PROMPT_4A_DETECT = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

Is there a relationship between these questions? If so, what is it?

Respond with JSON only:
{{"potentially_related": true | false, "mechanism": "<potential mechanism or 'none'>"}}"""

# Option 4: Adversarial Self-Check (second call - argue against)
PROMPT_4B_ARGUE_AGAINST = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

A colleague claims these questions are related because: "{mechanism}"

YOUR TASK: Argue why these questions are actually INDEPENDENT.
- What assumptions does the claimed mechanism rely on?
- Why might the connection be spurious or too indirect?
- Are there confounding factors that break the link?

After making your counter-argument, judge: Is the original relationship claim STRONG enough to survive your critique?

Respond with JSON only:
{{"counter_argument": "<your argument for independence>", "relationship_survives": true | false}}"""


# Option 5: Bidirectional Test
PROMPT_5_BIDIRECTIONAL = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

BIDIRECTIONAL TEST:
1. If you learned Q resolved YES, would you update your belief about X?
2. If you learned X resolved YES, would you update your belief about Q?

A REAL relationship should work in BOTH directions. If the connection only makes sense one way, it's likely spurious.

Answer both questions, then decide if this is a genuine bidirectional relationship.

Respond with JSON only:
{{"q_affects_x": true | false, "x_affects_q": true | false, "bidirectional": true | false, "is_related": true | false, "mechanism": "<mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float, 0 if not bidirectional>, "direction": "positive" | "negative" | "none"}}"""


# Option 6: Prior Anchoring
PROMPT_6_PRIOR = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

IMPORTANT CALIBRATION: In prediction market datasets, over 90% of question pairs have NO meaningful relationship. You should claim a relationship for fewer than 1 in 10 pairs.

Before deciding, ask yourself:
- Is this connection DIRECT and SPECIFIC, or vague and indirect?
- Would a professional forecaster actually update their X position based on Q?
- Am I seeing a real link, or pattern-matching on superficial similarities?

Default strongly to INDEPENDENT unless the relationship is unmistakable.

Respond with JSON only:
{{"is_related": true | false, "mechanism": "<specific mechanism or 'independent'>", "base_p_x": <float>, "shift_magnitude": <float, 0 if independent>, "direction": "positive" | "negative" | "none"}}"""


# Option 7: Specificity Requirement
PROMPT_7_SPECIFIC = """QUESTION Q: "{question_q}"
QUESTION X: "{question_x}"

SPECIFICITY TEST:
To claim these questions are related, you must name exactly ONE specific variable that:
1. Directly causes or influences Q's outcome
2. Directly causes or influences X's outcome
3. Is more specific than "the economy", "geopolitics", "market sentiment", or "world events"

Examples of SPECIFIC shared variables:
- "Federal Reserve interest rate decision on March 15"
- "Ukraine's territorial control of Kherson"
- "NVIDIA's Q4 2024 earnings report"

If you cannot name ONE specific shared variable, these questions are INDEPENDENT.

Respond with JSON only:
{{"shared_variable": "<specific variable name or 'none'>", "is_related": true | false, "base_p_x": <float>, "shift_magnitude": <float, 0 if independent>, "direction": "positive" | "negative" | "none"}}"""


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_standard(text: str) -> dict:
    """Parse standard JSON response."""
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except:
        return {}


def parse_option_2(text: str) -> tuple[bool, float, dict]:
    """Parse Option 2 (confidence gating). Returns (is_independent, spread, raw)."""
    data = parse_standard(text)
    confidence = data.get("confidence", 0)
    is_related = data.get("is_related", False) and confidence >= 80

    if not is_related:
        return True, 0, data

    base = float(data.get("base_p_x", 0.5))
    shift = float(data.get("shift_magnitude", 0))
    return False, shift, data


def parse_option_5(text: str) -> tuple[bool, float, dict]:
    """Parse Option 5 (bidirectional). Returns (is_independent, spread, raw)."""
    data = parse_standard(text)
    bidirectional = data.get("bidirectional", False)
    is_related = data.get("is_related", False) and bidirectional

    if not is_related:
        return True, 0, data

    shift = float(data.get("shift_magnitude", 0))
    return False, shift, data


def parse_option_6(text: str) -> tuple[bool, float, dict]:
    """Parse Option 6 (prior anchoring). Returns (is_independent, spread, raw)."""
    data = parse_standard(text)
    is_related = data.get("is_related", False)

    if not is_related:
        return True, 0, data

    shift = float(data.get("shift_magnitude", 0))
    return False, shift, data


def parse_option_7(text: str) -> tuple[bool, float, dict]:
    """Parse Option 7 (specificity). Returns (is_independent, spread, raw)."""
    data = parse_standard(text)
    shared_var = data.get("shared_variable", "none")
    is_related = data.get("is_related", False) and shared_var.lower() not in ["none", "", "n/a"]

    if not is_related:
        return True, 0, data

    shift = float(data.get("shift_magnitude", 0))
    return False, shift, data


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

async def test_option_2(litellm, pairs: list[dict]) -> list[dict]:
    """Test Option 2: Confidence Gating."""
    results = []
    for pair in pairs:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_2_CONFIDENCE.format(**pair)}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            is_independent, spread, raw = parse_option_2(text)
        except Exception as e:
            is_independent, spread, raw = True, 0, {"error": str(e)}

        results.append({
            "pair_idx": pair["idx"],
            "is_independent": is_independent,
            "spread": spread,
            "confidence": raw.get("confidence", 0),
            "raw": raw,
        })
    return results


async def test_option_4(litellm, pairs: list[dict]) -> list[dict]:
    """Test Option 4: Adversarial Self-Check (two calls)."""
    results = []
    for pair in pairs:
        try:
            # First call: detect potential relationship
            resp1 = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_4A_DETECT.format(**pair)}],
                max_tokens=300,
                temperature=0,
            )
            text1 = resp1.choices[0].message.content.strip()
            data1 = parse_standard(text1)

            potentially_related = data1.get("potentially_related", False)
            mechanism = data1.get("mechanism", "none")

            if not potentially_related or mechanism.lower() in ["none", ""]:
                # Already independent
                results.append({
                    "pair_idx": pair["idx"],
                    "is_independent": True,
                    "spread": 0,
                    "stage": "first_call",
                    "raw": data1,
                })
                continue

            # Second call: argue against
            resp2 = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_4B_ARGUE_AGAINST.format(
                    question_q=pair["question_q"],
                    question_x=pair["question_x"],
                    mechanism=mechanism,
                )}],
                max_tokens=500,
                temperature=0,
            )
            text2 = resp2.choices[0].message.content.strip()
            data2 = parse_standard(text2)

            survives = data2.get("relationship_survives", False)

            results.append({
                "pair_idx": pair["idx"],
                "is_independent": not survives,
                "spread": 0.1 if survives else 0,  # Placeholder
                "stage": "second_call",
                "mechanism": mechanism,
                "counter_argument": data2.get("counter_argument", ""),
                "raw": {"first": data1, "second": data2},
            })

        except Exception as e:
            results.append({
                "pair_idx": pair["idx"],
                "is_independent": True,
                "spread": 0,
                "error": str(e),
            })

    return results


async def test_option_5(litellm, pairs: list[dict]) -> list[dict]:
    """Test Option 5: Bidirectional Test."""
    results = []
    for pair in pairs:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_5_BIDIRECTIONAL.format(**pair)}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            is_independent, spread, raw = parse_option_5(text)
        except Exception as e:
            is_independent, spread, raw = True, 0, {"error": str(e)}

        results.append({
            "pair_idx": pair["idx"],
            "is_independent": is_independent,
            "spread": spread,
            "q_affects_x": raw.get("q_affects_x", False),
            "x_affects_q": raw.get("x_affects_q", False),
            "bidirectional": raw.get("bidirectional", False),
            "raw": raw,
        })
    return results


async def test_option_6(litellm, pairs: list[dict]) -> list[dict]:
    """Test Option 6: Prior Anchoring."""
    results = []
    for pair in pairs:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_6_PRIOR.format(**pair)}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            is_independent, spread, raw = parse_option_6(text)
        except Exception as e:
            is_independent, spread, raw = True, 0, {"error": str(e)}

        results.append({
            "pair_idx": pair["idx"],
            "is_independent": is_independent,
            "spread": spread,
            "raw": raw,
        })
    return results


async def test_option_7(litellm, pairs: list[dict]) -> list[dict]:
    """Test Option 7: Specificity Requirement."""
    results = []
    for pair in pairs:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_7_SPECIFIC.format(**pair)}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            is_independent, spread, raw = parse_option_7(text)
        except Exception as e:
            is_independent, spread, raw = True, 0, {"error": str(e)}

        results.append({
            "pair_idx": pair["idx"],
            "is_independent": is_independent,
            "spread": spread,
            "shared_variable": raw.get("shared_variable", "none"),
            "raw": raw,
        })
    return results


# =============================================================================
# MAIN
# =============================================================================

async def main():
    import litellm

    print("=" * 80)
    print("AGGRESSIVE INDEPENDENCE THRESHOLD TEST")
    print("=" * 80)

    # Load false positives from H1 results
    print("\nLoading false positives from H1 results...")
    with open(H1_RESULTS) as f:
        h1_data = json.load(f)

    # Get low-rho pairs that H1 incorrectly marked as related
    false_positives = []
    for i, r in enumerate(h1_data["results"]):
        if abs(r.get("rho", 0) or 0) <= 0.5 and r.get("is_related"):
            false_positives.append({
                "idx": i,
                "question_q": r["question_q"],
                "question_x": r["question_x"],
                "rho": r.get("rho", 0),
                "h1_improvement": r["improvement"],
            })

    print(f"Total false positives: {len(false_positives)}")

    # Use subset for testing
    test_pairs = false_positives[:30]
    print(f"Testing on: {len(test_pairs)} pairs")
    print(f"Model: {MODEL}")

    # Run all tests
    all_results = {}

    print("\nTesting Option 2 (Confidence Gating)...")
    all_results["2_confidence"] = await test_option_2(litellm, test_pairs)

    print("Testing Option 4 (Adversarial Self-Check)...")
    all_results["4_adversarial"] = await test_option_4(litellm, test_pairs)

    print("Testing Option 5 (Bidirectional Test)...")
    all_results["5_bidirectional"] = await test_option_5(litellm, test_pairs)

    print("Testing Option 6 (Prior Anchoring)...")
    all_results["6_prior"] = await test_option_6(litellm, test_pairs)

    print("Testing Option 7 (Specificity Requirement)...")
    all_results["7_specificity"] = await test_option_7(litellm, test_pairs)

    # Compute summary
    print("\n" + "=" * 80)
    print(f"RESULTS (n={len(test_pairs)} false positives)")
    print("=" * 80)
    print("\nGoal: Correctly reclassify false positives as INDEPENDENT")
    print("Higher % independent = better at filtering spurious relationships\n")

    print("{:<25} {:>15} {:>15}".format("Option", "% Independent", "# Independent"))
    print("-" * 55)

    summaries = {}
    for name, results in sorted(all_results.items()):
        n_independent = sum(1 for r in results if r["is_independent"])
        pct = n_independent / len(results) if results else 0
        summaries[name] = {"n_independent": n_independent, "pct_independent": pct}
        print("{:<25} {:>15.1%} {:>15}".format(name, pct, f"{n_independent}/{len(results)}"))

    # H1 baseline (0% independent on these pairs by definition)
    print("{:<25} {:>15.1%} {:>15}".format("H1 (baseline)", 0.0, "0/30"))

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Option 2: Confidence distribution
    print("\n--- Option 2: Confidence Distribution ---")
    confidences = [r.get("confidence", 0) for r in all_results["2_confidence"]]
    print(f"Mean confidence: {np.mean(confidences):.1f}")
    print(f"Confidence < 80 (independent): {sum(1 for c in confidences if c < 80)}")
    print(f"Confidence >= 80 (related): {sum(1 for c in confidences if c >= 80)}")

    # Option 4: How many survived adversarial check?
    print("\n--- Option 4: Adversarial Check ---")
    first_call = sum(1 for r in all_results["4_adversarial"] if r.get("stage") == "first_call")
    survived = sum(1 for r in all_results["4_adversarial"] if not r["is_independent"])
    print(f"Rejected at first call: {first_call}")
    print(f"Sent to adversarial check: {len(test_pairs) - first_call}")
    print(f"Survived adversarial check: {survived}")

    # Option 5: Bidirectional analysis
    print("\n--- Option 5: Bidirectional Test ---")
    q_affects_x = sum(1 for r in all_results["5_bidirectional"] if r.get("q_affects_x"))
    x_affects_q = sum(1 for r in all_results["5_bidirectional"] if r.get("x_affects_q"))
    bidirectional = sum(1 for r in all_results["5_bidirectional"] if r.get("bidirectional"))
    print(f"Q affects X: {q_affects_x}")
    print(f"X affects Q: {x_affects_q}")
    print(f"Bidirectional (both): {bidirectional}")

    # Option 7: What shared variables were claimed?
    print("\n--- Option 7: Claimed Shared Variables ---")
    variables = [r.get("shared_variable", "none") for r in all_results["7_specificity"] if r.get("shared_variable", "none").lower() not in ["none", "", "n/a"]]
    print(f"Claimed specific variable: {len(variables)}/{len(test_pairs)}")
    for v in variables[:5]:
        print(f"  - {v[:60]}")

    # Save results
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "n_pairs": len(test_pairs),
            "source": "H1 false positives (low-rho pairs marked as related)",
        },
        "summary": summaries,
        "results": all_results,
        "test_pairs": test_pairs,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
