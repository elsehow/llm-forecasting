#!/usr/bin/env python3
"""Consistency check loop experiment.

Tests whether showing models their own Bayesian inconsistency and asking them
to reconcile improves coherence and accuracy.

Protocol:
- Turn 1: Forward direction (P(A), P(A|B=YES))
- Turn 2: Reverse direction (P(B), P(B|A=YES))
- Turn 3: Consistency check (if inconsistent > 0.05)
- Turn 4: Second reconciliation (if still inconsistent > 0.05)

Usage:
    uv run python experiments/fb-conditional/scaffolding/consistency-check-loop/run_experiment.py
    uv run python experiments/fb-conditional/scaffolding/consistency-check-loop/run_experiment.py --limit 5
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm


# Prompts for each turn
TURN1_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Estimate:
- P(A): probability A resolves YES
- P(A|B=YES): probability A resolves YES, given B resolved YES

Return only JSON: {{"p_a": 0.XX, "p_a_given_b1": 0.XX}}"""

TURN2_PROMPT = """Questions:
- A: "{q_a}"
- B: "{q_b}"

Estimate:
- P(B): probability B resolves YES
- P(B|A=YES): probability B resolves YES, given A resolved YES

Return only JSON: {{"p_b": 0.XX, "p_b_given_a1": 0.XX}}"""

TURN3_PROMPT = """Your previous answers imply two different values for P(A=YES and B=YES):

From P(A|B) × P(B): {joint_forward:.2f}
From P(B|A) × P(A): {joint_reverse:.2f}

These cannot both be correct. Please reconsider and provide revised estimates:

Return JSON: {{
  "p_a": 0.XX,
  "p_b": 0.XX,
  "p_a_given_b1": 0.XX,
  "p_b_given_a1": 0.XX,
  "reasoning": "one sentence on what you revised"
}}"""

TURN4_PROMPT = """Your revised estimates still imply different joint probabilities:

From P(A|B) × P(B): {joint_forward_v2:.2f}
From P(B|A) × P(A): {joint_reverse_v2:.2f}

The correct joint probability must satisfy BOTH directions.
What is your final estimate for P(A=YES and B=YES)?

Return JSON: {{"p_joint_final": 0.XX, "p_a": 0.XX, "p_b": 0.XX}}"""

CONSISTENCY_THRESHOLD = 0.05


def extract_json(content: str) -> dict | None:
    """Extract JSON object from response content."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


async def call_model(
    messages: list[dict],
    model: str = "claude-sonnet-4-20250514",
    thinking: bool = True,
) -> tuple[str | None, dict | None]:
    """Make a model call and return (raw_content, parsed_json)."""
    kwargs = {
        "model": model,
        "messages": messages,
    }

    if thinking and "claude" in model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
        kwargs["temperature"] = 1  # Required for thinking mode
    else:
        kwargs["temperature"] = 0.3

    try:
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        parsed = extract_json(content)
        return content, parsed
    except Exception as e:
        print(f"    Error: {e}")
        return None, None


def compute_inconsistency(p_a: float, p_b: float, p_a_given_b1: float, p_b_given_a1: float) -> tuple[float, float, float]:
    """Compute joint probabilities and inconsistency.

    Returns: (joint_forward, joint_reverse, inconsistency)
    """
    joint_forward = p_a_given_b1 * p_b  # P(A,B) from forward
    joint_reverse = p_b_given_a1 * p_a  # P(A,B) from reverse
    inconsistency = abs(joint_forward - joint_reverse)
    return joint_forward, joint_reverse, inconsistency


def sign(x: float) -> int:
    """Return sign of a number (-1, 0, or 1)."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


async def run_pair(pair: dict, model: str, thinking: bool) -> dict:
    """Run consistency check loop on a single pair."""
    q_a = pair["text_a"]
    q_b = pair["text_b"]

    result = {
        "pair_id": f"{pair['id_a']}_{pair['id_b']}",
        "category": pair.get("category"),
        "reason": pair.get("reason"),
        "q_a": q_a[:100],
        "q_b": q_b[:100],
        "resolution_a": pair["resolution_a"],
        "resolution_b": pair["resolution_b"],
    }

    # Maintain conversation history
    messages = []

    # ============= TURN 1: Forward Direction =============
    turn1_content = TURN1_PROMPT.format(q_a=q_a, q_b=q_b)
    messages.append({"role": "user", "content": turn1_content})

    raw1, turn1 = await call_model(messages, model, thinking)

    if turn1 is None or not all(k in turn1 for k in ["p_a", "p_a_given_b1"]):
        result["error"] = "Failed to get Turn 1 response"
        return result

    # Add assistant response to history
    messages.append({"role": "assistant", "content": raw1})
    result["turn1"] = {"p_a": turn1["p_a"], "p_a_given_b1": turn1["p_a_given_b1"]}

    # ============= TURN 2: Reverse Direction =============
    turn2_content = TURN2_PROMPT.format(q_a=q_a, q_b=q_b)
    messages.append({"role": "user", "content": turn2_content})

    raw2, turn2 = await call_model(messages, model, thinking)

    if turn2 is None or not all(k in turn2 for k in ["p_b", "p_b_given_a1"]):
        result["error"] = "Failed to get Turn 2 response"
        return result

    messages.append({"role": "assistant", "content": raw2})
    result["turn2"] = {"p_b": turn2["p_b"], "p_b_given_a1": turn2["p_b_given_a1"]}

    # ============= Compute Initial Inconsistency =============
    p_a = turn1["p_a"]
    p_b = turn2["p_b"]
    p_a_given_b1 = turn1["p_a_given_b1"]
    p_b_given_a1 = turn2["p_b_given_a1"]

    joint_forward, joint_reverse, inconsistency = compute_inconsistency(
        p_a, p_b, p_a_given_b1, p_b_given_a1
    )

    result["initial"] = {
        "joint_forward": joint_forward,
        "joint_reverse": joint_reverse,
        "inconsistency": inconsistency,
        "consistent": inconsistency < CONSISTENCY_THRESHOLD,
    }

    # Track values for analysis
    initial_p_a = p_a
    initial_p_a_given_b1 = p_a_given_b1

    # ============= TURN 3: Consistency Check (if needed) =============
    if inconsistency >= CONSISTENCY_THRESHOLD:
        turn3_content = TURN3_PROMPT.format(
            joint_forward=joint_forward,
            joint_reverse=joint_reverse
        )
        messages.append({"role": "user", "content": turn3_content})

        raw3, turn3 = await call_model(messages, model, thinking)

        if turn3 is None or not all(k in turn3 for k in ["p_a", "p_b", "p_a_given_b1", "p_b_given_a1"]):
            result["error"] = "Failed to get Turn 3 response"
            result["reconciled"] = None
            return result

        messages.append({"role": "assistant", "content": raw3})

        # Compute new inconsistency
        p_a_v2 = turn3["p_a"]
        p_b_v2 = turn3["p_b"]
        p_a_given_b1_v2 = turn3["p_a_given_b1"]
        p_b_given_a1_v2 = turn3["p_b_given_a1"]

        joint_forward_v2, joint_reverse_v2, inconsistency_v2 = compute_inconsistency(
            p_a_v2, p_b_v2, p_a_given_b1_v2, p_b_given_a1_v2
        )

        result["reconciled"] = {
            "p_a": p_a_v2,
            "p_b": p_b_v2,
            "p_a_given_b1": p_a_given_b1_v2,
            "p_b_given_a1": p_b_given_a1_v2,
            "joint_forward": joint_forward_v2,
            "joint_reverse": joint_reverse_v2,
            "inconsistency": inconsistency_v2,
            "consistent": inconsistency_v2 < CONSISTENCY_THRESHOLD,
            "reasoning": turn3.get("reasoning", ""),
        }

        # ============= TURN 4: Second Reconciliation (if still inconsistent) =============
        if inconsistency_v2 >= CONSISTENCY_THRESHOLD:
            turn4_content = TURN4_PROMPT.format(
                joint_forward_v2=joint_forward_v2,
                joint_reverse_v2=joint_reverse_v2
            )
            messages.append({"role": "user", "content": turn4_content})

            raw4, turn4 = await call_model(messages, model, thinking)

            if turn4 is None or "p_joint_final" not in turn4:
                result["error"] = "Failed to get Turn 4 response"
                return result

            # Derive conditionals from final joint
            p_joint_final = turn4["p_joint_final"]
            p_a_final = turn4.get("p_a", p_a_v2)
            p_b_final = turn4.get("p_b", p_b_v2)

            # P(A|B) = P(A,B) / P(B)
            p_a_given_b1_final = p_joint_final / p_b_final if p_b_final > 0.001 else 0.5
            # P(B|A) = P(A,B) / P(A)
            p_b_given_a1_final = p_joint_final / p_a_final if p_a_final > 0.001 else 0.5

            # By construction, should be consistent now
            joint_forward_final = p_a_given_b1_final * p_b_final
            joint_reverse_final = p_b_given_a1_final * p_a_final
            inconsistency_final = abs(joint_forward_final - joint_reverse_final)

            result["final"] = {
                "p_joint_final": p_joint_final,
                "p_a": p_a_final,
                "p_b": p_b_final,
                "p_a_given_b1": p_a_given_b1_final,
                "p_b_given_a1": p_b_given_a1_final,
                "joint_forward": joint_forward_final,
                "joint_reverse": joint_reverse_final,
                "inconsistency": inconsistency_final,
                "consistent": inconsistency_final < CONSISTENCY_THRESHOLD,
            }

            # Use final values for analysis
            p_a_v2 = p_a_final
            p_a_given_b1_v2 = p_a_given_b1_final
            inconsistency_v2 = inconsistency_final
    else:
        result["reconciled"] = None
        p_a_v2 = p_a
        p_a_given_b1_v2 = p_a_given_b1

    # ============= Analysis =============
    resolution_a = pair["resolution_a"]
    resolution_b = pair["resolution_b"]

    # Brier scores - use conditional based on actual B resolution
    brier_initial = (initial_p_a_given_b1 - resolution_a) ** 2 if resolution_b == 1.0 else (initial_p_a - resolution_a) ** 2

    if result["reconciled"] is not None:
        final_p_a_given_b1 = result.get("final", result["reconciled"])["p_a_given_b1"]
        final_p_a = result.get("final", result["reconciled"])["p_a"]
        brier_reconciled = (final_p_a_given_b1 - resolution_a) ** 2 if resolution_b == 1.0 else (final_p_a - resolution_a) ** 2
        brier_improved = brier_reconciled < brier_initial

        # Direction flip check
        direction_initial = sign(initial_p_a_given_b1 - initial_p_a)
        direction_final = sign(final_p_a_given_b1 - final_p_a)
        direction_flipped = direction_initial != direction_final and direction_initial != 0 and direction_final != 0

        # Get final inconsistency
        final_state = result.get("final", result["reconciled"])
        final_inconsistency = final_state["inconsistency"]
        became_consistent = final_state["consistent"]
        reconciliation_helped = final_inconsistency < inconsistency
    else:
        brier_reconciled = None
        brier_improved = None
        direction_flipped = None
        became_consistent = result["initial"]["consistent"]
        reconciliation_helped = None

    result["analysis"] = {
        "needed_reconciliation": not result["initial"]["consistent"],
        "reconciliation_helped": reconciliation_helped,
        "direction_flipped": direction_flipped,
        "brier_initial": brier_initial,
        "brier_reconciled": brier_reconciled,
        "brier_improved": brier_improved,
        "became_consistent": became_consistent,
        "needed_turn4": "final" in result,
    }

    return result


def print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK LOOP EXPERIMENT RESULTS")
    print("=" * 70)

    if not valid:
        print("\nNo valid results to analyze.")
        return

    n = len(valid)

    # Initial consistency
    initially_consistent = sum(1 for r in valid if r["initial"]["consistent"])
    needed_reconciliation = n - initially_consistent

    print(f"\n{'Initially consistent:':<45} {initially_consistent}/{n} ({100*initially_consistent/n:.0f}%)")
    print(f"{'Needed reconciliation:':<45} {needed_reconciliation}/{n} ({100*needed_reconciliation/n:.0f}%)")

    # Post-reconciliation analysis (only for those that needed it)
    needed_recon = [r for r in valid if not r["initial"]["consistent"]]
    if needed_recon:
        # After Turn 3
        became_consistent_t3 = sum(1 for r in needed_recon if r["reconciled"] and r["reconciled"]["consistent"])
        print(f"{'Became consistent after Turn 3:':<45} {became_consistent_t3}/{len(needed_recon)} ({100*became_consistent_t3/len(needed_recon):.0f}%)")

        # Needed Turn 4
        needed_t4 = sum(1 for r in needed_recon if "final" in r)
        print(f"{'Needed Turn 4:':<45} {needed_t4}/{len(needed_recon)} ({100*needed_t4/len(needed_recon):.0f}%)")

        # After Turn 4
        became_consistent_t4 = sum(1 for r in needed_recon if r.get("final", {}).get("consistent", False))
        print(f"{'Became consistent after Turn 4:':<45} {became_consistent_t4}/{needed_t4 if needed_t4 > 0 else 1} ({100*became_consistent_t4/max(needed_t4, 1):.0f}%)")

        # Final consistency (combined Turn 3 + Turn 4)
        final_consistent = sum(1 for r in needed_recon if r["analysis"]["became_consistent"])
        print(f"{'Final consistent (Turn 3 or 4):':<45} {final_consistent}/{len(needed_recon)} ({100*final_consistent/len(needed_recon):.0f}%)")

        # Reconciliation helped
        recon_helped = sum(1 for r in needed_recon if r["analysis"]["reconciliation_helped"])
        print(f"{'Reconciliation reduced inconsistency:':<45} {recon_helped}/{len(needed_recon)} ({100*recon_helped/len(needed_recon):.0f}%)")

        # Direction flipped
        direction_flipped = sum(1 for r in needed_recon if r["analysis"]["direction_flipped"])
        print(f"{'Direction flipped during reconciliation:':<45} {direction_flipped}/{len(needed_recon)} ({100*direction_flipped/len(needed_recon):.0f}%)")

        # Brier impact
        brier_improved = sum(1 for r in needed_recon if r["analysis"]["brier_improved"])
        brier_worsened = sum(1 for r in needed_recon if r["analysis"]["brier_improved"] is False)
        print(f"{'Brier improved after reconciliation:':<45} {brier_improved}/{len(needed_recon)} ({100*brier_improved/len(needed_recon):.0f}%)")
        print(f"{'Brier worsened after reconciliation:':<45} {brier_worsened}/{len(needed_recon)} ({100*brier_worsened/len(needed_recon):.0f}%)")

    # By category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    print("\n" + "-" * 70)
    print("BY CATEGORY")
    print("-" * 70)
    print(f"\n{'Category':<10} {'Initial Consistent':<20} {'Final Consistent':<18} {'Brier Δ':<10}")
    print("-" * 58)

    for cat in ["strong", "weak", "none"]:
        if cat not in by_category:
            continue

        cat_results = by_category[cat]
        n_cat = len(cat_results)

        initial_cons = sum(1 for r in cat_results if r["initial"]["consistent"])
        final_cons = sum(1 for r in cat_results if r["analysis"]["became_consistent"])

        # Compute mean Brier delta (only for those that were reconciled)
        reconciled = [r for r in cat_results if r["analysis"]["brier_reconciled"] is not None]
        if reconciled:
            brier_deltas = [r["analysis"]["brier_reconciled"] - r["analysis"]["brier_initial"] for r in reconciled]
            mean_delta = sum(brier_deltas) / len(brier_deltas)
            delta_str = f"{mean_delta:+.4f}"
        else:
            delta_str = "N/A"

        print(f"{cat.capitalize():<10} {initial_cons}/{n_cat} ({100*initial_cons/n_cat:3.0f}%)          {final_cons}/{n_cat} ({100*final_cons/n_cat:3.0f}%)         {delta_str}")


def print_analysis(results: list[dict]):
    """Print detailed analysis answering key questions."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Group by category
    by_category = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    n = len(valid)
    initially_consistent = sum(1 for r in valid if r["initial"]["consistent"])
    needed_recon = [r for r in valid if not r["initial"]["consistent"]]

    # 1. Baseline consistency
    print(f"\n1. BASELINE CONSISTENCY")
    print(f"   {initially_consistent}/{n} ({100*initially_consistent/n:.0f}%) are Bayes-consistent before feedback.")
    print(f"   Compare to 50-70% expected from original experiment.")

    # 2. Reconciliation success
    if needed_recon:
        final_consistent = sum(1 for r in needed_recon if r["analysis"]["became_consistent"])
        print(f"\n2. RECONCILIATION SUCCESS")
        print(f"   When shown inconsistency, {final_consistent}/{len(needed_recon)} ({100*final_consistent/len(needed_recon):.0f}%) become consistent.")

    # 3. Reconciliation attempts
    if needed_recon:
        needed_t4 = sum(1 for r in needed_recon if "final" in r)
        print(f"\n3. RECONCILIATION ATTEMPTS")
        print(f"   {needed_t4}/{len(needed_recon)} ({100*needed_t4/len(needed_recon):.0f}%) needed Turn 4 (second attempt).")
        if needed_t4 > 0:
            converged_t4 = sum(1 for r in needed_recon if r.get("final", {}).get("consistent", False))
            print(f"   Of those, {converged_t4}/{needed_t4} converged to consistency.")

    # 4. Direction stability
    if needed_recon:
        direction_flipped = sum(1 for r in needed_recon if r["analysis"]["direction_flipped"])
        print(f"\n4. DIRECTION STABILITY")
        print(f"   {direction_flipped}/{len(needed_recon)} ({100*direction_flipped/len(needed_recon):.0f}%) flipped their implied direction during reconciliation.")
        if direction_flipped / len(needed_recon) > 0.2:
            print("   -> HIGH flip rate suggests underlying correlation estimate is unstable.")
        else:
            print("   -> LOW flip rate suggests models preserve their directional intuition.")

    # 5. Brier impact
    if needed_recon:
        brier_improved = sum(1 for r in needed_recon if r["analysis"]["brier_improved"])
        brier_worsened = sum(1 for r in needed_recon if r["analysis"]["brier_improved"] is False)
        print(f"\n5. BRIER IMPACT")
        print(f"   Improved: {brier_improved}/{len(needed_recon)} ({100*brier_improved/len(needed_recon):.0f}%)")
        print(f"   Worsened: {brier_worsened}/{len(needed_recon)} ({100*brier_worsened/len(needed_recon):.0f}%)")
        if brier_improved > brier_worsened:
            print("   -> Reconciliation tends to IMPROVE accuracy.")
        elif brier_worsened > brier_improved:
            print("   -> Reconciliation tends to HURT accuracy (optimizing for surface coherence).")
        else:
            print("   -> Mixed impact on accuracy.")

    # 6. Reasoning quality (sample)
    print(f"\n6. REASONING QUALITY (sample)")
    reasonings = [(r["category"], r["reconciled"]["reasoning"])
                  for r in needed_recon
                  if r.get("reconciled") and r["reconciled"].get("reasoning")]
    for cat, reasoning in reasonings[:5]:
        print(f"   [{cat}] {reasoning[:80]}...")

    # 7. Key finding
    print(f"\n7. KEY FINDING")
    if needed_recon:
        final_consistent_pct = 100 * sum(1 for r in needed_recon if r["analysis"]["became_consistent"]) / len(needed_recon)
        brier_improved_pct = 100 * brier_improved / len(needed_recon)

        if final_consistent_pct > 70 and brier_improved_pct > 50:
            print("   Models CAN self-correct with prompting → latent knowledge accessible.")
        elif final_consistent_pct > 70 and brier_improved_pct <= 50:
            print("   Models achieve coherence but HURT accuracy → surface optimization.")
        elif final_consistent_pct <= 70:
            print("   Models STRUGGLE to achieve consistency even with explicit guidance.")
    else:
        print("   All pairs were initially consistent - no reconciliation needed.")


async def main():
    parser = argparse.ArgumentParser(description="Consistency check loop experiment")
    parser.add_argument("--pairs", type=str, default="experiments/fb-conditional/pairs_filtered.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_false", dest="thinking")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load pairs
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        return

    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])
    if args.limit:
        pairs = pairs[:args.limit]

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"experiments/fb-conditional/scaffolding/consistency-check-loop/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running consistency check loop experiment on {len(pairs)} pairs")
    print(f"Model: {args.model}")
    print(f"Thinking: {args.thinking}")
    print(f"Output: {args.output}")
    print()

    # Run with concurrency limit (lower due to multi-turn)
    semaphore = asyncio.Semaphore(3)

    async def run_with_semaphore(i: int, pair: dict) -> dict:
        async with semaphore:
            print(f"[{i+1}/{len(pairs)}] {pair.get('category', '?')}: {pair.get('reason', '')[:50]}...")
            result = await run_pair(pair, args.model, args.thinking)
            if "error" not in result:
                initial_cons = "consistent" if result["initial"]["consistent"] else f"inconsistent ({result['initial']['inconsistency']:.2f})"
                turns = "Turn 4" if "final" in result else ("Turn 3" if result["reconciled"] else "N/A")
                final_cons = "consistent" if result["analysis"]["became_consistent"] else "inconsistent"
                print(f"    Initial: {initial_cons} | {turns} | Final: {final_cons}")
            else:
                print(f"    Error: {result['error']}")
            return result

    results = await asyncio.gather(*[
        run_with_semaphore(i, pair) for i, pair in enumerate(pairs)
    ])

    # Print outputs
    print_summary(results)
    print_analysis(results)

    # Save results
    output = {
        "results": results,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "model": args.model,
            "thinking": args.thinking,
            "num_pairs": len(pairs),
            "num_successful": len([r for r in results if "error" not in r]),
            "consistency_threshold": CONSISTENCY_THRESHOLD,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
