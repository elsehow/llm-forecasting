#!/usr/bin/env python3
"""
Compare ρ estimation prompts on GDP 2050 signals.

Tests whether the validated Exp3 prompt reduces the ρ=0.0 failure rate
compared to the current voi.py prompt.
"""

import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load environment from repo root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Use same model as voi.py batch API
MODEL = "claude-3-haiku-20240307"

# =============================================================================
# The two prompts to compare
# =============================================================================

PROMPT_A_CURRENT = """Estimate the correlation coefficient (ρ) between these two forecasting questions.

ρ ranges from -1 to +1:
- ρ = +1: Perfect positive correlation (if A happens, B definitely happens)
- ρ = 0: Independent (A and B are unrelated)
- ρ = -1: Perfect negative correlation (if A happens, B definitely doesn't)

Question A (target): {question_a}
Question B (signal): {question_b}

Consider:
1. Is there a causal relationship?
2. Are they measuring the same underlying phenomenon?
3. Could they be mutually exclusive?
4. Are they truly independent?

Respond with JSON only: {{"rho": <number between -1 and 1>, "reasoning": "<brief explanation>"}}"""

PROMPT_B_EXP3 = """You are estimating the correlation between two prediction market questions.

Question A: "{question_a}"
Question B: "{question_b}"

Estimate the correlation coefficient (ρ) between these two questions. This measures how much knowing the outcome of one question tells you about the other:
- ρ = +1: Perfect positive correlation (if A is YES, B is definitely YES)
- ρ = 0: Independent (knowing A tells you nothing about B)
- ρ = -1: Perfect negative correlation (if A is YES, B is definitely NO)

Think about:
- Are these questions about related events?
- Would one outcome make the other more or less likely?
- Are they measuring the same underlying phenomenon?

Respond with JSON only:
{{"rho": <float from -1 to +1>, "reasoning": "<brief explanation>"}}"""


def load_gdp_signals() -> tuple[str, list[dict]]:
    """Load signals from the existing GDP 2050 top-down run."""
    results_dir = Path(__file__).parent / "gdp_2040" / "results" / "gdp_2050"
    topdown_files = list(results_dir.glob("topdown_v7_*.json"))

    if not topdown_files:
        raise FileNotFoundError(f"No topdown results found in {results_dir}")

    # Use most recent
    latest = max(topdown_files, key=lambda p: p.stat().st_mtime)

    with open(latest) as f:
        data = json.load(f)

    target_question = data["question"]["text"]
    signals = data["signals"]

    print(f"Loaded {len(signals)} signals from {latest.name}")
    print(f"Target: {target_question}")

    return target_question, signals


def parse_rho_response(text: str) -> tuple[float, str]:
    """Parse ρ and reasoning from LLM response."""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())
        rho = data.get("rho", data.get("rho_estimate", 0.0))
        reasoning = data.get("reasoning", "")
        return float(rho), reasoning
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return 0.0, f"Parse error: {e}"


async def estimate_rho_single(
    client: anthropic.Anthropic,
    prompt_template: str,
    question_a: str,
    question_b: str,
) -> tuple[float, str]:
    """Estimate ρ using a single API call."""
    prompt = prompt_template.format(question_a=question_a, question_b=question_b)

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    return parse_rho_response(text)


async def run_comparison(target: str, signals: list[dict]) -> dict:
    """Run both prompts on all signals and compare results."""
    client = anthropic.Anthropic()

    results_a = []
    results_b = []

    print(f"\nRunning comparison on {len(signals)} signals...")
    print("=" * 60)

    for i, signal in enumerate(signals):
        signal_text = signal["text"]
        original_rho = signal.get("rho", 0.0)

        print(f"\n[{i+1}/{len(signals)}] {signal_text[:60]}...")
        print(f"  Original ρ: {original_rho}")

        # Run Prompt A (current)
        rho_a, reasoning_a = await estimate_rho_single(
            client, PROMPT_A_CURRENT, target, signal_text
        )
        results_a.append({
            "signal_id": signal["id"],
            "signal_text": signal_text,
            "rho": rho_a,
            "reasoning": reasoning_a,
            "original_rho": original_rho,
        })
        print(f"  Prompt A (current): ρ={rho_a:.2f}")

        # Run Prompt B (Exp3)
        rho_b, reasoning_b = await estimate_rho_single(
            client, PROMPT_B_EXP3, target, signal_text
        )
        results_b.append({
            "signal_id": signal["id"],
            "signal_text": signal_text,
            "rho": rho_b,
            "reasoning": reasoning_b,
            "original_rho": original_rho,
        })
        print(f"  Prompt B (Exp3):    ρ={rho_b:.2f}")

    return {
        "target": target,
        "prompt_a_results": results_a,
        "prompt_b_results": results_b,
    }


def analyze_results(results: dict) -> dict:
    """Compute comparison metrics."""
    a_results = results["prompt_a_results"]
    b_results = results["prompt_b_results"]

    # ρ=0 rates
    a_zero_count = sum(1 for r in a_results if r["rho"] == 0.0)
    b_zero_count = sum(1 for r in b_results if r["rho"] == 0.0)

    # Mean |ρ|
    a_mean_abs_rho = sum(abs(r["rho"]) for r in a_results) / len(a_results)
    b_mean_abs_rho = sum(abs(r["rho"]) for r in b_results) / len(b_results)

    # Signals above VOI floor (assuming floor=0.1, which requires |ρ| > ~0.2)
    # Actually VOI depends on base rate too, but |ρ| > 0.1 is a rough proxy
    a_above_threshold = sum(1 for r in a_results if abs(r["rho"]) >= 0.2)
    b_above_threshold = sum(1 for r in b_results if abs(r["rho"]) >= 0.2)

    # "Truly independent" pattern in reasoning
    a_independent_mentions = sum(
        1 for r in a_results
        if "independent" in r["reasoning"].lower() or "unrelated" in r["reasoning"].lower()
    )
    b_independent_mentions = sum(
        1 for r in b_results
        if "independent" in r["reasoning"].lower() or "unrelated" in r["reasoning"].lower()
    )

    n = len(a_results)

    metrics = {
        "n_signals": n,
        "prompt_a": {
            "name": "Current voi.py",
            "rho_zero_count": a_zero_count,
            "rho_zero_rate": a_zero_count / n,
            "mean_abs_rho": a_mean_abs_rho,
            "above_threshold_count": a_above_threshold,
            "above_threshold_rate": a_above_threshold / n,
            "independent_mentions": a_independent_mentions,
        },
        "prompt_b": {
            "name": "Validated Exp3",
            "rho_zero_count": b_zero_count,
            "rho_zero_rate": b_zero_count / n,
            "mean_abs_rho": b_mean_abs_rho,
            "above_threshold_count": b_above_threshold,
            "above_threshold_rate": b_above_threshold / n,
            "independent_mentions": b_independent_mentions,
        },
        "comparison": {
            "rho_zero_reduction": a_zero_count - b_zero_count,
            "mean_abs_rho_improvement": b_mean_abs_rho - a_mean_abs_rho,
            "above_threshold_improvement": b_above_threshold - a_above_threshold,
        }
    }

    return metrics


def print_report(metrics: dict, results: dict):
    """Print comparison report."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    n = metrics["n_signals"]
    a = metrics["prompt_a"]
    b = metrics["prompt_b"]

    print(f"\nTotal signals: {n}")

    print("\n" + "-" * 40)
    print(f"{'Metric':<30} {'Prompt A':<12} {'Prompt B':<12}")
    print("-" * 40)
    print(f"{'ρ=0 count':<30} {a['rho_zero_count']:<12} {b['rho_zero_count']:<12}")
    print(f"{'ρ=0 rate':<30} {a['rho_zero_rate']:.1%}{'':>6} {b['rho_zero_rate']:.1%}")
    print(f"{'Mean |ρ|':<30} {a['mean_abs_rho']:.3f}{'':>7} {b['mean_abs_rho']:.3f}")
    print(f"{'|ρ| ≥ 0.2 count':<30} {a['above_threshold_count']:<12} {b['above_threshold_count']:<12}")
    print(f"{'|ρ| ≥ 0.2 rate':<30} {a['above_threshold_rate']:.1%}{'':>6} {b['above_threshold_rate']:.1%}")
    print(f"{'\"independent\" mentions':<30} {a['independent_mentions']:<12} {b['independent_mentions']:<12}")

    print("\n" + "-" * 40)
    print("VERDICT")
    print("-" * 40)

    comp = metrics["comparison"]

    # Check success criteria
    b_wins = True
    reasons = []

    if b["rho_zero_rate"] < 0.30:
        reasons.append(f"✓ ρ=0 rate {b['rho_zero_rate']:.1%} < 30% target")
    else:
        reasons.append(f"✗ ρ=0 rate {b['rho_zero_rate']:.1%} ≥ 30% target")
        b_wins = False

    if b["above_threshold_rate"] > 0.60:
        reasons.append(f"✓ |ρ| ≥ 0.2 rate {b['above_threshold_rate']:.1%} > 60% target")
    else:
        reasons.append(f"✗ |ρ| ≥ 0.2 rate {b['above_threshold_rate']:.1%} ≤ 60% target")
        b_wins = False

    if comp["rho_zero_reduction"] > 0:
        reasons.append(f"✓ Reduced ρ=0 by {comp['rho_zero_reduction']} signals")
    else:
        reasons.append(f"✗ Did not reduce ρ=0 count")
        b_wins = False

    for reason in reasons:
        print(reason)

    print()
    if b_wins:
        print("🎉 PROMPT B WINS - Proceed to Phase 2a (commit change)")
    else:
        print("⚠️  PROMPT B DOES NOT CLEARLY WIN - Proceed to Phase 2b (investigate)")

    # Show examples where they differ most
    print("\n" + "-" * 40)
    print("NOTABLE DIFFERENCES")
    print("-" * 40)

    a_results = results["prompt_a_results"]
    b_results = results["prompt_b_results"]

    diffs = []
    for ra, rb in zip(a_results, b_results):
        diff = abs(ra["rho"] - rb["rho"])
        diffs.append((diff, ra, rb))

    diffs.sort(reverse=True)

    for diff, ra, rb in diffs[:5]:
        print(f"\n  Signal: {ra['signal_text'][:50]}...")
        print(f"    Prompt A: ρ={ra['rho']:.2f} - {ra['reasoning'][:60]}...")
        print(f"    Prompt B: ρ={rb['rho']:.2f} - {rb['reasoning'][:60]}...")


async def main():
    target, signals = load_gdp_signals()

    results = await run_comparison(target, signals)

    # Save raw results
    output_dir = Path(__file__).parent / "gdp_2040" / "results"
    output_file = output_dir / "rho_prompt_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {output_file}")

    metrics = analyze_results(results)

    # Save metrics
    metrics_file = output_dir / "rho_prompt_comparison_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    print_report(metrics, results)


if __name__ == "__main__":
    asyncio.run(main())
