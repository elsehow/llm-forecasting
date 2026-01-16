#!/usr/bin/env python3
"""
Compare scenario construction approaches.

Loads results from all three approaches and scores against evaluation criteria:
- MECE: Are scenarios genuinely exclusive?
- Evaluable: Can experts in 2030 agree which scenario we're in?
- Cruxy: Do scenarios capture factors that most change GDP beliefs?
- Signal coverage: Can we find observables that update P(scenario)?

Usage:
    uv run python experiments/scenario-construction/gdp_2040/compare.py
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv
import litellm

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
MODEL = "claude-sonnet-4-20250514"


@dataclass
class EvaluationScore:
    """Score for a single criterion."""
    score: int  # 1-5
    reasoning: str


@dataclass
class ApproachEvaluation:
    """Full evaluation for an approach."""
    approach: str
    mece: EvaluationScore
    evaluable: EvaluationScore
    cruxy: EvaluationScore
    signal_coverage: EvaluationScore
    total: float  # Average score

    @property
    def scores_dict(self) -> dict:
        return {
            "mece": self.mece.score,
            "evaluable": self.evaluable.score,
            "cruxy": self.cruxy.score,
            "signal_coverage": self.signal_coverage.score,
            "total": self.total,
        }


def load_latest_results(approach: str) -> dict | None:
    """Load most recent results for an approach."""
    pattern = f"{approach}_*.json"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


async def evaluate_approach(approach: str, results: dict) -> ApproachEvaluation:
    """LLM evaluates an approach against criteria."""
    scenarios = results.get("scenarios", [])

    prompt = f"""You are evaluating scenario construction approaches for GDP 2040 forecasting.

Approach: {approach}

Scenarios:
{json.dumps(scenarios, indent=2)}

Evaluate this approach on four criteria (score 1-5, where 5 is best):

1. **MECE (Mutually Exclusive, Collectively Exhaustive)**
   - Can two scenarios co-occur? (Bad: yes)
   - Do scenarios cover the space of GDP outcomes? (Good: yes)

2. **Evaluable**
   - Could experts in 2030 agree which scenario we're in?
   - Are resolution criteria clear and measurable?

3. **Cruxy**
   - Do scenarios capture the factors that most change GDP beliefs?
   - Would knowing which scenario we're in significantly update GDP forecast?

4. **Signal Coverage**
   - Do signals resolve in 2-5 years (not 2040)?
   - Are signals concrete and observable?

Format your response as JSON:
{{
  "mece": {{"score": 1-5, "reasoning": "..."}},
  "evaluable": {{"score": 1-5, "reasoning": "..."}},
  "cruxy": {{"score": 1-5, "reasoning": "..."}},
  "signal_coverage": {{"score": 1-5, "reasoning": "..."}}
}}
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    mece = EvaluationScore(data["mece"]["score"], data["mece"]["reasoning"])
    evaluable = EvaluationScore(data["evaluable"]["score"], data["evaluable"]["reasoning"])
    cruxy = EvaluationScore(data["cruxy"]["score"], data["cruxy"]["reasoning"])
    signal_coverage = EvaluationScore(data["signal_coverage"]["score"], data["signal_coverage"]["reasoning"])

    total = (mece.score + evaluable.score + cruxy.score + signal_coverage.score) / 4

    return ApproachEvaluation(
        approach=approach,
        mece=mece,
        evaluable=evaluable,
        cruxy=cruxy,
        signal_coverage=signal_coverage,
        total=total,
    )


async def main():
    """Compare all approaches."""
    print("=" * 60)
    print("COMPARING SCENARIO CONSTRUCTION APPROACHES")
    print("=" * 60)

    approaches = ["hybrid", "topdown", "bottomup"]
    evaluations = []

    for approach in approaches:
        print(f"\nLoading {approach} results...")
        results = load_latest_results(approach)
        if results is None:
            print(f"  No results found for {approach}")
            continue

        print(f"  Found {len(results.get('scenarios', []))} scenarios")
        print(f"  Evaluating...")

        eval_result = await evaluate_approach(approach, results)
        evaluations.append(eval_result)

    if not evaluations:
        print("\nNo results to compare. Run the approach scripts first.")
        return

    # Display comparison
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Table header
    print(f"\n{'Approach':<12} {'MECE':>6} {'Eval':>6} {'Cruxy':>6} {'Signal':>6} {'TOTAL':>8}")
    print("-" * 50)

    for e in sorted(evaluations, key=lambda x: x.total, reverse=True):
        print(f"{e.approach:<12} {e.mece.score:>6} {e.evaluable.score:>6} {e.cruxy.score:>6} {e.signal_coverage.score:>6} {e.total:>8.2f}")

    # Winner
    winner = max(evaluations, key=lambda x: x.total)
    print(f"\nWinner: {winner.approach} (score: {winner.total:.2f})")

    # Detailed reasoning
    print("\n" + "-" * 60)
    print("DETAILED REASONING")
    print("-" * 60)

    for e in evaluations:
        print(f"\n{e.approach.upper()}")
        print(f"  MECE ({e.mece.score}/5): {e.mece.reasoning}")
        print(f"  Evaluable ({e.evaluable.score}/5): {e.evaluable.reasoning}")
        print(f"  Cruxy ({e.cruxy.score}/5): {e.cruxy.reasoning}")
        print(f"  Signal Coverage ({e.signal_coverage.score}/5): {e.signal_coverage.reasoning}")

    # Save comparison
    output_file = RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    comparison = {
        "evaluations": [
            {
                "approach": e.approach,
                "scores": e.scores_dict,
                "reasoning": {
                    "mece": e.mece.reasoning,
                    "evaluable": e.evaluable.reasoning,
                    "cruxy": e.cruxy.reasoning,
                    "signal_coverage": e.signal_coverage.reasoning,
                },
            }
            for e in evaluations
        ],
        "winner": winner.approach,
        "created_at": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {output_file}")

    # Next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"\nBased on results, the {winner.approach} approach performed best.")
    print("\nRecommended actions:")
    print("1. Review detailed reasoning above")
    print("2. Check if winner aligns with your intuition")
    print("3. If hybrid won → validates VOI validation findings")
    print("4. If bottom-up won → DML architecture is viable")
    print("5. If top-down won → simpler structure may suffice")
    print("\nAfter validation, update Tree of Life project note with findings.")


if __name__ == "__main__":
    asyncio.run(main())
