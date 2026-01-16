#!/usr/bin/env python3
"""
Evaluate scenario quality on three criteria:
1. Cruxiness - Do scenarios produce different outcome forecasts?
2. Evaluability - Are indicators specific enough to measure?
3. Signal Coverage - Are signals trackable and early-resolving?

Usage:
    uv run python experiments/scenario-construction/gdp_2040/evaluate.py --target gdp_2040 results/gdp_2040/*.json
    uv run python experiments/scenario-construction/gdp_2040/evaluate.py --target renewable_2050 results/renewable_2050/*.json
"""

import argparse
import json
import asyncio
import sys
from pathlib import Path
from statistics import mean, stdev
from datetime import datetime

from pydantic import BaseModel, Field
import litellm
from dotenv import load_dotenv

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import get_target, TARGETS

load_dotenv()

MODEL = "claude-sonnet-4-20250514"


# ============================================================
# Pydantic models for evaluation
# ============================================================

class ConditionalForecast(BaseModel):
    """Outcome forecast conditional on a scenario."""
    scenario_name: str
    point_estimate: float = Field(description="Expected outcome value")
    low: float = Field(description="10th percentile")
    high: float = Field(description="90th percentile")
    reasoning: str


class ConditionalForecastsResponse(BaseModel):
    """All conditional forecasts."""
    forecasts: list[ConditionalForecast]
    baseline_without_scenario: float = Field(description="Unconditional forecast without scenario info")


class IndicatorScore(BaseModel):
    """Evaluability score for one indicator."""
    indicator: str
    has_numeric_threshold: bool
    has_timeframe: bool
    has_data_source: bool
    specificity_score: int = Field(description="0-3 based on above booleans")
    suggested_improvement: str | None


class EvaluabilityResponse(BaseModel):
    """Evaluability assessment for all indicators."""
    scores: list[IndicatorScore]
    overall_assessment: str


class SignalScore(BaseModel):
    """Coverage score for one signal."""
    signal: str
    resolves_before_2030: bool
    has_clear_resolution: bool
    is_trackable: bool = Field(description="Maps to market, government stat, or verifiable event")
    trackable_source: str | None = Field(description="If trackable, what source?")
    coverage_score: int = Field(description="0-3 based on above booleans")


class SignalCoverageResponse(BaseModel):
    """Signal coverage assessment."""
    scores: list[SignalScore]
    overall_trackable_percent: float


# ============================================================
# Evaluation functions
# ============================================================

async def evaluate_cruxiness(scenarios: list[dict], config) -> dict:
    """
    Evaluate cruxiness by getting conditional outcome forecasts.

    High variance in E[outcome | scenario] = cruxy scenarios.
    """
    # Use outcome_range field if available, fall back to gdp_range for backward compat
    scenario_summaries = [
        {
            "name": s["name"],
            "description": s["description"],
            "outcome_range": s.get("outcome_range", s.get("gdp_range", "unknown")),
        }
        for s in scenarios
    ]

    # Get unit formatting
    unit_label = config.question.unit.short_label if config.question.unit else ""
    base_rate = config.question.base_rate

    prompt = f"""You are a superforecaster estimating: {config.question.text}

{config.context}

For EACH of the following scenarios, provide your forecast conditional on that scenario being true:

{json.dumps(scenario_summaries, indent=2)}

For each scenario:
1. Point estimate (expected value, numeric only, e.g., 45.0 for {unit_label})
2. 10th percentile (low end)
3. 90th percentile (high end)
4. Brief reasoning

Also provide your BASELINE forecast without knowing which scenario we're in.
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=ConditionalForecastsResponse,
    )

    result = ConditionalForecastsResponse.model_validate_json(response.choices[0].message.content)

    # Calculate cruxiness metrics
    point_estimates = [f.point_estimate for f in result.forecasts]
    baseline = result.baseline_without_scenario

    # Variance and range
    if len(point_estimates) > 1:
        variance = stdev(point_estimates) ** 2
        std_dev = stdev(point_estimates)
        spread = max(point_estimates) - min(point_estimates)
    else:
        variance = 0
        std_dev = 0
        spread = 0

    # Cruxiness score: how much do scenarios move the forecast vs baseline?
    deviations_from_baseline = [abs(p - baseline) for p in point_estimates]
    avg_deviation = mean(deviations_from_baseline) if deviations_from_baseline else 0

    # Normalize using config (e.g., $20T spread = 1.0 for GDP, 30pp spread = 1.0 for renewable)
    normalizer = config.cruxiness_normalizer

    return {
        "forecasts": [
            {
                "scenario": f.scenario_name,
                "point_estimate": f.point_estimate,
                "range": [f.low, f.high],
                "reasoning": f.reasoning,
            }
            for f in result.forecasts
        ],
        "baseline": baseline,
        "unit": unit_label,
        "metrics": {
            "spread": spread,
            "std_dev": std_dev,
            "avg_deviation_from_baseline": avg_deviation,
            "min_outcome": min(point_estimates) if point_estimates else 0,
            "max_outcome": max(point_estimates) if point_estimates else 0,
        },
        "cruxiness_score": min(spread / normalizer, 1.0),
    }


async def evaluate_evaluability(scenarios: list[dict]) -> dict:
    """
    Evaluate how specific/measurable the indicators are.
    """
    # Collect all indicators
    all_indicators = []
    for s in scenarios:
        bundle = s.get("indicator_bundle", {})
        for indicator, threshold in bundle.items():
            all_indicators.append({
                "scenario": s["name"],
                "indicator": indicator,
                "threshold": threshold,
            })

    if not all_indicators:
        return {"scores": [], "average_score": 0, "evaluability_score": 0}

    prompt = f"""You are assessing whether economic indicators are specific enough for experts to measure.

For each indicator, assess:
1. Does it have a NUMERIC threshold? (e.g., ">3%" vs "high")
2. Does it have a TIMEFRAME? (e.g., "by 2030" vs "eventually")
3. Does it have an implied DATA SOURCE? (e.g., "unemployment" implies BLS)

Indicators to assess:
{json.dumps(all_indicators, indent=2)}

Score each 0-3 based on how many criteria it meets.
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=EvaluabilityResponse,
    )

    result = EvaluabilityResponse.model_validate_json(response.choices[0].message.content)

    scores = [s.specificity_score for s in result.scores]
    avg_score = mean(scores) if scores else 0

    return {
        "indicator_scores": [
            {
                "indicator": s.indicator,
                "has_numeric": s.has_numeric_threshold,
                "has_timeframe": s.has_timeframe,
                "has_source": s.has_data_source,
                "score": s.specificity_score,
                "improvement": s.suggested_improvement,
            }
            for s in result.scores
        ],
        "average_score": avg_score,
        "evaluability_score": avg_score / 3,  # Normalize to 0-1
        "overall_assessment": result.overall_assessment,
    }


async def evaluate_signal_coverage(scenarios: list[dict], signals: list[dict] | None = None) -> dict:
    """
    Evaluate signal coverage: are signals trackable and early-resolving?
    """
    # Collect signals from scenarios or use provided signals
    if signals:
        signal_texts = [s.get("text", s) if isinstance(s, dict) else s for s in signals[:20]]  # Limit to 20
    else:
        signal_texts = []
        for s in scenarios:
            signal_texts.extend(s.get("signals", [])[:5])  # Up to 5 per scenario

    if not signal_texts:
        return {"scores": [], "trackable_percent": 0, "coverage_score": 0}

    prompt = f"""You are assessing whether economic signals are trackable and useful for forecasting.

For each signal, assess:
1. Does it RESOLVE BEFORE 2030? (early warning value)
2. Does it have CLEAR RESOLUTION CRITERIA? (unambiguous yes/no)
3. Is it TRACKABLE? (maps to prediction market, government stat, or verifiable event)

If trackable, identify the source (e.g., "Polymarket", "BLS unemployment", "Fed announcement").

Signals to assess:
{json.dumps(signal_texts, indent=2)}
"""

    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SignalCoverageResponse,
    )

    result = SignalCoverageResponse.model_validate_json(response.choices[0].message.content)

    scores = [s.coverage_score for s in result.scores]
    trackable_count = sum(1 for s in result.scores if s.is_trackable)
    early_count = sum(1 for s in result.scores if s.resolves_before_2030)

    return {
        "signal_scores": [
            {
                "signal": s.signal[:60] + "..." if len(s.signal) > 60 else s.signal,
                "early": s.resolves_before_2030,
                "clear": s.has_clear_resolution,
                "trackable": s.is_trackable,
                "source": s.trackable_source,
                "score": s.coverage_score,
            }
            for s in result.scores
        ],
        "total_signals": len(result.scores),
        "trackable_count": trackable_count,
        "trackable_percent": trackable_count / len(result.scores) if result.scores else 0,
        "early_resolving_percent": early_count / len(result.scores) if result.scores else 0,
        "average_score": mean(scores) if scores else 0,
        "coverage_score": mean(scores) / 3 if scores else 0,  # Normalize to 0-1
    }


async def evaluate_scenarios(results_file: Path, config) -> dict:
    """Run all evaluations on a results file."""
    with open(results_file) as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    signals = data.get("signals", [])
    approach = data.get("approach", "unknown")
    target = data.get("target", "unknown")

    print(f"\nEvaluating: {results_file.name}")
    print(f"Target: {target}")
    print(f"Approach: {approach}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Signals: {len(signals)}")

    # Run evaluations
    print("\n[1/3] Evaluating cruxiness...")
    cruxiness = await evaluate_cruxiness(scenarios, config)

    print("[2/3] Evaluating evaluability...")
    evaluability = await evaluate_evaluability(scenarios)

    print("[3/3] Evaluating signal coverage...")
    coverage = await evaluate_signal_coverage(scenarios, signals)

    return {
        "file": results_file.name,
        "target": target,
        "approach": approach,
        "cruxiness": cruxiness,
        "evaluability": evaluability,
        "signal_coverage": coverage,
        "summary": {
            "cruxiness_score": cruxiness["cruxiness_score"],
            "evaluability_score": evaluability["evaluability_score"],
            "coverage_score": coverage["coverage_score"],
        }
    }


def print_evaluation(eval_result: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print(f"EVALUATION: {eval_result.get('target', 'unknown')} / {eval_result['approach']}")
    print("=" * 60)

    # Cruxiness
    print("\n### CRUXINESS ###")
    crux = eval_result["cruxiness"]
    unit = crux.get("unit", "")
    print(f"Score: {crux['cruxiness_score']:.2f}")
    print(f"Outcome Spread: {crux['metrics']['min_outcome']:.1f} - {crux['metrics']['max_outcome']:.1f} {unit} ({crux['metrics']['spread']:.1f} range)")
    print(f"Baseline: {crux['baseline']:.1f} {unit}")
    print(f"Avg deviation from baseline: {crux['metrics']['avg_deviation_from_baseline']:.1f}")
    print("\nConditional forecasts:")
    for f in crux["forecasts"]:
        print(f"  {f['scenario']}: {f['point_estimate']:.1f} {unit} [{f['range'][0]:.1f}-{f['range'][1]:.1f}]")

    # Evaluability
    print("\n### EVALUABILITY ###")
    evalu = eval_result["evaluability"]
    print(f"Score: {evalu['evaluability_score']:.2f}")
    print(f"Average indicator specificity: {evalu['average_score']:.1f}/3")
    print(f"\n{evalu['overall_assessment']}")

    # Signal Coverage
    print("\n### SIGNAL COVERAGE ###")
    cov = eval_result["signal_coverage"]
    print(f"Score: {cov['coverage_score']:.2f}")
    print(f"Trackable: {cov['trackable_percent']:.0%} ({cov['trackable_count']}/{cov['total_signals']})")
    print(f"Early-resolving: {cov['early_resolving_percent']:.0%}")

    # Summary
    print("\n### SUMMARY ###")
    summary = eval_result["summary"]
    print(f"  Cruxiness:    {summary['cruxiness_score']:.2f}")
    print(f"  Evaluability: {summary['evaluability_score']:.2f}")
    print(f"  Coverage:     {summary['coverage_score']:.2f}")
    print(f"  OVERALL:      {mean(summary.values()):.2f}")


async def main():
    """Run evaluation on specified results file(s)."""
    parser = argparse.ArgumentParser(description="Evaluate scenario generation results")
    parser.add_argument(
        "--target",
        choices=list(TARGETS.keys()),
        required=True,
        help="Target question to evaluate for",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Results files to evaluate (defaults to most recent in target directory)",
    )
    args = parser.parse_args()

    # Load config
    config = get_target(args.target)
    print(f"Target: {config.question.text}")
    print(f"Cruxiness normalizer: {config.cruxiness_normalizer}")

    # Find files
    if not args.files:
        results_dir = Path(__file__).parent / "results" / args.target
        v4_files = sorted(results_dir.glob("*_v4_*.json"))
        if not v4_files:
            print(f"No v4 results found in {results_dir}. Run approach scripts first.")
            return
        files = v4_files  # All v4 results
    else:
        files = [Path(f) for f in args.files]

    all_results = []
    for f in files:
        if f.exists() and "evaluation" not in f.name:  # Skip existing evaluation files
            result = await evaluate_scenarios(f, config)
            all_results.append(result)
            print_evaluation(result)

    # Save evaluation
    if all_results:
        output_dir = Path(__file__).parent / "results" / args.target
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        eval_output = {
            "target": args.target,
            "question": {
                "id": config.question.id,
                "text": config.question.text,
                "unit": config.question.unit.type if config.question.unit else None,
            },
            "config": {
                "cruxiness_normalizer": config.cruxiness_normalizer,
            },
            "evaluations": all_results,
            "created_at": datetime.now().isoformat(),
        }

        with open(output_file, "w") as f:
            json.dump(eval_output, f, indent=2)
        print(f"\n\nEvaluation saved to: {output_file}")

        # Print summary comparison
        if len(all_results) > 1:
            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY")
            print("=" * 60)
            for r in all_results:
                s = r["summary"]
                overall = mean(s.values())
                print(f"  {r['approach'][:20]:<20} Crux: {s['cruxiness_score']:.2f}  Eval: {s['evaluability_score']:.2f}  Cov: {s['coverage_score']:.2f}  Overall: {overall:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
