#!/usr/bin/env python3
"""
MVP Benchmark for Question Generation (Phase 1)

Pipeline:
1. Select high-volume Polymarket ultimates
2. Generate cruxes for each
3. Score cruxes by VOI (ρ estimation → conditionals → Linear VOI)
4. Generate LLM ranking baseline
5. Compare: do VOI rankings beat naive LLM ranking?
6. Cross-validate: do generated cruxes match known high-ρ pairs?

Usage:
    uv run python experiments/question-generation/benchmark-mvp/run_benchmark.py
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
import litellm

from llm_forecasting.models import Signal
from llm_forecasting.voi import estimate_rho as _estimate_rho_core

load_dotenv()

# Paths
BENCHMARK_DIR = Path(__file__).parent
CONDITIONAL_DIR = BENCHMARK_DIR.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Config
MODEL = "anthropic/claude-sonnet-4-20250514"
MODEL_CHEAP = "anthropic/claude-3-haiku-20240307"  # For bulk operations
MIN_VOLUME = 1_000_000  # $1M minimum volume
N_ULTIMATES = 20  # Number of ultimates to test
N_CRUXES_PER_ULTIMATE = 10  # Cruxes to generate per ultimate


def load_data():
    """Load markets and pairs data."""
    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)

    # Build lookup from question to condition_id
    question_to_cond = {m["question"]: m["condition_id"] for m in markets}

    # Build lookup from condition_id pair to rho
    pair_rhos = {}
    for p in pairs:
        cond_a = p["market_a"]["condition_id"]
        cond_b = p["market_b"]["condition_id"]
        rho = p["rho"]
        if not np.isnan(rho):
            pair_rhos[(cond_a, cond_b)] = rho
            pair_rhos[(cond_b, cond_a)] = rho

    return markets, pairs, question_to_cond, pair_rhos


def select_ultimates(markets: list[dict], n: int = N_ULTIMATES) -> list[dict]:
    """Select diverse, high-volume ultimates for crux generation."""
    # Filter by volume
    high_volume = [m for m in markets if m.get("volume_total", 0) >= MIN_VOLUME]

    # Exclude overly specific questions (sports matches, individual nominations)
    excluded_patterns = [
        "win the 2025",  # Sports
        "win the 2024",
        "nomination",  # Too many similar 2028 nomination questions
        "win the 2028 Democratic",
        "win the 2028 Republican",
        "bps after",  # Fed rate changes (related but specific)
    ]

    def is_good_ultimate(m):
        q = m.get("question", "").lower()
        return not any(p in q for p in excluded_patterns)

    filtered = [m for m in high_volume if is_good_ultimate(m)]

    # Sort by volume and take top N
    sorted_markets = sorted(filtered, key=lambda m: m.get("volume_total", 0), reverse=True)

    return sorted_markets[:n]


CRUX_GENERATION_PROMPT = """You are a superforecaster. Given an ultimate question (a long-horizon prediction market question), generate {n} intermediate "crux" questions that would help predict the answer.

A good crux question:
1. Resolves BEFORE the ultimate question
2. Has a clear, unambiguous resolution criterion
3. Would significantly update your probability if answered
4. Is specific and measurable

Ultimate Question: {ultimate}

Generate exactly {n} crux questions. Format as a JSON array of strings.
Example: ["Will X happen by Y date?", "Will Z be announced before W?", ...]

JSON array only:"""


async def generate_cruxes(ultimate: str, n: int = N_CRUXES_PER_ULTIMATE) -> list[str]:
    """Generate crux questions for an ultimate."""
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": CRUX_GENERATION_PROMPT.format(ultimate=ultimate, n=n)
            }],
            max_tokens=1000,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        cruxes = json.loads(text)
        return cruxes[:n]
    except Exception as e:
        print(f"  Error generating cruxes: {e}")
        return []


async def estimate_rho(ultimate: str, crux: str) -> tuple[float, str]:
    """Estimate correlation between ultimate and crux.

    Delegates to core llm_forecasting.voi.estimate_rho().
    """
    return await _estimate_rho_core(ultimate, crux, model=MODEL_CHEAP)


CONDITIONAL_FORECAST_PROMPT = """You are a superforecaster. Estimate probabilities for these prediction market questions.

Ultimate Question: {ultimate}
Crux Question: {crux}
Estimated correlation (ρ): {rho:.2f}

Given this correlation, estimate:
1. P(Ultimate = YES) - your base probability
2. P(Crux = YES) - probability the crux resolves YES
3. P(Ultimate = YES | Crux = YES) - if the crux resolves YES
4. P(Ultimate = YES | Crux = NO) - if the crux resolves NO

The correlation ρ = {rho:.2f} means:
- If ρ > 0: crux YES should increase P(Ultimate)
- If ρ < 0: crux YES should decrease P(Ultimate)
- If ρ ≈ 0: crux answer shouldn't affect P(Ultimate) much

Respond with JSON only:
{{"p_ultimate": <0-1>, "p_crux": <0-1>, "p_ultimate_given_crux_yes": <0-1>, "p_ultimate_given_crux_no": <0-1>}}"""


async def elicit_conditionals(ultimate: str, crux: str, rho: float) -> dict:
    """Elicit conditional probabilities."""
    try:
        response = await litellm.acompletion(
            model=MODEL_CHEAP,  # Use cheaper model
            messages=[{
                "role": "user",
                "content": CONDITIONAL_FORECAST_PROMPT.format(
                    ultimate=ultimate, crux=crux, rho=rho
                )
            }],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def compute_linear_voi(p_ultimate: float, p_crux: float,
                       p_ult_given_yes: float, p_ult_given_no: float) -> float:
    """Compute Linear VOI (expected absolute belief shift)."""
    shift_yes = abs(p_ult_given_yes - p_ultimate)
    shift_no = abs(p_ult_given_no - p_ultimate)
    return p_crux * shift_yes + (1 - p_crux) * shift_no


def crux_to_signal(
    crux: str,
    ultimate_id: str,
    index: int,
    rho: float,
    voi: float,
    rho_reasoning: str | None = None,
    p_crux: float | None = None,
) -> Signal:
    """Convert a scored crux to a Signal object.

    Args:
        crux: The crux question text
        ultimate_id: ID of the ultimate question (used to generate signal ID)
        index: Index of this crux (for ID generation)
        rho: Estimated correlation with ultimate
        voi: Computed VOI score
        rho_reasoning: Explanation of rho estimate
        p_crux: P(crux=yes), stored as base_rate

    Returns:
        Signal object with VOI metadata
    """
    return Signal(
        id=f"{ultimate_id}_crux_{index}",
        source="llm",
        text=crux,
        voi=voi,
        rho=rho,
        rho_reasoning=rho_reasoning,
        base_rate=p_crux,
    )


LLM_RANKING_PROMPT = """You are a superforecaster. Rank these candidate crux questions by how useful they would be for predicting the ultimate question.

Ultimate Question: {ultimate}

Candidate Cruxes:
{cruxes_formatted}

Rank from most useful (1) to least useful ({n}). Consider:
- Does it resolve before the ultimate?
- Would the answer significantly update your probability?
- Is it measurable and clear?

Respond with JSON only: {{"rankings": [<list of crux indices in order from most to least useful>]}}"""


async def get_llm_ranking(ultimate: str, cruxes: list[str]) -> list[int]:
    """Get naive LLM ranking of cruxes."""
    cruxes_formatted = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cruxes))

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": LLM_RANKING_PROMPT.format(
                    ultimate=ultimate,
                    cruxes_formatted=cruxes_formatted,
                    n=len(cruxes)
                )
            }],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return result["rankings"]
    except Exception as e:
        print(f"  Error getting LLM ranking: {e}")
        return list(range(1, len(cruxes) + 1))


def compute_rank_correlation(voi_ranking: list[int], llm_ranking: list[int]) -> float:
    """Compute Spearman rank correlation between VOI and LLM rankings."""
    from scipy import stats

    # Convert to ranks
    n = len(voi_ranking)
    voi_ranks = [0] * n
    llm_ranks = [0] * n

    for rank, idx in enumerate(voi_ranking):
        voi_ranks[idx - 1] = rank + 1
    for rank, idx in enumerate(llm_ranking):
        llm_ranks[idx - 1] = rank + 1

    rho, p = stats.spearmanr(voi_ranks, llm_ranks)
    return rho, p


async def process_ultimate(ultimate_data: dict, pair_rhos: dict,
                           question_to_cond: dict) -> dict:
    """Process a single ultimate: generate cruxes, score, rank."""
    ultimate = ultimate_data["question"]
    ultimate_cond = ultimate_data["condition_id"]

    print(f"\n{'='*60}")
    print(f"Ultimate: {ultimate[:60]}...")

    # 1. Generate cruxes
    print("  Generating cruxes...")
    cruxes = await generate_cruxes(ultimate)
    if not cruxes:
        return {"error": "No cruxes generated"}

    print(f"  Generated {len(cruxes)} cruxes")

    # 2. Score each crux by VOI
    print("  Scoring cruxes by VOI...")
    crux_scores = []

    for i, crux in enumerate(cruxes):
        # Estimate ρ
        rho_est, rho_reason = await estimate_rho(ultimate, crux)

        # Check if this crux matches a known Polymarket question
        known_rho = None
        matched_cond = question_to_cond.get(crux)
        if matched_cond and (ultimate_cond, matched_cond) in pair_rhos:
            known_rho = pair_rhos[(ultimate_cond, matched_cond)]

        # Elicit conditionals
        conditionals = await elicit_conditionals(ultimate, crux, rho_est)

        if "error" in conditionals:
            voi = 0.0
        else:
            voi = compute_linear_voi(
                conditionals["p_ultimate"],
                conditionals["p_crux"],
                conditionals["p_ultimate_given_crux_yes"],
                conditionals["p_ultimate_given_crux_no"]
            )

        crux_scores.append({
            "crux": crux,
            "rho_estimated": rho_est,
            "rho_reasoning": rho_reason,
            "rho_known": known_rho,
            "conditionals": conditionals,
            "voi": voi,
        })

        print(f"    {i+1}. VOI={voi:.3f}, ρ={rho_est:.2f}: {crux[:40]}...")

    # Convert to Signal objects for type-safe access
    signals = [
        crux_to_signal(
            crux=cs["crux"],
            ultimate_id=ultimate_cond,
            index=i,
            rho=cs["rho_estimated"],
            voi=cs["voi"],
            rho_reasoning=cs["rho_reasoning"],
            p_crux=cs["conditionals"].get("p_crux") if "error" not in cs["conditionals"] else None,
        )
        for i, cs in enumerate(crux_scores)
    ]

    # 3. Rank by VOI
    voi_ranking = sorted(range(len(cruxes)), key=lambda i: -crux_scores[i]["voi"])
    voi_ranking = [i + 1 for i in voi_ranking]  # 1-indexed

    # 4. Get LLM ranking
    print("  Getting LLM ranking baseline...")
    llm_ranking = await get_llm_ranking(ultimate, cruxes)

    # 5. Compare rankings
    if len(voi_ranking) == len(llm_ranking):
        rho_corr, p_corr = compute_rank_correlation(voi_ranking, llm_ranking)
        print(f"  Rank correlation: ρ={rho_corr:.2f}, p={p_corr:.3f}")
    else:
        rho_corr, p_corr = None, None

    return {
        "ultimate": ultimate,
        "condition_id": ultimate_cond,
        "volume": ultimate_data.get("volume_total", 0),
        "crux_scores": crux_scores,
        "signals": signals,  # Signal objects for type-safe access
        "voi_ranking": voi_ranking,
        "llm_ranking": llm_ranking,
        "rank_correlation": {"rho": rho_corr, "p": p_corr},
    }


async def main():
    print("=" * 70)
    print("MVP BENCHMARK: Question Generation (Phase 1)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    markets, pairs, question_to_cond, pair_rhos = load_data()
    print(f"  {len(markets)} markets, {len(pairs)} pairs, {len(pair_rhos)} ρ values")

    # Select ultimates
    print("\nSelecting ultimates...")
    ultimates = select_ultimates(markets, N_ULTIMATES)
    print(f"  Selected {len(ultimates)} ultimates")

    for i, u in enumerate(ultimates):
        vol = u.get("volume_total", 0)
        q = u.get("question", "")[:50]
        print(f"  {i+1}. ${vol:,.0f} - {q}...")

    # Process each ultimate
    print("\n" + "=" * 70)
    print("PROCESSING ULTIMATES")
    print("=" * 70)

    results = []
    for ultimate in ultimates:
        result = await process_ultimate(ultimate, pair_rhos, question_to_cond)
        results.append(result)

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    valid_results = [r for r in results if "error" not in r]

    # Average VOI
    all_vois = []
    for r in valid_results:
        for cs in r["crux_scores"]:
            all_vois.append(cs["voi"])

    print(f"\nTotal cruxes generated: {len(all_vois)}")
    print(f"Mean VOI: {np.mean(all_vois):.3f}")
    print(f"Median VOI: {np.median(all_vois):.3f}")
    print(f"Max VOI: {np.max(all_vois):.3f}")

    # Rank correlation with LLM baseline
    correlations = [r["rank_correlation"]["rho"] for r in valid_results
                   if r["rank_correlation"]["rho"] is not None]

    print(f"\nVOI vs LLM Ranking:")
    print(f"  Mean ρ: {np.mean(correlations):.2f}")
    print(f"  Median ρ: {np.median(correlations):.2f}")

    # High correlation means VOI agrees with LLM intuition
    # Low correlation means VOI provides different signal
    if np.mean(correlations) > 0.7:
        print("  → High agreement: VOI aligns with LLM intuition")
    elif np.mean(correlations) > 0.3:
        print("  → Moderate agreement: VOI provides some different signal")
    else:
        print("  → Low agreement: VOI provides very different ranking than LLM")

    # Top cruxes by VOI (using Signal objects)
    print("\n" + "-" * 70)
    print("TOP 10 CRUXES BY VOI")
    print("-" * 70)

    # Collect all signals with their ultimate question
    all_signals: list[tuple[str, Signal]] = []
    for r in valid_results:
        for signal in r["signals"]:
            all_signals.append((r["ultimate"], signal))

    # Sort by VOI and take top 10
    top_signals = sorted(all_signals, key=lambda x: -x[1].voi)[:10]
    for i, (ultimate, signal) in enumerate(top_signals):
        print(f"\n{i+1}. VOI={signal.voi:.3f}, ρ={signal.rho:.2f}")
        print(f"   Ultimate: {ultimate[:50]}...")
        print(f"   Crux: {signal.text[:60]}...")
        if signal.is_synthetic:
            print(f"   Source: LLM-generated")

    # Save results (serialize Signal objects to dicts)
    serializable_results = []
    for r in results:
        if "error" in r:
            serializable_results.append(r)
        else:
            serializable_results.append({
                **{k: v for k, v in r.items() if k != "signals"},
                "signals": [s.model_dump(mode="json") for s in r["signals"]],
            })

    output = {
        "metadata": {
            "n_ultimates": len(ultimates),
            "n_cruxes_per_ultimate": N_CRUXES_PER_ULTIMATE,
            "total_cruxes": len(all_vois),
            "model": MODEL,
            "model_cheap": MODEL_CHEAP,
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": {
            "mean_voi": float(np.mean(all_vois)),
            "median_voi": float(np.median(all_vois)),
            "max_voi": float(np.max(all_vois)),
            "mean_rank_correlation": float(np.mean(correlations)) if correlations else None,
        },
        "results": serializable_results,
    }

    output_path = RESULTS_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
