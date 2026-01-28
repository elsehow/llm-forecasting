#!/usr/bin/env python3
"""
Contrastive Pairwise Experiment.

Tests whether LLM pairwise comparison ("Which of A vs B shifts the forecast more?")
can rank within-domain cruxes better than individual VOI scores (which give r~0 within domains).

Hypothesis: Direct comparison captures relative informativeness that absolute VOI scoring misses.
"""

import asyncio
import json
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path

import litellm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from prompts import FRAMINGS, format_prompt
from ranking import compute_win_rates, bradley_terry_mle, rank_by_scores

load_dotenv()

# Paths
RUSSELL_DATA_DIR = Path(__file__).parent.parent.parent / "russell-2000-crux" / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Config - can be overridden by command line
DEFAULT_MODEL = "anthropic/claude-3-haiku-20240307"
CONCURRENCY = 20  # Parallel requests


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Contrastive Pairwise Experiment")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--framing",
        type=str,
        default=None,
        choices=list(FRAMINGS.keys()),
        help="Run only a specific framing (default: all)"
    )
    return parser.parse_args()


def load_russell_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Russell 2000 cruxes and returns data."""
    cruxes_path = RUSSELL_DATA_DIR / "cruxes_with_voi.parquet"
    returns_path = RUSSELL_DATA_DIR / "stock_returns.parquet"

    if not cruxes_path.exists():
        raise FileNotFoundError(f"Cruxes not found: {cruxes_path}")
    if not returns_path.exists():
        raise FileNotFoundError(f"Returns not found: {returns_path}")

    cruxes = pd.read_parquet(cruxes_path)
    returns = pd.read_parquet(returns_path)

    return cruxes, returns


def generate_pairs(cruxes: pd.DataFrame) -> list[dict]:
    """Generate all pairwise comparisons within each stock-day.

    Args:
        cruxes: DataFrame with ticker, date, crux, crux_index, linear_voi

    Returns:
        List of pair dicts with stock-day context and crux pair info
    """
    pairs = []

    for (ticker, date), group in cruxes.groupby(["ticker", "date"]):
        group = group.reset_index(drop=True)

        # Ultimate question for this stock-day
        ultimate = f"Will {ticker} close higher than it opened on {date}?"

        # Generate all pairs within this group
        for i, j in combinations(range(len(group)), 2):
            row_a = group.iloc[i]
            row_b = group.iloc[j]

            pairs.append({
                "ticker": ticker,
                "date": str(date),
                "ultimate_question": ultimate,
                "crux_a": row_a["crux"],
                "crux_b": row_b["crux"],
                "crux_a_idx": int(row_a["crux_index"]),
                "crux_b_idx": int(row_b["crux_index"]),
                "voi_a": float(row_a["linear_voi"]),
                "voi_b": float(row_b["linear_voi"]),
                "rho_a": float(row_a["rho"]),
                "rho_b": float(row_b["rho"]),
            })

    return pairs


# Global model variable (set by main)
MODEL = DEFAULT_MODEL


async def compare_pair(
    pair: dict,
    framing: str,
    swap_order: bool = False,
    semaphore: asyncio.Semaphore = None,
    model: str = None,
) -> dict:
    """Run LLM pairwise comparison for a single pair.

    Args:
        pair: Dict with pair info
        framing: Which prompt framing to use
        swap_order: If True, swap A/B order (for order effect testing)
        semaphore: Concurrency limiter
        model: Model to use (defaults to global MODEL)

    Returns:
        Dict with comparison result
    """
    if swap_order:
        crux_a, crux_b = pair["crux_b"], pair["crux_a"]
    else:
        crux_a, crux_b = pair["crux_a"], pair["crux_b"]

    prompt = format_prompt(
        framing=framing,
        ultimate_question=pair["ultimate_question"],
        crux_a=crux_a,
        crux_b=crux_b,
    )

    use_model = model or MODEL

    async def call_llm():
        try:
            response = await litellm.acompletion(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            text = response.choices[0].message.content.strip().upper()

            # Parse answer - look for A or B anywhere in response
            winner = None
            # First check if it starts with A or B
            if text.startswith("A"):
                winner = 0
            elif text.startswith("B"):
                winner = 1
            else:
                # Check for A or B anywhere (handle "Answer: A" type responses)
                if " A" in text or text == "A" or ": A" in text or "\nA" in text:
                    winner = 0
                elif " B" in text or text == "B" or ": B" in text or "\nB" in text:
                    winner = 1

            # Adjust for swap
            if winner is not None and swap_order:
                winner = 1 - winner

            return {
                **pair,
                "framing": framing,
                "swapped": swap_order,
                "raw_answer": text,
                "winner": winner,  # 0 = original A wins, 1 = original B wins
            }
        except Exception as e:
            return {
                **pair,
                "framing": framing,
                "swapped": swap_order,
                "raw_answer": f"ERROR: {e}",
                "winner": None,
            }

    if semaphore:
        async with semaphore:
            return await call_llm()
    else:
        return await call_llm()


async def run_all_comparisons(
    pairs: list[dict],
    framings: list[str] = None,
) -> list[dict]:
    """Run all pairwise comparisons across all framings.

    To control for position bias, we run BOTH orders (A-B and B-A) for each pair
    and take majority vote. If the model truly prefers one crux, it should win
    in both orders. If it's just position bias, the votes will cancel out.

    Args:
        pairs: List of pair dicts
        framings: List of framings to test (default: all three)

    Returns:
        List of comparison results with de-biased winners
    """
    if framings is None:
        framings = list(FRAMINGS.keys())

    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_results = []

    for framing in framings:
        print(f"\n  Running framing: {framing} (both orders for each pair)")
        tasks = []

        for pair in pairs:
            # Run BOTH orders to cancel position bias
            tasks.append(compare_pair(pair, framing, swap_order=False, semaphore=semaphore))
            tasks.append(compare_pair(pair, framing, swap_order=True, semaphore=semaphore))

        results = await asyncio.gather(*tasks)

        # Aggregate results by pair
        by_pair = {}
        for r in results:
            key = (r["ticker"], r["date"], r["crux_a"], r["crux_b"])
            if key not in by_pair:
                by_pair[key] = {"normal": None, "swapped": None, "pair": r}
            if r.get("swapped", False):
                by_pair[key]["swapped"] = r["winner"]
            else:
                by_pair[key]["normal"] = r["winner"]

        # Compute de-biased winners
        debiased = []
        agreements = 0
        disagreements = 0
        for key, data in by_pair.items():
            pair = data["pair"].copy()
            normal = data["normal"]
            swapped = data["swapped"]

            if normal is None or swapped is None:
                continue

            if normal == swapped:
                # Both orders agree - this crux truly wins
                pair["debiased_winner"] = normal
                pair["agreement"] = True
                agreements += 1
            else:
                # Disagreement - position bias dominated, call it a tie/uncertain
                pair["debiased_winner"] = None  # Uncertain
                pair["agreement"] = False
                disagreements += 1

            debiased.append(pair)

        all_results.extend(debiased)

        # Progress update
        total = agreements + disagreements
        agree_rate = agreements / total if total > 0 else 0
        print(f"      Pairs: {total}, Agree: {agreements} ({agree_rate:.1%}), Disagree: {disagreements}")

    return all_results


def compute_rankings_per_stock(
    comparisons: list[dict],
    framing: str,
) -> dict[tuple[str, str], dict]:
    """Compute rankings for each stock-day.

    Args:
        comparisons: All comparison results
        framing: Which framing to use

    Returns:
        Dict mapping (ticker, date) -> ranking info
    """
    # Filter to this framing and valid de-biased results
    valid = [
        c for c in comparisons
        if c["framing"] == framing and c.get("debiased_winner") is not None
    ]

    # Group by stock-day
    by_stock = {}
    for c in valid:
        key = (c["ticker"], c["date"])
        if key not in by_stock:
            by_stock[key] = []
        # Use debiased_winner as the winner
        c_copy = c.copy()
        c_copy["winner"] = c["debiased_winner"]
        by_stock[key].append(c_copy)

    rankings = {}
    for key, stock_comps in by_stock.items():
        # Win rates
        win_rates = compute_win_rates(stock_comps)

        # Bradley-Terry (if enough comparisons)
        if len(stock_comps) >= 3:
            bt_scores = bradley_terry_mle(stock_comps)
        else:
            bt_scores = win_rates  # Fallback to win rates

        rankings[key] = {
            "win_rates": win_rates,
            "bt_scores": bt_scores,
            "n_comparisons": len(stock_comps),
        }

    return rankings


def merge_with_ground_truth(
    rankings: dict,
    cruxes: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Merge crux rankings with ground truth returns.

    Args:
        rankings: Dict from compute_rankings_per_stock
        cruxes: Original cruxes DataFrame
        returns: Returns DataFrame

    Returns:
        DataFrame with crux-level data including rankings and returns
    """
    rows = []

    for (ticker, date), ranking_info in rankings.items():
        # Get return for this stock-day (handle both string and datetime date formats)
        ret_row = returns[
            (returns["ticker"] == ticker) &
            (returns["date"].astype(str) == str(date))
        ]
        if len(ret_row) == 0:
            continue

        abs_return = abs(float(ret_row["return"].iloc[0]))

        # Get cruxes for this stock-day
        stock_cruxes = cruxes[
            (cruxes["ticker"] == ticker) &
            (cruxes["date"].astype(str) == str(date))
        ]

        for _, crux_row in stock_cruxes.iterrows():
            crux = crux_row["crux"]

            rows.append({
                "ticker": ticker,
                "date": date,
                "crux": crux,
                "crux_index": int(crux_row["crux_index"]),
                "linear_voi": float(crux_row["linear_voi"]),
                "rho": float(crux_row["rho"]),
                "abs_return": abs_return,
                "win_rate": ranking_info["win_rates"].get(crux, 0.5),
                "bt_score": ranking_info["bt_scores"].get(crux, 0.0),
            })

    return pd.DataFrame(rows)


def analyze_position_bias(comparisons: list[dict]) -> dict:
    """Analyze position bias in comparisons.

    Since we randomize A/B order, we can measure the inherent position bias
    by looking at the overall A vs B choice rate (should be ~50% if unbiased).

    Args:
        comparisons: All comparison results

    Returns:
        Dict with position bias analysis
    """
    results = {}

    for framing in FRAMINGS.keys():
        frame_comps = [c for c in comparisons if c["framing"] == framing and c["winner"] is not None]

        if not frame_comps:
            continue

        # Count raw A vs B responses (before swap correction)
        a_chosen = sum(1 for c in frame_comps if c.get("raw_answer", "").startswith("A"))
        b_chosen = sum(1 for c in frame_comps if c.get("raw_answer", "").startswith("B"))
        total = a_chosen + b_chosen

        results[framing] = {
            "a_chosen": a_chosen,
            "b_chosen": b_chosen,
            "a_rate": a_chosen / total if total > 0 else 0.5,
            "b_rate": b_chosen / total if total > 0 else 0.5,
            "bias": abs(a_chosen - b_chosen) / total if total > 0 else 0,
        }

    return results


def analyze_framing_agreement(comparisons: list[dict]) -> dict:
    """Analyze agreement across different framings.

    Args:
        comparisons: All comparison results

    Returns:
        Dict with framing agreement analysis
    """
    # Group by pair using de-biased winners
    by_pair = {}
    for c in comparisons:
        debiased = c.get("debiased_winner")
        if debiased is None:
            continue
        key = (c["ticker"], c["date"], c["crux_a"], c["crux_b"])
        if key not in by_pair:
            by_pair[key] = {}
        by_pair[key][c["framing"]] = debiased

    # Count agreement patterns
    three_way_agree = 0
    two_way_agree = 0
    no_agree = 0

    framings = list(FRAMINGS.keys())

    for key, votes in by_pair.items():
        if len(votes) < 3:
            continue

        winners = [votes.get(f) for f in framings if f in votes]
        if len(set(winners)) == 1:
            three_way_agree += 1
        elif len(set(winners)) == 2:
            two_way_agree += 1
        else:
            no_agree += 1

    total = three_way_agree + two_way_agree + no_agree
    if total == 0:
        return {"error": "No pairs with all three framings"}

    return {
        "n_pairs": total,
        "three_way_agreement": three_way_agree,
        "two_way_agreement": two_way_agree,
        "no_agreement": no_agree,
        "three_way_rate": three_way_agree / total,
        "two_way_rate": two_way_agree / total,
        "no_agreement_rate": no_agree / total,
    }


def validate_rankings(merged: pd.DataFrame) -> dict:
    """Compute correlations between rankings and ground truth.

    Args:
        merged: DataFrame with crux rankings and ground truth

    Returns:
        Dict with correlation results
    """
    results = {}

    if len(merged) < 3:
        return {
            "error": f"Too few observations ({len(merged)})",
            "win_rate": {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0},
            "bt_score": {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0},
            "voi_baseline": {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0},
        }

    # Correlation of win_rate with abs_return
    r_wr, p_wr = stats.pearsonr(merged["win_rate"], merged["abs_return"])
    rho_wr, _ = stats.spearmanr(merged["win_rate"], merged["abs_return"])

    # Correlation of bt_score with abs_return
    r_bt, p_bt = stats.pearsonr(merged["bt_score"], merged["abs_return"])
    rho_bt, _ = stats.spearmanr(merged["bt_score"], merged["abs_return"])

    # VOI baseline (for comparison)
    r_voi, p_voi = stats.pearsonr(merged["linear_voi"], merged["abs_return"])
    rho_voi, _ = stats.spearmanr(merged["linear_voi"], merged["abs_return"])

    results["win_rate"] = {
        "pearson_r": float(r_wr),
        "pearson_p": float(p_wr),
        "spearman_rho": float(rho_wr),
    }
    results["bt_score"] = {
        "pearson_r": float(r_bt),
        "pearson_p": float(p_bt),
        "spearman_rho": float(rho_bt),
    }
    results["voi_baseline"] = {
        "pearson_r": float(r_voi),
        "pearson_p": float(p_voi),
        "spearman_rho": float(rho_voi),
    }

    return results


async def main():
    global MODEL

    args = parse_args()
    MODEL = args.model
    framings_to_run = [args.framing] if args.framing else None

    print("=" * 70)
    print("CONTRASTIVE PAIRWISE EXPERIMENT")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print("\nHypothesis: LLM pairwise comparison ranks within-domain cruxes")
    print("            better than independent VOI scores (r~0 within earnings)")

    # Load data
    print("\n[1/5] Loading Russell 2000 data...")
    cruxes, returns = load_russell_data()
    print(f"      Cruxes: {len(cruxes)}, Stock-days: {cruxes.groupby(['ticker', 'date']).ngroups}")

    # Generate pairs
    print("\n[2/5] Generating pairwise comparisons...")
    pairs = generate_pairs(cruxes)
    print(f"      Generated {len(pairs)} pairs (C(5,2) = 10 per stock-day × 80)")

    # Run comparisons
    print(f"\n[3/5] Running LLM comparisons (model: {MODEL})...")
    print(f"      Framings: {list(FRAMINGS.keys())}")
    print(f"      Total calls: ~{len(pairs) * 3 * 1.1:.0f} (3 framings + 10% order tests)")

    comparisons = await run_all_comparisons(pairs, framings=framings_to_run)

    # Save raw comparisons
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "pairwise_results.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=str)
    print(f"      Saved raw results to pairwise_results.json")

    # Analyze
    print("\n[4/5] Computing rankings...")
    results_by_framing = {}

    for framing in FRAMINGS.keys():
        rankings = compute_rankings_per_stock(comparisons, framing)
        merged = merge_with_ground_truth(rankings, cruxes, returns)
        validation = validate_rankings(merged)

        results_by_framing[framing] = {
            "n_cruxes_ranked": len(merged),
            "validation": validation,
        }

        print(f"\n      {framing}:")
        print(f"        Win-rate vs |return|: r={validation['win_rate']['pearson_r']:.3f}, p={validation['win_rate']['pearson_p']:.4f}")
        print(f"        BT-score vs |return|: r={validation['bt_score']['pearson_r']:.3f}, p={validation['bt_score']['pearson_p']:.4f}")

    # Ablations
    print("\n[5/5] Running ablations...")
    position_bias = analyze_position_bias(comparisons)
    framing_agreement = analyze_framing_agreement(comparisons)

    print("\n      Position bias (raw A/B choices, should be ~50%):")
    for framing, stats in position_bias.items():
        print(f"        {framing}: A={stats['a_rate']:.1%}, B={stats['b_rate']:.1%}")
    print(f"      Framing agreement: {framing_agreement.get('three_way_rate', 0):.1%} 3-way, {framing_agreement.get('two_way_rate', 0):.1%} 2-way")

    # Final summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Find best framing
    best_framing = None
    best_r = -1
    voi_baseline_r = None

    for framing, data in results_by_framing.items():
        r = data["validation"]["win_rate"]["pearson_r"]
        if r > best_r:
            best_r = r
            best_framing = framing
        if voi_baseline_r is None:
            voi_baseline_r = data["validation"]["voi_baseline"]["pearson_r"]

    print(f"\n{'Framing':<15} {'r(win_rate)':<12} {'p-value':<10} {'r(BT)':<10}")
    print("-" * 50)
    for framing, data in results_by_framing.items():
        wr = data["validation"]["win_rate"]
        bt = data["validation"]["bt_score"]
        marker = " *" if framing == best_framing else ""
        print(f"{framing:<15} {wr['pearson_r']:>10.3f} {wr['pearson_p']:>10.4f} {bt['pearson_r']:>10.3f}{marker}")

    print(f"\nVOI baseline:   r={voi_baseline_r:.3f}")

    # Success evaluation
    delta_r = best_r - voi_baseline_r if voi_baseline_r else 0
    print("\n" + "-" * 50)
    if best_r > 0.10 and results_by_framing[best_framing]["validation"]["win_rate"]["pearson_p"] < 0.10:
        print(f"SUCCESS: {best_framing} framing achieves r={best_r:.3f}, ΔR=+{delta_r:.3f} over VOI")
    elif best_r > voi_baseline_r:
        print(f"PARTIAL: {best_framing} beats VOI but effect weak (r={best_r:.3f})")
    else:
        print(f"FAIL: No framing beats VOI baseline (best r={best_r:.3f} vs VOI r={voi_baseline_r:.3f})")

    # Save full results
    output = {
        "metadata": {
            "experiment": "contrastive_pairwise",
            "model": MODEL,
            "n_pairs": len(pairs),
            "n_comparisons": len(comparisons),
            "run_at": datetime.now().isoformat(),
        },
        "results_by_framing": results_by_framing,
        "ablations": {
            "position_bias": position_bias,
            "framing_agreement": framing_agreement,
        },
        "summary": {
            "best_framing": best_framing,
            "best_r": float(best_r),
            "voi_baseline_r": float(voi_baseline_r) if voi_baseline_r else None,
            "delta_r": float(delta_r),
        },
    }

    with open(OUTPUT_DIR / "contrastive_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'contrastive_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
