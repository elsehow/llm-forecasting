#!/usr/bin/env python3
"""
Contrarian Flip Experiment.

Tests whether inverting LLM "informative" rankings yields positive correlation with |return|.

Hypothesis: The original contrastive experiment showed r=-0.127 with extreme position bias
(82% chose B). If we flip the rankings, we should get r≈+0.127, which would beat the r>0.1 baseline.

Two theories:
1. Position bias theory: Model knows which cruxes matter, but B-bias inverted the rankings
2. Inverted intuition theory: Model genuinely doesn't understand earnings prediction

This experiment tests theory #1 by simply negating the Bradley-Terry scores.
"""

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from ranking import bradley_terry_mle

# Paths
DATA_DIR = Path(__file__).parent / "data"
RUSSELL_DATA_DIR = Path(__file__).parent.parent.parent / "russell-2000-crux" / "data"


def load_pairwise_results() -> list[dict]:
    """Load existing pairwise comparison results."""
    path = DATA_DIR / "pairwise_results.json"
    with open(path) as f:
        return json.load(f)


def load_returns() -> pd.DataFrame:
    """Load stock returns data."""
    return pd.read_parquet(RUSSELL_DATA_DIR / "stock_returns.parquet")


def compute_bt_scores_per_stockday(
    comparisons: list[dict],
    framing: str = "informative",
    use_debiased: bool = True,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute Bradley-Terry scores for each stock-day.

    Args:
        comparisons: Pairwise comparison results
        framing: Which framing to filter on
        use_debiased: If True, use debiased_winner (order-corrected), else raw winner

    Returns:
        Dict mapping (ticker, date) -> {crux: bt_score}
    """
    # Filter to framing and valid results
    winner_key = "debiased_winner" if use_debiased else "winner"
    valid = [
        c for c in comparisons
        if c["framing"] == framing and c.get(winner_key) is not None
    ]

    print(f"Filtered to {len(valid)} valid comparisons for framing='{framing}'")

    # Group by stock-day
    by_stockday = {}
    for c in valid:
        key = (c["ticker"], c["date"])
        if key not in by_stockday:
            by_stockday[key] = []
        # Use selected winner for BT computation
        c_copy = c.copy()
        c_copy["winner"] = c[winner_key]
        by_stockday[key].append(c_copy)

    # Compute BT scores for each stock-day
    bt_scores = {}
    for key, stock_comps in by_stockday.items():
        if len(stock_comps) >= 3:  # Need enough comparisons for BT
            bt_scores[key] = bradley_terry_mle(stock_comps)
        else:
            # Fallback: use simple win-rate as pseudo-score
            from ranking import compute_win_rates
            bt_scores[key] = compute_win_rates(stock_comps)

    return bt_scores


def merge_bt_with_returns(
    bt_scores: dict[tuple[str, str], dict[str, float]],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Merge BT scores with ground truth returns.

    Args:
        bt_scores: Dict mapping (ticker, date) -> {crux: score}
        returns: Returns DataFrame

    Returns:
        DataFrame with crux-level bt_score and abs_return
    """
    rows = []

    for (ticker, date), crux_scores in bt_scores.items():
        # Get return for this stock-day
        ret_row = returns[
            (returns["ticker"] == ticker) &
            (returns["date"].astype(str) == str(date))
        ]
        if len(ret_row) == 0:
            continue

        abs_return = abs(float(ret_row["return"].iloc[0]))

        for crux, score in crux_scores.items():
            rows.append({
                "ticker": ticker,
                "date": date,
                "crux": crux,
                "bt_score": score,
                "abs_return": abs_return,
            })

    return pd.DataFrame(rows)


def run_experiment():
    """Run the contrarian flip experiment."""
    print("=" * 70)
    print("CONTRARIAN FLIP EXPERIMENT")
    print("=" * 70)
    print("\nHypothesis: Flipping LLM rankings (r=-0.127) gives r≈+0.127")

    # Load data
    print("\n[1/4] Loading existing data...")
    comparisons = load_pairwise_results()
    returns = load_returns()
    print(f"      Loaded {len(comparisons)} comparisons, {len(returns)} returns")

    # Compute original BT scores for "informative" framing
    print("\n[2/4] Computing Bradley-Terry scores (informative framing)...")
    bt_scores_original = compute_bt_scores_per_stockday(comparisons, framing="informative")
    print(f"      Computed BT scores for {len(bt_scores_original)} stock-days")

    # Merge with returns
    print("\n[3/4] Merging with ground truth returns...")
    merged_original = merge_bt_with_returns(bt_scores_original, returns)
    print(f"      Merged {len(merged_original)} crux observations")

    # Compute original correlation
    r_original, p_original = stats.pearsonr(
        merged_original["bt_score"], merged_original["abs_return"]
    )
    rho_original, _ = stats.spearmanr(
        merged_original["bt_score"], merged_original["abs_return"]
    )

    print(f"\n      Original BT vs |return|:")
    print(f"        Pearson r  = {r_original:.4f} (p={p_original:.4f})")
    print(f"        Spearman ρ = {rho_original:.4f}")

    # FLIP: negate the scores
    print("\n[4/4] Flipping rankings (negating BT scores)...")
    bt_scores_flipped = {
        key: {crux: -score for crux, score in crux_scores.items()}
        for key, crux_scores in bt_scores_original.items()
    }

    merged_flipped = merge_bt_with_returns(bt_scores_flipped, returns)

    # Compute flipped correlation
    r_flipped, p_flipped = stats.pearsonr(
        merged_flipped["bt_score"], merged_flipped["abs_return"]
    )
    rho_flipped, _ = stats.spearmanr(
        merged_flipped["bt_score"], merged_flipped["abs_return"]
    )

    print(f"\n      Flipped BT vs |return|:")
    print(f"        Pearson r  = {r_flipped:.4f} (p={p_flipped:.4f})")
    print(f"        Spearman ρ = {rho_flipped:.4f}")

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Original':<15} {'Flipped':<15}")
    print("-" * 50)
    print(f"{'Pearson r':<20} {r_original:>12.4f} {r_flipped:>15.4f}")
    print(f"{'p-value':<20} {p_original:>12.4f} {p_flipped:>15.4f}")
    print(f"{'Spearman ρ':<20} {rho_original:>12.4f} {rho_flipped:>15.4f}")

    # Interpretation
    print("\n" + "-" * 50)

    if r_flipped > 0.10 and p_flipped < 0.05:
        interpretation = "STRONG SUCCESS"
        explanation = "Flipping achieves r>0.1 with p<0.05. Position bias was the problem."
    elif r_flipped > 0 and p_flipped < 0.10:
        interpretation = "WEAK SUCCESS"
        explanation = "Flipping is directionally correct but needs more data."
    elif abs(r_flipped) < 0.05:
        interpretation = "NULL"
        explanation = "Flipping gives r≈0. Model intuition is random within-domain."
    else:
        interpretation = "STILL INVERTED"
        explanation = "Flipping doesn't help. Deeper problem with the approach."

    print(f"Interpretation: {interpretation}")
    print(f"  {explanation}")

    # Save results
    results = {
        "experiment": "contrarian_flip",
        "hypothesis": "Flipping LLM rankings reverses negative correlation",
        "n_comparisons": len([c for c in comparisons if c["framing"] == "informative"]),
        "n_crux_observations": len(merged_original),
        "n_stockdays": len(bt_scores_original),
        "original": {
            "pearson_r": float(r_original),
            "pearson_p": float(p_original),
            "spearman_rho": float(rho_original),
        },
        "flipped": {
            "pearson_r": float(r_flipped),
            "pearson_p": float(p_flipped),
            "spearman_rho": float(rho_flipped),
        },
        "interpretation": interpretation,
        "explanation": explanation,
        "success_criteria": {
            "strong_success": "r>0.1, p<0.05",
            "weak_success": "r>0, p<0.1",
            "null": "r≈0",
            "still_inverted": "r<0",
        },
    }

    output_path = DATA_DIR / "contrarian_flip_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
