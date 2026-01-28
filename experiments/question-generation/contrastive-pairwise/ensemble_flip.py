#!/usr/bin/env python3
"""
Ensemble Flip Experiment.

Combines flipped pairwise BT scores with flipped VOI scores to test whether
ensemble of inverted signals yields more robust predictions.

Background:
- Pairwise BT flip: r=+0.162, p=0.015 (significant)
- VOI flip (crux): r=+0.087, p=0.082 (marginal)

Question: Does combining them improve or degrade signal?
"""

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from ranking import bradley_terry_mle, compute_win_rates

# Paths
DATA_DIR = Path(__file__).parent / "data"
RUSSELL_DATA_DIR = Path(__file__).parent.parent.parent / "russell-2000-crux" / "data"


def load_pairwise_results() -> list[dict]:
    """Load existing pairwise comparison results."""
    path = DATA_DIR / "pairwise_results.json"
    with open(path) as f:
        return json.load(f)


def load_cruxes_with_voi() -> pd.DataFrame:
    """Load cruxes with VOI scores."""
    return pd.read_parquet(RUSSELL_DATA_DIR / "cruxes_with_voi.parquet")


def load_returns() -> pd.DataFrame:
    """Load stock returns data."""
    return pd.read_parquet(RUSSELL_DATA_DIR / "stock_returns.parquet")


def compute_bt_scores_per_stockday(
    comparisons: list[dict],
    framing: str = "informative",
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute Bradley-Terry scores for each stock-day.

    Returns:
        Dict mapping (ticker, date) -> {crux: bt_score}
    """
    valid = [
        c for c in comparisons
        if c["framing"] == framing and c.get("debiased_winner") is not None
    ]

    by_stockday = {}
    for c in valid:
        key = (c["ticker"], c["date"])
        if key not in by_stockday:
            by_stockday[key] = []
        c_copy = c.copy()
        c_copy["winner"] = c["debiased_winner"]
        by_stockday[key].append(c_copy)

    bt_scores = {}
    for key, stock_comps in by_stockday.items():
        if len(stock_comps) >= 3:
            bt_scores[key] = bradley_terry_mle(stock_comps)
        else:
            bt_scores[key] = compute_win_rates(stock_comps)

    return bt_scores


def zscore_within_stockday(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Z-score normalize scores within each stock-day."""
    def zscore(x):
        if len(x) <= 1 or x.std() == 0:
            return x * 0  # Return zeros if can't normalize
        return (x - x.mean()) / x.std()

    df = df.copy()
    df[f"{score_col}_z"] = df.groupby(["ticker", "date"])[score_col].transform(zscore)
    return df


def run_experiment():
    """Run the ensemble flip experiment."""
    print("=" * 70)
    print("ENSEMBLE FLIP EXPERIMENT")
    print("=" * 70)
    print("\nHypothesis: Combining flipped pairwise + VOI improves signal")

    # Load data
    print("\n[1/5] Loading data...")
    comparisons = load_pairwise_results()
    cruxes = load_cruxes_with_voi()
    returns = load_returns()

    # Filter to earnings days
    earnings_returns = returns[returns["is_earnings_day"]].copy()
    earnings_returns["abs_return"] = earnings_returns["return"].abs()

    print(f"      Pairwise comparisons: {len(comparisons)}")
    print(f"      Cruxes with VOI: {len(cruxes)}")
    print(f"      Earnings days: {len(earnings_returns)}")

    # Compute flipped BT scores
    print("\n[2/5] Computing flipped Bradley-Terry scores...")
    bt_scores = compute_bt_scores_per_stockday(comparisons, framing="informative")
    print(f"      BT scores for {len(bt_scores)} stock-days")

    # Build pairwise dataframe
    pairwise_rows = []
    for (ticker, date), crux_scores in bt_scores.items():
        for crux, score in crux_scores.items():
            pairwise_rows.append({
                "ticker": ticker,
                "date": date,
                "crux": crux,
                "bt_score": -score,  # FLIP: negate
            })
    pairwise_df = pd.DataFrame(pairwise_rows)
    print(f"      Pairwise observations: {len(pairwise_df)}")

    # Merge VOI with earnings returns
    print("\n[3/5] Preparing VOI data...")
    voi_df = cruxes.merge(
        earnings_returns[["ticker", "date", "abs_return"]],
        on=["ticker", "date"],
        how="inner"
    )
    voi_df["flipped_voi"] = -voi_df["linear_voi"]  # FLIP: negate
    print(f"      VOI observations: {len(voi_df)}")

    # Merge pairwise with VOI on crux
    print("\n[4/5] Merging signals...")
    # Need to match cruxes between datasets - use text matching
    merged = pairwise_df.merge(
        voi_df[["ticker", "date", "crux", "flipped_voi", "abs_return"]],
        on=["ticker", "date", "crux"],
        how="inner"
    )
    print(f"      Matched observations: {len(merged)}")
    print(f"      Stock-days: {merged.groupby(['ticker', 'date']).ngroups}")

    # Aggregate to stock-day level (since abs_return is constant within stock-day)
    # Z-scoring within stock-day removes cross-stock-day variation which is what we need
    print("\n      Aggregating to stock-day level...")

    pairwise_agg = pairwise_df.groupby(["ticker", "date"])["bt_score"].mean().reset_index()
    pairwise_agg.columns = ["ticker", "date", "bt_score_mean"]

    voi_agg = voi_df.groupby(["ticker", "date"]).agg({
        "flipped_voi": "mean",
        "abs_return": "first"
    }).reset_index()

    merged = pairwise_agg.merge(voi_agg, on=["ticker", "date"], how="inner")
    print(f"      Stock-day observations: {len(merged)}")

    # Z-score globally (not within stock-day) to put on same scale
    merged["bt_z"] = (merged["bt_score_mean"] - merged["bt_score_mean"].mean()) / merged["bt_score_mean"].std()
    merged["voi_z"] = (merged["flipped_voi"] - merged["flipped_voi"].mean()) / merged["flipped_voi"].std()

    # Compute ensemble
    merged["ensemble"] = (merged["bt_z"] + merged["voi_z"]) / 2

    # Compute correlations
    print("\n[5/5] Computing correlations...")

    r_bt, p_bt = stats.pearsonr(merged["bt_z"], merged["abs_return"])
    r_voi, p_voi = stats.pearsonr(merged["voi_z"], merged["abs_return"])
    r_ensemble, p_ensemble = stats.pearsonr(merged["ensemble"], merged["abs_return"])

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Signal':<25} {'r':<12} {'p':<12}")
    print("-" * 50)
    print(f"{'Pairwise BT flip (z)':<25} {r_bt:>10.4f} {p_bt:>12.4f}")
    print(f"{'VOI flip (z)':<25} {r_voi:>10.4f} {p_voi:>12.4f}")
    print(f"{'Ensemble (avg z)':<25} {r_ensemble:>10.4f} {p_ensemble:>12.4f}")

    print(f"\n{'n observations:':<25} {len(merged)}")

    # Interpretation
    print("\n" + "-" * 50)

    # Compare ensemble to pairwise at same level (stock-day)
    # Original crux-level r=0.162 is not comparable to stock-day level

    if r_ensemble > r_bt + 0.02:
        interpretation = "IMPROVED"
        explanation = f"Ensemble ({r_ensemble:.3f}) beats pairwise alone ({r_bt:.3f}) at stock-day level"
    elif abs(r_ensemble - r_bt) < 0.02:
        interpretation = "EQUIVALENT"
        explanation = f"Ensemble ({r_ensemble:.3f}) ≈ pairwise ({r_bt:.3f}). VOI adds minimal value."
    else:
        interpretation = "DEGRADED"
        explanation = f"Ensemble ({r_ensemble:.3f}) < pairwise ({r_bt:.3f}). VOI adds noise."

    print(f"Interpretation: {interpretation}")
    print(f"  {explanation}")

    # Recommendation
    print("\n" + "-" * 50)
    print("RECOMMENDATION")
    print("-" * 50)
    if interpretation == "DEGRADED":
        print(f"  Use pairwise flip alone (r={r_bt:.3f})")
        print("  VOI flip adds noise, not signal")
        surprise_test = False
    elif r_ensemble > 0.20 and p_ensemble < 0.05:
        print(f"  Strong signal (r={r_ensemble:.3f}, p={p_ensemble:.3f}). Proceed to surprise hypothesis.")
        surprise_test = True
    else:
        print(f"  VOI adds minimal incremental value. Pairwise flip remains primary method.")
        surprise_test = r_ensemble > 0.15

    # Save results
    results = {
        "experiment": "ensemble_flip",
        "n_stockdays": len(merged),
        "signals": {
            "pairwise_bt_z": {"pearson_r": float(r_bt), "pearson_p": float(p_bt)},
            "voi_z": {"pearson_r": float(r_voi), "pearson_p": float(p_voi)},
            "ensemble": {"pearson_r": float(r_ensemble), "pearson_p": float(p_ensemble)},
        },
        "comparison": {
            "pairwise_stockday_r": float(r_bt),
            "ensemble_delta": float(r_ensemble - r_bt),
            "note": "Comparison at stock-day level (n=62). Original crux-level r=0.162 not comparable.",
        },
        "interpretation": interpretation,
        "explanation": explanation,
        "surprise_test_warranted": surprise_test,
    }

    output_path = DATA_DIR / "ensemble_flip_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
