#!/usr/bin/env python3
"""
Validate MMR selection on Russell 2000 data.

Tests whether MMR selection (VOI + diversity) predicts returns
as well as naive top-K selection.

Hypothesis: MMR should maintain predictive power while adding diversity.
- If MMR r ≈ naive r → diversity is free
- If MMR r << naive r → diversity hurts prediction
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    """Load cruxes and returns."""
    cruxes = pd.read_parquet(DATA_DIR / "cruxes_with_voi.parquet")
    returns = pd.read_parquet(DATA_DIR / "stock_returns.parquet")
    return cruxes, returns


def compute_similarity_matrix(texts: list[str]) -> np.ndarray:
    """Compute pairwise cosine similarity using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


def select_top_k_naive(cruxes_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Naive selection: top-K by VOI for each stock-day."""
    return cruxes_df.groupby(["ticker", "date"]).apply(
        lambda g: g.nlargest(k, "linear_voi")
    ).reset_index(drop=True)


def select_top_k_mmr(cruxes_df: pd.DataFrame, k: int, lambda_param: float = 0.7) -> pd.DataFrame:
    """MMR selection: balance VOI with diversity for each stock-day."""
    results = []

    for (ticker, date), group in cruxes_df.groupby(["ticker", "date"]):
        if len(group) <= k:
            results.append(group)
            continue

        # Compute similarity matrix for this group
        texts = group["crux"].tolist()
        sim_matrix = compute_similarity_matrix(texts)

        # Normalize VOI to [0, 1]
        vois = group["linear_voi"].values
        voi_min, voi_max = vois.min(), vois.max()
        if voi_max > voi_min:
            vois_norm = (vois - voi_min) / (voi_max - voi_min)
        else:
            vois_norm = np.ones(len(vois)) * 0.5

        # MMR selection
        indices = list(range(len(group)))
        selected = []

        while len(selected) < k and indices:
            best_score = -float("inf")
            best_idx = None

            for idx in indices:
                relevance = vois_norm[idx]
                if selected:
                    redundancy = max(sim_matrix[idx, s] for s in selected)
                else:
                    redundancy = 0
                score = lambda_param * relevance - (1 - lambda_param) * redundancy

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                indices.remove(best_idx)

        selected_df = group.iloc[selected]
        results.append(selected_df)

    return pd.concat(results, ignore_index=True)


def compute_correlation(selected_df: pd.DataFrame, returns_df: pd.DataFrame,
                        earnings_only: bool = True) -> dict:
    """Compute correlation between aggregated VOI and |return|."""
    # Filter returns
    if earnings_only:
        returns_filtered = returns_df[returns_df["is_earnings_day"]].copy()
    else:
        returns_filtered = returns_df.copy()

    if len(returns_filtered) == 0:
        return {"error": "No matching returns"}

    # Aggregate VOI per stock-day
    voi_agg = selected_df.groupby(["ticker", "date"]).agg({
        "linear_voi": ["mean", "sum", "std"],
        "crux": "count",
    }).reset_index()
    voi_agg.columns = ["ticker", "date", "voi_mean", "voi_sum", "voi_std", "n_cruxes"]

    # Merge with returns
    merged = voi_agg.merge(
        returns_filtered[["ticker", "date", "return"]],
        on=["ticker", "date"],
        how="inner"
    )

    if len(merged) < 5:
        return {"error": f"Too few observations ({len(merged)})"}

    merged["abs_return"] = merged["return"].abs()

    # Compute correlations
    r_mean, p_mean = stats.pearsonr(merged["voi_mean"], merged["abs_return"])
    r_sum, p_sum = stats.pearsonr(merged["voi_sum"], merged["abs_return"])

    return {
        "n": len(merged),
        "r_voi_mean": float(r_mean),
        "p_voi_mean": float(p_mean),
        "r_voi_sum": float(r_sum),
        "p_voi_sum": float(p_sum),
        "mean_voi": float(merged["voi_mean"].mean()),
        "mean_abs_return": float(merged["abs_return"].mean()),
    }


def compute_diversity_metrics(selected_df: pd.DataFrame) -> dict:
    """Compute diversity metrics for selection."""
    diversities = []

    for (ticker, date), group in selected_df.groupby(["ticker", "date"]):
        if len(group) < 2:
            continue
        texts = group["crux"].tolist()
        sim_matrix = compute_similarity_matrix(texts)
        # Average pairwise similarity (lower = more diverse)
        n = len(texts)
        pairwise_sims = []
        for i in range(n):
            for j in range(i+1, n):
                pairwise_sims.append(sim_matrix[i, j])
        if pairwise_sims:
            diversities.append(1 - np.mean(pairwise_sims))

    return {
        "mean_diversity": float(np.mean(diversities)) if diversities else 0,
        "n_groups": len(diversities),
    }


def main():
    print("=" * 70)
    print("MMR SELECTION VALIDATION ON RUSSELL 2000")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    cruxes, returns = load_data()
    print(f"      Cruxes: {len(cruxes)}")
    print(f"      Returns: {len(returns)}")

    # Check earnings days
    earnings_days = returns[returns["is_earnings_day"]]
    print(f"      Earnings days: {len(earnings_days)}")

    # Test different K values and lambda values
    print("\n[2/4] Comparing selection methods...")

    k_values = [1, 3, 5]
    lambda_values = [0.5, 0.7, 0.9]

    results = []

    for k in k_values:
        print(f"\n  K = {k}:")

        # Naive top-K
        naive = select_top_k_naive(cruxes, k)
        naive_corr = compute_correlation(naive, returns, earnings_only=True)
        naive_div = compute_diversity_metrics(naive)

        if "error" not in naive_corr:
            print(f"    Naive:      r={naive_corr['r_voi_mean']:.3f}, "
                  f"diversity={naive_div['mean_diversity']:.3f}")
            results.append({
                "method": "naive",
                "k": k,
                "lambda": None,
                **naive_corr,
                **naive_div,
            })

        # MMR with different lambda
        for lam in lambda_values:
            mmr = select_top_k_mmr(cruxes, k, lambda_param=lam)
            mmr_corr = compute_correlation(mmr, returns, earnings_only=True)
            mmr_div = compute_diversity_metrics(mmr)

            if "error" not in mmr_corr:
                print(f"    MMR(λ={lam}): r={mmr_corr['r_voi_mean']:.3f}, "
                      f"diversity={mmr_div['mean_diversity']:.3f}")
                results.append({
                    "method": "mmr",
                    "k": k,
                    "lambda": lam,
                    **mmr_corr,
                    **mmr_div,
                })

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Best method for each K
    print("\n  Best method by correlation (for each K):")
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        if k_results:
            best = max(k_results, key=lambda x: x["r_voi_mean"])
            method = f"MMR(λ={best['lambda']})" if best["lambda"] else "Naive"
            print(f"    K={k}: {method} → r={best['r_voi_mean']:.3f}")

    # Trade-off analysis for K=5
    print("\n  Trade-off (K=5):")
    k5_results = [r for r in results if r["k"] == 5]
    for r in sorted(k5_results, key=lambda x: x.get("lambda") or 999):
        method = f"λ={r['lambda']}" if r["lambda"] else "naive"
        print(f"    {method:8}: r={r['r_voi_mean']:.3f}, diversity={r['mean_diversity']:.3f}")

    # Key comparison
    print("\n" + "=" * 70)
    print("KEY COMPARISON (K=3)")
    print("=" * 70)

    k3_naive = [r for r in results if r["k"] == 3 and r["method"] == "naive"]
    k3_mmr07 = [r for r in results if r["k"] == 3 and r.get("lambda") == 0.7]

    if k3_naive and k3_mmr07:
        naive = k3_naive[0]
        mmr = k3_mmr07[0]

        print(f"\n  Naive Top-3:")
        print(f"    Correlation (r):  {naive['r_voi_mean']:.3f}")
        print(f"    Diversity:        {naive['mean_diversity']:.3f}")

        print(f"\n  MMR Top-3 (λ=0.7):")
        print(f"    Correlation (r):  {mmr['r_voi_mean']:.3f}")
        print(f"    Diversity:        {mmr['mean_diversity']:.3f}")

        delta_r = mmr["r_voi_mean"] - naive["r_voi_mean"]
        delta_div = mmr["mean_diversity"] - naive["mean_diversity"]

        print(f"\n  Change:")
        print(f"    Correlation: {delta_r:+.3f}")
        print(f"    Diversity:   {delta_div:+.3f}")

        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if abs(delta_r) < 0.05:
            print(f"\n✓ MMR maintains prediction (Δr={delta_r:+.3f})")
            print("  → Diversity is essentially free")
        elif delta_r > 0:
            print(f"\n✓ MMR IMPROVES prediction (Δr={delta_r:+.3f})")
            print("  → Diversity helps!")
        else:
            print(f"\n⚠ MMR hurts prediction (Δr={delta_r:+.3f})")
            print("  → Trade-off exists; may need different λ")

    # Save results
    print("\n[4/4] Saving results...")
    output = {
        "metadata": {
            "experiment": "mmr_russell_validation",
            "n_cruxes": len(cruxes),
            "n_earnings_days": len(earnings_days),
            "k_values": k_values,
            "lambda_values": lambda_values,
            "run_at": datetime.now().isoformat(),
        },
        "results": results,
    }

    output_path = DATA_DIR / "mmr_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"      Saved to {output_path}")


if __name__ == "__main__":
    main()
