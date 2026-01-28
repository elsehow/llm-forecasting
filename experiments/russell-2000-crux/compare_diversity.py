"""Compare baseline vs diverse crux generation for VOI variance and predictive power.

This experiment tests whether forcing crux diversity across categories
(financial, operational, strategic, macro, sector-specific) improves:
1. VOI variance (baseline has std ~0.027, very tight)
2. Within-earnings correlation with |return| (baseline r = -0.15)

Success criteria:
- VOI std > 0.05 (diversity creates variance)
- Within-earnings r > 0.15 (diversity improves prediction)
- No single category > 50% (prompt actually diversifies)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"


def load_datasets(
    baseline_file: str = "cruxes_with_voi.parquet",
    diverse_file: str = "cruxes_diverse_with_voi.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline and diverse datasets."""
    baseline_path = DATA_DIR / baseline_file
    diverse_path = DATA_DIR / diverse_file

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    if not diverse_path.exists():
        raise FileNotFoundError(f"Diverse file not found: {diverse_path}")

    baseline = pd.read_parquet(baseline_path)
    diverse = pd.read_parquet(diverse_path)

    return baseline, diverse


def compare_voi_variance(baseline: pd.DataFrame, diverse: pd.DataFrame) -> dict:
    """Compare VOI variance between baseline and diverse conditions."""
    results = {
        "baseline": {
            "linear_voi_mean": baseline["linear_voi"].mean(),
            "linear_voi_std": baseline["linear_voi"].std(),
            "linear_voi_min": baseline["linear_voi"].min(),
            "linear_voi_max": baseline["linear_voi"].max(),
            "entropy_voi_std": baseline["entropy_voi"].std(),
            "rho_std": baseline["rho"].std(),
            "n_cruxes": len(baseline),
        },
        "diverse": {
            "linear_voi_mean": diverse["linear_voi"].mean(),
            "linear_voi_std": diverse["linear_voi"].std(),
            "linear_voi_min": diverse["linear_voi"].min(),
            "linear_voi_max": diverse["linear_voi"].max(),
            "entropy_voi_std": diverse["entropy_voi"].std(),
            "rho_std": diverse["rho"].std(),
            "n_cruxes": len(diverse),
        },
    }

    # Calculate improvement ratios
    results["variance_ratio"] = (
        results["diverse"]["linear_voi_std"] / results["baseline"]["linear_voi_std"]
        if results["baseline"]["linear_voi_std"] > 0 else float("inf")
    )

    # Success check
    results["success_variance"] = results["diverse"]["linear_voi_std"] > 0.05

    return results


def analyze_category_distribution(diverse: pd.DataFrame) -> dict:
    """Analyze whether diverse prompting actually produces category diversity."""
    if "category" not in diverse.columns:
        return {"error": "No category column - not a diverse dataset"}

    category_counts = diverse["category"].value_counts()
    total = len(diverse)

    results = {
        "category_distribution": category_counts.to_dict(),
        "category_percentages": (category_counts / total * 100).to_dict(),
        "max_category_pct": category_counts.max() / total * 100,
        "n_categories": len(category_counts),
    }

    # Success: no single category > 50%
    results["success_diversity"] = results["max_category_pct"] < 50

    # VOI by category
    if "linear_voi" in diverse.columns:
        voi_by_cat = diverse.groupby("category")["linear_voi"].agg(["mean", "std", "max"])
        results["voi_by_category"] = voi_by_cat.to_dict()

    return results


def compare_within_earnings(
    baseline: pd.DataFrame,
    diverse: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> dict:
    """Compare within-earnings correlation for baseline vs diverse."""

    def compute_within_earnings_r(cruxes_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
        """Compute within-earnings correlation for one dataset."""
        # Filter to earnings days only
        earnings_returns = returns_df[returns_df["is_earnings_day"]].copy()

        if len(earnings_returns) == 0:
            return {"error": "No earnings days"}

        # Aggregate VOI per stock-day
        voi_by_day = cruxes_df.groupby(["ticker", "date"]).agg({
            "linear_voi": "mean",
            "entropy_voi": "mean",
        }).reset_index()

        # Merge
        merged = voi_by_day.merge(
            earnings_returns[["ticker", "date", "return"]],
            on=["ticker", "date"],
            how="inner"
        )

        if len(merged) < 5:
            return {"error": f"Too few observations ({len(merged)})"}

        merged["abs_return"] = merged["return"].abs()

        r_linear, p_linear = stats.pearsonr(merged["linear_voi"], merged["abs_return"])
        rho_linear, _ = stats.spearmanr(merged["linear_voi"], merged["abs_return"])

        return {
            "n": len(merged),
            "pearson_r": r_linear,
            "pearson_p": p_linear,
            "spearman_rho": rho_linear,
        }

    results = {
        "baseline": compute_within_earnings_r(baseline, returns_df),
        "diverse": compute_within_earnings_r(diverse, returns_df),
    }

    # Success check
    if "pearson_r" in results["diverse"]:
        results["success_correlation"] = results["diverse"]["pearson_r"] > 0.15
        results["improvement"] = (
            results["diverse"]["pearson_r"] - results["baseline"].get("pearson_r", 0)
        )

    return results


def classify_crux_category(crux: str) -> str:
    """Classify a crux into categories based on keywords (for baseline analysis)."""
    crux_lower = crux.lower()

    # Financial keywords
    if any(w in crux_lower for w in ["earnings", "revenue", "profit", "margin", "eps",
                                      "guidance", "consensus", "beat", "miss", "quarter"]):
        return "financial"

    # Operational keywords
    if any(w in crux_lower for w in ["production", "inventory", "supply chain", "efficiency",
                                      "operations", "manufacturing", "capacity"]):
        return "operational"

    # Strategic keywords
    if any(w in crux_lower for w in ["partnership", "acquisition", "m&a", "merger",
                                      "contract", "deal", "ceo", "management", "strategy"]):
        return "strategic"

    # Macro keywords
    if any(w in crux_lower for w in ["fed", "interest rate", "inflation", "gdp",
                                      "economic", "macro", "market", "sector"]):
        return "macro"

    # Default to financial (most common)
    return "financial"


def analyze_baseline_categories(baseline: pd.DataFrame) -> dict:
    """Classify baseline cruxes by keyword and analyze distribution."""
    if "category" not in baseline.columns or baseline["category"].isna().all():
        baseline["inferred_category"] = baseline["crux"].apply(classify_crux_category)
        cat_col = "inferred_category"
    else:
        cat_col = "category"

    category_counts = baseline[cat_col].value_counts()
    total = len(baseline)

    return {
        "category_distribution": category_counts.to_dict(),
        "category_percentages": (category_counts / total * 100).to_dict(),
        "max_category_pct": category_counts.max() / total * 100,
        "dominant_category": category_counts.idxmax(),
    }


def create_comparison_plots(baseline: pd.DataFrame, diverse: pd.DataFrame, save_dir: Path):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. VOI distribution comparison
    ax = axes[0, 0]
    ax.hist(baseline["linear_voi"], bins=30, alpha=0.5, label="Baseline", color="blue")
    ax.hist(diverse["linear_voi"], bins=30, alpha=0.5, label="Diverse", color="orange")
    ax.set_xlabel("Linear VOI")
    ax.set_ylabel("Count")
    ax.set_title(f"VOI Distribution\nBaseline std={baseline['linear_voi'].std():.4f}, "
                 f"Diverse std={diverse['linear_voi'].std():.4f}")
    ax.legend()

    # 2. Category distribution (diverse only)
    ax = axes[0, 1]
    if "category" in diverse.columns:
        cat_counts = diverse["category"].value_counts()
        ax.bar(cat_counts.index, cat_counts.values, color="steelblue")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_title("Diverse Crux Category Distribution")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No category data", ha="center", va="center")
        ax.set_title("Category Distribution (N/A)")

    # 3. VOI by category (diverse)
    ax = axes[1, 0]
    if "category" in diverse.columns:
        voi_by_cat = diverse.groupby("category")["linear_voi"].mean().sort_values(ascending=False)
        ax.bar(voi_by_cat.index, voi_by_cat.values, color="coral")
        ax.set_xlabel("Category")
        ax.set_ylabel("Mean Linear VOI")
        ax.set_title("Mean VOI by Category (Diverse)")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No category data", ha="center", va="center")

    # 4. Rho distribution comparison
    ax = axes[1, 1]
    ax.hist(baseline["rho"], bins=30, alpha=0.5, label="Baseline", color="blue")
    ax.hist(diverse["rho"], bins=30, alpha=0.5, label="Diverse", color="orange")
    ax.set_xlabel("Rho (ρ)")
    ax.set_ylabel("Count")
    ax.set_title(f"Rho Distribution\nBaseline std={baseline['rho'].std():.4f}, "
                 f"Diverse std={diverse['rho'].std():.4f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "diversity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_dir / 'diversity_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs diverse crux generation")
    parser.add_argument(
        "--baseline",
        type=str,
        default="cruxes_with_voi.parquet",
        help="Baseline cruxes file (default: cruxes_with_voi.parquet)"
    )
    parser.add_argument(
        "--diverse",
        type=str,
        default="cruxes_diverse_with_voi.parquet",
        help="Diverse cruxes file (default: cruxes_diverse_with_voi.parquet)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CRUX DIVERSITY EXPERIMENT - BASELINE VS DIVERSE COMPARISON")
    print("=" * 70)

    # Load datasets
    try:
        baseline, diverse = load_datasets(args.baseline, args.diverse)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    print(f"\nDatasets loaded:")
    print(f"  Baseline: {len(baseline)} cruxes ({args.baseline})")
    print(f"  Diverse:  {len(diverse)} cruxes ({args.diverse})")

    # 1. VOI Variance Comparison
    print("\n" + "=" * 70)
    print("1. VOI VARIANCE COMPARISON")
    print("=" * 70)
    print("Target: Diverse std > 0.05 (baseline ~0.027)")

    variance_results = compare_voi_variance(baseline, diverse)

    print(f"\nBaseline:")
    print(f"  Linear VOI std:  {variance_results['baseline']['linear_voi_std']:.4f}")
    print(f"  Linear VOI mean: {variance_results['baseline']['linear_voi_mean']:.4f}")
    print(f"  Range: [{variance_results['baseline']['linear_voi_min']:.4f}, "
          f"{variance_results['baseline']['linear_voi_max']:.4f}]")

    print(f"\nDiverse:")
    print(f"  Linear VOI std:  {variance_results['diverse']['linear_voi_std']:.4f}")
    print(f"  Linear VOI mean: {variance_results['diverse']['linear_voi_mean']:.4f}")
    print(f"  Range: [{variance_results['diverse']['linear_voi_min']:.4f}, "
          f"{variance_results['diverse']['linear_voi_max']:.4f}]")

    print(f"\nVariance ratio (diverse/baseline): {variance_results['variance_ratio']:.2f}x")
    status = "[PASS]" if variance_results["success_variance"] else "[FAIL]"
    print(f"{status} Diversity increases variance: {variance_results['success_variance']}")

    # 2. Category Distribution
    print("\n" + "=" * 70)
    print("2. CATEGORY DISTRIBUTION")
    print("=" * 70)
    print("Target: No single category > 50%")

    # Baseline (inferred)
    baseline_cats = analyze_baseline_categories(baseline)
    print(f"\nBaseline (inferred from keywords):")
    for cat, pct in sorted(baseline_cats["category_percentages"].items(),
                          key=lambda x: -x[1]):
        print(f"  {cat}: {pct:.1f}%")
    print(f"  Dominant: {baseline_cats['dominant_category']} "
          f"({baseline_cats['max_category_pct']:.1f}%)")

    # Diverse (explicit)
    diverse_cats = analyze_category_distribution(diverse)
    if "error" not in diverse_cats:
        print(f"\nDiverse (explicit categories):")
        for cat, pct in sorted(diverse_cats["category_percentages"].items(),
                              key=lambda x: -x[1]):
            print(f"  {cat}: {pct:.1f}%")
        status = "[PASS]" if diverse_cats["success_diversity"] else "[FAIL]"
        print(f"{status} Category diversity achieved: {diverse_cats['success_diversity']}")
    else:
        print(f"\nDiverse: {diverse_cats['error']}")

    # 3. Within-Earnings Correlation
    print("\n" + "=" * 70)
    print("3. WITHIN-EARNINGS CORRELATION (VOI vs |return|)")
    print("=" * 70)
    print("Target: r > 0.15")

    # Load returns for correlation analysis
    returns_path = DATA_DIR / "stock_returns.parquet"
    if returns_path.exists():
        returns_df = pd.read_parquet(returns_path)
        corr_results = compare_within_earnings(baseline, diverse, returns_df)

        if "error" not in corr_results.get("baseline", {}):
            print(f"\nBaseline:")
            print(f"  n = {corr_results['baseline']['n']}")
            print(f"  Pearson r  = {corr_results['baseline']['pearson_r']:.4f}")
            print(f"  p-value    = {corr_results['baseline']['pearson_p']:.4f}")
            print(f"  Spearman ρ = {corr_results['baseline']['spearman_rho']:.4f}")
        else:
            print(f"\nBaseline: {corr_results['baseline'].get('error', 'N/A')}")

        if "error" not in corr_results.get("diverse", {}):
            print(f"\nDiverse:")
            print(f"  n = {corr_results['diverse']['n']}")
            print(f"  Pearson r  = {corr_results['diverse']['pearson_r']:.4f}")
            print(f"  p-value    = {corr_results['diverse']['pearson_p']:.4f}")
            print(f"  Spearman ρ = {corr_results['diverse']['spearman_rho']:.4f}")

            if "improvement" in corr_results:
                print(f"\nImprovement: {corr_results['improvement']:+.4f}")
                status = "[PASS]" if corr_results["success_correlation"] else "[FAIL]"
                print(f"{status} Correlation improved: {corr_results['success_correlation']}")
        else:
            print(f"\nDiverse: {corr_results['diverse'].get('error', 'N/A')}")
    else:
        print("\nReturns data not found - skipping correlation analysis")
        corr_results = {}

    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_variance = variance_results.get("success_variance", False)
    success_diversity = diverse_cats.get("success_diversity", False) if "error" not in diverse_cats else False
    success_correlation = corr_results.get("success_correlation", False)

    print(f"\n  VOI variance increased (std > 0.05):     {'[PASS]' if success_variance else '[FAIL]'}")
    print(f"  Category diversity (<50% max):           {'[PASS]' if success_diversity else '[FAIL]'}")
    print(f"  Within-earnings correlation (r > 0.15):  {'[PASS]' if success_correlation else '[FAIL]'}")

    all_pass = success_variance and success_diversity and success_correlation
    print("\n" + "-" * 70)
    if all_pass:
        print("CONCLUSION: Diversity prompting WORKS - update Question Generation paper framing")
    elif success_variance and not success_correlation:
        print("CONCLUSION: Categories create noise, not signal")
    else:
        print("CONCLUSION: Crux homogeneity is deeper than prompting can fix")

    # Create visualizations
    create_comparison_plots(baseline, diverse, DATA_DIR)

    # Save results
    results = {
        "variance": {k: v for k, v in variance_results.items() if not isinstance(v, pd.DataFrame)},
        "baseline_categories": baseline_cats,
        "diverse_categories": {k: v for k, v in diverse_cats.items() if not isinstance(v, pd.DataFrame)} if "error" not in diverse_cats else diverse_cats,
        "correlation": {k: v for k, v in corr_results.items() if not isinstance(v, pd.DataFrame)},
        "success": {
            "variance": success_variance,
            "diversity": success_diversity,
            "correlation": success_correlation,
            "all_pass": all_pass,
        },
    }

    with open(DATA_DIR / "diversity_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {DATA_DIR / 'diversity_comparison_results.json'}")


if __name__ == "__main__":
    main()
