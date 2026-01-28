#!/usr/bin/env python3
"""
Analyze within-category VOI validation results.

Key question: Does VOI discriminate *within* a category, or only *between* categories?

Compares:
- Overall Polymarket: r=0.65 (baseline)
- Russell 2000 earnings: r=-0.15 (within-category failure)
- Polymarket Fed/Monetary: ??? (test case)
"""

import json
from pathlib import Path
from scipy import stats
import numpy as np
from datetime import datetime

# Paths
INPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Reference baselines
BASELINES = {
    "overall_polymarket": {
        "r": 0.65,
        "description": "All Polymarket pairs (heterogeneous categories)"
    },
    "russell_2000_earnings": {
        "r": -0.15,
        "description": "Russell 2000 earnings (single category, LLM rho)"
    }
}


def analyze_category(cat_name: str, records: list) -> dict:
    """Analyze VOI validation for a category."""
    if not records:
        return None

    # Extract arrays
    linear_voi = np.array([r["linear_voi"] for r in records])
    entropy_voi = np.array([r["entropy_voi"] for r in records])
    actual_shift = np.array([r["actual_shift"] for r in records])
    abs_rho = np.array([r["abs_rho"] for r in records])

    # Compute correlations
    results = {
        "n_pairs": len(records),
        "linear_voi": {
            "mean": float(np.mean(linear_voi)),
            "std": float(np.std(linear_voi)),
        },
        "entropy_voi": {
            "mean": float(np.mean(entropy_voi)),
            "std": float(np.std(entropy_voi)),
        },
        "actual_shift": {
            "mean": float(np.mean(actual_shift)),
            "std": float(np.std(actual_shift)),
        },
        "abs_rho": {
            "mean": float(np.mean(abs_rho)),
            "std": float(np.std(abs_rho)),
        }
    }

    # Correlation: linear_voi vs actual_shift
    if len(records) >= 3:
        r_linear, p_linear = stats.pearsonr(linear_voi, actual_shift)
        r_entropy, p_entropy = stats.pearsonr(entropy_voi, actual_shift)
        r_rho, p_rho = stats.pearsonr(abs_rho, actual_shift)

        results["correlations"] = {
            "linear_voi_vs_shift": {
                "r": float(r_linear),
                "p": float(p_linear),
            },
            "entropy_voi_vs_shift": {
                "r": float(r_entropy),
                "p": float(p_entropy),
            },
            "abs_rho_vs_shift": {
                "r": float(r_rho),
                "p": float(p_rho),
            }
        }
    else:
        results["correlations"] = None

    return results


def main():
    print("=" * 70)
    print("WITHIN-CATEGORY VOI VALIDATION ANALYSIS")
    print("=" * 70)

    categories_to_analyze = ["fed_monetary", "politics"]
    all_results = {}

    for cat_name in categories_to_analyze:
        input_path = INPUT_DIR / f"{cat_name}_voi_validation.json"
        if not input_path.exists():
            print(f"\n{cat_name}: No validation data found, skipping")
            continue

        with open(input_path) as f:
            data = json.load(f)

        records = data["records"]
        print(f"\n{'='*70}")
        print(f"CATEGORY: {cat_name.upper()}")
        print(f"{'='*70}")

        analysis = analyze_category(cat_name, records)
        if not analysis:
            print("  No data to analyze")
            continue

        all_results[cat_name] = analysis

        print(f"\nN pairs: {analysis['n_pairs']}")

        print(f"\nDescriptive Statistics:")
        print(f"  Linear VOI:   mean={analysis['linear_voi']['mean']:.3f}, std={analysis['linear_voi']['std']:.3f}")
        print(f"  Entropy VOI:  mean={analysis['entropy_voi']['mean']:.3f}, std={analysis['entropy_voi']['std']:.3f}")
        print(f"  Actual shift: mean={analysis['actual_shift']['mean']:.3f}, std={analysis['actual_shift']['std']:.3f}")
        print(f"  |rho|:        mean={analysis['abs_rho']['mean']:.3f}, std={analysis['abs_rho']['std']:.3f}")

        if analysis["correlations"]:
            corr = analysis["correlations"]
            print(f"\nCorrelations (VOI vs Actual Shift):")
            print(f"  Linear VOI:  r={corr['linear_voi_vs_shift']['r']:.3f}, p={corr['linear_voi_vs_shift']['p']:.4f}")
            print(f"  Entropy VOI: r={corr['entropy_voi_vs_shift']['r']:.3f}, p={corr['entropy_voi_vs_shift']['p']:.4f}")
            print(f"  |rho|:       r={corr['abs_rho_vs_shift']['r']:.3f}, p={corr['abs_rho_vs_shift']['p']:.4f}")

    # Comparison with baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)

    print(f"\nReference baselines:")
    for name, baseline in BASELINES.items():
        print(f"  {name}: r={baseline['r']:.2f} ({baseline['description']})")

    print(f"\nWithin-category results:")
    for cat_name, analysis in all_results.items():
        if analysis and analysis["correlations"]:
            r = analysis["correlations"]["linear_voi_vs_shift"]["r"]
            p = analysis["correlations"]["linear_voi_vs_shift"]["p"]
            print(f"  {cat_name}: r={r:.2f} (p={p:.4f})")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    for cat_name, analysis in all_results.items():
        if not analysis or not analysis["correlations"]:
            continue

        r = analysis["correlations"]["linear_voi_vs_shift"]["r"]
        p = analysis["correlations"]["linear_voi_vs_shift"]["p"]
        n = analysis["n_pairs"]

        print(f"\n{cat_name}:")

        if n < 10:
            print(f"  LOW N WARNING: Only {n} pairs - interpret with caution")

        if r > 0.15 and p < 0.10:
            print(f"  POSITIVE: Within-category discrimination works (r={r:.2f}, p={p:.3f})")
            print(f"  -> Russell 2000 failure may be domain-specific (equities confounds)")
        elif abs(r) < 0.10:
            print(f"  NULL: No within-category discrimination (r={r:.2f})")
            print(f"  -> VOI may fundamentally be a category detector, not a within-category ranker")
        elif r < 0:
            print(f"  NEGATIVE: Inverse relationship (r={r:.2f})")
            print(f"  -> Something may be wrong with experimental design")
        else:
            print(f"  WEAK: Positive but not significant (r={r:.2f}, p={p:.3f})")
            print(f"  -> More data needed, trend direction is encouraging")

    # Save combined results
    output_path = OUTPUT_DIR / "within_category_analysis.json"
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "experiment": "within_category_voi_validation",
                "analyzed_at": datetime.now().isoformat(),
            },
            "baselines": BASELINES,
            "results": all_results,
        }, f, indent=2)
    print(f"\n\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
