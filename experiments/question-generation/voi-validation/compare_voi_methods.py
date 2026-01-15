#!/usr/bin/env python3
"""
Compare Linear vs Entropy VOI across experiments.

Aggregates results from:
- VOI validation (Q1): Does VOI predict actual market shifts?
- MVP benchmark (Q2a): Does VOI differ from naive LLM ranking?

Usage:
    uv run python experiments/question-generation/voi-validation/compare_voi_methods.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

# Paths
VOI_DIR = Path(__file__).parent
BENCHMARK_DIR = VOI_DIR.parent / "benchmark-mvp"
RESULTS_DIR = VOI_DIR / "results"
BENCHMARK_RESULTS = BENCHMARK_DIR / "results" / "benchmark_results.json"
VALIDATION_RESULTS = RESULTS_DIR / "voi_validation_nontrivial.json"


def load_results():
    """Load results from both experiments."""
    validation = None
    benchmark = None

    if VALIDATION_RESULTS.exists():
        with open(VALIDATION_RESULTS) as f:
            validation = json.load(f)

    if BENCHMARK_RESULTS.exists():
        with open(BENCHMARK_RESULTS) as f:
            benchmark = json.load(f)

    return validation, benchmark


def analyze_validation_results(data: dict) -> dict:
    """Analyze VOI validation experiment results."""
    if not data:
        return {"error": "No validation data available"}

    corrs = data.get("correlations", {})

    return {
        "n_pairs": data["metadata"]["n_pairs"],
        "linear_voi_vs_shift": corrs.get("linear_voi_vs_shift", {}),
        "entropy_voi_vs_shift": corrs.get("entropy_voi_vs_shift", {}),
        "entropy_voi_norm_vs_shift": corrs.get("entropy_voi_normalized_vs_shift", {}),
        "comparison": data.get("voi_comparison", {}),
    }


def analyze_benchmark_results(data: dict) -> dict:
    """Analyze MVP benchmark experiment results."""
    if not data:
        return {"error": "No benchmark data available"}

    stats_data = data.get("statistics", {})

    return {
        "n_cruxes": data["metadata"]["total_cruxes"],
        "linear_voi": {
            "mean": stats_data.get("mean_voi_linear"),
            "median": stats_data.get("median_voi_linear"),
            "max": stats_data.get("max_voi_linear"),
        },
        "entropy_voi": {
            "mean": stats_data.get("mean_voi_entropy"),
            "median": stats_data.get("median_voi_entropy"),
            "max": stats_data.get("max_voi_entropy"),
        },
        "entropy_voi_normalized": {
            "mean": stats_data.get("mean_voi_entropy_normalized"),
            "median": stats_data.get("median_voi_entropy_normalized"),
            "max": stats_data.get("max_voi_entropy_normalized"),
        },
        "ranking_correlations": {
            "linear_vs_llm": stats_data.get("mean_linear_vs_llm"),
            "entropy_vs_llm": stats_data.get("mean_entropy_vs_llm"),
            "linear_vs_entropy": stats_data.get("mean_linear_vs_entropy"),
        },
    }


def main():
    print("=" * 70)
    print("VOI METHOD COMPARISON: Linear vs Entropy")
    print("=" * 70)

    validation, benchmark = load_results()

    # Q1: Validation results
    print("\n" + "=" * 70)
    print("Q1: VOI VALIDATION (Does VOI predict actual market shifts?)")
    print("=" * 70)

    val_analysis = analyze_validation_results(validation)
    if "error" in val_analysis:
        print(f"\n{val_analysis['error']}")
        print("Run validate_voi_nontrivial.py first.")
    else:
        print(f"\nN pairs analyzed: {val_analysis['n_pairs']}")

        print("\n--- Correlation with Actual Shifts ---")
        if val_analysis["linear_voi_vs_shift"]:
            r = val_analysis["linear_voi_vs_shift"].get("r", "N/A")
            p = val_analysis["linear_voi_vs_shift"].get("p", "N/A")
            print(f"Linear VOI:     r = {r:.3f}, p = {p:.3f}")

        if val_analysis["entropy_voi_vs_shift"]:
            r = val_analysis["entropy_voi_vs_shift"].get("r", "N/A")
            p = val_analysis["entropy_voi_vs_shift"].get("p", "N/A")
            print(f"Entropy VOI:    r = {r:.3f}, p = {p:.3f}")

        if val_analysis["entropy_voi_norm_vs_shift"]:
            r = val_analysis["entropy_voi_norm_vs_shift"].get("r", "N/A")
            p = val_analysis["entropy_voi_norm_vs_shift"].get("p", "N/A")
            print(f"Entropy (norm): r = {r:.3f}, p = {p:.3f}")

        if val_analysis["comparison"]:
            print(f"\n--- Head-to-Head ---")
            winner = val_analysis["comparison"].get("better_predictor", "Unknown")
            delta = val_analysis["comparison"].get("advantage_delta_r", 0)
            tau = val_analysis["comparison"].get("linear_vs_entropy_tau", 0)
            print(f"Better predictor: {winner} (Δr = {delta:.3f})")
            print(f"Ranking agreement (τ): {tau:.3f}")

    # Q2a: Benchmark results
    print("\n" + "=" * 70)
    print("Q2a: MVP BENCHMARK (VOI divergence from LLM ranking)")
    print("=" * 70)

    bench_analysis = analyze_benchmark_results(benchmark)
    if "error" in bench_analysis:
        print(f"\n{bench_analysis['error']}")
        print("Run run_benchmark.py first.")
    else:
        print(f"\nN cruxes generated: {bench_analysis['n_cruxes']}")

        print("\n--- VOI Statistics ---")
        print(f"{'Metric':<20} {'Linear':<12} {'Entropy':<12} {'Entropy (norm)':<15}")
        print("-" * 60)

        linear = bench_analysis["linear_voi"]
        entropy = bench_analysis["entropy_voi"]
        entropy_norm = bench_analysis["entropy_voi_normalized"]

        if linear.get("mean") is not None:
            print(f"{'Mean':<20} {linear['mean']:<12.3f} {entropy['mean']:<12.3f} {entropy_norm['mean']*100:<12.1f}%")
            print(f"{'Median':<20} {linear['median']:<12.3f} {entropy['median']:<12.3f} {entropy_norm['median']*100:<12.1f}%")
            print(f"{'Max':<20} {linear['max']:<12.3f} {entropy['max']:<12.3f} {entropy_norm['max']*100:<12.1f}%")

        print("\n--- Ranking Correlations ---")
        corrs = bench_analysis["ranking_correlations"]
        if corrs.get("linear_vs_llm") is not None:
            print(f"Linear VOI vs LLM:  ρ = {corrs['linear_vs_llm']:.2f}")
        if corrs.get("entropy_vs_llm") is not None:
            print(f"Entropy VOI vs LLM: ρ = {corrs['entropy_vs_llm']:.2f}")
        if corrs.get("linear_vs_entropy") is not None:
            print(f"Linear vs Entropy:  ρ = {corrs['linear_vs_entropy']:.2f}")

        if corrs.get("linear_vs_llm") is not None and corrs.get("entropy_vs_llm") is not None:
            diff = corrs["linear_vs_llm"] - corrs["entropy_vs_llm"]
            if abs(diff) > 0.05:
                if diff > 0:
                    print(f"\n→ Linear VOI more aligned with LLM intuition (Δρ = {diff:.2f})")
                else:
                    print(f"\n→ Entropy VOI more aligned with LLM intuition (Δρ = {-diff:.2f})")
            else:
                print(f"\n→ Both methods similarly aligned with LLM intuition")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nBased on theoretical expectations:")
    print("  - Linear VOI: Constant gradient, stable under noise")
    print("  - Entropy VOI: Steep gradients at extremes, amplifies errors")
    print("  - Prior work: Linear VOI has +0.160 τ stability advantage")
    print("              +0.352 τ at extreme base rates (<0.10 or >0.90)")

    if "error" not in val_analysis and val_analysis.get("comparison"):
        winner = val_analysis["comparison"].get("better_predictor", "Unknown")
        print(f"\nEmpirical result (Q1 validation): {winner} predicts shifts better")

    if "error" not in bench_analysis:
        corrs = bench_analysis["ranking_correlations"]
        if corrs.get("linear_vs_entropy") is not None:
            tau = corrs["linear_vs_entropy"]
            if tau > 0.7:
                print(f"\nRanking agreement: HIGH (ρ = {tau:.2f}) - methods produce similar rankings")
            elif tau > 0.4:
                print(f"\nRanking agreement: MODERATE (ρ = {tau:.2f}) - some divergence in rankings")
            else:
                print(f"\nRanking agreement: LOW (ρ = {tau:.2f}) - methods produce different rankings")


if __name__ == "__main__":
    main()
