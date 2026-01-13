#!/usr/bin/env python3
"""Linear VOI vs Entropy VOI stability experiment.

Tests whether linear VOI produces more stable and useful rankings than
entropy-based VOI when conditional probability estimates have magnitude errors.

Hypothesis: Entropy-based VOI has steep gradients at probability extremes,
amplifying magnitude errors. Linear VOI may be more robust to miscalibration
while preserving useful rankings.

Usage:
    uv run python experiments/magnitude/linear-voi/run_experiment.py
    uv run python experiments/magnitude/linear-voi/run_experiment.py --trials 50
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


def entropy(p: float) -> float:
    """Binary entropy in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def entropy_voi(p_x: float, p_q: float, p_x_given_q_yes: float, p_x_given_q_no: float) -> float:
    """
    Standard entropy-based VOI.

    p_x: prior P(X)
    p_q: P(Q=yes) - probability the signal question resolves yes
    p_x_given_q_yes: P(X|Q=yes)
    p_x_given_q_no: P(X|Q=no)

    Returns expected reduction in entropy (information gain).
    """
    h_prior = entropy(p_x)
    h_posterior = p_q * entropy(p_x_given_q_yes) + (1 - p_q) * entropy(p_x_given_q_no)
    return h_prior - h_posterior


def linear_voi(p_x: float, p_q: float, p_x_given_q_yes: float, p_x_given_q_no: float) -> float:
    """
    Linear VOI: expected absolute belief shift.

    More robust to estimation errors because gradients are constant.
    """
    shift_yes = abs(p_x_given_q_yes - p_x)
    shift_no = abs(p_x_given_q_no - p_x)
    return p_q * shift_yes + (1 - p_q) * shift_no


def add_noise(conditionals: dict, noise_std: float, rng: np.random.Generator) -> dict:
    """Add Gaussian noise to conditional estimates, clipped to [0.01, 0.99]."""
    return {
        "p_a": np.clip(conditionals["p_a"] + rng.normal(0, noise_std), 0.01, 0.99),
        "p_a_given_b1": np.clip(conditionals["p_a_given_b1"] + rng.normal(0, noise_std), 0.01, 0.99),
        "p_a_given_b0": np.clip(conditionals["p_a_given_b0"] + rng.normal(0, noise_std), 0.01, 0.99),
    }


def compute_rankings(pairs: list[dict], voi_func) -> list[float]:
    """Compute VOI scores for all pairs using the given VOI function."""
    scores = []
    for pair in pairs:
        # Use P(B) = 0.5 as a reasonable prior for the signal question
        # (we don't have elicited P(B) in the data)
        p_q = 0.5
        score = voi_func(
            pair["p_a"],
            p_q,
            pair["p_a_given_b1"],
            pair["p_a_given_b0"],
        )
        scores.append(score)
    return scores


def ranking_stability(original_scores: list[float], noisy_scores: list[float]) -> dict:
    """
    Measure how much rankings change under noise.

    Returns Kendall's tau, number of rank switches, top-5 preservation rate.
    """
    # Get rankings (higher score = higher rank)
    original_ranks = np.argsort(np.argsort(original_scores))
    noisy_ranks = np.argsort(np.argsort(noisy_scores))

    # Kendall's tau
    tau, _ = kendalltau(original_ranks, noisy_ranks)

    # Rank switches (pairs that swap relative order)
    switches = sum(1 for i, j in zip(original_ranks, noisy_ranks) if i != j)

    # Top-5 preservation: what fraction of original top-5 remain in noisy top-5?
    n = len(original_scores)
    k = min(5, n)
    original_top5 = set(np.argsort(original_scores)[-k:])
    noisy_top5 = set(np.argsort(noisy_scores)[-k:])
    top5_preservation = len(original_top5 & noisy_top5) / k

    return {
        "tau": tau,
        "switches": switches,
        "top5_preservation": top5_preservation,
    }


def create_synthetic_examples() -> list[dict]:
    """
    Create 10 synthetic examples with varying base rates to test stability at extremes.

    These represent "ultimate questions" with extreme probabilities where entropy
    gradients are steepest.
    """
    synthetics = []
    base_rates = [0.05, 0.05, 0.10, 0.10, 0.30, 0.30, 0.50, 0.50, 0.70, 0.70]

    for i, base_rate in enumerate(base_rates):
        # Alternate between strong positive and strong negative correlations
        if i % 2 == 0:
            # Positive correlation: B=YES increases A
            p_a_given_b1 = min(0.95, base_rate + 0.25)
            p_a_given_b0 = max(0.02, base_rate - 0.20)
        else:
            # Negative correlation: B=YES decreases A
            p_a_given_b1 = max(0.02, base_rate - 0.20)
            p_a_given_b0 = min(0.95, base_rate + 0.25)

        synthetics.append({
            "pair_id": f"synthetic_{i}",
            "category": "synthetic",
            "p_a": base_rate,
            "p_a_given_b1": p_a_given_b1,
            "p_a_given_b0": p_a_given_b0,
            "description": f"Synthetic pair with base rate {base_rate}",
        })

    return synthetics


def load_pairs_from_twostage(results_path: str) -> list[dict]:
    """Load pairs with conditional estimates from two-stage results."""
    with open(results_path) as f:
        data = json.load(f)

    pairs = []
    for result in data["results"]:
        # Skip errors and baseline-only results (no conditionals)
        if "error" in result:
            continue
        if result["stage2"]["method"] != "bracket":
            continue
        if result["stage2"]["p_a_given_b1"] is None:
            continue

        pairs.append({
            "pair_id": result["pair_id"],
            "category": result.get("category", "unknown"),
            "reason": result.get("reason", ""),
            "q_a": result.get("q_a", ""),
            "q_b": result.get("q_b", ""),
            "p_a": result["stage2"]["p_a"],
            "p_a_given_b1": result["stage2"]["p_a_given_b1"],
            "p_a_given_b0": result["stage2"]["p_a_given_b0"],
            "direction": result["stage2"]["direction"],
        })

    return pairs


def run_monte_carlo_stability(
    pairs: list[dict],
    noise_levels: list[float],
    n_trials: int,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo simulation to measure VOI stability under noise.

    Returns stability metrics for both entropy and linear VOI at each noise level.
    """
    rng = np.random.default_rng(seed)

    # Compute true (noiseless) VOI scores
    true_entropy_scores = compute_rankings(pairs, entropy_voi)
    true_linear_scores = compute_rankings(pairs, linear_voi)

    results = {}

    for noise_std in noise_levels:
        entropy_stabilities = []
        linear_stabilities = []

        for _ in range(n_trials):
            # Add noise to all pairs
            noisy_pairs = [add_noise(p, noise_std, rng) for p in pairs]

            # Compute noisy VOI scores
            noisy_entropy_scores = compute_rankings(noisy_pairs, entropy_voi)
            noisy_linear_scores = compute_rankings(noisy_pairs, linear_voi)

            # Measure stability
            entropy_stability = ranking_stability(true_entropy_scores, noisy_entropy_scores)
            linear_stability = ranking_stability(true_linear_scores, noisy_linear_scores)

            entropy_stabilities.append(entropy_stability)
            linear_stabilities.append(linear_stability)

        # Aggregate across trials
        results[f"noise_{noise_std}"] = {
            "entropy": {
                "mean_tau": np.mean([s["tau"] for s in entropy_stabilities]),
                "std_tau": np.std([s["tau"] for s in entropy_stabilities]),
                "mean_switches": np.mean([s["switches"] for s in entropy_stabilities]),
                "mean_top5_preservation": np.mean([s["top5_preservation"] for s in entropy_stabilities]),
            },
            "linear": {
                "mean_tau": np.mean([s["tau"] for s in linear_stabilities]),
                "std_tau": np.std([s["tau"] for s in linear_stabilities]),
                "mean_switches": np.mean([s["switches"] for s in linear_stabilities]),
                "mean_top5_preservation": np.mean([s["top5_preservation"] for s in linear_stabilities]),
            },
        }

    return results


def analyze_extreme_base_rates(pairs: list[dict], noise_std: float, n_trials: int, seed: int = 42) -> dict:
    """Analyze stability specifically for extreme vs moderate base rate pairs."""
    rng = np.random.default_rng(seed)

    # Split into extreme (< 0.10 or > 0.90) and moderate (0.30 - 0.70)
    extreme_pairs = [p for p in pairs if p["p_a"] < 0.10 or p["p_a"] > 0.90]
    moderate_pairs = [p for p in pairs if 0.30 <= p["p_a"] <= 0.70]

    def analyze_subset(subset_pairs: list[dict]) -> dict:
        if len(subset_pairs) < 2:
            return {"entropy_tau": None, "linear_tau": None, "n": len(subset_pairs)}

        true_entropy = compute_rankings(subset_pairs, entropy_voi)
        true_linear = compute_rankings(subset_pairs, linear_voi)

        entropy_taus = []
        linear_taus = []

        for _ in range(n_trials):
            noisy_pairs = [add_noise(p, noise_std, rng) for p in subset_pairs]
            noisy_entropy = compute_rankings(noisy_pairs, entropy_voi)
            noisy_linear = compute_rankings(noisy_pairs, linear_voi)

            entropy_tau, _ = kendalltau(true_entropy, noisy_entropy)
            linear_tau, _ = kendalltau(true_linear, noisy_linear)

            entropy_taus.append(entropy_tau)
            linear_taus.append(linear_tau)

        return {
            "entropy_tau": np.mean(entropy_taus),
            "linear_tau": np.mean(linear_taus),
            "n": len(subset_pairs),
        }

    return {
        "extreme": analyze_subset(extreme_pairs),
        "moderate": analyze_subset(moderate_pairs),
    }


def compute_voi_comparison(pairs: list[dict]) -> list[dict]:
    """Compute both VOI formulations for each pair and compare."""
    results = []
    for pair in pairs:
        p_q = 0.5  # Prior on signal question

        e_voi = entropy_voi(pair["p_a"], p_q, pair["p_a_given_b1"], pair["p_a_given_b0"])
        l_voi = linear_voi(pair["p_a"], p_q, pair["p_a_given_b1"], pair["p_a_given_b0"])

        results.append({
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "p_a": pair["p_a"],
            "p_a_given_b1": pair["p_a_given_b1"],
            "p_a_given_b0": pair["p_a_given_b0"],
            "entropy_voi": e_voi,
            "linear_voi": l_voi,
        })

    # Add rankings
    entropy_scores = [r["entropy_voi"] for r in results]
    linear_scores = [r["linear_voi"] for r in results]

    entropy_ranks = np.argsort(np.argsort(entropy_scores)[::-1]) + 1  # 1 = highest
    linear_ranks = np.argsort(np.argsort(linear_scores)[::-1]) + 1

    for i, r in enumerate(results):
        r["entropy_rank"] = int(entropy_ranks[i])
        r["linear_rank"] = int(linear_ranks[i])

    return results


def print_summary(stability_results: dict, extreme_analysis: dict, voi_comparison: list[dict]):
    """Print summary tables."""
    print("\n" + "=" * 70)
    print("LINEAR VOI VS ENTROPY VOI STABILITY EXPERIMENT")
    print("=" * 70)

    # Stability summary table
    print("\n" + "-" * 70)
    print("STABILITY UNDER NOISE (Kendall's τ)")
    print("-" * 70)

    print("\n| Noise Level (σ) | Entropy τ | Linear τ | Entropy Top-5 | Linear Top-5 | Winner |")
    print("|-----------------|-----------|----------|---------------|--------------|--------|")

    for noise_key, data in stability_results.items():
        noise_level = noise_key.replace("noise_", "")
        entropy_tau = data["entropy"]["mean_tau"]
        linear_tau = data["linear"]["mean_tau"]
        entropy_top5 = data["entropy"]["mean_top5_preservation"]
        linear_top5 = data["linear"]["mean_top5_preservation"]

        winner = "Linear" if linear_tau > entropy_tau else "Entropy" if entropy_tau > linear_tau else "Tie"

        print(f"| {noise_level:>15} | {entropy_tau:>9.3f} | {linear_tau:>8.3f} | {entropy_top5:>13.1%} | {linear_top5:>12.1%} | {winner:>6} |")

    # Extreme base rate analysis
    print("\n" + "-" * 70)
    print("EXTREME BASE RATE ANALYSIS (σ = 0.05)")
    print("-" * 70)

    print("\n| Base Rate       | Entropy Stability | Linear Stability | Winner |")
    print("|-----------------|-------------------|------------------|--------|")

    for group, data in extreme_analysis.items():
        if data["entropy_tau"] is not None:
            winner = "Linear" if data["linear_tau"] > data["entropy_tau"] else "Entropy"
            print(f"| {group:>15} (n={data['n']}) | {data['entropy_tau']:>17.3f} | {data['linear_tau']:>16.3f} | {winner:>6} |")
        else:
            print(f"| {group:>15} (n={data['n']}) | {'N/A':>17} | {'N/A':>16} | — |")

    # VOI ranking comparison
    print("\n" + "-" * 70)
    print("VOI RANKING COMPARISON (Top 10)")
    print("-" * 70)

    # Sort by entropy VOI
    sorted_by_entropy = sorted(voi_comparison, key=lambda x: x["entropy_voi"], reverse=True)[:10]

    print("\n| Rank | Entropy VOI | Linear VOI | Linear Rank | Category | P(A) |")
    print("|------|-------------|------------|-------------|----------|------|")

    for i, pair in enumerate(sorted_by_entropy, 1):
        print(f"| {i:>4} | {pair['entropy_voi']:>11.4f} | {pair['linear_voi']:>10.4f} | {pair['linear_rank']:>11} | {pair['category']:>8} | {pair['p_a']:.2f} |")

    # Correlation between rankings
    entropy_ranks = [p["entropy_rank"] for p in voi_comparison]
    linear_ranks = [p["linear_rank"] for p in voi_comparison]
    rank_correlation, _ = kendalltau(entropy_ranks, linear_ranks)

    print(f"\nRanking correlation (Kendall's τ): {rank_correlation:.3f}")


def print_analysis(stability_results: dict, extreme_analysis: dict, voi_comparison: list[dict]):
    """Print analysis answering key questions."""
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # 1. Is linear VOI more stable?
    print("\n1. IS LINEAR VOI MORE STABLE?")
    for noise_key, data in stability_results.items():
        noise_level = noise_key.replace("noise_", "")
        entropy_tau = data["entropy"]["mean_tau"]
        linear_tau = data["linear"]["mean_tau"]
        diff = linear_tau - entropy_tau

        if diff > 0.01:
            verdict = f"YES, linear is more stable by {diff:.3f}"
        elif diff < -0.01:
            verdict = f"NO, entropy is more stable by {-diff:.3f}"
        else:
            verdict = "Roughly equal stability"

        print(f"   σ = {noise_level}: {verdict}")

    # 2. Does the gap widen at extremes?
    print("\n2. DOES THE GAP WIDEN AT EXTREMES?")
    extreme = extreme_analysis.get("extreme", {})
    moderate = extreme_analysis.get("moderate", {})

    if extreme.get("entropy_tau") is not None and moderate.get("entropy_tau") is not None:
        gap_extreme = extreme["linear_tau"] - extreme["entropy_tau"]
        gap_moderate = moderate["linear_tau"] - moderate["entropy_tau"]

        print(f"   Extreme base rates (n={extreme['n']}): linear advantage = {gap_extreme:+.3f}")
        print(f"   Moderate base rates (n={moderate['n']}): linear advantage = {gap_moderate:+.3f}")

        if gap_extreme > gap_moderate:
            print("   -> YES, linear VOI advantage is larger at extremes")
        else:
            print("   -> NO, linear VOI advantage does not increase at extremes")
    else:
        print("   Insufficient data for extreme/moderate comparison")

    # 3. Top-5 preservation
    print("\n3. TOP-5 PRESERVATION")
    for noise_key, data in stability_results.items():
        noise_level = noise_key.replace("noise_", "")
        entropy_top5 = data["entropy"]["mean_top5_preservation"]
        linear_top5 = data["linear"]["mean_top5_preservation"]

        better = "Linear" if linear_top5 > entropy_top5 else "Entropy"
        print(f"   σ = {noise_level}: Entropy={entropy_top5:.1%}, Linear={linear_top5:.1%} ({better} better)")

    # 4. Do rankings differ qualitatively?
    print("\n4. DO RANKINGS DIFFER QUALITATIVELY?")
    entropy_ranks = [p["entropy_rank"] for p in voi_comparison]
    linear_ranks = [p["linear_rank"] for p in voi_comparison]
    rank_correlation, _ = kendalltau(entropy_ranks, linear_ranks)

    if rank_correlation > 0.8:
        print(f"   Rankings are highly correlated (τ = {rank_correlation:.3f})")
        print("   -> Both VOI formulations identify similar high-value questions")
    elif rank_correlation > 0.5:
        print(f"   Rankings are moderately correlated (τ = {rank_correlation:.3f})")
        print("   -> Some differences in which questions are identified as valuable")
    else:
        print(f"   Rankings are weakly correlated (τ = {rank_correlation:.3f})")
        print("   -> VOI formulations identify different questions as valuable")

    # Check for rank disagreements in top-5
    top5_entropy = sorted(voi_comparison, key=lambda x: x["entropy_rank"])[:5]
    top5_linear = sorted(voi_comparison, key=lambda x: x["linear_rank"])[:5]

    top5_entropy_ids = {p["pair_id"] for p in top5_entropy}
    top5_linear_ids = {p["pair_id"] for p in top5_linear}
    overlap = len(top5_entropy_ids & top5_linear_ids)

    print(f"   Top-5 overlap: {overlap}/5 questions in common")

    # 5. Practical recommendation
    print("\n5. PRACTICAL RECOMMENDATION")

    # Average advantage across noise levels
    advantages = []
    for data in stability_results.values():
        advantages.append(data["linear"]["mean_tau"] - data["entropy"]["mean_tau"])

    mean_advantage = np.mean(advantages)

    if mean_advantage > 0.02:
        print(f"   Linear VOI is more stable (avg τ advantage: {mean_advantage:+.3f})")
        print("   -> RECOMMEND: Use linear VOI for question ranking under uncertainty")
    elif mean_advantage < -0.02:
        print(f"   Entropy VOI is more stable (avg τ advantage: {-mean_advantage:+.3f})")
        print("   -> RECOMMEND: Use entropy VOI (standard information gain)")
    else:
        print(f"   Stability is roughly equal (avg τ difference: {mean_advantage:+.3f})")
        print("   -> RECOMMEND: Either formulation works; entropy has theoretical grounding")

    # 6. Key finding
    print("\n6. KEY FINDING")
    print("   Does changing the VOI formula solve the magnitude problem for question ranking?")

    # Check if linear stabilizes extreme pairs more
    if extreme.get("linear_tau") is not None:
        extreme_gap = extreme["linear_tau"] - extreme["entropy_tau"]
        if extreme_gap > 0.05:
            print(f"   -> YES for extreme probabilities: linear VOI is significantly more stable ({extreme_gap:+.3f})")
        else:
            print(f"   -> PARTIAL: linear VOI provides modest improvements at extremes ({extreme_gap:+.3f})")
    else:
        print("   -> Insufficient extreme probability pairs to evaluate")


def main():
    parser = argparse.ArgumentParser(description="Linear VOI vs Entropy VOI stability experiment")
    parser.add_argument("--twostage-results", type=str,
                        default="experiments/fb-conditional/scaffolding/two-stage/results_20260112_113429.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per noise level")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load pairs from two-stage results
    pairs_path = Path(args.twostage_results)
    if not pairs_path.exists():
        print(f"Two-stage results not found: {pairs_path}")
        return

    pairs = load_pairs_from_twostage(args.twostage_results)
    print(f"Loaded {len(pairs)} pairs with conditional estimates")

    # Add synthetic examples
    synthetics = create_synthetic_examples()
    all_pairs = pairs + synthetics
    print(f"Added {len(synthetics)} synthetic examples (total: {len(all_pairs)})")

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"experiments/magnitude/linear-voi/results_{timestamp}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run stability analysis
    print("\nRunning Monte Carlo stability analysis...")
    noise_levels = [0.02, 0.05, 0.10]
    stability_results = run_monte_carlo_stability(all_pairs, noise_levels, args.trials, args.seed)

    # Extreme base rate analysis
    print("Analyzing extreme base rates...")
    extreme_analysis = analyze_extreme_base_rates(all_pairs, 0.05, args.trials, args.seed)

    # VOI comparison
    print("Computing VOI comparisons...")
    voi_comparison = compute_voi_comparison(all_pairs)

    # Print results
    print_summary(stability_results, extreme_analysis, voi_comparison)
    print_analysis(stability_results, extreme_analysis, voi_comparison)

    # Save results
    output = {
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "category": p["category"],
                "p_a": p["p_a"],
                "p_a_given_b1": p["p_a_given_b1"],
                "p_a_given_b0": p["p_a_given_b0"],
            }
            for p in all_pairs
        ],
        "voi_comparison": voi_comparison,
        "stability_analysis": stability_results,
        "extreme_base_rate_analysis": extreme_analysis,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "n_trials": args.trials,
            "seed": args.seed,
            "noise_levels": noise_levels,
            "num_real_pairs": len(pairs),
            "num_synthetic_pairs": len(synthetics),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
