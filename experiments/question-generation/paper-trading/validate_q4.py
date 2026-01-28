#!/usr/bin/env python3
"""
Q4 Validation: Do high-VOI generated cruxes predict actual market movement?

This script tracks generated cruxes from the MVP benchmark and measures
whether high-VOI cruxes correctly predict ultimate market movements.

Check dates:
- Feb 3, 2026: First check
- March 2, 2026: Follow-up if more statistical power needed

Usage:
    uv run python experiments/question-generation/paper-trading/validate_q4.py
    uv run python experiments/question-generation/paper-trading/validate_q4.py --results calibrated

Prerequisites:
    - Run benchmark first: experiments/question-generation/benchmark-mvp/run_benchmark.py
    - For calibrated results: experiments/question-generation/benchmark-mvp/rescore_with_calibrated.py
    - Ensure price history is up to date
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Paths
PAPER_TRADING_DIR = Path(__file__).parent
BENCHMARK_DIR = PAPER_TRADING_DIR.parent / "benchmark-mvp"
CONDITIONAL_DIR = PAPER_TRADING_DIR.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
RESULTS_DIR = PAPER_TRADING_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Result files
BENCHMARK_RESULTS_FILES = {
    "baseline": BENCHMARK_DIR / "results" / "benchmark_results.json",
    "calibrated": BENCHMARK_DIR / "results" / "benchmark_results_calibrated.json",
    "enhanced": PAPER_TRADING_DIR / "results" / "q4_enhanced_cruxes.json",
}
BENCHMARK_RESULTS = BENCHMARK_RESULTS_FILES["baseline"]  # Default


@dataclass
class CruxTracker:
    """Track a generated crux and its predictions."""
    crux_text: str
    ultimate_text: str
    ultimate_condition_id: str
    voi_linear: float
    voi_entropy: float
    voi_entropy_normalized: float
    rho: float
    p_ultimate_initial: float
    p_crux_initial: float
    p_ultimate_given_crux_yes: float
    p_ultimate_given_crux_no: float

    # Filled in during validation
    p_ultimate_current: float | None = None
    crux_resolved: bool = False
    crux_outcome: bool | None = None  # True = YES, False = NO
    actual_shift: float | None = None
    predicted_shift: float | None = None


def load_benchmark_results(results_type: str = "baseline") -> tuple[dict, str]:
    """Load generated cruxes from benchmark.

    Args:
        results_type: "baseline", "calibrated", or "enhanced"

    Returns:
        Tuple of (benchmark data dict, results type label)
    """
    results_file = BENCHMARK_RESULTS_FILES.get(results_type)
    if results_file is None:
        raise ValueError(f"Unknown results type: {results_type}. Use 'baseline', 'calibrated', or 'enhanced'")

    if not results_file.exists():
        if results_type == "calibrated":
            raise FileNotFoundError(
                f"Calibrated results not found at {results_file}\n"
                "Run rescore first: uv run python experiments/question-generation/benchmark-mvp/rescore_with_calibrated.py"
            )
        elif results_type == "enhanced":
            raise FileNotFoundError(
                f"Enhanced results not found at {results_file}\n"
                "Run enhanced crux generation first."
            )
        raise FileNotFoundError(
            f"Benchmark results not found at {results_file}\n"
            "Run benchmark first: uv run python experiments/question-generation/benchmark-mvp/run_benchmark.py"
        )

    with open(results_file) as f:
        data = json.load(f)

    return data, results_type


def load_price_history(condition_id: str) -> list[dict] | None:
    """Load price history for a market."""
    path = PRICE_HISTORY_DIR / f"{condition_id}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    return data.get("candles", [])


def get_current_price(candles: list[dict]) -> float | None:
    """Get most recent price from candles."""
    if not candles:
        return None
    return candles[-1].get("close")


def get_price_at_date(candles: list[dict], target_date: datetime) -> float | None:
    """Get price at or before a specific date."""
    target_ts = int(target_date.timestamp())
    for c in reversed(candles):
        if c["timestamp"] <= target_ts:
            return c["close"]
    return candles[0]["close"] if candles else None


def extract_trackers(benchmark_data: dict) -> list[CruxTracker]:
    """Extract crux trackers from benchmark results."""
    trackers = []

    for result in benchmark_data.get("results", []):
        if "error" in result:
            continue

        ultimate = result["ultimate"]
        condition_id = result["condition_id"]

        for cs in result.get("crux_scores", []):
            conditionals = cs.get("conditionals", {})
            if "error" in conditionals:
                continue

            tracker = CruxTracker(
                crux_text=cs["crux"],
                ultimate_text=ultimate,
                ultimate_condition_id=condition_id,
                voi_linear=cs.get("voi_linear", cs.get("voi", 0)),
                voi_entropy=cs.get("voi_entropy", 0),
                voi_entropy_normalized=cs.get("voi_entropy_normalized", 0),
                rho=cs.get("rho_estimated", 0),
                p_ultimate_initial=conditionals.get("p_ultimate", 0.5),
                p_crux_initial=conditionals.get("p_crux", 0.5),
                p_ultimate_given_crux_yes=conditionals.get("p_ultimate_given_crux_yes", 0.5),
                p_ultimate_given_crux_no=conditionals.get("p_ultimate_given_crux_no", 0.5),
            )
            trackers.append(tracker)

    return trackers


def extract_trackers_enhanced(data: dict) -> list[CruxTracker]:
    """Extract trackers from q4_enhanced_cruxes.json format.

    The enhanced format has different field names:
    - predictions (not results)
    - enhanced_cruxes (not crux_scores)
    - p_crux_yes, p_ult_given_yes, p_ult_given_no (not nested conditionals)
    """
    trackers = []

    for pred in data.get("predictions", []):
        ultimate = pred["ultimate"]
        condition_id = pred["condition_id"]

        for crux in pred.get("enhanced_cruxes", []):
            # Enhanced format has different field names
            p_crux = crux.get("p_crux_yes", 0.5)
            p_ult_given_yes = crux.get("p_ult_given_yes", 0.5)
            p_ult_given_no = crux.get("p_ult_given_no", 0.5)
            voi_linear = crux.get("voi_linear", 0)

            # Derive p_ultimate from conditionals using Bayesian logic
            # p_ult = p_crux * p_ult_given_yes + (1-p_crux) * p_ult_given_no
            p_ultimate = p_crux * p_ult_given_yes + (1 - p_crux) * p_ult_given_no

            tracker = CruxTracker(
                crux_text=crux["text"],
                ultimate_text=ultimate,
                ultimate_condition_id=condition_id,
                voi_linear=voi_linear,
                voi_entropy=0,  # Not available in enhanced format
                voi_entropy_normalized=0,
                rho=0,  # Not available
                p_ultimate_initial=p_ultimate,
                p_crux_initial=p_crux,
                p_ultimate_given_crux_yes=p_ult_given_yes,
                p_ultimate_given_crux_no=p_ult_given_no,
            )
            trackers.append(tracker)

    return trackers


def update_tracker_prices(trackers: list[CruxTracker]) -> list[CruxTracker]:
    """Update trackers with current market prices."""
    # Cache price histories
    price_cache: dict[str, list[dict]] = {}

    for tracker in trackers:
        cond_id = tracker.ultimate_condition_id

        if cond_id not in price_cache:
            price_cache[cond_id] = load_price_history(cond_id) or []

        candles = price_cache[cond_id]
        if candles:
            tracker.p_ultimate_current = get_current_price(candles)
            if tracker.p_ultimate_current is not None:
                tracker.actual_shift = abs(tracker.p_ultimate_current - tracker.p_ultimate_initial)

    return trackers


def compute_predicted_shifts(trackers: list[CruxTracker]) -> list[CruxTracker]:
    """Compute predicted shift for each tracker based on VOI logic.

    Predicted shift = expected absolute change in P(ultimate)
    This is exactly what linear VOI measures, but we can also compute
    what the shift would be if crux resolves YES vs NO.
    """
    for tracker in trackers:
        # Expected shift (this is linear VOI)
        shift_if_yes = abs(tracker.p_ultimate_given_crux_yes - tracker.p_ultimate_initial)
        shift_if_no = abs(tracker.p_ultimate_given_crux_no - tracker.p_ultimate_initial)
        tracker.predicted_shift = (
            tracker.p_crux_initial * shift_if_yes +
            (1 - tracker.p_crux_initial) * shift_if_no
        )

    return trackers


def analyze_results(trackers: list[CruxTracker]) -> dict:
    """Analyze correlation between VOI and actual shifts."""
    # Filter to trackers with actual shift data
    valid = [t for t in trackers if t.actual_shift is not None]

    if len(valid) < 5:
        return {
            "error": f"Only {len(valid)} trackers have shift data. Need at least 5.",
            "n_valid": len(valid),
            "n_total": len(trackers),
        }

    # Extract arrays
    linear_vois = np.array([t.voi_linear for t in valid])
    entropy_vois = np.array([t.voi_entropy for t in valid])
    entropy_vois_norm = np.array([t.voi_entropy_normalized for t in valid])
    actual_shifts = np.array([t.actual_shift for t in valid])
    predicted_shifts = np.array([t.predicted_shift for t in valid])

    # Correlations
    r_linear, p_linear = stats.pearsonr(linear_vois, actual_shifts)
    r_entropy, p_entropy = stats.pearsonr(entropy_vois, actual_shifts)
    r_entropy_norm, p_entropy_norm = stats.pearsonr(entropy_vois_norm, actual_shifts)
    r_predicted, p_predicted = stats.pearsonr(predicted_shifts, actual_shifts)

    # Spearman for robustness
    rho_linear, p_rho_linear = stats.spearmanr(linear_vois, actual_shifts)
    rho_entropy, p_rho_entropy = stats.spearmanr(entropy_vois, actual_shifts)

    # Compare methods
    tau_voi, p_tau = stats.kendalltau(linear_vois, entropy_vois)

    # Determine winner
    if r_linear > r_entropy:
        winner = "Linear VOI"
        advantage = r_linear - r_entropy
    else:
        winner = "Entropy VOI"
        advantage = r_entropy - r_linear

    return {
        "n_valid": len(valid),
        "n_total": len(trackers),
        "mean_actual_shift": float(np.mean(actual_shifts)),
        "mean_linear_voi": float(np.mean(linear_vois)),
        "mean_entropy_voi": float(np.mean(entropy_vois)),
        "correlations": {
            "linear_voi_vs_shift": {"r": float(r_linear), "p": float(p_linear)},
            "entropy_voi_vs_shift": {"r": float(r_entropy), "p": float(p_entropy)},
            "entropy_voi_norm_vs_shift": {"r": float(r_entropy_norm), "p": float(p_entropy_norm)},
            "predicted_shift_vs_actual": {"r": float(r_predicted), "p": float(p_predicted)},
        },
        "spearman": {
            "linear_voi_vs_shift": {"rho": float(rho_linear), "p": float(p_rho_linear)},
            "entropy_voi_vs_shift": {"rho": float(rho_entropy), "p": float(p_rho_entropy)},
        },
        "voi_comparison": {
            "linear_vs_entropy_tau": float(tau_voi),
            "better_predictor": winner,
            "advantage_delta_r": float(advantage),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate Q4 benchmark: Do high-VOI cruxes predict market movement?"
    )
    parser.add_argument(
        "--results",
        choices=["baseline", "calibrated", "enhanced"],
        default="baseline",
        help="Which benchmark results to validate (default: baseline)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Q4 VALIDATION: Do High-VOI Cruxes Predict Market Movement?")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Results: {args.results}")

    # Load benchmark results
    print(f"\nLoading {args.results} benchmark results...")
    try:
        benchmark_data, results_type = load_benchmark_results(args.results)
    except FileNotFoundError as e:
        print(f"\n{e}")
        return

    metadata = benchmark_data.get("metadata", {})
    print(f"  Benchmark from: {metadata.get('timestamp', 'unknown')}")
    print(f"  Total cruxes: {metadata.get('total_cruxes', 'unknown')}")

    # Extract trackers
    print("\nExtracting crux trackers...")
    if args.results == "enhanced":
        trackers = extract_trackers_enhanced(benchmark_data)
    else:
        trackers = extract_trackers(benchmark_data)
    print(f"  {len(trackers)} cruxes with valid conditionals")

    # Update with current prices
    print("\nFetching current market prices...")
    trackers = update_tracker_prices(trackers)
    with_prices = [t for t in trackers if t.p_ultimate_current is not None]
    print(f"  {len(with_prices)} ultimates have price history")

    # Compute predicted shifts
    trackers = compute_predicted_shifts(trackers)

    # Analyze
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    results = analyze_results(trackers)

    if "error" in results:
        print(f"\n⚠️ {results['error']}")
        print("Check back when more price data is available.")
    else:
        print(f"\nValid cruxes: {results['n_valid']} / {results['n_total']}")
        print(f"Mean actual shift: {results['mean_actual_shift']:.3f}")

        print("\n--- Correlation with Actual Shifts ---")
        corrs = results["correlations"]
        print(f"Linear VOI:     r = {corrs['linear_voi_vs_shift']['r']:.3f}, p = {corrs['linear_voi_vs_shift']['p']:.3f}")
        print(f"Entropy VOI:    r = {corrs['entropy_voi_vs_shift']['r']:.3f}, p = {corrs['entropy_voi_vs_shift']['p']:.3f}")
        print(f"Entropy (norm): r = {corrs['entropy_voi_norm_vs_shift']['r']:.3f}, p = {corrs['entropy_voi_norm_vs_shift']['p']:.3f}")
        print(f"Predicted shift: r = {corrs['predicted_shift_vs_actual']['r']:.3f}, p = {corrs['predicted_shift_vs_actual']['p']:.3f}")

        print("\n--- Spearman (Rank Correlation) ---")
        spearman = results["spearman"]
        print(f"Linear VOI:  ρ = {spearman['linear_voi_vs_shift']['rho']:.3f}, p = {spearman['linear_voi_vs_shift']['p']:.3f}")
        print(f"Entropy VOI: ρ = {spearman['entropy_voi_vs_shift']['rho']:.3f}, p = {spearman['entropy_voi_vs_shift']['p']:.3f}")

        print("\n--- VOI Method Comparison ---")
        comp = results["voi_comparison"]
        print(f"Better predictor: {comp['better_predictor']} (Δr = {comp['advantage_delta_r']:.3f})")
        print(f"Method agreement: τ = {comp['linear_vs_entropy_tau']:.3f}")

        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        best_r = max(
            corrs['linear_voi_vs_shift']['r'],
            corrs['entropy_voi_vs_shift']['r']
        )
        best_p = min(
            corrs['linear_voi_vs_shift']['p'],
            corrs['entropy_voi_vs_shift']['p']
        )

        if best_p < 0.05:
            print(f"\n✅ Q4 VALIDATED")
            print(f"   High-VOI generated cruxes predict actual market movement")
            print(f"   Best correlation: r = {best_r:.3f}, p = {best_p:.3f}")
        elif best_p < 0.10:
            print(f"\n⚠️ MARGINALLY SIGNIFICANT")
            print(f"   Directional evidence, but need more data")
            print(f"   Best correlation: r = {best_r:.3f}, p = {best_p:.3f}")
        elif best_r > 0.2:
            print(f"\n📊 DIRECTIONALLY CORRECT BUT UNDERPOWERED")
            print(f"   Effect exists but need more cruxes/time")
            print(f"   Best correlation: r = {best_r:.3f}, p = {best_p:.3f}")
        else:
            print(f"\n❌ NO CLEAR RELATIONSHIP YET")
            print(f"   r = {best_r:.3f}, p = {best_p:.3f}")
            print("   May need to wait for cruxes to resolve")

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_timestamp": metadata.get("timestamp"),
            "n_cruxes": len(trackers),
            "results_type": results_type,
            "rho_method": metadata.get("rho_method", "baseline"),
        },
        "results": results,
        "top_movers": [
            {
                "crux": t.crux_text[:80],
                "ultimate": t.ultimate_text[:60],
                "voi_linear": t.voi_linear,
                "voi_entropy": t.voi_entropy,
                "actual_shift": t.actual_shift,
                "p_initial": t.p_ultimate_initial,
                "p_current": t.p_ultimate_current,
            }
            for t in sorted(
                [t for t in trackers if t.actual_shift is not None],
                key=lambda t: t.actual_shift,
                reverse=True
            )[:10]
        ],
    }

    output_path = RESULTS_DIR / f"q4_validation_{results_type}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
