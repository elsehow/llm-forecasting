#!/usr/bin/env python3
"""
Validate that VOI (computed from historical ρ) predicts belief shifts (ΔP) on Metaculus.

Success criteria:
- r(VOI, |ΔP|) > 0.15  (VOI magnitude predicts belief shift magnitude)
- Direction accuracy > 55%  (sign(ρ) == sign(ΔP))
- n_pairs > 100

Input: metaculus_comovement_pairs.json (from collect_comovement_pairs.py)
Output: metaculus_voi_validation.json
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats
from llm_forecasting.voi import linear_voi_from_rho

# Configuration
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "metaculus_comovement_pairs.json"
CURATED_FILE = DATA_DIR / "metaculus_curated_pairs.json"
OUTPUT_FILE = DATA_DIR / "metaculus_voi_validation.json"

# Success criteria
MIN_PAIRS = 100
MIN_VOI_CORRELATION = 0.15
MIN_DIRECTION_ACCURACY = 0.55


def load_pairs() -> list[dict]:
    """Load co-movement pairs from JSON."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Run collect_comovement_pairs.py first."
        )

    with open(INPUT_FILE) as f:
        data = json.load(f)

    return data.get("pairs", [])


def load_curated_pairs() -> list[dict]:
    """Load hand-curated semantically-related pairs."""
    if not CURATED_FILE.exists():
        raise FileNotFoundError(f"Curated file not found: {CURATED_FILE}")

    with open(CURATED_FILE) as f:
        data = json.load(f)

    # Normalize field names to match comovement format
    pairs = []
    for p in data.get("pairs", []):
        pairs.append({
            "q_id": p["q_id"],
            "q_title": p["q_title"],
            "x_id": p["x_id"],
            "x_title": p["x_title"],
            "rho": p["rho"],
            "x_delta_p": p["x_delta_p"],
            "x_prob_before": p["x_prob_before"],
            "q_resolution": p["q_resolution"],
            "classification": p.get("classification"),
        })
    return pairs


def compute_voi_for_pair(pair: dict) -> float | None:
    """Compute linear VOI for a single pair.

    Uses linear_voi_from_rho which handles the rho → posterior conversion.

    Args:
        pair: Dict with rho, x_prob_before (used as p_a and p_b)

    Returns:
        VOI value or None if invalid
    """
    rho = pair.get("rho")
    p_x = pair.get("x_prob_before")  # P(X) before Q resolved

    if rho is None or p_x is None:
        return None

    if np.isnan(rho) or np.isnan(p_x):
        return None

    # For VOI, we use:
    # - p_a = P(X) = probability of X (what we're trying to predict)
    # - p_b = P(Q) - we don't have this directly, but for ranking by |ρ| we can use 0.5
    # Actually, for this validation, VOI ≈ |ρ| * σ(X) is sufficient for ranking
    # Let's use the full formula with p_b = 0.5 as a reasonable default

    try:
        voi = linear_voi_from_rho(rho, p_a=p_x, p_b=0.5)
        return voi
    except Exception:
        return None


def validate_pairs(pairs: list[dict], min_delta_p: float = 0.0) -> dict:
    """
    Validate VOI predictions against observed belief shifts.

    Args:
        pairs: List of pair dicts with rho, x_delta_p, etc.
        min_delta_p: Minimum |ΔP| to include pair (filters noise)

    Returns validation results including correlations and direction accuracy.
    """
    # Filter pairs with valid data
    valid_pairs = []
    for p in pairs:
        rho = p.get("rho")
        delta_p = p.get("x_delta_p")
        p_x = p.get("x_prob_before")

        if rho is None or delta_p is None or p_x is None:
            continue
        if np.isnan(rho) or np.isnan(delta_p) or np.isnan(p_x):
            continue

        # Filter for meaningful belief shifts
        if abs(delta_p) < min_delta_p:
            continue

        voi = compute_voi_for_pair(p)
        if voi is None:
            continue

        valid_pairs.append({
            **p,
            "voi": voi,
        })

    n = len(valid_pairs)
    print(f"Valid pairs for analysis: {n}")

    if n < 10:
        return {
            "status": "insufficient_data",
            "n_pairs": n,
            "message": "Need at least 10 valid pairs for analysis",
        }

    # Extract arrays
    rhos = np.array([p["rho"] for p in valid_pairs])
    delta_ps = np.array([p["x_delta_p"] for p in valid_pairs])
    vois = np.array([p["voi"] for p in valid_pairs])
    abs_delta_ps = np.abs(delta_ps)

    # Primary metric: Spearman correlation between |ρ| (or VOI) and |ΔP|
    # VOI is monotonic in |ρ|, so these are equivalent for ranking
    r_voi_dp, p_voi_dp = stats.spearmanr(vois, abs_delta_ps)
    r_rho_dp, p_rho_dp = stats.spearmanr(np.abs(rhos), abs_delta_ps)

    # Also compute Pearson for completeness
    r_voi_dp_pearson, _ = stats.pearsonr(vois, abs_delta_ps)
    r_rho_dp_pearson, _ = stats.pearsonr(np.abs(rhos), abs_delta_ps)

    # Direction accuracy: does sign(ρ) match sign(ΔP)?
    # Exclude pairs where either is ~0
    direction_mask = (np.abs(rhos) > 0.05) & (np.abs(delta_ps) > 0.01)
    if np.sum(direction_mask) > 0:
        matched_signs = np.sign(rhos[direction_mask]) == np.sign(delta_ps[direction_mask])
        direction_accuracy = np.mean(matched_signs)
        n_direction_valid = np.sum(direction_mask)
    else:
        direction_accuracy = None
        n_direction_valid = 0

    # Summary statistics
    stats_summary = {
        "rho": {
            "mean": float(np.mean(rhos)),
            "std": float(np.std(rhos)),
            "median": float(np.median(rhos)),
            "min": float(np.min(rhos)),
            "max": float(np.max(rhos)),
        },
        "delta_p": {
            "mean": float(np.mean(delta_ps)),
            "std": float(np.std(delta_ps)),
            "median": float(np.median(delta_ps)),
            "abs_mean": float(np.mean(abs_delta_ps)),
            "abs_median": float(np.median(abs_delta_ps)),
        },
        "voi": {
            "mean": float(np.mean(vois)),
            "std": float(np.std(vois)),
            "median": float(np.median(vois)),
            "min": float(np.min(vois)),
            "max": float(np.max(vois)),
        },
    }

    # Check success criteria
    passes_n = bool(n >= MIN_PAIRS)
    passes_voi_corr = bool(r_voi_dp is not None and r_voi_dp > MIN_VOI_CORRELATION)
    passes_direction = bool(direction_accuracy is not None and direction_accuracy > MIN_DIRECTION_ACCURACY)

    success = passes_n and passes_voi_corr and passes_direction

    return {
        "status": "success" if success else "below_criteria",
        "n_pairs": n,
        "n_direction_valid": int(n_direction_valid),

        # Primary metrics
        "r_voi_dp_spearman": float(r_voi_dp) if r_voi_dp is not None else None,
        "p_voi_dp_spearman": float(p_voi_dp) if p_voi_dp is not None else None,
        "r_rho_dp_spearman": float(r_rho_dp) if r_rho_dp is not None else None,
        "p_rho_dp_spearman": float(p_rho_dp) if p_rho_dp is not None else None,

        # Pearson for reference
        "r_voi_dp_pearson": float(r_voi_dp_pearson) if r_voi_dp_pearson is not None else None,
        "r_rho_dp_pearson": float(r_rho_dp_pearson) if r_rho_dp_pearson is not None else None,

        # Direction accuracy
        "direction_accuracy": float(direction_accuracy) if direction_accuracy is not None else None,

        # Summary stats
        "statistics": stats_summary,

        # Success criteria check
        "criteria": {
            "n_pairs": {"threshold": MIN_PAIRS, "actual": n, "passes": passes_n},
            "r_voi_dp": {"threshold": MIN_VOI_CORRELATION, "actual": r_voi_dp, "passes": passes_voi_corr},
            "direction_accuracy": {"threshold": MIN_DIRECTION_ACCURACY, "actual": direction_accuracy, "passes": passes_direction},
        },

        # Per-pair results for debugging
        "pairs": valid_pairs,
    }


def print_results(results: dict):
    """Print validation results in human-readable format."""
    print("\n" + "=" * 70)
    print("VOI VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nPairs analyzed: {results['n_pairs']}")
    print(f"Pairs with clear direction: {results['n_direction_valid']}")

    print("\n--- Primary Metrics ---")
    r_voi = results.get("r_voi_dp_spearman")
    p_voi = results.get("p_voi_dp_spearman")
    print(f"r(VOI, |ΔP|) Spearman: {r_voi:.3f}" if r_voi else "r(VOI, |ΔP|): N/A")
    if p_voi:
        print(f"  p-value: {p_voi:.4f}")

    r_rho = results.get("r_rho_dp_spearman")
    print(f"r(|ρ|, |ΔP|) Spearman: {r_rho:.3f}" if r_rho else "r(|ρ|, |ΔP|): N/A")

    dir_acc = results.get("direction_accuracy")
    print(f"Direction accuracy: {dir_acc:.1%}" if dir_acc else "Direction accuracy: N/A")

    print("\n--- Statistics ---")
    stats = results.get("statistics", {})

    rho_stats = stats.get("rho", {})
    print(f"ρ: mean={rho_stats.get('mean', 0):.3f}, std={rho_stats.get('std', 0):.3f}")

    dp_stats = stats.get("delta_p", {})
    print(f"|ΔP|: mean={dp_stats.get('abs_mean', 0):.3f}, median={dp_stats.get('abs_median', 0):.3f}")

    voi_stats = stats.get("voi", {})
    print(f"VOI: mean={voi_stats.get('mean', 0):.3f}, std={voi_stats.get('std', 0):.3f}")

    print("\n--- Success Criteria ---")
    criteria = results.get("criteria", {})
    for name, c in criteria.items():
        status = "✓" if c.get("passes") else "✗"
        threshold = c.get("threshold")
        actual = c.get("actual")
        if actual is not None:
            if isinstance(actual, float):
                print(f"{status} {name}: {actual:.3f} (threshold: {threshold})")
            else:
                print(f"{status} {name}: {actual} (threshold: {threshold})")
        else:
            print(f"✗ {name}: N/A (threshold: {threshold})")

    print("\n" + "=" * 70)
    if results.get("status") == "success":
        print("✓ ALL CRITERIA PASSED - VOI validated on Metaculus")
    else:
        print("✗ VALIDATION FAILED - See criteria above")
    print("=" * 70)


def main():
    print("=" * 70)
    print("METACULUS VOI VALIDATION")
    print("=" * 70)

    all_results = {}

    # === PART 1: Curated Pairs (semantic relationships) ===
    print("\n" + "=" * 70)
    print("PART 1: CURATED PAIRS (semantic relationships)")
    print("=" * 70)

    try:
        curated_pairs = load_curated_pairs()
        print(f"Loaded {len(curated_pairs)} curated pairs")

        # Validate curated pairs (no threshold - all are semantically related)
        curated_results = validate_pairs(curated_pairs, min_delta_p=0.0)
        print_results(curated_results)

        # By classification
        classifications = {}
        for p in curated_results.get("pairs", []):
            cls = p.get("classification", "unknown")
            if cls not in classifications:
                classifications[cls] = []
            classifications[cls].append(p)

        print("\n--- By Classification ---")
        for cls, cls_pairs in sorted(classifications.items()):
            if len(cls_pairs) >= 3:
                rhos = [abs(p["rho"]) for p in cls_pairs]
                dps = [abs(p["x_delta_p"]) for p in cls_pairs]
                if len(rhos) == len(dps) and np.std(rhos) > 0 and np.std(dps) > 0:
                    r, p_val = stats.spearmanr(rhos, dps)
                    print(f"  {cls}: n={len(cls_pairs)}, r(|ρ|,|ΔP|)={r:.3f}")
                else:
                    print(f"  {cls}: n={len(cls_pairs)}, insufficient variance")
            else:
                print(f"  {cls}: n={len(cls_pairs)} (too few)")

        all_results["curated"] = {k: v for k, v in curated_results.items() if k != "pairs"}
        all_results["curated"]["n_pairs"] = len(curated_results.get("pairs", []))

    except FileNotFoundError as e:
        print(f"Skipping curated pairs: {e}")

    # === PART 2: All Pairs (temporal overlap) ===
    print("\n" + "=" * 70)
    print("PART 2: ALL PAIRS (temporal overlap, for comparison)")
    print("=" * 70)

    try:
        pairs = load_pairs()
        print(f"Loaded {len(pairs)} pairs from temporal overlap")

        # Validate at multiple thresholds
        print("\n--- Validation at multiple ΔP thresholds ---")
        thresholds = [0.0, 0.01, 0.02, 0.05]
        for thresh in thresholds:
            filtered = [p for p in pairs if abs(p.get("x_delta_p", 0)) >= thresh]
            n = len(filtered)
            if n >= 10:
                rhos = np.array([p["rho"] for p in filtered if p.get("rho") is not None])
                dps = np.array([abs(p["x_delta_p"]) for p in filtered if p.get("x_delta_p") is not None])
                if len(rhos) == len(dps) and len(rhos) >= 10:
                    r, _ = stats.spearmanr(np.abs(rhos), dps)
                    print(f"  |ΔP| >= {thresh:.0%}: n={n}, r(|ρ|,|ΔP|)={r:.3f}")

        # Main validation with threshold
        results = validate_pairs(pairs, min_delta_p=0.01)
        print_results(results)

        all_results["temporal_overlap"] = {k: v for k, v in results.items() if k != "pairs"}

    except FileNotFoundError as e:
        print(f"Skipping temporal overlap pairs: {e}")

    # Save combined results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
