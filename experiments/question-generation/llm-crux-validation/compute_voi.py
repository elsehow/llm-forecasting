#!/usr/bin/env python3
"""
Step 4: Compute VOI for matched (ultimate, crux) pairs.

Uses LLM two-step ρ estimation (structural reasoning) and linear VOI.
This matches the methodology from the human-curated baseline for fair comparison.

Usage:
    uv run python experiments/question-generation/llm-crux-validation/compute_voi.py
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

# Add package to path
PACKAGE_DIR = Path(__file__).parent.parent.parent.parent / "packages" / "llm-forecasting" / "src"
sys.path.insert(0, str(PACKAGE_DIR))

from llm_forecasting.voi import (
    estimate_rho_two_step,
    linear_voi_from_rho,
    entropy_voi_from_rho,
    compare_voi_metrics_from_rho,
)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting" / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
RESULTS_DIR = Path(__file__).parent / "results"

# Use a better model for ρ estimation to match the human-curated baseline
RHO_MODEL = "anthropic/claude-3-5-sonnet-20241022"


def get_current_price(condition_id: str) -> float | None:
    """Get current price from price history."""
    price_file = PRICE_HISTORY_DIR / f"{condition_id[:40]}.json"
    if not price_file.exists():
        return None

    try:
        with open(price_file) as f:
            data = json.load(f)
        candles = data.get("candles", [])
        if candles:
            return candles[-1]["close"]
    except Exception:
        pass
    return None


async def compute_voi_for_pair(pair: dict) -> dict:
    """Compute VOI for a single (ultimate, crux/market) pair."""
    ultimate_question = pair["ultimate_question"]
    crux_market_question = pair["match"]["market_question"]

    # Get prices
    ultimate_id = pair["ultimate_id"]
    crux_market_id = pair["match"]["market_id"]

    ultimate_price = get_current_price(ultimate_id)
    crux_price = pair["match"]["market_price"]

    if ultimate_price is None:
        return {**pair, "voi_computed": False, "error": "No ultimate price"}

    # Estimate ρ using two-step approach
    try:
        rho, reasoning = await estimate_rho_two_step(
            question_a=ultimate_question,
            question_b=crux_market_question,
            model=RHO_MODEL,
        )
    except Exception as e:
        return {**pair, "voi_computed": False, "error": f"ρ estimation error: {e}"}

    # Compute VOI metrics
    voi_metrics = compare_voi_metrics_from_rho(rho, ultimate_price, crux_price)

    return {
        **pair,
        "voi_computed": True,
        "error": None,
        "ultimate_price": ultimate_price,
        "crux_price": crux_price,
        "rho": rho,
        "rho_reasoning": reasoning,
        "linear_voi": voi_metrics["linear_voi"],
        "entropy_voi": voi_metrics["entropy_voi"],
        "entropy_voi_normalized": voi_metrics["entropy_voi_normalized"],
    }


async def main():
    print("=" * 70)
    print("COMPUTE VOI FOR MATCHED PAIRS")
    print("=" * 70)

    # Load matched pairs
    matched_path = RESULTS_DIR / "matched_pairs.json"
    if not matched_path.exists():
        print(f"\n❌ {matched_path} not found. Run match_cruxes.py first.")
        return

    with open(matched_path) as f:
        matched_data = json.load(f)
    pairs = matched_data["matched_pairs"]
    print(f"\nLoaded {len(pairs)} matched pairs")

    # Compute VOI
    print(f"\nComputing VOI using two-step ρ estimation ({RHO_MODEL})...")
    results = []
    batch_size = 5

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            compute_voi_for_pair(pair)
            for pair in batch
        ])
        results.extend(batch_results)
        print(f"  Processed {min(i+batch_size, len(pairs))}/{len(pairs)}")

    # Stats
    successful = [r for r in results if r["voi_computed"]]
    failed = [r for r in results if not r["voi_computed"]]

    print(f"\n{'='*70}")
    print("VOI COMPUTATION RESULTS")
    print("=" * 70)
    print(f"\nSuccessfully computed: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed[:3]:
            print(f"  - {f['error']}")

    if not successful:
        print("\n❌ No VOI computed!")
        return

    # ρ distribution
    rhos = [r["rho"] for r in successful]
    import numpy as np
    print(f"\nρ distribution:")
    print(f"  Mean: {np.mean(rhos):.3f}")
    print(f"  Std: {np.std(rhos):.3f}")
    print(f"  Min: {np.min(rhos):.3f}")
    print(f"  Max: {np.max(rhos):.3f}")

    # ρ sign distribution
    positive_rho = sum(1 for r in rhos if r > 0.05)
    negative_rho = sum(1 for r in rhos if r < -0.05)
    near_zero = sum(1 for r in rhos if -0.05 <= r <= 0.05)
    print(f"\nρ sign distribution:")
    print(f"  Positive (>0.05): {positive_rho}")
    print(f"  Negative (<-0.05): {negative_rho}")
    print(f"  Near zero: {near_zero}")

    # Linear VOI distribution
    linear_vois = [r["linear_voi"] for r in successful]
    print(f"\nLinear VOI distribution:")
    print(f"  Mean: {np.mean(linear_vois):.4f}")
    print(f"  Std: {np.std(linear_vois):.4f}")
    print(f"  Min: {np.min(linear_vois):.4f}")
    print(f"  Max: {np.max(linear_vois):.4f}")

    # Entropy VOI distribution
    entropy_vois = [r["entropy_voi"] for r in successful]
    print(f"\nEntropy VOI distribution (bits):")
    print(f"  Mean: {np.mean(entropy_vois):.4f}")
    print(f"  Std: {np.std(entropy_vois):.4f}")
    print(f"  Min: {np.min(entropy_vois):.4f}")
    print(f"  Max: {np.max(entropy_vois):.4f}")

    # Correlation between ρ magnitude and VOI (sanity check)
    abs_rhos = np.abs(rhos)
    from scipy import stats
    corr, pval = stats.pearsonr(abs_rhos, linear_vois)
    print(f"\nCorrelation |ρ| vs Linear VOI: r={corr:.3f} (p={pval:.4f})")

    # VOI by crux magnitude
    print(f"\nLinear VOI by crux magnitude:")
    for mag in ["high", "medium", "low"]:
        mag_vois = [r["linear_voi"] for r in successful if r["crux"].get("magnitude") == mag]
        if mag_vois:
            print(f"  {mag}: mean={np.mean(mag_vois):.4f}, n={len(mag_vois)}")

    # Top VOI pairs
    sorted_by_voi = sorted(successful, key=lambda x: -x["linear_voi"])
    print(f"\nTop 5 by Linear VOI:")
    for i, r in enumerate(sorted_by_voi[:5]):
        print(f"\n{i+1}. VOI={r['linear_voi']:.4f}, ρ={r['rho']:.2f}")
        print(f"   Ultimate: {r['ultimate_question'][:50]}...")
        print(f"   Crux: {r['crux']['crux'][:50]}...")
        print(f"   Match: {r['match']['market_question'][:50]}...")

    # Save
    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rho_model": RHO_MODEL,
            "n_pairs": len(pairs),
            "n_successful": len(successful),
            "rho_stats": {
                "mean": float(np.mean(rhos)),
                "std": float(np.std(rhos)),
                "min": float(np.min(rhos)),
                "max": float(np.max(rhos)),
                "positive_count": positive_rho,
                "negative_count": negative_rho,
                "near_zero_count": near_zero,
            },
            "linear_voi_stats": {
                "mean": float(np.mean(linear_vois)),
                "std": float(np.std(linear_vois)),
                "min": float(np.min(linear_vois)),
                "max": float(np.max(linear_vois)),
            },
            "entropy_voi_stats": {
                "mean": float(np.mean(entropy_vois)),
                "std": float(np.std(entropy_vois)),
                "min": float(np.min(entropy_vois)),
                "max": float(np.max(entropy_vois)),
            },
        },
        "pairs_with_voi": results,
    }

    output_path = RESULTS_DIR / "pairs_with_voi.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {output_path}")

    # Summary for comparison
    print(f"\n{'='*70}")
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    print(f"\nHuman-curated baseline (from voi-validation):")
    print(f"  Correlation with actual shifts: r=0.65")
    print(f"\nLLM-generated cruxes (this experiment):")
    print(f"  Total pairs: {len(successful)}")
    print(f"  Mean |ρ|: {np.mean(np.abs(rhos)):.3f}")
    print(f"  Mean Linear VOI: {np.mean(linear_vois):.4f}")
    print(f"\n⏳ Next step: Wait for crux market resolutions, then run validate.py")


if __name__ == "__main__":
    asyncio.run(main())
