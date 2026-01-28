#!/usr/bin/env python3
"""
Compute VOI and actual shifts for within-category resolved pairs.

For each resolved pair:
1. Compute VOI using market-derived rho (not LLM rho)
2. Compute actual shift when the paired market resolved
3. Output validation data for analysis
"""

import json
import math
from pathlib import Path
from datetime import datetime

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho, entropy_voi_from_rho

# Paths
INPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data"


def compute_voi_metrics(rho: float, p_resolved: float, p_other: float) -> dict:
    """
    Compute VOI metrics for a pair.

    Args:
        rho: Market-derived correlation between the two markets
        p_resolved: Prior probability of the market that resolved
        p_other: Prior probability of the other market

    Returns:
        Dict with linear_voi and entropy_voi
    """
    if math.isnan(rho):
        return {"linear_voi": None, "entropy_voi": None}

    linear = linear_voi_from_rho(rho, p_other, p_resolved)
    entropy = entropy_voi_from_rho(rho, p_other, p_resolved)

    return {
        "linear_voi": float(linear) if not math.isnan(linear) else None,
        "entropy_voi": float(entropy) if not math.isnan(entropy) else None,
    }


def main():
    print("=" * 70)
    print("COMPUTE WITHIN-CATEGORY VOI AND ACTUAL SHIFTS")
    print("=" * 70)

    categories_to_process = ["fed_monetary", "politics"]
    results = {}

    for cat_name in categories_to_process:
        input_path = INPUT_DIR / f"resolved_{cat_name}_pairs.json"
        if not input_path.exists():
            print(f"\n{cat_name}: No resolved pairs file found, skipping")
            continue

        print(f"\n[Processing {cat_name}]")

        with open(input_path) as f:
            resolved_pairs = json.load(f)

        print(f"  Input pairs: {len(resolved_pairs)}")

        validation_records = []

        for rp in resolved_pairs:
            pair = rp["pair"]
            rho = pair["rho"]

            # Get prices
            p_resolved = rp["resolution"]["price_before"]
            p_other = rp["other_market"]["price_before"]

            # Compute VOI
            voi_metrics = compute_voi_metrics(rho, p_resolved, p_other)

            if voi_metrics["linear_voi"] is None:
                continue

            # Compute actual shift
            actual_shift = abs(rp["other_market"]["price_after"] - rp["other_market"]["price_before"])

            validation_records.append({
                "pair_id": f"{pair['market_a']['condition_id'][:8]}_{pair['market_b']['condition_id'][:8]}",
                "resolved_question": rp["resolution"]["question"],
                "other_question": rp["other_market"]["question"],
                "rho": rho,
                "abs_rho": abs(rho),
                "resolution_outcome": rp["resolution"]["outcome"],
                "resolution_date": rp["resolution"]["date"],
                "p_resolved_before": p_resolved,
                "p_other_before": p_other,
                "p_other_after": rp["other_market"]["price_after"],
                "linear_voi": voi_metrics["linear_voi"],
                "entropy_voi": voi_metrics["entropy_voi"],
                "actual_shift": actual_shift,
                "n_observations": pair.get("n_observations", 0),
            })

        results[cat_name] = validation_records
        print(f"  Valid records: {len(validation_records)}")

        # Quick stats
        if validation_records:
            vois = [r["linear_voi"] for r in validation_records]
            shifts = [r["actual_shift"] for r in validation_records]
            print(f"  Linear VOI: mean={sum(vois)/len(vois):.3f}, range=[{min(vois):.3f}, {max(vois):.3f}]")
            print(f"  Actual shift: mean={sum(shifts)/len(shifts):.3f}, range=[{min(shifts):.3f}, {max(shifts):.3f}]")

    # Save results
    print("\n" + "-" * 70)
    print("Saving results...")

    for cat_name, records in results.items():
        if not records:
            continue

        output_path = OUTPUT_DIR / f"{cat_name}_voi_validation.json"
        with open(output_path, "w") as f:
            json.dump({
                "metadata": {
                    "category": cat_name,
                    "n_pairs": len(records),
                    "computed_at": datetime.now().isoformat(),
                },
                "records": records
            }, f, indent=2)
        print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()
