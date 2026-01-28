"""Apply timeliness rubric to all v2 cruxes.

Loads cruxes_all.parquet from v2 experiment, evaluates each crux
against the timeliness criterion, and saves results.
"""

import asyncio
from pathlib import Path

import pandas as pd

from timeliness import evaluate_timeliness_batch

# Paths
DATA_DIR = Path(__file__).parent / "data"
V2_DATA_DIR = Path(__file__).parent.parent / "russell-2000-crux" / "data"


async def main():
    # Load v2 cruxes
    cruxes_path = V2_DATA_DIR / "cruxes_all.parquet"
    if not cruxes_path.exists():
        print(f"Error: {cruxes_path} not found")
        print("Run russell-2000-crux experiment first!")
        return

    cruxes_df = pd.read_parquet(cruxes_path)
    print(f"\n=== Applying Timeliness Rubric ===")
    print(f"Cruxes to evaluate: {len(cruxes_df)}")

    # Convert to list of dicts for batch processing
    cruxes_list = cruxes_df.to_dict("records")

    # Evaluate timeliness
    results = await evaluate_timeliness_batch(cruxes_list)

    # Convert back to DataFrame
    results_df = pd.DataFrame(results)

    # Summary
    n_timely = results_df["timely"].sum()
    n_not_timely = len(results_df) - n_timely
    print(f"\n=== Results ===")
    print(f"Timely: {n_timely} ({n_timely/len(results_df)*100:.1f}%)")
    print(f"Not timely: {n_not_timely} ({n_not_timely/len(results_df)*100:.1f}%)")

    # Breakdown by earnings vs non-earnings
    earnings_df = results_df[results_df["is_earnings_day"]]
    non_earnings_df = results_df[~results_df["is_earnings_day"]]

    earnings_timely = earnings_df["timely"].sum() / len(earnings_df) if len(earnings_df) > 0 else 0
    non_earnings_timely = non_earnings_df["timely"].sum() / len(non_earnings_df) if len(non_earnings_df) > 0 else 0

    print(f"\nBreakdown:")
    print(f"  Earnings days: {earnings_timely*100:.1f}% timely ({len(earnings_df)} cruxes)")
    print(f"  Non-earnings days: {non_earnings_timely*100:.1f}% timely ({len(non_earnings_df)} cruxes)")

    # Show examples of NOT timely cruxes (moonshot detection)
    print(f"\n=== Examples of NOT TIMELY cruxes (moonshots) ===")
    not_timely_high_voi = results_df[~results_df["timely"]].nlargest(5, "linear_voi")
    for _, row in not_timely_high_voi.iterrows():
        print(f"\n  {row['ticker']} on {row['date']} (VOI={row['linear_voi']:.2f}):")
        print(f"    Crux: {row['crux'][:80]}...")
        print(f"    Reason: {row['timeliness_reason'][:80]}")

    # Show examples of TIMELY cruxes
    print(f"\n=== Examples of TIMELY cruxes ===")
    timely_samples = results_df[results_df["timely"]].head(5)
    for _, row in timely_samples.iterrows():
        print(f"\n  {row['ticker']} on {row['date']} (VOI={row['linear_voi']:.2f}):")
        print(f"    Crux: {row['crux'][:80]}...")
        print(f"    Reason: {row['timeliness_reason'][:80]}")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(DATA_DIR / "cruxes_evaluated.parquet", index=False)
    print(f"\nSaved to {DATA_DIR / 'cruxes_evaluated.parquet'}")


if __name__ == "__main__":
    asyncio.run(main())
