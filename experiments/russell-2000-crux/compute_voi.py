"""Compute VOI for each crux using LLM-based conditional forecasting.

For each (stock-day, crux) pair:
1. Estimate ρ (correlation) between stock direction and crux outcome
2. Elicit P(stock up | crux=yes) and P(stock up | crux=no)
3. Compute Linear VOI and Entropy VOI
"""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

# Add package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "llm-forecasting" / "src"))

from llm_forecasting.voi import (
    estimate_rho_two_step,
    linear_voi,
    entropy_voi,
    rho_to_posteriors,
)

from config import RHO_ESTIMATION_MODEL

DATA_DIR = Path(__file__).parent / "data"

# Global to be set by args
_model_override = None


async def compute_voi_for_crux(
    ticker: str,
    date: str,
    crux: str,
    p_stock_up: float = 0.5,  # Prior: stock equally likely up or down
    model: str | None = None,
) -> dict:
    """Compute VOI for a single crux.

    Args:
        ticker: Stock symbol
        date: Date string
        crux: Crux question
        p_stock_up: Prior probability stock closes higher than open
        model: Model to use for rho estimation (defaults to RHO_ESTIMATION_MODEL)

    Returns:
        Dict with rho, linear_voi, entropy_voi, reasoning
    """
    # Frame the ultimate question
    ultimate = f"Will {ticker} close higher than it opened on {date}?"

    # Use model override if set
    model_to_use = model or _model_override or RHO_ESTIMATION_MODEL

    try:
        # Two-step rho estimation (direction then magnitude)
        rho, reasoning = await estimate_rho_two_step(
            ultimate,
            crux,
            model=model_to_use,
        )

        # Estimate P(crux = yes) - assume 0.5 as prior if unknown
        p_crux = 0.5

        # Convert rho to conditional probabilities
        p_up_given_yes, p_up_given_no = rho_to_posteriors(rho, p_stock_up, p_crux)

        # Compute VOI metrics
        voi_linear = linear_voi(p_stock_up, p_crux, p_up_given_yes, p_up_given_no)
        voi_entropy = entropy_voi(p_stock_up, p_crux, p_up_given_yes, p_up_given_no)

        return {
            "rho": rho,
            "reasoning": reasoning,
            "p_up_given_yes": p_up_given_yes,
            "p_up_given_no": p_up_given_no,
            "linear_voi": voi_linear,
            "entropy_voi": voi_entropy,
        }

    except Exception as e:
        return {
            "rho": 0.0,
            "reasoning": f"Error: {e}",
            "p_up_given_yes": 0.5,
            "p_up_given_no": 0.5,
            "linear_voi": 0.0,
            "entropy_voi": 0.0,
        }


async def compute_voi_batch(cruxes_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VOI for all cruxes in batch.

    Args:
        cruxes_df: DataFrame with ticker, date, crux columns

    Returns:
        DataFrame with VOI columns added
    """
    results = []

    # Process in batches
    batch_size = 10
    total = len(cruxes_df)

    for i in range(0, total, batch_size):
        batch = cruxes_df.iloc[i:i+batch_size]
        print(f"  Processing {i+1}-{min(i+batch_size, total)} of {total}...")

        tasks = [
            compute_voi_for_crux(
                row["ticker"],
                row["date"],
                row["crux"],
            )
            for _, row in batch.iterrows()
        ]

        voi_results = await asyncio.gather(*tasks)

        for (_, row), voi in zip(batch.iterrows(), voi_results):
            results.append({
                **row.to_dict(),
                **voi,
            })

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute VOI for cruxes")
    parser.add_argument(
        "--input",
        type=str,
        default="cruxes_pilot.parquet",
        help="Input cruxes file (default: cruxes_pilot.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: input_with_voi.parquet)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model for rho estimation (default: {RHO_ESTIMATION_MODEL})"
    )
    return parser.parse_args()


async def main():
    global _model_override

    args = parse_args()

    # Set model override
    if args.model:
        _model_override = args.model

    # Load cruxes
    input_file = args.input
    if not input_file.endswith(".parquet"):
        input_file += ".parquet"
    cruxes_path = DATA_DIR / input_file

    if not cruxes_path.exists():
        print(f"Input file not found: {cruxes_path}")
        print("Run generate_cruxes.py first!")
        return

    cruxes_df = pd.read_parquet(cruxes_path)

    model_to_use = _model_override or RHO_ESTIMATION_MODEL
    print(f"\n=== Computing VOI ===")
    print(f"Input: {input_file}")
    print(f"Cruxes to process: {len(cruxes_df)}")
    print(f"Model: {model_to_use}")

    # Compute VOI
    results_df = await compute_voi_batch(cruxes_df)

    # Summary
    print(f"\n=== VOI Summary ===")
    print(f"Linear VOI: mean={results_df['linear_voi'].mean():.4f}, std={results_df['linear_voi'].std():.4f}, max={results_df['linear_voi'].max():.4f}")
    print(f"Entropy VOI: mean={results_df['entropy_voi'].mean():.4f}, max={results_df['entropy_voi'].max():.4f}")
    print(f"Rho: mean={results_df['rho'].mean():.4f}, std={results_df['rho'].std():.4f}")

    # Show top cruxes by VOI
    print(f"\n=== Top 10 Cruxes by Linear VOI ===")
    top_cruxes = results_df.nlargest(10, "linear_voi")
    for _, row in top_cruxes.iterrows():
        print(f"\n{row['ticker']} on {row['date']} (VOI={row['linear_voi']:.4f}, ρ={row['rho']:.3f}):")
        print(f"  {row['crux']}")

    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Default: replace .parquet with _with_voi.parquet
        output_file = input_file.replace(".parquet", "_with_voi.parquet")

    if not output_file.endswith(".parquet"):
        output_file += ".parquet"

    results_df.to_parquet(DATA_DIR / output_file, index=False)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
