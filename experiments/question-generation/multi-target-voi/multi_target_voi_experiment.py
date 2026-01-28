#!/usr/bin/env python3
"""
Multi-Target VOI Experiment.

Hypothesis: Single-target VOI fails within domains (r~0 for earnings). Multi-target
VOI might reveal which cruxes have broad portfolio relevance - cruxes that shift
beliefs about MULTIPLE stocks should rank higher than cruxes that only matter for one.

Design:
- Dataset: Russell 2000 earnings (80 targets, existing cruxes)
- For each crux, compute VOI against ALL targets (not just home target)
- Aggregation: portfolio_voi = mean(VOI(crux, target_i)) for all targets
- Compare: single-target vs portfolio VOI correlation with actual returns

Success: r_portfolio > r_single (multi-target beats single-target)
Strong success: r_portfolio > 0.2, p < 0.1

Usage:
    cd /Users/elsehow/Projects/llm-forecasting

    # Full run (expensive: ~32,000 ρ estimations)
    uv run python experiments/question-generation/multi-target-voi/multi_target_voi_experiment.py

    # Sampled run (cheap: ~400 ρ estimations, 10 targets)
    uv run python experiments/question-generation/multi-target-voi/multi_target_voi_experiment.py --sample 10

    # Dry run (no API calls, just show what would happen)
    uv run python experiments/question-generation/multi-target-voi/multi_target_voi_experiment.py --dry-run
"""

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from tqdm import tqdm

# Load .env from monorepo root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from llm_forecasting.voi import (
    estimate_rho_two_step,
    estimate_rho_two_step_batch,
    linear_voi_from_rho,
    DEFAULT_RHO_MODEL,
)

# Paths
RUSSELL_CRUX_DIR = Path(__file__).parent.parent.parent / "russell-2000-crux" / "data"
OUTPUT_DIR = Path(__file__).parent / "data"


@dataclass
class Target:
    """A forecasting target (stock on earnings day)."""
    ticker: str
    date: str
    company_name: str
    sector: str
    context: str

    @property
    def question(self) -> str:
        """Generate the implicit target question."""
        return f"Will {self.ticker} close higher than it opened on {self.date}?"

    @property
    def key(self) -> tuple[str, str]:
        return (self.ticker, self.date)


@dataclass
class Crux:
    """A crux question with VOI data."""
    ticker: str  # home ticker
    date: str    # home date
    crux_index: int
    crux_text: str
    method: str

    # Single-target VOI (from original experiment)
    single_rho: float
    single_voi: float

    @property
    def home_key(self) -> tuple[str, str]:
        return (self.ticker, self.date)


def load_data() -> tuple[list[Target], list[Crux], pd.DataFrame]:
    """Load targets, cruxes, and returns data."""
    # Load cruxes
    cruxes_df = pd.read_parquet(RUSSELL_CRUX_DIR / "cruxes_with_voi.parquet")

    # Load returns
    returns_df = pd.read_parquet(RUSSELL_CRUX_DIR / "stock_returns.parquet")
    earnings_returns = returns_df[returns_df["is_earnings_day"]].copy()

    # Build targets list
    targets = []
    seen = set()
    for _, row in cruxes_df.iterrows():
        key = (row["ticker"], str(row["date"]))
        if key not in seen:
            targets.append(Target(
                ticker=row["ticker"],
                date=str(row["date"]),
                company_name=row["company_name"],
                sector=row["sector"],
                context=row["context"],
            ))
            seen.add(key)

    # Build cruxes list
    cruxes = []
    for _, row in cruxes_df.iterrows():
        cruxes.append(Crux(
            ticker=row["ticker"],
            date=str(row["date"]),
            crux_index=row["crux_index"],
            crux_text=row["crux"],
            method=row["method"],
            single_rho=row["rho"],
            single_voi=row["linear_voi"],
        ))

    return targets, cruxes, earnings_returns


async def compute_all_cross_target_vois(
    cruxes: list[Crux],
    targets: list[Target],
    p_target: float = 0.5,  # prior P(stock goes up)
    p_crux: float = 0.5,    # prior P(crux resolves yes)
    use_batch_api: bool = True,
) -> list[dict]:
    """Compute portfolio VOI for all cruxes against all targets.

    Uses batch API for efficiency (50% cost savings, faster).

    Returns list of dicts with:
    - portfolio_voi: mean VOI across all targets
    - home_voi: VOI for home target
    - cross_target_vois: list of (target_key, rho, voi) for all targets
    """
    # Build all (target_q, crux_text) pairs that need ρ estimation
    # Skip home targets - we already have those
    pairs_to_estimate = []
    pair_indices = []  # (crux_idx, target_idx)

    for ci, crux in enumerate(cruxes):
        for ti, target in enumerate(targets):
            if crux.home_key != target.key:  # Skip home - use existing
                pairs_to_estimate.append((target.question, crux.crux_text))
                pair_indices.append((ci, ti))

    print(f"      Total pairs to estimate: {len(pairs_to_estimate)}")
    print(f"      Home pairs (cached): {len(cruxes)}")

    if not pairs_to_estimate:
        # No cross-target pairs to estimate
        pass
    elif use_batch_api:
        # Use batch API for all estimations
        print(f"      Submitting batch request...")
        results = await estimate_rho_two_step_batch(pairs_to_estimate)
        print(f"      Batch complete: {len(results)} results")
    else:
        # Sequential estimation (for testing/debugging)
        print(f"      Running sequential estimation...")
        results = []
        for pair in tqdm(pairs_to_estimate, desc="Estimating ρ"):
            rho, reasoning = await estimate_rho_two_step(pair[0], pair[1])
            results.append((rho, reasoning))

    # Build lookup: (crux_idx, target_idx) -> (rho, reasoning)
    rho_lookup = {}
    for (ci, ti), (rho, reasoning) in zip(pair_indices, results):
        rho_lookup[(ci, ti)] = (rho, reasoning)

    # Now build portfolio VOI for each crux
    portfolio_results = []

    for ci, crux in enumerate(cruxes):
        cross_target_vois = []

        for ti, target in enumerate(targets):
            if crux.home_key == target.key:
                # Home target - use existing single_rho
                rho = crux.single_rho
                is_home = True
            else:
                # Cross target - look up estimated rho
                rho, _ = rho_lookup.get((ci, ti), (0.0, "missing"))
                is_home = False

            voi = linear_voi_from_rho(rho, p_target, p_crux)
            cross_target_vois.append({
                "target_ticker": target.ticker,
                "target_date": target.date,
                "rho": rho,
                "voi": voi,
                "is_home": is_home,
            })

        # Aggregate
        all_vois = [x["voi"] for x in cross_target_vois]
        non_home_vois = [x["voi"] for x in cross_target_vois if not x["is_home"]]
        home_entry = next((x for x in cross_target_vois if x["is_home"]), None)

        portfolio_results.append({
            "crux_ticker": crux.ticker,
            "crux_date": crux.date,
            "crux_index": crux.crux_index,
            "crux_text": crux.crux_text,
            "portfolio_voi": float(np.mean(all_vois)),
            "portfolio_voi_excl_home": float(np.mean(non_home_vois)) if non_home_vois else 0.0,
            "home_voi": home_entry["voi"] if home_entry else crux.single_voi,
            "single_voi": crux.single_voi,
            "n_nonzero_targets": sum(1 for x in cross_target_vois if abs(x["rho"]) > 0.05),
            "cross_target_vois": cross_target_vois,
        })

    return portfolio_results


def analyze_results(
    portfolio_results: list[dict],
    returns_df: pd.DataFrame,
) -> dict:
    """Analyze correlation between VOI variants and actual returns."""

    # Merge with returns
    rows = []
    for r in portfolio_results:
        key = (r["crux_ticker"], r["crux_date"])
        # Find return for this target
        match = returns_df[
            (returns_df["ticker"] == r["crux_ticker"]) &
            (returns_df["date"].astype(str) == r["crux_date"])
        ]
        if len(match) == 0:
            continue

        actual_return = match.iloc[0]["return"]
        went_up = actual_return > 0

        rows.append({
            "crux_ticker": r["crux_ticker"],
            "crux_date": r["crux_date"],
            "crux_index": r["crux_index"],
            "crux_text": r["crux_text"],
            "single_voi": r["single_voi"],
            "home_voi": r["home_voi"],
            "portfolio_voi": r["portfolio_voi"],
            "portfolio_voi_excl_home": r["portfolio_voi_excl_home"],
            "n_nonzero_targets": r["n_nonzero_targets"],
            "actual_return": actual_return,
            "went_up": went_up,
            "abs_return": abs(actual_return),
        })

    df = pd.DataFrame(rows)

    # Compute correlations
    # We want: does higher VOI predict larger belief shifts (abs return)?
    results = {}

    for voi_col in ["single_voi", "home_voi", "portfolio_voi", "portfolio_voi_excl_home"]:
        r, p = stats.pearsonr(df[voi_col], df["abs_return"])
        results[voi_col] = {"r": r, "p": p}

    # Also test: does n_nonzero_targets correlate with abs_return?
    r_nz, p_nz = stats.pearsonr(df["n_nonzero_targets"], df["abs_return"])
    results["n_nonzero_targets"] = {"r": r_nz, "p": p_nz}

    return {
        "correlations": results,
        "df": df,
        "n": len(df),
    }


def print_results(analysis: dict):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("MULTI-TARGET VOI EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"\nN cruxes analyzed: {analysis['n']}")

    print("\n" + "-" * 70)
    print("CORRELATION: VOI variant vs |actual return|")
    print("-" * 70)

    corrs = analysis["correlations"]

    print("\n{:<30} {:>10} {:>12}".format("VOI Variant", "r", "p-value"))
    print("-" * 52)

    for name, vals in corrs.items():
        sig = "***" if vals["p"] < 0.01 else "**" if vals["p"] < 0.05 else "*" if vals["p"] < 0.1 else ""
        print(f"{name:<30} {vals['r']:>10.3f} {vals['p']:>10.4f} {sig}")

    # Key comparison
    print("\n" + "-" * 70)
    print("KEY COMPARISON: Portfolio VOI vs Single-target VOI")
    print("-" * 70)

    r_portfolio = corrs["portfolio_voi"]["r"]
    r_single = corrs["single_voi"]["r"]

    print(f"\nSingle-target VOI:  r = {r_single:.3f}")
    print(f"Portfolio VOI:      r = {r_portfolio:.3f}")
    print(f"Improvement:        Δr = {r_portfolio - r_single:+.3f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if r_portfolio > r_single + 0.05 and corrs["portfolio_voi"]["p"] < 0.1:
        print("\n✅ SUCCESS: Portfolio VOI outperforms single-target VOI")
        print("   Multi-target aggregation improves within-domain ranking")
    elif r_portfolio > r_single:
        print("\n⚠️  MARGINAL: Portfolio VOI slightly better but not significant")
        print("   Direction is right but more data needed")
    else:
        print("\n❌ FAILED: Portfolio VOI does not beat single-target VOI")
        print("   Aggregation does not help within-domain discrimination")

    # Bucket analysis
    print("\n" + "-" * 70)
    print("BUCKET ANALYSIS: Mean |return| by portfolio VOI quintile")
    print("-" * 70)

    df = analysis["df"]
    df["portfolio_voi_quintile"] = pd.qcut(df["portfolio_voi"], 5, labels=False, duplicates="drop")

    bucket_stats = df.groupby("portfolio_voi_quintile").agg({
        "portfolio_voi": "mean",
        "abs_return": ["mean", "count"],
    }).round(4)
    print(bucket_stats.to_string())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Target VOI Experiment")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N targets for testing (reduces cost)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making API calls",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use sequential estimation instead of batch API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if cache exists",
    )
    return parser.parse_args()


async def main():
    """Run the multi-target VOI experiment."""
    args = parse_args()

    print("=" * 70)
    print("MULTI-TARGET VOI EXPERIMENT")
    print("=" * 70)
    print("\nHypothesis: Cruxes with broad portfolio relevance (high avg VOI across")
    print("all targets) predict better than single-target VOI within earnings domain.")

    # Load data
    print("\n[1/4] Loading data...")
    targets, cruxes, returns_df = load_data()
    print(f"      Targets: {len(targets)}")
    print(f"      Cruxes: {len(cruxes)}")
    print(f"      Returns: {len(returns_df)}")

    # Sample if requested
    if args.sample:
        print(f"\n      Sampling {args.sample} targets...")
        np.random.seed(42)  # Reproducible
        sample_indices = np.random.choice(len(targets), min(args.sample, len(targets)), replace=False)
        targets = [targets[i] for i in sample_indices]

        # Filter cruxes to only those with home targets in sample
        target_keys = {t.key for t in targets}
        cruxes = [c for c in cruxes if c.home_key in target_keys]
        returns_df = returns_df[
            returns_df.apply(lambda r: (r["ticker"], str(r["date"])) in target_keys, axis=1)
        ]
        print(f"      Sampled: {len(targets)} targets, {len(cruxes)} cruxes")

    # Cost estimation
    n_pairs = len(cruxes) * (len(targets) - 1)  # Exclude home target
    est_cost = n_pairs * 0.00001  # ~$0.01 per 1000 haiku calls

    print(f"\n      Pairs to estimate: {n_pairs}")
    print(f"      Est. cost (Haiku): ~${est_cost:.2f}")

    if args.dry_run:
        print("\n      [DRY RUN] Stopping before API calls.")
        print("      Run without --dry-run to execute.")
        return None

    # Check for cached results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = f"cross_target_vois{'_sample' + str(args.sample) if args.sample else ''}.json"
    cache_path = OUTPUT_DIR / cache_name

    if cache_path.exists() and not args.force:
        print(f"\n[2/4] Loading cached results from {cache_path}...")
        with open(cache_path) as f:
            portfolio_results = json.load(f)
        print(f"      Loaded {len(portfolio_results)} cached results")
    else:
        # Compute cross-target VOIs
        print("\n[2/4] Computing cross-target VOIs...")

        portfolio_results = await compute_all_cross_target_vois(
            cruxes,
            targets,
            use_batch_api=not args.no_batch,
        )

        # Save intermediate results
        print("\n[3/4] Saving intermediate results...")
        with open(cache_path, "w") as f:
            json.dump(portfolio_results, f, indent=2)
        print(f"      Saved to {cache_path}")

    # Analyze results
    print("\n[4/4] Analyzing results...")
    analysis = analyze_results(portfolio_results, returns_df)

    # Print results
    print_results(analysis)

    # Save final results
    output = {
        "metadata": {
            "experiment": "multi_target_voi",
            "n_targets": len(targets),
            "n_cruxes": len(cruxes),
            "sampled": args.sample,
            "run_at": datetime.now().isoformat(),
        },
        "correlations": {k: v for k, v in analysis["correlations"].items()},
        "summary": {
            "r_single": analysis["correlations"]["single_voi"]["r"],
            "r_portfolio": analysis["correlations"]["portfolio_voi"]["r"],
            "improvement": analysis["correlations"]["portfolio_voi"]["r"] - analysis["correlations"]["single_voi"]["r"],
        },
    }

    results_name = f"multi_target_voi_results{'_sample' + str(args.sample) if args.sample else ''}.json"
    output_path = OUTPUT_DIR / results_name
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved results to {output_path}")

    return analysis


if __name__ == "__main__":
    asyncio.run(main())
