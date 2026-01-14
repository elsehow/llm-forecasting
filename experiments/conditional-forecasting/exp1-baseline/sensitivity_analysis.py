#!/usr/bin/env python3
"""Sensitivity analysis: re-run results excluding pairs that resolved before cutoff.

This addresses the potential confound where questions resolved before a model's
knowledge cutoff could be testing memorization rather than reasoning.

Usage:
    uv run python experiments/fb-conditional/sensitivity_analysis.py
    uv run python experiments/fb-conditional/sensitivity_analysis.py --cutoff 2025-06-01
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from llm_forecasting.storage.sqlite import SQLiteStorage


async def get_resolution_dates(storage: SQLiteStorage, pairs: list[dict]) -> dict[str, date | None]:
    """Get resolution dates for all questions in pairs."""
    # Collect all unique question IDs with their sources
    questions_to_lookup = {}
    for p in pairs:
        questions_to_lookup[p["id_a"]] = p["source_a"]
        questions_to_lookup[p["id_b"]] = p["source_b"]

    resolution_dates = {}
    for qid, source in questions_to_lookup.items():
        resolution = await storage.get_resolution(source, qid)
        if resolution:
            resolution_dates[qid] = resolution.date
        else:
            resolution_dates[qid] = None

    return resolution_dates


def filter_pairs_by_cutoff(pairs: list[dict], resolution_dates: dict[str, date | None], cutoff: date) -> list[dict]:
    """Filter pairs where BOTH questions resolved on or after cutoff."""
    filtered = []
    excluded = []

    for p in pairs:
        date_a = resolution_dates.get(p["id_a"])
        date_b = resolution_dates.get(p["id_b"])

        if date_a is None or date_b is None:
            excluded.append((p, "missing resolution date"))
            continue

        if date_a >= cutoff and date_b >= cutoff:
            filtered.append(p)
        else:
            reason = []
            if date_a < cutoff:
                reason.append(f"A resolved {date_a}")
            if date_b < cutoff:
                reason.append(f"B resolved {date_b}")
            excluded.append((p, ", ".join(reason)))

    return filtered, excluded


def compute_stats(results: list[dict], valid_pair_ids: set[str]) -> dict:
    """Compute summary statistics for filtered results."""
    categories = {}

    for r in results:
        if "error" in r:
            continue
        if r["pair_id"] not in valid_pair_ids:
            continue

        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    summary = {}
    for cat, cat_results in sorted(categories.items()):
        n = len(cat_results)
        if n == 0:
            continue
        mean_improvement = sum(r["improvement"] for r in cat_results) / n
        mean_sensitivity = sum(r["sensitivity"] for r in cat_results) / n
        wins = sum(1 for r in cat_results if r["improvement"] > 0)

        summary[cat] = {
            "n": n,
            "mean_improvement": mean_improvement,
            "mean_sensitivity": mean_sensitivity,
            "win_rate": wins / n,
        }

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis with date cutoff")
    parser.add_argument(
        "--db",
        type=str,
        default="data/forecastbench.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="experiments/fb-conditional/pairs.json",
        help="Path to pairs file",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2025-06-01",
        help="Exclude pairs where either question resolved before this date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    cutoff = date.fromisoformat(args.cutoff)

    # Load pairs
    with open(args.pairs) as f:
        pairs_data = json.load(f)
    pairs = pairs_data["pairs"]

    # Get resolution dates from database
    storage = SQLiteStorage(args.db)
    print(f"Looking up resolution dates for {len(pairs)} pairs...")
    resolution_dates = await get_resolution_dates(storage, pairs)
    await storage.close()

    # Filter pairs
    filtered_pairs, excluded = filter_pairs_by_cutoff(pairs, resolution_dates, cutoff)

    print(f"\n{'='*70}")
    print(f"CUTOFF: {cutoff}")
    print(f"{'='*70}")
    print(f"Original pairs: {len(pairs)}")
    print(f"Filtered pairs: {len(filtered_pairs)} (both resolved on/after {cutoff})")
    print(f"Excluded pairs: {len(excluded)}")

    # Show excluded pairs
    if excluded:
        print(f"\nExcluded pairs:")
        for p, reason in excluded:
            print(f"  - [{p['category']}] {p['text_a'][:40]}... x {p['text_b'][:40]}...")
            print(f"    Reason: {reason}")

    # Count filtered by category
    filtered_by_cat = {}
    for p in filtered_pairs:
        cat = p["category"]
        filtered_by_cat[cat] = filtered_by_cat.get(cat, 0) + 1
    print(f"\nFiltered pairs by category: {filtered_by_cat}")

    # Build set of valid pair IDs
    valid_pair_ids = {f"{p['id_a']}_{p['id_b']}" for p in filtered_pairs}

    # Find and analyze result files
    results_dir = Path("experiments/fb-conditional/results")
    result_files = sorted(results_dir.glob("*.json"))

    # Filter to main experiment files (exclude prompt tests)
    main_results = [f for f in result_files if "prompt_test" not in f.name]

    print(f"\n{'='*70}")
    print("COMPARISON: Original vs Filtered")
    print(f"{'='*70}")

    for result_file in main_results:
        with open(result_file) as f:
            data = json.load(f)

        if "results" not in data or "summary" not in data:
            continue

        model = data.get("metadata", {}).get("model", result_file.stem)
        thinking = data.get("metadata", {}).get("thinking", False)

        # Compute filtered stats
        filtered_summary = compute_stats(data["results"], valid_pair_ids)
        original_summary = data["summary"]

        if not filtered_summary:
            continue

        model_label = f"{model}" + (" (thinking)" if thinking else "")
        print(f"\n{model_label}")
        print("-" * 60)
        print(f"{'Category':<10} {'Metric':<20} {'Original':>12} {'Filtered':>12} {'Delta':>10}")
        print("-" * 60)

        for cat in ["strong", "weak", "none"]:
            if cat not in filtered_summary:
                continue
            orig = original_summary.get(cat, {})
            filt = filtered_summary[cat]

            # n
            orig_n = orig.get("n", 0)
            filt_n = filt["n"]
            print(f"{cat:<10} {'n':<20} {orig_n:>12} {filt_n:>12} {filt_n - orig_n:>+10}")

            # mean improvement
            orig_imp = orig.get("mean_improvement", 0)
            filt_imp = filt["mean_improvement"]
            delta_imp = filt_imp - orig_imp
            print(f"{'':<10} {'mean_improvement':<20} {orig_imp:>+12.4f} {filt_imp:>+12.4f} {delta_imp:>+10.4f}")

            # win rate
            orig_wr = orig.get("conditional_win_rate", 0)
            filt_wr = filt["win_rate"]
            delta_wr = filt_wr - orig_wr
            print(f"{'':<10} {'win_rate':<20} {orig_wr:>12.0%} {filt_wr:>12.0%} {delta_wr:>+10.0%}")


if __name__ == "__main__":
    asyncio.run(main())
