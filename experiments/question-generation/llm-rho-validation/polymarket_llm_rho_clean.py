#!/usr/bin/env python3
"""
LLM ρ estimation on Polymarket flagship pairs with contamination controls.

Measures LLM ρ gap on the SAME pairs that produced flagship r=0.27/0.21 results
(from validate_delta_p.py), with proper contamination filtering.

Key design decisions:
- Uses pairs from validate_delta_p.py (n=4,791 total)
- Filters to pairs with data after MODEL_KNOWLEDGE_CUTOFF (2025-10-01)
- Stratified sample (~150 pairs) for cost control
- Uses market-aware prompt (estimate_rho_market_aware from Phase 4)

Success criteria:
- LLM r > 0.15: Market-aware prompt generalizes beyond curated pairs
- LLM r ≈ 0.10-0.15: ~50% signal loss confirmed on flagship data
- LLM r < 0.10: Market-aware prompt doesn't generalize; need market data
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from scipy import stats
import numpy as np

# Load .env from monorepo root
_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

from llm_forecasting.voi import (
    estimate_rho,
    estimate_rho_market_aware,
    linear_voi_from_rho,
)

# Paths
SCRIPT_DIR = Path(__file__).parent
CONDITIONAL_DIR = SCRIPT_DIR.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = SCRIPT_DIR / "data"

# Import config from conditional-forecasting
import sys
sys.path.insert(0, str(CONDITIONAL_DIR))
from config import MODEL_KNOWLEDGE_CUTOFF

# Model to use for LLM estimation
MODEL = "anthropic/claude-sonnet-4-20250514"

# Time constants
SECONDS_PER_DAY = 86400
DELTA_P_WINDOW_DAYS = 7

# Sample sizes
TARGET_SAMPLE_SIZE = 150
TARGET_WITHIN_CATEGORY = 30
TARGET_CROSS_CATEGORY = 120

# Category patterns (from validate_delta_p.py)
CATEGORIES = {
    "crypto": [
        r"\bBitcoin\b", r"\bBTC\b", r"\bEthereum\b", r"\bETH\b",
        r"\bSolana\b", r"\bSOL\b", r"\bcrypto",
    ],
    "fed_monetary": [
        r"\bFed\b.*\b(rate|interest|bps|meeting|decrease|increase)",
        r"\bFOMC\b", r"\binterest rate",
    ],
    "politics": [
        r"\belection\b", r"\bpresident", r"\bnominate",
        r"\bTrump\b", r"\bBiden\b", r"\bRepublican", r"\bDemocrat",
    ],
    "sports": [
        r"\bwin on 20\d{2}\b", r"\bvs\.?\b", r"\bNFL\b", r"\bNBA\b",
        r"\bMLB\b", r"\bNHL\b", r"\bFC\b", r"Super Bowl", r"World Cup",
        r"Championship", r"Finals", r"Lakers", r"Warriors", r"Chiefs",
        r"Eagles", r"MVP", r"Playoffs", r"Stanley Cup",
    ],
}


class PriceHistory:
    """Price history for a market."""

    def __init__(self, condition_id: str, question: str, candles: list):
        self.condition_id = condition_id
        self.question = question
        self.candles = candles

    @property
    def first_timestamp(self) -> int:
        return self.candles[0]["timestamp"] if self.candles else 0

    @property
    def last_timestamp(self) -> int:
        return self.candles[-1]["timestamp"] if self.candles else 0

    def price_at(self, timestamp: int) -> float | None:
        """Get price at or before timestamp."""
        for candle in reversed(self.candles):
            if candle["timestamp"] <= timestamp:
                return candle["close"]
        return None

    def delta_p(self, start_ts: int, window_seconds: int) -> float | None:
        """Compute absolute price change over window."""
        start_price = self.price_at(start_ts)
        if start_price is None:
            return None

        end_ts = start_ts + window_seconds
        end_price = self.price_at(end_ts)
        if end_price is None:
            if self.last_timestamp > start_ts:
                end_price = self.candles[-1]["close"]
            else:
                return None

        return abs(end_price - start_price)


def categorize_market(question: str) -> str:
    """Categorize a market. Returns 'other' if no match."""
    for cat_name, patterns in CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return cat_name
    return "other"


def load_price_histories() -> dict[str, PriceHistory]:
    """Load all price histories."""
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = PriceHistory(
            data["condition_id"], data["question"], data["candles"]
        )
    return histories


def filter_pairs_post_cutoff(
    pairs: list[dict],
    histories: dict[str, PriceHistory],
    cutoff_ts: int,
) -> list[dict]:
    """Filter pairs to those with data after the knowledge cutoff."""
    filtered = []
    window_seconds = DELTA_P_WINDOW_DAYS * SECONDS_PER_DAY

    for pair in pairs:
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        rho = pair.get("rho")

        # Skip invalid rho
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            continue

        hist_a = histories.get(cond_a)
        hist_b = histories.get(cond_b)

        if not hist_a or not hist_b:
            continue

        # Find overlapping time window after cutoff
        start_ts = max(cutoff_ts, hist_a.first_timestamp, hist_b.first_timestamp)
        end_ts = start_ts + window_seconds

        # Both must have enough data after cutoff
        if hist_a.last_timestamp < end_ts or hist_b.last_timestamp < end_ts:
            continue

        # Get prices at start
        price_a = hist_a.price_at(start_ts)
        price_b = hist_b.price_at(start_ts)

        if price_a is None or price_b is None:
            continue

        # Compute delta_p
        delta_p_a = hist_a.delta_p(start_ts, window_seconds)
        delta_p_b = hist_b.delta_p(start_ts, window_seconds)

        if delta_p_a is None or delta_p_b is None:
            continue

        ground_truth = max(delta_p_a, delta_p_b)

        # Categorize
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        cat_a = categorize_market(q_a)
        cat_b = categorize_market(q_b)
        is_within = (cat_a == cat_b) and (cat_a != "other")

        filtered.append({
            **pair,
            "question_a": q_a,
            "question_b": q_b,
            "price_a": price_a,
            "price_b": price_b,
            "delta_p_a": delta_p_a,
            "delta_p_b": delta_p_b,
            "ground_truth": ground_truth,
            "category_a": cat_a,
            "category_b": cat_b,
            "is_within_category": is_within,
        })

    return filtered


def stratified_sample(pairs: list[dict], seed: int = 42) -> list[dict]:
    """Sample pairs proportionally by category."""
    random.seed(seed)

    # Split by within/cross
    within = [p for p in pairs if p["is_within_category"]]
    cross = [p for p in pairs if not p["is_within_category"]]

    print(f"\n  Available: {len(within)} within-category, {len(cross)} cross-category")

    # Sample within-category by category
    within_by_cat = defaultdict(list)
    for p in within:
        within_by_cat[p["category_a"]].append(p)

    # Target distribution for within-category (~30 total)
    within_targets = {
        "sports": 15,
        "politics": 10,
        "crypto": 3,
        "fed_monetary": 2,
    }

    sampled_within = []
    for cat, target in within_targets.items():
        available = within_by_cat.get(cat, [])
        n = min(target, len(available))
        if n > 0:
            sampled_within.extend(random.sample(available, n))
            print(f"    {cat}: sampled {n}/{len(available)}")

    # Sample cross-category
    n_cross = min(TARGET_CROSS_CATEGORY, len(cross))
    sampled_cross = random.sample(cross, n_cross) if n_cross > 0 else []
    print(f"    cross-category: sampled {n_cross}/{len(cross)}")

    return sampled_within + sampled_cross


async def estimate_llm_rho_for_pairs(pairs: list[dict], use_original: bool = False) -> list[tuple[float, str]]:
    """Estimate ρ using LLM prompt for all pairs."""
    prompt_name = "original single-step" if use_original else "market-aware"
    print(f"\nEstimating ρ for {len(pairs)} pairs using {prompt_name} prompt ({MODEL})...")

    results = []
    for i, pair in enumerate(pairs):
        q_a = pair["question_a"][:50]
        print(f"  [{i+1}/{len(pairs)}] {q_a}...")
        try:
            if use_original:
                rho, reasoning = await estimate_rho(
                    pair["question_a"],
                    pair["question_b"],
                    model=MODEL,
                )
            else:
                rho, reasoning = await estimate_rho_market_aware(
                    pair["question_a"],
                    pair["question_b"],
                    model=MODEL,
                )
            results.append((rho, reasoning))
        except Exception as e:
            print(f"    Error: {e}")
            results.append((0.0, f"Error: {e}"))

    return results


def compute_metrics(pairs: list[dict]) -> dict:
    """Compute correlation metrics."""
    if len(pairs) < 3:
        return {"r": None, "p": None, "n": len(pairs), "msg": "Too few pairs"}

    # Extract arrays
    voi_market = np.array([p["voi_market"] for p in pairs])
    voi_llm = np.array([p["voi_llm"] for p in pairs])
    ground_truth = np.array([p["ground_truth"] for p in pairs])
    rho_market = np.array([p["rho"] for p in pairs])
    rho_llm = np.array([p["llm_rho"] for p in pairs])

    # Filter out any NaN
    mask = ~(np.isnan(voi_market) | np.isnan(voi_llm) | np.isnan(ground_truth))
    voi_market = voi_market[mask]
    voi_llm = voi_llm[mask]
    ground_truth = ground_truth[mask]
    rho_market = rho_market[mask]
    rho_llm = rho_llm[mask]

    if len(voi_market) < 3:
        return {"r": None, "p": None, "n": len(voi_market), "msg": "Too few valid pairs"}

    # Check for zero variance
    if np.std(ground_truth) == 0:
        return {"r": None, "p": None, "n": len(voi_market), "msg": "Zero variance in ground truth"}

    # VOI correlations
    r_market, p_market = stats.pearsonr(voi_market, ground_truth)
    r_llm, p_llm = stats.pearsonr(voi_llm, ground_truth)

    # Rho calibration
    rho_mask = ~np.isnan(rho_market) & ~np.isnan(rho_llm)
    if np.sum(rho_mask) >= 3:
        r_rho, _ = stats.pearsonr(rho_llm[rho_mask], rho_market[rho_mask])
        mae_rho = float(np.mean(np.abs(rho_llm[rho_mask] - rho_market[rho_mask])))
    else:
        r_rho = None
        mae_rho = None

    # Direction accuracy
    direction_correct = np.sum(np.sign(rho_llm[rho_mask]) == np.sign(rho_market[rho_mask]))
    direction_accuracy = float(direction_correct / np.sum(rho_mask)) if np.sum(rho_mask) > 0 else None

    return {
        "n": int(len(voi_market)),
        "ceiling_r": float(r_market),
        "ceiling_p": float(p_market),
        "llm_r": float(r_llm),
        "llm_p": float(p_llm),
        "rho_calibration_r": float(r_rho) if r_rho is not None else None,
        "rho_mae": mae_rho,
        "direction_accuracy": direction_accuracy,
        "voi_market_mean": float(np.mean(voi_market)),
        "voi_llm_mean": float(np.mean(voi_llm)),
        "ground_truth_mean": float(np.mean(ground_truth)),
    }


async def main():
    """Run the clean LLM ρ validation experiment."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true",
                       help="Use original single-step prompt instead of market-aware")
    args = parser.parse_args()

    use_original = args.original
    prompt_name = "original_single_step" if use_original else "market_aware_calibrated"

    print("=" * 70)
    print("LLM ρ VALIDATION ON POLYMARKET FLAGSHIP DATA")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Prompt: {prompt_name}")
    print(f"Knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")

    # Load data
    print("\n[1/6] Loading data...")
    histories = load_price_histories()
    print(f"      Price histories: {len(histories)}")

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Total pairs: {len(pairs)}")

    # Convert cutoff to timestamp
    cutoff_ts = int(datetime.combine(
        MODEL_KNOWLEDGE_CUTOFF,
        datetime.min.time(),
        tzinfo=timezone.utc
    ).timestamp())

    # Filter pairs
    print("\n[2/6] Filtering pairs post-cutoff...")
    clean_pairs = filter_pairs_post_cutoff(pairs, histories, cutoff_ts)
    print(f"      Valid pairs after cutoff: {len(clean_pairs)}")

    # Stratified sample
    print("\n[3/6] Stratified sampling...")
    sample = stratified_sample(clean_pairs)
    print(f"      Sampled: {len(sample)} pairs")

    # Split stats
    within = [p for p in sample if p["is_within_category"]]
    cross = [p for p in sample if not p["is_within_category"]]
    print(f"      Within-category: {len(within)}")
    print(f"      Cross-category: {len(cross)}")

    # LLM ρ estimation
    print("\n[4/6] LLM ρ estimation...")
    llm_results = await estimate_llm_rho_for_pairs(sample, use_original=use_original)

    # Compute VOI for each pair
    print("\n[5/6] Computing VOI metrics...")
    for pair, (llm_rho, reasoning) in zip(sample, llm_results):
        pair["llm_rho"] = llm_rho
        pair["llm_reasoning"] = reasoning

        # Market VOI (ceiling)
        pair["voi_market"] = linear_voi_from_rho(
            pair["rho"], pair["price_a"], pair["price_b"]
        )

        # LLM VOI
        pair["voi_llm"] = linear_voi_from_rho(
            llm_rho, pair["price_a"], pair["price_b"]
        )

    # Compute correlations
    print("\n[6/6] Computing correlations...")

    # Overall
    all_metrics = compute_metrics(sample)

    # By category split
    within_metrics = compute_metrics(within)
    cross_metrics = compute_metrics(cross)

    # By category type
    by_category = {}
    for cat in ["sports", "politics", "crypto", "fed_monetary"]:
        cat_pairs = [p for p in sample if p["category_a"] == cat and p["is_within_category"]]
        if cat_pairs:
            by_category[cat] = compute_metrics(cat_pairs)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def print_metrics(name: str, m: dict):
        if m.get("r") is None or m.get("ceiling_r") is None:
            print(f"\n{name}: {m.get('msg', 'N/A')} (n={m.get('n', 0)})")
            return

        print(f"\n{name} (n={m['n']}):")
        print(f"  Ceiling r(VOI_market, Δp): {m['ceiling_r']:.3f} (p={m['ceiling_p']:.4f})")
        print(f"  LLM r(VOI_llm, Δp):        {m['llm_r']:.3f} (p={m['llm_p']:.4f})")
        print(f"  ρ calibration r:           {m['rho_calibration_r']:.3f}" if m['rho_calibration_r'] else "  ρ calibration r:           N/A")
        print(f"  ρ MAE:                     {m['rho_mae']:.3f}" if m['rho_mae'] else "  ρ MAE:                     N/A")
        print(f"  Direction accuracy:        {m['direction_accuracy']:.1%}" if m['direction_accuracy'] else "  Direction accuracy:        N/A")

    print_metrics("ALL PAIRS", all_metrics)
    print_metrics("WITHIN-CATEGORY", within_metrics)
    print_metrics("CROSS-CATEGORY", cross_metrics)

    print("\n" + "-" * 70)
    print("BY CATEGORY (within-category only)")
    for cat, m in sorted(by_category.items(), key=lambda x: -(x[1].get("n") or 0)):
        print_metrics(f"  {cat}", m)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if all_metrics.get("llm_r") is not None:
        llm_r = all_metrics["llm_r"]
        ceiling_r = all_metrics["ceiling_r"]

        print(f"\nLLM VOI r = {llm_r:.3f}")
        print(f"Ceiling r = {ceiling_r:.3f}")
        print(f"Retained power: {llm_r/ceiling_r:.1%}" if ceiling_r > 0 else "")

        if llm_r > 0.15:
            print("\n✓ GENERALIZES: Market-aware prompt works on flagship data")
        elif llm_r > 0.10:
            print("\n~ MARGINAL: ~50% signal loss confirmed")
        else:
            print("\n✗ DOESN'T GENERALIZE: Need market data for good ρ estimates")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "experiment": "polymarket_llm_rho_clean",
            "model": MODEL,
            "prompt": prompt_name,
            "n_total_pairs": len(pairs),
            "n_post_cutoff": len(clean_pairs),
            "n_sampled": len(sample),
            "n_within_category": len(within),
            "n_cross_category": len(cross),
            "knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "run_at": datetime.now().isoformat(),
        },
        "summary": all_metrics,
        "by_split": {
            "within_category": within_metrics,
            "cross_category": cross_metrics,
        },
        "by_category": by_category,
        "pairs": [
            {
                "question_a": p["question_a"],
                "question_b": p["question_b"],
                "category_a": p["category_a"],
                "category_b": p["category_b"],
                "is_within_category": p["is_within_category"],
                "rho_market": p["rho"],
                "rho_llm": p["llm_rho"],
                "voi_market": p["voi_market"],
                "voi_llm": p["voi_llm"],
                "ground_truth": p["ground_truth"],
                "llm_reasoning": p["llm_reasoning"],
            }
            for p in sample
        ],
    }

    suffix = "_original" if use_original else "_market_aware"
    output_path = OUTPUT_DIR / f"polymarket_llm_rho_clean{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
