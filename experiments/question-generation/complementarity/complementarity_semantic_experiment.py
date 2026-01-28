#!/usr/bin/env python3
"""
Complementarity Scoring with Semantic Within-Domain Definition.

Extends complementarity_experiment.py by defining "within-domain" using
semantic similarity thresholds instead of regex categories.

This should increase n from 11 to ~50-100 depending on threshold.

Key insight: Instead of discrete categories (crypto, fed, politics),
we use embedding cosine similarity > threshold as "within-domain".
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent / "data"

# Model for semantic similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Resolution thresholds
RESOLUTION_THRESHOLD_HIGH = 0.95
RESOLUTION_THRESHOLD_LOW = 0.05

# Windows
WINDOW_BEFORE_DAYS = 3
WINDOW_AFTER_DAYS = 3
SECONDS_PER_DAY = 86400


def get_embedder():
    """Load sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


class PriceHistory:
    def __init__(self, condition_id: str, question: str, candles: list):
        self.condition_id = condition_id
        self.question = question
        self.candles = candles

    @property
    def last_price(self) -> float:
        return self.candles[-1]["close"] if self.candles else 0.5

    @property
    def last_timestamp(self) -> int:
        return self.candles[-1]["timestamp"] if self.candles else 0

    def price_at(self, timestamp: int) -> float | None:
        for candle in reversed(self.candles):
            if candle["timestamp"] <= timestamp:
                return candle["close"]
        return None

    def price_after(self, timestamp: int, window_seconds: int) -> float | None:
        prices = []
        for candle in self.candles:
            if timestamp < candle["timestamp"] <= timestamp + window_seconds:
                prices.append(candle["close"])
        return np.mean(prices) if prices else None


def load_price_histories() -> dict[str, PriceHistory]:
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = PriceHistory(
            data["condition_id"], data["question"], data["candles"]
        )
    return histories


def detect_resolution(history: PriceHistory):
    """Detect if and when a market resolved."""
    if len(history.candles) < 5:
        return None

    final_price = history.last_price
    if not (final_price < RESOLUTION_THRESHOLD_LOW or final_price > RESOLUTION_THRESHOLD_HIGH):
        return None

    outcome = "YES" if final_price > RESOLUTION_THRESHOLD_HIGH else "NO"
    threshold = RESOLUTION_THRESHOLD_HIGH if outcome == "YES" else RESOLUTION_THRESHOLD_LOW

    for i, candle in enumerate(history.candles):
        if (outcome == "YES" and candle["close"] >= threshold) or \
           (outcome == "NO" and candle["close"] <= threshold):
            if i < 3:
                return None
            before_prices = [c["close"] for c in history.candles[max(0, i-3):i]]
            price_before = np.mean(before_prices) if before_prices else 0.5
            if abs(final_price - price_before) < 0.10:
                return None
            return {
                "timestamp": candle["timestamp"],
                "outcome": outcome,
                "price_before": price_before,
            }
    return None


def compute_all_embeddings(questions: list[str], embedder) -> np.ndarray:
    """Compute embeddings for all questions."""
    embeddings = embedder.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def run_semantic_within_experiment(
    similarity_threshold: float,
    embedder,
    histories: dict[str, PriceHistory],
    resolutions: dict,
    pairs: list[dict],
    question_embeddings: dict[str, np.ndarray],
) -> dict:
    """
    Run complementarity experiment with semantic within-domain definition.

    Args:
        similarity_threshold: Pairs with cosine sim > threshold are "within-domain"
    """
    print(f"\n{'='*70}")
    print(f"SEMANTIC WITHIN-DOMAIN (threshold={similarity_threshold})")
    print("="*70)

    results = []

    for pair in pairs:
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        rho = pair["rho"]

        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            continue

        res_a = resolutions.get(cond_a)
        res_b = resolutions.get(cond_b)

        if not res_a and not res_b:
            continue

        # Use whichever resolved
        if res_a:
            resolution = res_a
            resolved_cond = cond_a
            resolved_q = pair["market_a"]["question"]
            other_cond = cond_b
            other_q = pair["market_b"]["question"]
        else:
            resolution = res_b
            resolved_cond = cond_b
            resolved_q = pair["market_b"]["question"]
            other_cond = cond_a
            other_q = pair["market_a"]["question"]

        other_history = histories.get(other_cond)
        if not other_history:
            continue

        resolution_ts = resolution["timestamp"]

        # Get prices for other market
        price_before = other_history.price_at(resolution_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY)
        if price_before is None:
            price_before = other_history.price_at(resolution_ts)
        if price_before is None:
            continue

        price_after = other_history.price_after(resolution_ts, WINDOW_AFTER_DAYS * SECONDS_PER_DAY)
        if price_after is None:
            if other_history.last_timestamp > resolution_ts:
                price_after = other_history.last_price
            else:
                continue

        # Compute VOI and actual shift
        linear_voi = linear_voi_from_rho(rho, price_before, resolution["price_before"])
        actual_shift = abs(price_after - price_before)

        # Compute semantic similarity
        emb_resolved = question_embeddings.get(resolved_q)
        emb_other = question_embeddings.get(other_q)

        if emb_resolved is None or emb_other is None:
            continue

        semantic_sim = float(emb_resolved @ emb_other)

        # Determine if within-domain
        is_within_domain = semantic_sim >= similarity_threshold

        # Complementarity score: VOI penalized by similarity
        complementarity = linear_voi - 0.5 * semantic_sim

        results.append({
            "resolved_question": resolved_q[:80],
            "other_question": other_q[:80],
            "rho": rho,
            "linear_voi": float(linear_voi),
            "actual_shift": actual_shift,
            "semantic_sim": semantic_sim,
            "is_within_domain": is_within_domain,
            "complementarity": complementarity,
        })

    # Split by within/cross domain
    within = [r for r in results if r["is_within_domain"]]
    cross = [r for r in results if not r["is_within_domain"]]

    print(f"\n  Total pairs: {len(results)}")
    print(f"  Within-domain (sim >= {similarity_threshold}): {len(within)}")
    print(f"  Cross-domain (sim < {similarity_threshold}): {len(cross)}")

    # Compute correlations
    summary = {"threshold": similarity_threshold}

    def analyze_group(name: str, group: list) -> dict | None:
        if len(group) < 5:
            print(f"\n  {name}: Too few pairs ({len(group)})")
            return None

        df = pd.DataFrame(group)
        r_voi, p_voi = stats.pearsonr(df["linear_voi"], df["actual_shift"])
        r_comp, p_comp = stats.pearsonr(df["complementarity"], df["actual_shift"])
        r_sim, p_sim = stats.pearsonr(df["semantic_sim"], df["actual_shift"])

        print(f"\n  {name} (n={len(df)}):")
        print(f"    VOI vs shift:           r={r_voi:+.3f} (p={p_voi:.3f})")
        print(f"    Complementarity vs shift: r={r_comp:+.3f} (p={p_comp:.3f})")
        print(f"    Similarity vs shift:    r={r_sim:+.3f} (p={p_sim:.3f})")

        return {
            "n": len(df),
            "voi_vs_shift": {"r": float(r_voi), "p": float(p_voi)},
            "complementarity_vs_shift": {"r": float(r_comp), "p": float(p_comp)},
            "similarity_vs_shift": {"r": float(r_sim), "p": float(p_sim)},
            "mean_voi": float(df["linear_voi"].mean()),
            "mean_shift": float(df["actual_shift"].mean()),
            "mean_sim": float(df["semantic_sim"].mean()),
        }

    summary["all"] = analyze_group("ALL", results)
    summary["within_domain"] = analyze_group("WITHIN-DOMAIN", within)
    summary["cross_domain"] = analyze_group("CROSS-DOMAIN", cross)

    # Verdict
    if summary["within_domain"]:
        r_voi_within = summary["within_domain"]["voi_vs_shift"]["r"]
        r_comp_within = summary["within_domain"]["complementarity_vs_shift"]["r"]

        print(f"\n  Within-domain comparison:")
        print(f"    VOI:           r={r_voi_within:+.3f}")
        print(f"    Complementarity: r={r_comp_within:+.3f}")
        print(f"    Delta:         {r_comp_within - r_voi_within:+.3f}")

        if r_comp_within > r_voi_within + 0.05 and r_comp_within > 0:
            verdict = "SUCCESS: Complementarity improves within-domain ranking"
        elif r_comp_within > 0 and r_voi_within <= 0:
            verdict = "SUCCESS: Complementarity reverses negative correlation"
        elif abs(r_comp_within - r_voi_within) < 0.05:
            verdict = "NO EFFECT: Complementarity doesn't change ranking"
        else:
            verdict = "FAIL: Complementarity hurts ranking"

        print(f"\n  Verdict: {verdict}")
        summary["verdict"] = verdict

    return {"summary": summary, "pairs": results}


def main():
    print("=" * 70)
    print("COMPLEMENTARITY WITH SEMANTIC WITHIN-DOMAIN DEFINITION")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")
    print("\nHypothesis: Using semantic similarity threshold instead of")
    print("regex categories will increase n and reveal complementarity signal.")

    # Load embedder
    print("\n[1/5] Loading embedding model...")
    embedder = get_embedder()

    # Load price histories
    print("\n[2/5] Loading price histories...")
    histories = load_price_histories()
    print(f"      Loaded {len(histories)} histories")

    # Detect resolutions
    print("\n[3/5] Detecting resolutions...")
    resolutions = {}
    for cond_id, history in histories.items():
        res = detect_resolution(history)
        if res:
            res_date = datetime.fromtimestamp(res["timestamp"], tz=timezone.utc)
            if res_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                resolutions[cond_id] = res
    print(f"      Found {len(resolutions)} resolved markets")

    # Load pairs
    print("\n[4/5] Loading pairs and computing embeddings...")
    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Loaded {len(pairs)} pairs")

    # Get all unique questions
    all_questions = set()
    for p in pairs:
        all_questions.add(p["market_a"]["question"])
        all_questions.add(p["market_b"]["question"])
    question_list = list(all_questions)
    print(f"      Unique questions: {len(question_list)}")

    # Compute embeddings
    embeddings = compute_all_embeddings(question_list, embedder)
    question_embeddings = {q: embeddings[i] for i, q in enumerate(question_list)}

    # Test multiple thresholds
    print("\n[5/5] Running experiments...")
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    all_results = {}

    for threshold in thresholds:
        result = run_semantic_within_experiment(
            threshold, embedder, histories, resolutions,
            pairs, question_embeddings
        )
        all_results[str(threshold)] = result["summary"]

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS THRESHOLDS")
    print("=" * 70)

    print("\n| Threshold | n_within | r(VOI) | r(Comp) | Delta |")
    print("|-----------|----------|--------|---------|-------|")

    for threshold in thresholds:
        summary = all_results[str(threshold)]
        within = summary.get("within_domain")
        if within:
            r_voi = within["voi_vs_shift"]["r"]
            r_comp = within["complementarity_vs_shift"]["r"]
            delta = r_comp - r_voi
            print(f"| {threshold:.1f} | {within['n']:>8} | {r_voi:+.3f}  | {r_comp:+.3f}   | {delta:+.3f} |")
        else:
            print(f"| {threshold:.1f} | <5 | - | - | - |")

    # Best threshold
    best_threshold = None
    best_delta = -float("inf")
    for threshold in thresholds:
        summary = all_results[str(threshold)]
        within = summary.get("within_domain")
        if within and within["n"] >= 10:
            r_voi = within["voi_vs_shift"]["r"]
            r_comp = within["complementarity_vs_shift"]["r"]
            delta = r_comp - r_voi
            if delta > best_delta:
                best_delta = delta
                best_threshold = threshold

    if best_threshold:
        print(f"\n  Best threshold: {best_threshold} (delta = {best_delta:+.3f})")
        print(f"  n = {all_results[str(best_threshold)]['within_domain']['n']}")

    # Overall interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check if any threshold shows improvement
    any_improvement = any(
        all_results[str(t)].get("within_domain", {}).get("complementarity_vs_shift", {}).get("r", -1)
        > all_results[str(t)].get("within_domain", {}).get("voi_vs_shift", {}).get("r", 0)
        for t in thresholds
        if all_results[str(t)].get("within_domain")
    )

    if any_improvement:
        print("\nCOMPLEMENTARITY SHOWS IMPROVEMENT at some thresholds.")
        print("Penalizing semantic similarity helps within-domain ranking.")
    else:
        print("\nCOMPLEMENTARITY DOES NOT HELP across all thresholds.")
        print("The within-domain VOI problem is fundamental, not fixable by diversity.")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "experiment": "complementarity_semantic_within",
            "embedding_model": EMBEDDING_MODEL,
            "thresholds": thresholds,
            "run_at": datetime.now().isoformat(),
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
        },
        "results_by_threshold": all_results,
        "best_threshold": best_threshold,
        "best_delta": float(best_delta) if best_delta != -float("inf") else None,
    }

    output_path = OUTPUT_DIR / "complementarity_semantic_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
