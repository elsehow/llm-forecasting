#!/usr/bin/env python3
"""
Co-movement Validation for High-Similarity Pairs.

Instead of requiring resolution, validate VOI by checking if high-similarity
pairs co-move during ANY significant price change.

This increases n from 5 (resolution-only) to potentially 50+ events.

Method:
1. Find significant price moves (>10% in 3 days) in any market
2. For each move, check high-similarity partners
3. Compute whether partner moved in VOI-predicted direction
4. Correlate VOI with actual co-movement magnitude
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

# Import canonical VOI
from llm_forecasting.voi import linear_voi_from_rho

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting" / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
OUTPUT_DIR = Path(__file__).parent / "data"

# Parameters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_PRICE_MOVE = 0.10  # 10% move to count as "significant"
MOVE_WINDOW_DAYS = 3
RESPONSE_WINDOW_DAYS = 5
SIMILARITY_THRESHOLD = 0.7


def load_price_histories() -> dict:
    """Load all price histories."""
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = {
            "question": data["question"],
            "candles": data["candles"],
        }
    return histories


def find_significant_moves(history: dict, min_move: float, window_days: int) -> list[dict]:
    """
    Find all significant price moves in a market's history.

    Returns list of {timestamp, price_before, price_after, direction, magnitude}
    """
    candles = history.get("candles", [])
    if len(candles) < 5:
        return []

    moves = []
    window_seconds = window_days * 86400

    # Slide through candles looking for significant moves
    for i, candle in enumerate(candles):
        # Look back window_days to find price_before
        target_ts = candle["timestamp"] - window_seconds
        price_before = None
        for j in range(i-1, -1, -1):
            if candles[j]["timestamp"] <= target_ts:
                price_before = candles[j]["close"]
                break

        if price_before is None:
            continue

        price_after = candle["close"]
        magnitude = abs(price_after - price_before)

        if magnitude >= min_move:
            # Check this isn't too close to a previous move
            if moves and candle["timestamp"] - moves[-1]["timestamp"] < window_seconds:
                # Update if this move is larger
                if magnitude > moves[-1]["magnitude"]:
                    moves[-1] = {
                        "timestamp": candle["timestamp"],
                        "price_before": price_before,
                        "price_after": price_after,
                        "direction": 1 if price_after > price_before else -1,
                        "magnitude": magnitude,
                    }
            else:
                moves.append({
                    "timestamp": candle["timestamp"],
                    "price_before": price_before,
                    "price_after": price_after,
                    "direction": 1 if price_after > price_before else -1,
                    "magnitude": magnitude,
                })

    return moves


def get_price_response(history: dict, event_ts: int, window_days: int) -> dict | None:
    """
    Get price response in a market after an event timestamp.

    Returns {price_before, price_after, change} or None if no data.
    """
    candles = history.get("candles", [])
    if not candles:
        return None

    window_seconds = window_days * 86400

    # Price at event time (or closest before)
    price_before = None
    for c in reversed(candles):
        if c["timestamp"] <= event_ts:
            price_before = c["close"]
            break

    if price_before is None:
        return None

    # Price after window
    prices_after = []
    for c in candles:
        if event_ts < c["timestamp"] <= event_ts + window_seconds:
            prices_after.append(c["close"])

    if not prices_after:
        # Try last available price if it's after event
        for c in reversed(candles):
            if c["timestamp"] > event_ts:
                prices_after.append(c["close"])
                break

    if not prices_after:
        return None

    price_after = np.mean(prices_after)

    return {
        "price_before": price_before,
        "price_after": price_after,
        "change": price_after - price_before,
    }


def main():
    print("=" * 70)
    print("CO-MOVEMENT VALIDATION FOR HIGH-SIMILARITY PAIRS")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Min price move: {MIN_PRICE_MOVE*100:.0f}%")
    print(f"  Move window: {MOVE_WINDOW_DAYS} days")
    print(f"  Response window: {RESPONSE_WINDOW_DAYS} days")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")

    # Load data
    print("\n[1/5] Loading data...")
    histories = load_price_histories()
    print(f"      Markets: {len(histories)}")

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Pairs: {len(pairs)}")

    # Build condition_id -> question mapping
    cond_to_question = {}
    for h_id, h in histories.items():
        cond_to_question[h_id] = h["question"]

    # Compute embeddings
    print("\n[2/5] Computing embeddings...")
    questions = list(set(cond_to_question.values()))

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    q_to_emb = {q: embeddings_norm[i] for i, q in enumerate(questions)}

    # Build pair lookup with similarity
    print("\n[3/5] Building high-similarity pair index...")

    # Group pairs by condition_id for fast lookup
    pairs_by_cond = defaultdict(list)
    high_sim_count = 0

    for p in pairs:
        cond_a = p["market_a"]["condition_id"]
        cond_b = p["market_b"]["condition_id"]
        q_a = p["market_a"]["question"]
        q_b = p["market_b"]["question"]
        rho = p["rho"]

        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            continue

        if q_a not in q_to_emb or q_b not in q_to_emb:
            continue

        sim = float(q_to_emb[q_a] @ q_to_emb[q_b])

        if sim >= SIMILARITY_THRESHOLD:
            high_sim_count += 1
            pairs_by_cond[cond_a].append({
                "other_cond": cond_b,
                "other_question": q_b,
                "rho": rho,
                "similarity": sim,
            })
            pairs_by_cond[cond_b].append({
                "other_cond": cond_a,
                "other_question": q_a,
                "rho": rho,
                "similarity": sim,
            })

    print(f"      High-similarity pairs: {high_sim_count}")
    print(f"      Markets with high-sim partners: {len(pairs_by_cond)}")

    # Find significant moves and check partner responses
    print("\n[4/5] Finding significant moves and partner responses...")

    events = []
    markets_with_moves = 0

    for cond_id, history in histories.items():
        if cond_id not in pairs_by_cond:
            continue

        moves = find_significant_moves(history, MIN_PRICE_MOVE, MOVE_WINDOW_DAYS)
        if not moves:
            continue

        markets_with_moves += 1

        for move in moves:
            # Check each high-similarity partner
            for partner in pairs_by_cond[cond_id]:
                other_cond = partner["other_cond"]
                other_history = histories.get(other_cond)

                if not other_history:
                    continue

                response = get_price_response(other_history, move["timestamp"], RESPONSE_WINDOW_DAYS)
                if response is None:
                    continue

                # Compute VOI
                voi = linear_voi_from_rho(
                    partner["rho"],
                    response["price_before"],
                    move["price_before"]
                )

                # Predicted direction based on rho sign
                predicted_direction = move["direction"] * np.sign(partner["rho"])
                actual_direction = np.sign(response["change"]) if response["change"] != 0 else 0

                events.append({
                    "mover_cond": cond_id,
                    "mover_question": cond_to_question.get(cond_id, "")[:60],
                    "partner_cond": other_cond,
                    "partner_question": partner["other_question"][:60],
                    "move_timestamp": move["timestamp"],
                    "move_magnitude": move["magnitude"],
                    "move_direction": move["direction"],
                    "rho": partner["rho"],
                    "similarity": partner["similarity"],
                    "voi": float(voi),
                    "partner_change": response["change"],
                    "partner_abs_change": abs(response["change"]),
                    "predicted_direction": predicted_direction,
                    "actual_direction": actual_direction,
                    "direction_correct": int(predicted_direction == actual_direction) if actual_direction != 0 else None,
                })

    print(f"      Markets with significant moves: {markets_with_moves}")
    print(f"      Total move-partner events: {len(events)}")

    # Filter to unique events (avoid double-counting symmetric pairs)
    seen = set()
    unique_events = []
    for e in events:
        key = tuple(sorted([e["mover_cond"], e["partner_cond"]])) + (e["move_timestamp"],)
        if key not in seen:
            seen.add(key)
            unique_events.append(e)

    print(f"      Unique events (after dedup): {len(unique_events)}")

    if len(unique_events) < 10:
        print("\n  Too few events for analysis")
        return

    # Analysis
    print("\n[5/5] Analyzing results...")
    df = pd.DataFrame(unique_events)

    # Remove events where partner didn't move
    df_moved = df[df["actual_direction"] != 0].copy()
    print(f"\n  Events where partner moved: {len(df_moved)}")

    # Correlation: VOI vs partner absolute change
    r_voi, p_voi = stats.pearsonr(df_moved["voi"], df_moved["partner_abs_change"])
    r_rho, p_rho = stats.pearsonr(df_moved["rho"].abs(), df_moved["partner_abs_change"])
    r_sim, p_sim = stats.pearsonr(df_moved["similarity"], df_moved["partner_abs_change"])

    print(f"\n  Correlations with partner |change|:")
    print(f"    VOI:        r={r_voi:+.3f} (p={p_voi:.3f})")
    print(f"    |rho|:      r={r_rho:+.3f} (p={p_rho:.3f})")
    print(f"    similarity: r={r_sim:+.3f} (p={p_sim:.3f})")

    # Direction accuracy
    df_direction = df_moved[df_moved["direction_correct"].notna()]
    if len(df_direction) > 0:
        accuracy = df_direction["direction_correct"].mean()
        print(f"\n  Direction prediction accuracy: {accuracy:.1%} (n={len(df_direction)})")
        print(f"    (baseline = 50%)")

        # Accuracy by VOI quartile
        df_direction["voi_quartile"] = pd.qcut(df_direction["voi"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        acc_by_quartile = df_direction.groupby("voi_quartile")["direction_correct"].mean()
        print(f"\n  Direction accuracy by VOI quartile:")
        for q, acc in acc_by_quartile.items():
            n = (df_direction["voi_quartile"] == q).sum()
            print(f"    {q}: {acc:.1%} (n={n})")

    # Compare to overall Polymarket baseline
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n  This experiment (co-movement, sim >= {SIMILARITY_THRESHOLD}):")
    print(f"    n = {len(df_moved)}")
    print(f"    r(VOI, |change|) = {r_voi:+.3f}")

    print(f"\n  Previous results:")
    print(f"    Resolution-based, sim >= 0.7: n=5, r=+0.95")
    print(f"    Resolution-based, all pairs: n=168, r=+0.43")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if r_voi > 0.3:
        print("\n  STRONG: VOI predicts co-movement magnitude in high-sim pairs")
    elif r_voi > 0.15:
        print("\n  MODERATE: VOI shows some signal for co-movement")
    elif r_voi > 0:
        print("\n  WEAK: VOI shows minimal positive signal")
    else:
        print("\n  NONE: VOI does not predict co-movement in high-sim pairs")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "experiment": "comovement_validation",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "min_price_move": MIN_PRICE_MOVE,
            "move_window_days": MOVE_WINDOW_DAYS,
            "response_window_days": RESPONSE_WINDOW_DAYS,
            "run_at": datetime.now().isoformat(),
        },
        "summary": {
            "n_events": len(df_moved),
            "r_voi_change": float(r_voi),
            "p_voi_change": float(p_voi),
            "r_rho_change": float(r_rho),
            "r_sim_change": float(r_sim),
            "direction_accuracy": float(accuracy) if len(df_direction) > 0 else None,
        },
        "events": unique_events[:100],  # Save sample
    }

    output_path = OUTPUT_DIR / "comovement_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
