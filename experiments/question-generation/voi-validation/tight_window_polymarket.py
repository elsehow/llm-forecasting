"""Tight-window Polymarket conditional validation.

Component 2 of the real-world CivBench anchoring study.

For each recently-resolved market A, find co-active markets B and measure:
1. Market shift: B's price 1hr before vs 1hr after A resolves (5m granularity)
2. LLM anchoring: ask models for P(B) and P(B|A resolved), compare shift to market shift

Pipeline phases:
  Phase 1: Fetch resolved markets, detect resolution timestamps
  Phase 2: For each resolution event, find co-active markets with price data
  Phase 3: Compute tight-window market shifts, filter by proportional threshold
  Phase 4: LLM evaluation (unconditional + conditional)
  Phase 5: Anchoring analysis (sensitivity ratio, direction, magnitude)

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/question-generation/voi-validation/tight_window_polymarket.py --phase 1
    uv run python experiments/question-generation/voi-validation/tight_window_polymarket.py --phase 2
    uv run python experiments/question-generation/voi-validation/tight_window_polymarket.py --phase 3
    uv run python experiments/question-generation/voi-validation/tight_window_polymarket.py --phase 4
    uv run python experiments/question-generation/voi-validation/tight_window_polymarket.py --phase 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

_monorepo_root = Path(__file__).resolve().parents[4]
load_dotenv(_monorepo_root / ".env")

import numpy as np

from llm_forecasting.market_data.polymarket import GAMMA_API_URL, PolymarketData

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "tight_window"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training cutoff — only use markets resolved after this to avoid contamination
TRAINING_CUTOFF = datetime(2025, 10, 1, tzinfo=timezone.utc)

# Window parameters
WINDOW_HOURS = 1  # hours before/after resolution to measure
PRICE_INTERVAL = "1h"  # granularity for tight window

# Proportional threshold: |shift| > THRESHOLD_FRAC * min(p, 1-p)
THRESHOLD_FRAC = 0.10

# Models for LLM evaluation
MODELS = [
    "o3-2025-04-16",
    "claude-opus-4-5-20251101",
    "gpt-4.1-2025-04-14",
    "claude-sonnet-4-20250514",
]


# ============================================================
# Phase 1: Fetch resolved markets and detect resolution timestamps
# ============================================================

async def phase1_fetch_resolved():
    """Fetch recently-resolved markets and detect their resolution timestamps."""
    print("\n=== Phase 1: Fetching resolved markets ===\n")

    provider = PolymarketData()

    # Fetch closed/resolved markets from gamma API
    resolved_markets = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        offset = 0
        batch_size = 100

        while len(resolved_markets) < 1000:
            params = {
                "limit": batch_size,
                "offset": offset,
                "closed": "true",
                "order": "volume24hr",
                "ascending": "false",
            }

            response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            for raw in batch:
                market = provider._parse_market(raw)
                if (market
                    and market.resolved_value is not None
                    and market.clob_token_ids):
                    resolved_markets.append({
                        "condition_id": market.id,
                        "title": market.title,
                        "resolved_value": market.resolved_value,
                        "resolution_date": market.resolution_date.isoformat() if market.resolution_date else None,
                        "clob_token_id": market.clob_token_ids[0],
                        "description": market.description or "",
                    })

            print(f"  Fetched {offset + len(batch)} markets, {len(resolved_markets)} resolved...")

            if len(batch) < batch_size:
                break
            offset += batch_size
            await asyncio.sleep(0.2)

    print(f"\n  Total resolved markets: {len(resolved_markets)}")

    # Detect resolution timestamps from price history
    print("\n  Detecting resolution timestamps from price history...")

    for i, market in enumerate(resolved_markets):
        if i % 20 == 0:
            print(f"    Processing {i}/{len(resolved_markets)}...")

        try:
            # Fetch 1h price history around the resolution date
            res_date = market.get("resolution_date")
            if res_date:
                end = datetime.fromisoformat(res_date).replace(tzinfo=timezone.utc) + timedelta(days=2)
                start = end - timedelta(days=7)
            else:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=30)

            history = await provider.fetch_price_history_by_token(
                market["clob_token_id"],
                start=start,
                end=end,
                interval="1h",
            )

            if not history:
                market["resolution_ts"] = None
                continue

            # Find resolution timestamp: when price first hits >= 0.95 or <= 0.05
            threshold = 0.95 if market["resolved_value"] > 0.5 else 0.05
            res_ts = None
            for point in history:
                if market["resolved_value"] > 0.5 and point.price >= threshold:
                    res_ts = point.timestamp.isoformat()
                    break
                elif market["resolved_value"] <= 0.5 and point.price <= threshold:
                    res_ts = point.timestamp.isoformat()
                    break

            market["resolution_ts"] = res_ts

        except Exception as e:
            market["resolution_ts"] = None

        await asyncio.sleep(0.1)

    # Filter to markets with detected resolution timestamps after training cutoff
    with_ts = [m for m in resolved_markets if m["resolution_ts"]]
    post_cutoff = [m for m in with_ts
                   if datetime.fromisoformat(m["resolution_ts"]) > TRAINING_CUTOFF]

    print(f"\n  Markets with detected timestamps: {len(with_ts)}")
    print(f"  After training cutoff ({TRAINING_CUTOFF.date()}): {len(post_cutoff)}")

    out_path = OUTPUT_DIR / "phase1_resolved_markets.json"
    with open(out_path, "w") as f:
        json.dump({"markets": post_cutoff, "total_fetched": len(resolved_markets)}, f, indent=2)
    print(f"  Saved to {out_path}")


# ============================================================
# Phase 2: Find co-active markets with tight-window price data
# ============================================================

async def phase2_find_pairs():
    """For each resolved market, find co-active markets with meaningful price shifts."""
    print("\n=== Phase 2: Finding co-active market pairs ===\n")

    with open(OUTPUT_DIR / "phase1_resolved_markets.json") as f:
        data = json.load(f)
    resolved = data["markets"]
    print(f"  Loaded {len(resolved)} resolved markets")

    provider = PolymarketData()

    # Fetch pool of candidate B markets (active + recently closed)
    print("  Fetching candidate B markets (active markets)...")
    b_candidates = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        offset = 0
        batch_size = 100
        while len(b_candidates) < 500:
            params = {
                "limit": batch_size,
                "offset": offset,
                "active": "true",
                "order": "volume24hr",
                "ascending": "false",
            }
            response = await client.get(f"{GAMMA_API_URL}/markets", params=params)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for raw in batch:
                market = provider._parse_market(raw)
                if market and market.clob_token_ids and market.current_probability:
                    b_candidates.append({
                        "condition_id": market.id,
                        "title": market.title,
                        "current_prob": market.current_probability,
                        "clob_token_id": market.clob_token_ids[0],
                    })
            if len(batch) < batch_size:
                break
            offset += batch_size
            await asyncio.sleep(0.2)

    print(f"  Found {len(b_candidates)} candidate B markets")

    # For each resolved A, check B's price around A's resolution time
    pairs = []
    checked = 0

    for a_idx, a_market in enumerate(resolved[:100]):  # Limit to top 100 by volume
        res_ts = datetime.fromisoformat(a_market["resolution_ts"])
        window_start = res_ts - timedelta(hours=WINDOW_HOURS)
        window_end = res_ts + timedelta(hours=WINDOW_HOURS)

        if a_idx % 10 == 0:
            print(f"  Checking resolved market {a_idx}/{min(len(resolved), 100)}: {a_market['title'][:60]}...")

        for b_market in b_candidates:
            if b_market["condition_id"] == a_market["condition_id"]:
                continue

            checked += 1
            try:
                history = await provider.fetch_price_history_by_token(
                    b_market["clob_token_id"],
                    start=window_start - timedelta(hours=3),
                    end=window_end + timedelta(hours=3),
                    interval=PRICE_INTERVAL,
                )

                if len(history) < 4:
                    continue

                # Find prices closest to T-1hr and T+1hr
                before_points = [p for p in history if p.timestamp <= res_ts]
                after_points = [p for p in history if p.timestamp >= res_ts]

                if not before_points or not after_points:
                    continue

                p_before = before_points[-1].price  # latest before resolution
                p_after = after_points[0].price      # earliest after resolution

                market_shift = p_after - p_before

                # Proportional threshold
                uncertainty = min(p_before, 1 - p_before)
                if uncertainty < 0.05:
                    continue  # B is too certain, skip
                threshold = THRESHOLD_FRAC * uncertainty

                if abs(market_shift) < threshold:
                    continue

                pairs.append({
                    "a_condition_id": a_market["condition_id"],
                    "a_title": a_market["title"],
                    "a_resolved_value": a_market["resolved_value"],
                    "a_resolution_ts": a_market["resolution_ts"],
                    "b_condition_id": b_market["condition_id"],
                    "b_title": b_market["title"],
                    "b_clob_token_id": b_market["clob_token_id"],
                    "b_price_before": p_before,
                    "b_price_after": p_after,
                    "market_shift": market_shift,
                    "b_uncertainty": uncertainty,
                    "shift_over_threshold": abs(market_shift) / threshold,
                })

                if len(pairs) % 5 == 0:
                    print(f"    Found {len(pairs)} pairs so far (checked {checked} combinations)...")

            except Exception:
                pass

            await asyncio.sleep(0.05)  # Rate limit

        if len(pairs) >= 80:
            print(f"  Reached {len(pairs)} pairs, stopping early")
            break

    # Sort by shift magnitude (most informative first)
    pairs.sort(key=lambda p: abs(p["market_shift"]), reverse=True)

    print(f"\n  Total pairs found: {len(pairs)} (from {checked} checked)")
    if pairs:
        shifts = [abs(p["market_shift"]) for p in pairs]
        print(f"  Market shift distribution:")
        print(f"    Mean: {np.mean(shifts):.4f}")
        print(f"    Median: {np.median(shifts):.4f}")
        print(f"    Max: {max(shifts):.4f}")

    out_path = OUTPUT_DIR / "phase2_pairs.json"
    with open(out_path, "w") as f:
        json.dump({"pairs": pairs, "total_checked": checked}, f, indent=2)
    print(f"  Saved to {out_path}")


# ============================================================
# Phase 3: LLM evaluation
# ============================================================

async def phase3_llm_eval():
    """Run unconditional + conditional prompts on selected pairs."""
    print("\n=== Phase 3: LLM evaluation ===\n")

    with open(OUTPUT_DIR / "phase2_pairs.json") as f:
        data = json.load(f)
    pairs = data["pairs"][:50]  # Cap at 50 pairs
    print(f"  Evaluating {len(pairs)} pairs across {len(MODELS)} models")

    import anthropic
    import openai

    anthropic_client = anthropic.Anthropic()
    openai_client = openai.OpenAI()

    results = []

    for pair_idx, pair in enumerate(pairs):
        a_resolved = "YES" if pair["a_resolved_value"] > 0.5 else "NO"

        unconditional_prompt = (
            f"What is the probability that the following will resolve YES?\n\n"
            f"\"{pair['b_title']}\"\n\n"
            f"Respond with ONLY a number between 0.0 and 1.0, nothing else."
        )

        conditional_prompt = (
            f"The prediction market question \"{pair['a_title']}\" has just resolved {a_resolved}.\n\n"
            f"Given this information, what is the probability that the following will resolve YES?\n\n"
            f"\"{pair['b_title']}\"\n\n"
            f"Respond with ONLY a number between 0.0 and 1.0, nothing else."
        )

        pair_results = {
            "pair_idx": pair_idx,
            "a_title": pair["a_title"],
            "b_title": pair["b_title"],
            "a_resolved": a_resolved,
            "b_price_before": pair["b_price_before"],
            "b_price_after": pair["b_price_after"],
            "market_shift": pair["market_shift"],
            "model_results": {},
        }

        for model in MODELS:
            try:
                if "claude" in model:
                    # Anthropic API
                    uncond_resp = anthropic_client.messages.create(
                        model=model,
                        max_tokens=20,
                        messages=[{"role": "user", "content": unconditional_prompt}],
                    )
                    p_uncond = _parse_prob(uncond_resp.content[0].text)

                    cond_resp = anthropic_client.messages.create(
                        model=model,
                        max_tokens=20,
                        messages=[{"role": "user", "content": conditional_prompt}],
                    )
                    p_cond = _parse_prob(cond_resp.content[0].text)
                else:
                    # OpenAI API — o3 uses max_completion_tokens
                    oai_kwargs = {"max_completion_tokens": 20} if "o3" in model else {"max_tokens": 20}
                    uncond_resp = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": unconditional_prompt}],
                        **oai_kwargs,
                    )
                    p_uncond = _parse_prob(uncond_resp.choices[0].message.content)

                    cond_resp = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": conditional_prompt}],
                        **oai_kwargs,
                    )
                    p_cond = _parse_prob(cond_resp.choices[0].message.content)

                llm_shift = p_cond - p_uncond

                pair_results["model_results"][model] = {
                    "p_unconditional": p_uncond,
                    "p_conditional": p_cond,
                    "llm_shift": llm_shift,
                    "spread": abs(llm_shift),
                }

            except Exception as e:
                print(f"    Error with {model} on pair {pair_idx}: {e}")
                pair_results["model_results"][model] = {"error": str(e)}

            time.sleep(0.5)  # Rate limit between calls

        results.append(pair_results)

        if pair_idx % 5 == 0:
            print(f"  Completed pair {pair_idx}/{len(pairs)}")
            # Incremental save
            out_path = OUTPUT_DIR / "phase3_llm_results.json"
            with open(out_path, "w") as f:
                json.dump({"results": results, "models": MODELS}, f, indent=2)

    out_path = OUTPUT_DIR / "phase3_llm_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "models": MODELS}, f, indent=2)
    print(f"\n  Saved {len(results)} results to {out_path}")


def _parse_prob(text: str) -> float:
    """Extract a probability from LLM response text."""
    text = text.strip()
    # Try to extract a number
    for token in text.split():
        token = token.strip(".,;:()[]{}\"'")
        try:
            val = float(token)
            if 0 <= val <= 1:
                return val
        except ValueError:
            continue
    return 0.5  # fallback


# ============================================================
# Phase 4: Anchoring analysis
# ============================================================

def phase4_analysis():
    """Compute anchoring metrics comparing LLM shifts to market shifts."""
    print("\n=== Phase 4: Anchoring analysis ===\n")

    with open(OUTPUT_DIR / "phase3_llm_results.json") as f:
        data = json.load(f)
    results = data["results"]
    models = data["models"]

    print(f"  Loaded {len(results)} pairs, {len(models)} models\n")

    # Per-model analysis
    for model in models:
        spreads = []
        sensitivity_ratios = []
        directions_correct = []
        magnitude_captures = []

        for r in results:
            mr = r["model_results"].get(model, {})
            if "error" in mr or "llm_shift" not in mr:
                continue

            market_shift = r["market_shift"]
            llm_shift = mr["llm_shift"]
            spread = mr["spread"]
            spreads.append(spread)

            if abs(market_shift) > 0.01:
                ratio = spread / abs(market_shift)
                sensitivity_ratios.append(ratio)

                # Direction
                correct = np.sign(llm_shift) == np.sign(market_shift)
                directions_correct.append(correct)

                # Magnitude (correct direction only)
                if correct and abs(market_shift) > 0:
                    mag = abs(llm_shift) / abs(market_shift)
                    magnitude_captures.append(min(mag, 3.0))

        if not spreads:
            print(f"  [{model}]: no valid results")
            continue

        print(f"  {model}:")
        print(f"    N pairs: {len(spreads)}")
        print(f"    Spread: mean={np.mean(spreads):.4f}, median={np.median(spreads):.4f}")
        print(f"    % near-zero (<0.05): {100*np.mean([s < 0.05 for s in spreads]):.0f}%")
        if sensitivity_ratios:
            print(f"    Sensitivity ratio: mean={np.mean(sensitivity_ratios):.3f}, median={np.median(sensitivity_ratios):.3f}")
        if directions_correct:
            print(f"    Direction accuracy: {100*np.mean(directions_correct):.1f}%")
        if magnitude_captures:
            print(f"    Magnitude capture: {100*np.mean(magnitude_captures):.1f}% (correct-direction only)")
        print()

    # Aggregate across models
    print("  --- Aggregate (all models) ---")
    all_spreads = []
    all_ratios = []
    all_dirs = []
    all_mags = []

    for r in results:
        for model in models:
            mr = r["model_results"].get(model, {})
            if "error" in mr or "llm_shift" not in mr:
                continue

            spread = mr["spread"]
            all_spreads.append(spread)

            ms = r["market_shift"]
            ls = mr["llm_shift"]
            if abs(ms) > 0.01:
                all_ratios.append(spread / abs(ms))
                correct = np.sign(ls) == np.sign(ms)
                all_dirs.append(correct)
                if correct and abs(ms) > 0:
                    all_mags.append(min(abs(ls) / abs(ms), 3.0))

    if all_spreads:
        print(f"    Total observations: {len(all_spreads)}")
        print(f"    Spread: mean={np.mean(all_spreads):.4f}")
        print(f"    % near-zero (<0.05): {100*np.mean([s < 0.05 for s in all_spreads]):.0f}%")
    if all_ratios:
        print(f"    Sensitivity ratio: mean={np.mean(all_ratios):.3f}, median={np.median(all_ratios):.3f}")
        print(f"    (CivBench reference: 0.16-0.19)")
    if all_dirs:
        print(f"    Direction accuracy: {100*np.mean(all_dirs):.1f}%")
        print(f"    (CivBench reference: 61%)")
    if all_mags:
        print(f"    Magnitude capture: {100*np.mean(all_mags):.1f}%")
        print(f"    (CivBench reference: 2-3%)")

    # Save
    out_path = OUTPUT_DIR / "phase4_analysis.json"
    analysis = {
        "n_pairs": len(results),
        "models": models,
        "aggregate": {
            "n_observations": len(all_spreads),
            "mean_spread": float(np.mean(all_spreads)) if all_spreads else None,
            "pct_near_zero": float(np.mean([s < 0.05 for s in all_spreads])) if all_spreads else None,
            "mean_sensitivity_ratio": float(np.mean(all_ratios)) if all_ratios else None,
            "median_sensitivity_ratio": float(np.median(all_ratios)) if all_ratios else None,
            "direction_accuracy": float(np.mean(all_dirs)) if all_dirs else None,
            "magnitude_capture": float(np.mean(all_mags)) if all_mags else None,
        },
        "civbench_reference": {
            "signal_capture": "16-19%",
            "direction_accuracy": "61%",
            "magnitude_capture": "2-3%",
        },
    }
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved to {out_path}")


# ============================================================
# Main
# ============================================================

async def async_main(phase: int):
    if phase == 1:
        await phase1_fetch_resolved()
    elif phase == 2:
        await phase2_find_pairs()
    elif phase == 3:
        await phase3_llm_eval()
    elif phase == 4:
        phase4_analysis()
    elif phase == 0:
        # Run all phases
        await phase1_fetch_resolved()
        await phase2_find_pairs()
        await phase3_llm_eval()
        phase4_analysis()


def main():
    parser = argparse.ArgumentParser(description="Tight-window Polymarket conditional validation")
    parser.add_argument("--phase", type=int, default=0,
                        help="Phase to run (1-4, or 0 for all)")
    args = parser.parse_args()
    asyncio.run(async_main(args.phase))


if __name__ == "__main__":
    main()
