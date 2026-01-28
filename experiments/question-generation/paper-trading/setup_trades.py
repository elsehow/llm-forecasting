#!/usr/bin/env python3
"""
Set up paper trades from Phase 1 generated cruxes.

For each high-VOI crux:
1. Record current ultimate price (from market data)
2. Forecast crux outcome
3. Predict ultimate price movement
4. Create trade record

Usage:
    uv run python experiments/question-generation/paper-trading/setup_trades.py
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
import litellm

load_dotenv()

PAPER_DIR = Path(__file__).parent
BENCHMARK_DIR = PAPER_DIR.parent / "benchmark-mvp"
CONDITIONAL_DIR = PAPER_DIR.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"

MODEL = "anthropic/claude-sonnet-4-20250514"
VOI_THRESHOLD = 0.15  # Only trade high-VOI cruxes


def load_data():
    """Load benchmark results and market data."""
    with open(BENCHMARK_DIR / "results" / "benchmark_results.json") as f:
        benchmark = json.load(f)

    with open(DATA_DIR / "markets.json") as f:
        markets = json.load(f)

    # Get latest prices from price history
    prices = {}
    price_dir = DATA_DIR / "price_history"
    for path in price_dir.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        cond_id = data["condition_id"]
        candles = data.get("candles", [])
        if candles:
            prices[cond_id] = candles[-1]["close"]

    return benchmark, markets, prices


FORECAST_PROMPT = """You are a superforecaster. Estimate the probability that this question resolves YES.

Question: {question}

Consider:
- Current date: {date}
- Base rates for similar events
- Recent news and trends
- Resolution criteria

Respond with JSON only: {{"probability": <0-1>, "reasoning": "<brief explanation>"}}"""


async def forecast_crux(crux: str) -> tuple[float, str]:
    """Forecast probability of crux resolving YES."""
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": FORECAST_PROMPT.format(
                    question=crux,
                    date=datetime.now().strftime("%Y-%m-%d")
                )
            }],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return result["probability"], result.get("reasoning", "")
    except Exception as e:
        return 0.5, f"Error: {e}"


async def main():
    print("=" * 70)
    print("PAPER TRADING: Setup Trades")
    print("=" * 70)

    benchmark, markets, prices = load_data()

    # Build condition_id lookup
    cond_to_market = {m["condition_id"]: m for m in markets}

    trades = []
    trade_id = 0

    for result in benchmark["results"]:
        if "error" in result:
            continue

        ultimate = result["ultimate"]
        ultimate_cond = result["condition_id"]
        ultimate_price = prices.get(ultimate_cond)

        if ultimate_price is None:
            continue

        # Filter to high-VOI cruxes
        high_voi_cruxes = [cs for cs in result["crux_scores"] if cs["voi"] >= VOI_THRESHOLD]

        if not high_voi_cruxes:
            continue

        print(f"\n{'='*60}")
        print(f"Ultimate: {ultimate[:55]}...")
        print(f"  Current price: {ultimate_price:.2f}")
        print(f"  High-VOI cruxes: {len(high_voi_cruxes)}")

        for cs in high_voi_cruxes[:3]:  # Limit to top 3 per ultimate
            crux = cs["crux"]
            voi = cs["voi"]
            rho_est = cs["rho_estimated"]
            conditionals = cs["conditionals"]

            if "error" in conditionals:
                continue

            # Forecast crux
            print(f"\n  Crux: {crux[:50]}...")
            p_crux_forecast, reasoning = await forecast_crux(crux)
            print(f"    Forecast P(crux=YES): {p_crux_forecast:.2f}")

            # Get conditional prices from benchmark
            p_ult_given_yes = conditionals.get("p_ultimate_given_crux_yes", ultimate_price)
            p_ult_given_no = conditionals.get("p_ultimate_given_crux_no", ultimate_price)

            # Compute expected price after crux resolves
            expected_price = (p_crux_forecast * p_ult_given_yes +
                            (1 - p_crux_forecast) * p_ult_given_no)

            expected_move = expected_price - ultimate_price

            # Determine trade direction
            if abs(expected_move) < 0.02:  # Skip if expected move < 2%
                print(f"    Expected move too small: {expected_move:+.3f}")
                continue

            direction = "BUY" if expected_move > 0 else "SELL"

            print(f"    Expected move: {expected_move:+.3f}")
            print(f"    Direction: {direction}")

            trade = {
                "trade_id": trade_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "open",
                "ultimate": {
                    "question": ultimate,
                    "condition_id": ultimate_cond,
                    "price_at_entry": ultimate_price,
                },
                "crux": {
                    "question": crux,
                    "voi": voi,
                    "rho_estimated": rho_est,
                    "p_forecast": p_crux_forecast,
                    "forecast_reasoning": reasoning,
                },
                "conditionals": {
                    "p_ult_given_yes": p_ult_given_yes,
                    "p_ult_given_no": p_ult_given_no,
                },
                "trade": {
                    "direction": direction,
                    "expected_move": expected_move,
                    "expected_price": expected_price,
                },
                "outcome": None,  # Filled when crux resolves
            }

            trades.append(trade)
            trade_id += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal trades set up: {len(trades)}")

    buys = sum(1 for t in trades if t["trade"]["direction"] == "BUY")
    sells = len(trades) - buys
    print(f"  BUY: {buys}")
    print(f"  SELL: {sells}")

    avg_voi = sum(t["crux"]["voi"] for t in trades) / len(trades) if trades else 0
    print(f"  Avg VOI: {avg_voi:.3f}")

    avg_expected_move = sum(abs(t["trade"]["expected_move"]) for t in trades) / len(trades) if trades else 0
    print(f"  Avg expected |move|: {avg_expected_move:.3f}")

    # Save trades
    output = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "voi_threshold": VOI_THRESHOLD,
            "n_trades": len(trades),
        },
        "trades": trades,
    }

    output_path = PAPER_DIR / "trades.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Show sample trades
    print("\n" + "-" * 70)
    print("SAMPLE TRADES")
    print("-" * 70)

    for t in trades[:5]:
        print(f"\n[{t['trade']['direction']}] {t['ultimate']['question'][:45]}...")
        print(f"  Crux: {t['crux']['question'][:45]}...")
        print(f"  VOI: {t['crux']['voi']:.3f}, P(crux): {t['crux']['p_forecast']:.2f}")
        print(f"  Entry: {t['ultimate']['price_at_entry']:.2f} → Expected: {t['trade']['expected_price']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
