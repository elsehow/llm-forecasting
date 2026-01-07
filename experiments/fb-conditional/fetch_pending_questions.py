#!/usr/bin/env python3
"""Fetch questions and generate pairs for longitudinal conditional forecasting experiment.

Creates stock/crypto questions with a fixed resolution date and generates pairs.
"""

import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import litellm

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "llm-forecasting" / "src"))

from llm_forecasting.sources import ManifoldSource, YahooFinanceSource

# Extended list of tickers for more question diversity
TICKERS = [
    # Major indices
    {"symbol": "^GSPC", "name": "S&P 500"},
    {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
    {"symbol": "^IXIC", "name": "NASDAQ Composite"},
    {"symbol": "^RUT", "name": "Russell 2000"},
    {"symbol": "^VIX", "name": "VIX Volatility Index"},
    # Tech stocks
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOGL", "name": "Alphabet"},
    {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "META", "name": "Meta"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "AMD", "name": "AMD"},
    {"symbol": "INTC", "name": "Intel"},
    {"symbol": "CRM", "name": "Salesforce"},
    # Finance
    {"symbol": "JPM", "name": "JPMorgan Chase"},
    {"symbol": "BAC", "name": "Bank of America"},
    {"symbol": "GS", "name": "Goldman Sachs"},
    {"symbol": "V", "name": "Visa"},
    {"symbol": "MA", "name": "Mastercard"},
    # Healthcare
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "PFE", "name": "Pfizer"},
    {"symbol": "UNH", "name": "UnitedHealth"},
    {"symbol": "MRK", "name": "Merck"},
    # Energy
    {"symbol": "XOM", "name": "Exxon Mobil"},
    {"symbol": "CVX", "name": "Chevron"},
    {"symbol": "CL=F", "name": "Crude Oil Futures"},
    # Commodities
    {"symbol": "GC=F", "name": "Gold Futures"},
    {"symbol": "SI=F", "name": "Silver Futures"},
    # Crypto
    {"symbol": "BTC-USD", "name": "Bitcoin"},
    {"symbol": "ETH-USD", "name": "Ethereum"},
    {"symbol": "SOL-USD", "name": "Solana"},
    # Consumer
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "KO", "name": "Coca-Cola"},
    {"symbol": "PG", "name": "Procter & Gamble"},
    {"symbol": "DIS", "name": "Disney"},
    {"symbol": "NFLX", "name": "Netflix"},
]


async def fetch_stock_questions(resolution_date: date):
    """Fetch stock questions with a fixed resolution date."""
    print(f"Generating stock questions resolving on {resolution_date}")
    print("=" * 60)

    source = YahooFinanceSource(tickers=TICKERS)
    questions = await source.fetch_questions()

    # Set resolution date and create proper question text
    updated_questions = []
    for q in questions:
        # Update question text with actual dates
        text = q.text.replace("{resolution_date}", resolution_date.strftime("%B %d, %Y"))
        text = text.replace("{forecast_due_date}", date.today().strftime("%B %d, %Y"))

        updated_questions.append({
            "id": q.id,
            "source": "yfinance",
            "text": text,
            "background": q.background,
            "url": q.url,
            "resolution_date": resolution_date.isoformat(),
            "baseline_price": q.base_rate,  # Current price as baseline
            "baseline_date": date.today().isoformat(),
        })

    await source.close()
    print(f"  Generated {len(updated_questions)} stock questions")
    return updated_questions


async def fetch_market_questions(min_days: int, max_days: int):
    """Fetch market questions from Manifold in the given window."""
    today = date.today()
    min_date = today + timedelta(days=min_days)
    max_date = today + timedelta(days=max_days)

    print(f"\nFetching market questions resolving {min_date} to {max_date}")

    source = ManifoldSource()
    questions = await source.fetch_questions()

    # Filter to resolution window
    filtered = []
    for q in questions:
        if q.resolution_date and min_date <= q.resolution_date <= max_date:
            filtered.append({
                "id": q.id,
                "source": "manifold",
                "text": q.text,
                "background": q.background[:500] if q.background else None,
                "url": q.url,
                "resolution_date": q.resolution_date.isoformat(),
                "base_rate": q.base_rate,
            })

    await source.close()
    print(f"  Found {len(filtered)} market questions in window")
    return filtered


async def generate_pairs(questions: list[dict], model: str = "claude-sonnet-4-20250514"):
    """Use LLM to generate correlated pairs from questions."""
    print(f"\nGenerating pairs from {len(questions)} questions...")

    # Format questions for the prompt
    q_list = "\n".join([
        f"{i+1}. [{q['source']}] {q['text'][:100]}..."
        for i, q in enumerate(questions)
    ])

    prompt = f"""You are helping design a forecasting experiment. Below are {len(questions)} questions that will resolve in the next 7-14 days.

Your task: Find pairs of questions that are likely CORRELATED - where knowing the outcome of one would help predict the other.

Questions:
{q_list}

Return a JSON array of pairs. For each pair include:
- "pair_id": unique identifier (e.g., "1_2" for questions 1 and 2)
- "q1_idx": index of first question (1-based)
- "q2_idx": index of second question (1-based)
- "category": "strong" (clear causal/logical link), "weak" (same domain, unclear link), or "none" (unrelated - include a few as controls)
- "reason": brief explanation of the relationship

Find at least 30 pairs if possible, with a mix of strong/weak/none categories.
Prioritize finding strong pairs - questions where outcomes should genuinely correlate.

Return ONLY valid JSON, no other text."""

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
    )

    content = response.choices[0].message.content

    # Extract JSON
    import re
    json_match = re.search(r'\[[\s\S]*\]', content)
    if not json_match:
        print("  ERROR: Could not parse pairs from LLM response")
        return []

    pairs_raw = json.loads(json_match.group())

    # Build full pair objects
    pairs = []
    for p in pairs_raw:
        q1_idx = p["q1_idx"] - 1
        q2_idx = p["q2_idx"] - 1
        if 0 <= q1_idx < len(questions) and 0 <= q2_idx < len(questions):
            pairs.append({
                "pair_id": p["pair_id"],
                "category": p["category"],
                "reason": p["reason"],
                "question_a": questions[q1_idx],
                "question_b": questions[q2_idx],
            })

    # Count by category
    by_cat = {}
    for p in pairs:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    print(f"  Generated {len(pairs)} pairs: {by_cat}")

    return pairs


async def main():
    # Resolution date: 7 days from now
    resolution_date = date.today() + timedelta(days=7)
    output_dir = Path(__file__).parent

    # Fetch questions
    stock_questions = await fetch_stock_questions(resolution_date)
    market_questions = await fetch_market_questions(min_days=5, max_days=10)

    all_questions = stock_questions + market_questions
    print(f"\nTotal questions: {len(all_questions)}")

    if len(all_questions) < 10:
        print("Not enough questions to generate pairs.")
        return

    # Save questions
    questions_file = output_dir / f"pending_questions_{resolution_date.isoformat()}.json"
    with open(questions_file, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"\nSaved questions to {questions_file}")

    # Generate pairs
    pairs = await generate_pairs(all_questions)

    if pairs:
        pairs_file = output_dir / f"pending_pairs_{resolution_date.isoformat()}.json"
        with open(pairs_file, "w") as f:
            json.dump({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "resolution_date": resolution_date.isoformat(),
                "num_questions": len(all_questions),
                "pairs": pairs,
            }, f, indent=2)
        print(f"Saved pairs to {pairs_file}")
        print(f"\n*** Set a reminder to run the experiment on {resolution_date} ***")


if __name__ == "__main__":
    asyncio.run(main())
