"""Generate cruxes for stock-day pairs using LLM.

For each stock on each day, generate cruxes that might explain or predict
whether the stock will close higher than it opened.

Methods:
- baseline: Pure LLM generation without web search (safe for retrospective data)
- enhanced: Uses web search for context (may contaminate retrospective analysis)
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import litellm
import pandas as pd

from config import (
    CRUX_GENERATION_MODEL,
    PILOT_N_CRUXES,
    MODEL_KNOWLEDGE_CUTOFF,
)

DATA_DIR = Path(__file__).parent / "data"


CRUX_GENERATION_PROMPT = """You are generating cruxes (key uncertainties) that could explain whether a stock will go up or down on a specific day.

**Stock:** {ticker} ({company_name})
**Sector:** {sector}
**Date:** {date}
**Context:** {context}

Generate {n_cruxes} cruxes - questions whose answers would most affect whether {ticker} closes higher than it opened on {date}.

Good cruxes are:
- Specific and resolvable (not vague like "will the market be bullish?")
- Relevant to this stock on this date
- About events or information that could move the stock

Examples of good cruxes:
- "Will {ticker}'s Q4 earnings beat analyst expectations?"
- "Will the Fed announce a rate cut today?"
- "Will {ticker}'s CEO announce any major strategic changes?"
- "Will {ticker}'s main competitor report disappointing results?"

Bad cruxes (avoid these):
- Self-referential: "Will {ticker}'s stock price go up?" (circular)
- Too vague: "Will market sentiment be positive?"
- Already known: "Did {ticker} report earnings last week?" (past event)

Return a JSON array of {n_cruxes} crux questions:
["<crux 1>", "<crux 2>", ...]"""


# Sector-specific guidance for diverse crux generation
SECTOR_GUIDANCE = {
    "Healthcare": "Clinical trial results, FDA decisions, drug pricing, patient outcomes",
    "Technology": "Product launches, AI adoption, chip supply, customer concentration",
    "Financial Services": "Interest rate sensitivity, loan quality, capital ratios, deposit flows",
    "Industrials": "Order backlog, capacity utilization, labor costs, capex announcements",
    "Consumer Cyclical": "Same-store sales, inventory levels, consumer sentiment, pricing power",
    "Consumer Defensive": "Market share, commodity costs, promotional activity",
    "Energy": "Oil/gas prices, production volumes, drilling activity, renewable transition",
    "Utilities": "Rate case outcomes, weather impact, renewable capacity",
    "Basic Materials": "Commodity prices, demand forecasts, cost inflation",
    "Communication Services": "Subscriber growth, content costs, advertising revenue",
    "Real Estate": "Occupancy rates, rental income, interest rate sensitivity",
}


DIVERSE_CRUX_PROMPT = """You are generating cruxes (key uncertainties) that could explain whether a stock will go up or down on a specific day.

**Stock:** {ticker} ({company_name})
**Sector:** {sector}
**Date:** {date}
**Context:** {context}

Generate exactly 5 cruxes - ONE from each category below. Each crux should be a question whose answer would most affect whether {ticker} closes higher than it opened on {date}.

**REQUIRED CATEGORIES (one crux each):**

1. FINANCIAL: About earnings, revenue, margins, or guidance
   Example: "Will {ticker}'s Q4 revenue beat analyst consensus?"

2. OPERATIONAL: About production, efficiency, supply chain, or execution
   Example: "Will {ticker} report improved inventory turnover?"

3. STRATEGIC: About partnerships, M&A, competitive positioning, or management
   Example: "Will {ticker} announce any major contracts or partnerships?"

4. MACRO/MARKET: About Fed policy, sector trends, or economic factors
   Example: "Will today's macro data favor {sector} stocks?"

5. SECTOR-SPECIFIC ({sector}): {sector_specific_guidance}
   Example: Focus on factors unique to {sector} companies

For each crux:
- Be specific and resolvable (not vague)
- Focus on information that could realistically emerge on {date}
- Avoid self-referential questions ("Will the stock go up?")

Return a JSON object with categories as keys:
{{"financial": "...", "operational": "...", "strategic": "...", "macro": "...", "sector_specific": "..."}}"""


async def generate_cruxes_for_stock_day(
    ticker: str,
    company_name: str,
    sector: str,
    date: str,
    context: str,
    n_cruxes: int = PILOT_N_CRUXES,
) -> list[str]:
    """Generate cruxes for a single stock-day.

    Args:
        ticker: Stock symbol
        company_name: Company name
        sector: Industry sector
        date: Date string (YYYY-MM-DD)
        context: Additional context (e.g., "Earnings release today")
        n_cruxes: Number of cruxes to generate

    Returns:
        List of crux question strings
    """
    prompt = CRUX_GENERATION_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        date=date,
        context=context,
        n_cruxes=n_cruxes,
    )

    try:
        response = await litellm.acompletion(
            model=CRUX_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,  # Some creativity for diverse cruxes
        )

        text = response.choices[0].message.content.strip()

        # Parse JSON array
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        cruxes = json.loads(text)
        return cruxes[:n_cruxes]

    except Exception as e:
        print(f"  Error generating cruxes for {ticker} on {date}: {e}")
        return []


async def generate_diverse_cruxes_for_stock_day(
    ticker: str,
    company_name: str,
    sector: str,
    date: str,
    context: str,
) -> list[dict]:
    """Generate category-constrained diverse cruxes for a single stock-day.

    Args:
        ticker: Stock symbol
        company_name: Company name
        sector: Industry sector
        date: Date string (YYYY-MM-DD)
        context: Additional context (e.g., "Earnings release today")

    Returns:
        List of dicts with 'crux' and 'category' keys
    """
    sector_guidance = SECTOR_GUIDANCE.get(sector, "Industry-specific factors and trends")

    prompt = DIVERSE_CRUX_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        date=date,
        context=context,
        sector_specific_guidance=sector_guidance,
    )

    try:
        response = await litellm.acompletion(
            model=CRUX_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
        )

        text = response.choices[0].message.content.strip()

        # Parse JSON object
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        crux_dict = json.loads(text)

        # Convert to list with category labels
        categories = ["financial", "operational", "strategic", "macro", "sector_specific"]
        result = []
        for cat in categories:
            if cat in crux_dict:
                result.append({
                    "crux": crux_dict[cat],
                    "category": cat,
                })

        return result

    except Exception as e:
        print(f"  Error generating diverse cruxes for {ticker} on {date}: {e}")
        return []


async def generate_cruxes_batch(
    stock_days: list[dict],
    n_cruxes: int = PILOT_N_CRUXES,
    diverse: bool = False,
) -> list[dict]:
    """Generate cruxes for a batch of stock-day pairs.

    Args:
        stock_days: List of dicts with ticker, company_name, sector, date, context
        n_cruxes: Number of cruxes per stock-day (ignored if diverse=True)
        diverse: If True, use category-constrained diverse generation

    Returns:
        List of dicts with ticker, date, cruxes (and categories if diverse)
    """
    results = []

    # Process in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(stock_days), batch_size):
        batch = stock_days[i:i+batch_size]
        print(f"  Processing {i+1}-{min(i+batch_size, len(stock_days))} of {len(stock_days)}...")

        if diverse:
            tasks = [
                generate_diverse_cruxes_for_stock_day(
                    sd["ticker"],
                    sd["company_name"],
                    sd["sector"],
                    sd["date"],
                    sd.get("context", ""),
                )
                for sd in batch
            ]
        else:
            tasks = [
                generate_cruxes_for_stock_day(
                    sd["ticker"],
                    sd["company_name"],
                    sd["sector"],
                    sd["date"],
                    sd.get("context", ""),
                    n_cruxes,
                )
                for sd in batch
            ]

        cruxes_list = await asyncio.gather(*tasks)

        for sd, cruxes in zip(batch, cruxes_list):
            results.append({
                "ticker": sd["ticker"],
                "date": sd["date"],
                "company_name": sd["company_name"],
                "sector": sd["sector"],
                "context": sd.get("context", ""),
                "cruxes": cruxes,
                "diverse": diverse,
            })

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    return results


def build_stock_day_list(
    returns_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    universe: list[dict],
    pilot_mode: bool = True,
) -> list[dict]:
    """Build list of stock-day pairs to generate cruxes for.

    In pilot mode, only generates for earnings days.
    In full mode, generates for all days.
    """
    # Build lookup for company info
    company_info = {u["ticker"]: u for u in universe}

    stock_days = []

    if pilot_mode:
        # Only earnings days
        for _, row in earnings_df.iterrows():
            ticker = row["ticker"]
            date = row["earnings_date"]

            # Check if we have return data for this day
            has_return = len(returns_df[
                (returns_df["ticker"] == ticker) &
                (returns_df["date"] == date)
            ]) > 0

            if has_return:
                info = company_info.get(ticker, {})
                stock_days.append({
                    "ticker": ticker,
                    "date": date,
                    "company_name": info.get("company_name", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "context": f"This is {ticker}'s earnings release day.",
                })
    else:
        # All days for all stocks
        for ticker in returns_df["ticker"].unique():
            info = company_info.get(ticker, {})
            ticker_returns = returns_df[returns_df["ticker"] == ticker]

            for _, row in ticker_returns.iterrows():
                date = row["date"]

                # Check if earnings day
                is_earnings = row.get("is_earnings_day", False)
                context = f"This is {ticker}'s earnings release day." if is_earnings else ""

                stock_days.append({
                    "ticker": ticker,
                    "date": date,
                    "company_name": info.get("company_name", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "context": context,
                })

    return stock_days


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cruxes for stock-day pairs")
    parser.add_argument(
        "--earnings-only",
        action="store_true",
        help="Only generate cruxes for earnings days (default behavior)"
    )
    parser.add_argument(
        "--all-days",
        action="store_true",
        help="Generate cruxes for all trading days, not just earnings"
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "enhanced"],
        default="baseline",
        help="Generation method: 'baseline' (no web search, safe for retrospective) or 'enhanced' (with web search)"
    )
    parser.add_argument(
        "--n-cruxes",
        type=int,
        default=PILOT_N_CRUXES,
        help=f"Number of cruxes per stock-day (default: {PILOT_N_CRUXES})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: cruxes_pilot.parquet)"
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Use category-constrained diversity prompt (5 cruxes, 1 per category)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    returns_path = DATA_DIR / "stock_returns.parquet"
    earnings_path = DATA_DIR / "earnings_calendar.json"
    universe_path = DATA_DIR / "stock_universe.json"

    if not all(p.exists() for p in [returns_path, earnings_path, universe_path]):
        print("Run fetch_earnings_calendar.py and fetch_stock_data.py first!")
        return

    returns_df = pd.read_parquet(returns_path)
    with open(earnings_path) as f:
        earnings = json.load(f)
    earnings_df = pd.DataFrame(earnings)
    with open(universe_path) as f:
        universe = json.load(f)

    # Determine mode: earnings-only is default unless --all-days specified
    earnings_only = not args.all_days

    # Build stock-day list
    stock_days = build_stock_day_list(returns_df, earnings_df, universe, pilot_mode=earnings_only)

    print(f"\n=== Generating Cruxes ===")
    print(f"Mode: {'earnings-only' if earnings_only else 'all days'}")
    print(f"Method: {args.method}")
    print(f"Diverse: {args.diverse}")
    print(f"Stock-day pairs: {len(stock_days)}")
    print(f"Model: {CRUX_GENERATION_MODEL}")
    if args.diverse:
        print(f"Cruxes per stock-day: 5 (1 per category)")
    else:
        print(f"Cruxes per stock-day: {args.n_cruxes}")

    if args.method == "enhanced":
        print("WARNING: Enhanced method uses web search - may contaminate retrospective analysis!")

    if len(stock_days) == 0:
        print("No stock-days to process!")
        return

    # Generate cruxes
    results = await generate_cruxes_batch(stock_days, args.n_cruxes, diverse=args.diverse)

    # Summary
    total_cruxes = sum(len(r["cruxes"]) for r in results)
    print(f"\n=== Results ===")
    print(f"Generated {total_cruxes} cruxes for {len(results)} stock-days")

    # Show some examples
    print(f"\nExample cruxes:")
    for r in results[:3]:
        print(f"\n{r['ticker']} on {r['date']} ({r['context']}):")
        for i, crux in enumerate(r["cruxes"][:3], 1):
            print(f"  {i}. {crux}")

    # Save
    # Flatten for easier processing
    flat_results = []
    for r in results:
        if r.get("diverse", False):
            # Diverse format: list of dicts with 'crux' and 'category'
            for i, crux_item in enumerate(r["cruxes"]):
                flat_results.append({
                    "ticker": r["ticker"],
                    "date": r["date"],
                    "company_name": r["company_name"],
                    "sector": r["sector"],
                    "context": r["context"],
                    "crux_index": i,
                    "crux": crux_item["crux"],
                    "category": crux_item["category"],
                    "method": args.method,
                    "diverse": True,
                })
        else:
            # Baseline format: list of strings
            for i, crux in enumerate(r["cruxes"]):
                flat_results.append({
                    "ticker": r["ticker"],
                    "date": r["date"],
                    "company_name": r["company_name"],
                    "sector": r["sector"],
                    "context": r["context"],
                    "crux_index": i,
                    "crux": crux,
                    "category": None,
                    "method": args.method,
                    "diverse": False,
                })

    flat_df = pd.DataFrame(flat_results)

    # Determine output filename
    if args.output:
        output_file = args.output
    elif args.diverse:
        output_file = "cruxes_diverse.parquet"
    else:
        output_file = "cruxes_pilot.parquet"

    if not output_file.endswith(".parquet"):
        output_file += ".parquet"

    flat_df.to_parquet(DATA_DIR / output_file, index=False)
    print(f"\nSaved {len(flat_df)} cruxes to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
