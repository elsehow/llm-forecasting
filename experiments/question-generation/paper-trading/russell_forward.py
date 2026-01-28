"""Russell 2000 Forward-Looking Prospective Validation.

Generate cruxes (baseline + enhanced) for future stock-dates, compute VOI,
then validate against actual returns after resolution.

Timeline:
- Generate: Jan 26, 2026
- Resolution: Jan 27 - Feb 9, 2026 (10 trading days)
- Validation: Compare VOI vs |return|
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import dotenv_values
from openai import OpenAI

# Load env from specific path
env = dotenv_values("/Users/elsehow/Projects/llm-forecasting/.env")

# Paths
RUSSELL_DATA_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/russell-2000-crux/data")
OUTPUT_DIR = Path(__file__).parent / "results"

# API keys
SERP_API_KEY = env.get("SERP_API_KEY")
client = OpenAI(api_key=env.get("OPENAI_API_KEY"))


# =============================================================================
# Stock Selection
# =============================================================================

def load_stock_universe() -> list[dict]:
    """Load the Russell 2000 stock universe."""
    with open(RUSSELL_DATA_DIR / "stock_universe.json") as f:
        return json.load(f)


def select_stocks(n_stocks: int = None) -> list[dict]:
    """Select stocks from universe.

    If n_stocks is None, returns all stocks.
    Otherwise, returns diverse sample across sectors.
    """
    stocks = load_stock_universe()

    if n_stocks is None or n_stocks >= len(stocks):
        return stocks

    # Group by sector for diverse sampling
    by_sector = {}
    for stock in stocks:
        sector = stock.get("sector", "Unknown")
        if sector not in by_sector:
            by_sector[sector] = []
        by_sector[sector].append(stock)

    # Take proportionally from each sector
    selected = []
    per_sector = max(2, n_stocks // len(by_sector))

    for sector, sector_stocks in by_sector.items():
        for stock in sector_stocks[:per_sector]:
            if len(selected) < n_stocks:
                selected.append(stock)

    return selected


def get_trading_days(start_date: datetime, n_days: int = 10) -> list[str]:
    """Get next N trading days (simple: skip weekends)."""
    days = []
    current = start_date

    while len(days) < n_days:
        # Skip weekends
        if current.weekday() < 5:  # 0=Monday, 4=Friday
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return days


# =============================================================================
# Web Search
# =============================================================================

def search_web(query: str, num_results: int = 5) -> list[dict]:
    """Search web using Serper API."""
    if not SERP_API_KEY:
        return [{"title": "Web search unavailable", "href": "", "body": f"No API key for: {query}"}]

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERP_API_KEY, "Content-Type": "application/json"},
            data=json.dumps({"q": query, "num": num_results}),
            timeout=10,
        )

        if response.status_code != 200:
            return [{"error": f"Search failed: {response.status_code}"}]

        data = response.json()
        results = []
        for r in data.get("organic", []):
            results.append({
                "title": r.get("title", ""),
                "href": r.get("link", ""),
                "body": r.get("snippet", ""),
            })

        return results if results else [{"title": "No results", "href": "", "body": f"No results for: {query}"}]

    except Exception as e:
        return [{"error": str(e)}]


# =============================================================================
# Baseline Crux Generation (no web search)
# =============================================================================

BASELINE_PROMPT = """Generate 5 binary yes/no crux questions for this stock prediction.

Ultimate question: Will {ticker} ({company_name}) close higher than it opened on {date}?
Sector: {sector}

Generate cruxes that are:
1. Specific to this company or sector
2. Resolvable before the target date
3. Relevant to the stock's short-term price movement

Return JSON object: {{"cruxes": [{{"text": "question", "rationale": "why relevant"}}]}}"""


def generate_baseline_cruxes(
    ticker: str,
    date: str,
    company_name: str,
    sector: str,
    n_cruxes: int = 5,
) -> list[dict]:
    """Generate cruxes without web search."""
    prompt = BASELINE_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        date=date,
        sector=sector,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = json.loads(response.choices[0].message.content)
        cruxes = content.get("cruxes", content.get("questions", []))
        if isinstance(content, list):
            cruxes = content

        # Normalize format
        result = []
        for c in cruxes[:n_cruxes]:
            text = c.get("text") or c.get("question") or ""
            if text:
                result.append({
                    "text": text,
                    "rationale": c.get("rationale", ""),
                    "pipeline": "baseline"
                })
        return result
    except Exception as e:
        print(f"  Baseline error: {e}")
        return []


# =============================================================================
# Enhanced Crux Generation (with web search)
# =============================================================================

ENHANCED_SYSTEM_PROMPT = """You are a forecasting assistant generating cruxes for stock price predictions.

Use web search to find REAL SCHEDULED EVENTS that could affect the stock. Only generate cruxes grounded in what you find.

Good cruxes:
- Based on real scheduled events (earnings, Fed meetings, product launches)
- Specific to this company and date
- Resolvable with clear criteria

Bad cruxes to avoid:
- Speculative events without evidence
- Generic market cruxes that apply to all stocks equally"""

ENHANCED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for scheduled events and news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "return_cruxes",
            "description": "Return generated cruxes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cruxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["text", "rationale"],
                        }
                    }
                },
                "required": ["cruxes"],
            }
        }
    }
]


def generate_enhanced_cruxes(
    ticker: str,
    date: str,
    company_name: str,
    sector: str,
    n_cruxes: int = 5,
    max_iterations: int = 8,
) -> tuple[list[dict], int]:
    """Generate cruxes with web search grounding.

    Returns (cruxes, n_searches_used).
    """
    user_prompt = f"""Generate {n_cruxes} cruxes for:

Stock: {ticker} ({company_name})
Sector: {sector}
Date: {date}

Ultimate question: Will {ticker} close higher than it opened on {date}?

Search for scheduled events around this date, then generate grounded cruxes."""

    messages = [
        {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    n_searches = 0

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=ENHANCED_TOOLS,
            tool_choice="auto" if iteration < max_iterations - 1 else {"type": "function", "function": {"name": "return_cruxes"}},
        )

        message = response.choices[0].message

        if message.tool_calls:
            # Check for return_cruxes
            for tool_call in message.tool_calls:
                if tool_call.function.name == "return_cruxes":
                    args = json.loads(tool_call.function.arguments)
                    cruxes = args.get("cruxes", [])
                    return ([{**c, "pipeline": "enhanced"} for c in cruxes[:n_cruxes]], n_searches)

            # Handle search calls
            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                if tool_call.function.name == "search_web":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")

                    results = search_web(query)
                    n_searches += 1

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(results[:3], indent=2),
                    })
        else:
            messages.append(message.model_dump())
            messages.append({
                "role": "user",
                "content": "Please use return_cruxes to return your cruxes now.",
            })

    return ([], n_searches)


# =============================================================================
# VOI Computation
# =============================================================================

VOI_PROMPT = """Estimate probabilities for this forecasting crux.

Ultimate question: {ultimate}
Crux: {crux}

Estimate:
1. P(crux = yes): probability the crux resolves "yes"
2. P(ultimate | crux = yes): probability of ultimate given crux is "yes"
3. P(ultimate | crux = no): probability of ultimate given crux is "no"

Return JSON: {{"p_crux_yes": 0.XX, "p_ult_given_yes": 0.XX, "p_ult_given_no": 0.XX}}"""


def compute_voi(crux: dict, ultimate: str) -> dict:
    """Compute VOI for a crux using LLM probability estimates."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": VOI_PROMPT.format(
                ultimate=ultimate,
                crux=crux.get("text", ""),
            )}],
            response_format={"type": "json_object"},
        )

        probs = json.loads(response.choices[0].message.content)

        p_yes = probs.get("p_crux_yes", 0.5)
        p_no = 1 - p_yes
        p_ult_yes = probs.get("p_ult_given_yes", 0.5)
        p_ult_no = probs.get("p_ult_given_no", 0.5)
        p_ult = p_yes * p_ult_yes + p_no * p_ult_no

        voi = p_yes * abs(p_ult_yes - p_ult) + p_no * abs(p_ult_no - p_ult)

        crux["p_crux_yes"] = p_yes
        crux["p_ult_given_yes"] = p_ult_yes
        crux["p_ult_given_no"] = p_ult_no
        crux["voi_linear"] = voi

        return crux

    except Exception as e:
        print(f"  VOI error: {e}")
        crux["voi_linear"] = 0.0
        return crux


# =============================================================================
# Main Experiment
# =============================================================================

def run_forward_looking(
    n_stocks: int = None,  # None = all stocks
    n_trading_days: int = 10,
    dates_per_stock: int = 5,  # How many dates each stock gets
    n_cruxes: int = 5,
    search_budget: int = 1000,  # Higher default
):
    """Run forward-looking prospective validation experiment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Russell 2000 Forward-Looking Experiment ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Select stocks
    stocks = select_stocks(n_stocks)
    print(f"\nSelected {len(stocks)} stocks:")
    for s in stocks[:5]:
        print(f"  {s['ticker']} ({s['company_name'][:30]}) - {s['sector']}")
    print("  ...")

    # Get trading days starting tomorrow (Jan 27, 2026)
    start_date = datetime(2026, 1, 27)
    trading_days = get_trading_days(start_date, n_trading_days)
    print(f"\nTrading days: {trading_days[0]} to {trading_days[-1]}")

    # Generate stock-day combinations
    stock_days = []
    for i, stock in enumerate(stocks):
        # Spread stocks across dates for diversity, then give each dates_per_stock days
        start_idx = i % len(trading_days)
        for j in range(dates_per_stock):
            date_idx = (start_idx + j) % len(trading_days)
            stock_days.append({
                "ticker": stock["ticker"],
                "company_name": stock["company_name"],
                "sector": stock["sector"],
                "date": trading_days[date_idx],
            })

    print(f"\nTotal stock-days: {len(stock_days)}")

    # Generate cruxes
    total_searches = 0
    predictions = []

    for i, sd in enumerate(stock_days):
        print(f"\n[{i+1}/{len(stock_days)}] {sd['ticker']} on {sd['date']}")

        ultimate = f"Will {sd['ticker']} close higher than it opened on {sd['date']}?"

        # Baseline cruxes
        baseline_cruxes = generate_baseline_cruxes(
            ticker=sd["ticker"],
            date=sd["date"],
            company_name=sd["company_name"],
            sector=sd["sector"],
            n_cruxes=n_cruxes,
        )
        print(f"  Baseline: {len(baseline_cruxes)} cruxes")

        # Enhanced cruxes (if within budget)
        enhanced_cruxes = []
        if total_searches < search_budget:
            enhanced_cruxes, n_searches = generate_enhanced_cruxes(
                ticker=sd["ticker"],
                date=sd["date"],
                company_name=sd["company_name"],
                sector=sd["sector"],
                n_cruxes=n_cruxes,
            )
            total_searches += n_searches
            print(f"  Enhanced: {len(enhanced_cruxes)} cruxes ({n_searches} searches, total: {total_searches})")
        else:
            print(f"  Enhanced: skipped (search budget exhausted)")

        # Compute VOI for all cruxes
        all_cruxes = baseline_cruxes + enhanced_cruxes
        for crux in all_cruxes:
            compute_voi(crux, ultimate)

        # Store prediction
        predictions.append({
            "ticker": sd["ticker"],
            "company_name": sd["company_name"],
            "sector": sd["sector"],
            "date": sd["date"],
            "ultimate": ultimate,
            "baseline_cruxes": baseline_cruxes,
            "enhanced_cruxes": enhanced_cruxes,
            "resolution": None,  # Filled after date resolves
        })

        # Print sample
        if baseline_cruxes:
            print(f"  Sample baseline: {baseline_cruxes[0].get('text', '')[:60]}...")
        if enhanced_cruxes:
            print(f"  Sample enhanced: {enhanced_cruxes[0].get('text', '')[:60]}...")

    # Save results
    output = {
        "metadata": {
            "experiment": "russell_forward",
            "generated_at": datetime.now().isoformat(),
            "n_stocks": len(stocks),
            "n_stock_days": len(stock_days),
            "trading_days": trading_days,
            "total_searches": total_searches,
        },
        "predictions": predictions,
    }

    output_file = OUTPUT_DIR / "russell_forward_cruxes.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Stock-days: {len(stock_days)}")
    print(f"Total searches: {total_searches}")
    print(f"Saved to: {output_file}")

    # Compute summary stats
    all_baseline_vois = []
    all_enhanced_vois = []
    for pred in predictions:
        for c in pred.get("baseline_cruxes", []):
            if "voi_linear" in c:
                all_baseline_vois.append(c["voi_linear"])
        for c in pred.get("enhanced_cruxes", []):
            if "voi_linear" in c:
                all_enhanced_vois.append(c["voi_linear"])

    if all_baseline_vois:
        print(f"\nBaseline VOI: mean={sum(all_baseline_vois)/len(all_baseline_vois):.3f}, n={len(all_baseline_vois)}")
    if all_enhanced_vois:
        print(f"Enhanced VOI: mean={sum(all_enhanced_vois)/len(all_enhanced_vois):.3f}, n={len(all_enhanced_vois)}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stocks", type=int, default=None, help="Number of stocks (default: all)")
    parser.add_argument("--n-days", type=int, default=10, help="Trading days to consider")
    parser.add_argument("--dates-per-stock", type=int, default=5, help="Dates per stock")
    parser.add_argument("--n-cruxes", type=int, default=5, help="Cruxes per stock-day")
    parser.add_argument("--search-budget", type=int, default=1000, help="Max Serper searches")

    args = parser.parse_args()

    run_forward_looking(
        n_stocks=args.n_stocks,
        n_trading_days=args.n_days,
        dates_per_stock=args.dates_per_stock,
        n_cruxes=args.n_cruxes,
        search_budget=args.search_budget,
    )
