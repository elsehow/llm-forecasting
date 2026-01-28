"""Enhanced crux pipeline adapter for Russell 2000 stock-days.

This adapter tests whether web-search grounding improves crux quality over
the simple LLM prompt used in the baseline Russell experiment.

Key hypothesis: Web search grounding should reduce "moonshot" cruxes
(speculative high-impact events like FDA approvals that aren't actually scheduled)
and improve timeliness pass rate.

Pipeline:
1. Take stock ticker + date
2. Use web agent with live search to find real scheduled events
3. Generate cruxes grounded in what the search finds
4. Apply 4-dimension evaluation (resolvability, specificity, research_quality, relevance)
5. Output cruxes compatible with existing validation scripts
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
RUSSELL_DATA_DIR = Path("/Users/elsehow/Projects/llm-forecasting/experiments/russell-2000-crux/data")
OUTPUT_DIR = Path(__file__).parent / "data"

# Serper API for web search
SERP_API_KEY = os.getenv("SERP_API_KEY")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================================================================
# Web Search Tools
# =============================================================================

def search_web(query: str, num_results: int = 10) -> list[dict]:
    """Search the web using Serper API.

    Returns list of dicts with 'title', 'href', 'body' (snippet).
    Falls back to a note if API key is missing.
    """
    if not SERP_API_KEY:
        # Graceful fallback: return a note that search is unavailable
        # The model can still generate cruxes based on its knowledge
        return [{
            "title": "Web search unavailable",
            "href": "",
            "body": f"SERP_API_KEY not set. Cannot search for: {query}. Generate cruxes based on general knowledge of this stock and typical events for this sector. Focus on scheduled events like earnings dates, Fed meetings, or sector-specific events.",
        }]

    payload = json.dumps({"q": query, "num": num_results})
    headers = {"X-API-KEY": SERP_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            data=payload,
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
        return [{"error": f"Search error: {str(e)}"}]


# =============================================================================
# Stock Crux Generation with Web Search
# =============================================================================

STOCK_CRUX_SYSTEM_PROMPT = """You are a forecasting research assistant generating cruxes (key uncertainties) for stock price predictions.

Your task: Given a stock and date, use web search to find REAL SCHEDULED EVENTS that could affect the stock, then generate cruxes based on those events.

CRITICAL: Only generate cruxes about events that are ACTUALLY SCHEDULED or PLAUSIBLE based on your web search results. Do NOT generate speculative "moonshot" cruxes about unlikely events (like FDA approvals for non-pharma companies, or major acquisitions without any news).

Good cruxes are:
1. GROUNDED in real events you found through search (earnings dates, Fed meetings, product launches, regulatory filings)
2. SPECIFIC to this company and date
3. RESOLVABLE with clear criteria
4. HIGH-IMPACT if they occur

Bad cruxes to avoid:
- "Will [TICKER] receive FDA approval?" (unless it's a pharma company with pending approvals)
- "Will [TICKER] announce a major acquisition?" (unless there are actual rumors)
- Generic market cruxes that apply to all stocks equally
- Self-referential cruxes like "Will the stock go up?"

For each crux, provide:
- question: Binary yes/no question
- background_information: What you found from web search (with sources)
- resolution_criteria: How to determine the answer
- rationale: Why this is relevant to the stock price

Process:
1. Search for scheduled events for this stock around the date
2. Search for company-specific news that might affect the date
3. Check for broader market events (Fed meetings, economic data releases)
4. Generate cruxes based ONLY on what you find
"""

STOCK_CRUX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information about scheduled events, news, or data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant information"
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "return_cruxes",
            "description": "Return the generated cruxes once you have gathered enough information from web search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cruxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "background_information": {"type": "string"},
                                "resolution_criteria": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["question", "background_information", "resolution_criteria", "rationale"],
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
    context: str,
    n_cruxes: int = 5,
    model: str = "gpt-4o-mini",
    max_iterations: int = 10,
) -> list[dict]:
    """Generate cruxes using web search to ground them in real events.

    Returns list of crux dicts, each with:
    - question, background_information, resolution_criteria, rationale
    """
    user_prompt = f"""Generate {n_cruxes} cruxes for the following stock-day:

Stock: {ticker} ({company_name})
Sector: {sector}
Date: {date}
Context: {context}

Ultimate question: Will {ticker} close higher than it opened on {date}?

Start by searching for:
1. "{ticker} {company_name} {date} earnings" or similar company events
2. "{ticker} news {date}" for any scheduled announcements
3. "Fed meeting {date}" or "economic calendar {date}" for macro events

Then generate cruxes based on what you find. If you don't find specific events, acknowledge this and focus on sector-level or macro events that could affect this stock."""

    messages = [
        {"role": "system", "content": STOCK_CRUX_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=STOCK_CRUX_TOOLS,
            tool_choice="auto" if iteration < max_iterations - 1 else {"type": "function", "function": {"name": "return_cruxes"}},
        )

        message = response.choices[0].message

        # Check if model returned cruxes
        if message.tool_calls:
            # First check if any tool call is return_cruxes
            for tool_call in message.tool_calls:
                if tool_call.function.name == "return_cruxes":
                    args = json.loads(tool_call.function.arguments)
                    return args.get("cruxes", [])

            # Handle all search_web calls (may be parallel)
            # Add assistant message with all tool calls
            messages.append(message.model_dump())

            # Execute each search and add tool results
            for tool_call in message.tool_calls:
                if tool_call.function.name == "search_web":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")

                    # Execute search
                    results = search_web(query)
                    result_text = json.dumps(results[:5], indent=2)  # Limit results

                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })
        else:
            # No tool call, model returned text
            # Force a final crux return
            messages.append(message.model_dump())
            messages.append({
                "role": "user",
                "content": "Please use the return_cruxes function to return your cruxes now.",
            })

    # If we hit max iterations, return empty
    return []


# =============================================================================
# Evaluation (simplified version)
# =============================================================================

EVALUATION_PROMPT = """Evaluate this forecasting crux on 4 dimensions (score 1-10 each):

1. RESOLVABILITY: Can this be definitively resolved? Are criteria clear?
2. SPECIFICITY: Are all terms precisely defined? No ambiguity?
3. RESEARCH_QUALITY: Is it grounded in sources? Good background info?
4. RELEVANCE: Will knowing the answer meaningfully update the stock forecast?

Crux: {question}
Background: {background}
Resolution: {resolution}
Ultimate question: {ultimate}

Return JSON with scores and brief reasoning:
{{"resolvability": {{"score": N, "reason": "..."}}, "specificity": {{"score": N, "reason": "..."}}, "research_quality": {{"score": N, "reason": "..."}}, "relevance": {{"score": N, "reason": "..."}}, "overall_score": N}}"""


def evaluate_crux(
    crux: dict,
    ultimate_question: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate a single crux on 4 dimensions."""
    prompt = EVALUATION_PROMPT.format(
        question=crux.get("question", ""),
        background=crux.get("background_information", ""),
        resolution=crux.get("resolution_criteria", ""),
        ultimate=ultimate_question,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Data Loading and Comparison
# =============================================================================

def load_sample_stock_days(n_earnings: int = 5, n_non_earnings: int = 8) -> list[dict]:
    """Load sample stock-days for proof-of-concept."""
    cruxes = pd.read_parquet(RUSSELL_DATA_DIR / "cruxes_all.parquet")

    stock_days = cruxes.groupby(
        ["ticker", "date", "company_name", "sector", "context", "is_earnings_day"]
    ).size().reset_index(name="n_cruxes")

    # Sample earnings days (where baseline works)
    earnings = stock_days[stock_days["is_earnings_day"] == True]
    sampled_earnings = earnings.sample(n=min(n_earnings, len(earnings)), random_state=42)

    # Sample non-earnings days (where baseline fails - moonshot problem)
    non_earnings = stock_days[stock_days["is_earnings_day"] == False]
    sampled_non_earnings = non_earnings.sample(n=min(n_non_earnings, len(non_earnings)), random_state=42)

    sampled = pd.concat([sampled_earnings, sampled_non_earnings])

    return sampled.to_dict("records")


def load_baseline_cruxes(stock_days: list[dict]) -> pd.DataFrame:
    """Load existing baseline cruxes for the same stock-days."""
    cruxes = pd.read_parquet(RUSSELL_DATA_DIR / "cruxes_all.parquet")

    baseline = []
    for sd in stock_days:
        mask = (cruxes["ticker"] == sd["ticker"]) & (cruxes["date"] == sd["date"])
        stock_cruxes = cruxes[mask].copy()
        if len(stock_cruxes) > 0:
            stock_cruxes["pipeline"] = "baseline"
            baseline.append(stock_cruxes)

    return pd.concat(baseline, ignore_index=True) if baseline else pd.DataFrame()


def compare_pipelines(enhanced: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """Compare crux quality between pipelines."""

    def analyze_cruxes(df: pd.DataFrame, crux_col: str) -> dict:
        if len(df) == 0:
            return {}

        # Keywords indicating real scheduled events
        scheduled_keywords = [
            "earnings", "report", "release", "announcement",
            "guidance", "conference", "call",
        ]

        # Keywords for speculative moonshots
        moonshot_keywords = [
            "fda approval", "major acquisition", "breakthrough",
            "partnership announcement", "ceo resignation",
        ]

        # Fed-related (market-wide, not stock-specific)
        fed_keywords = ["fed", "fomc", "federal reserve", "rate cut", "rate hike"]

        cruxes = df[crux_col].str.lower()

        return {
            "n_cruxes": len(df),
            "scheduled_event_rate": cruxes.apply(
                lambda x: any(kw in x for kw in scheduled_keywords)
            ).mean(),
            "moonshot_rate": cruxes.apply(
                lambda x: any(kw in x for kw in moonshot_keywords)
            ).mean(),
            "fed_rate": cruxes.apply(
                lambda x: any(kw in x for kw in fed_keywords)
            ).mean(),
        }

    comparison = {
        "enhanced": analyze_cruxes(enhanced, "question") if "question" in enhanced.columns else {},
        "baseline": analyze_cruxes(baseline, "crux") if "crux" in baseline.columns else {},
    }

    # Add evaluation scores if available
    if "overall_score" in enhanced.columns:
        comparison["enhanced"]["mean_overall_score"] = enhanced["overall_score"].mean()
        for dim in ["resolvability", "specificity", "research_quality", "relevance"]:
            if f"{dim}_score" in enhanced.columns:
                comparison["enhanced"][f"mean_{dim}"] = enhanced[f"{dim}_score"].mean()

    return comparison


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    n_earnings: int = 5,
    n_non_earnings: int = 8,
    n_cruxes_per_stock: int = 5,
    model: str = "gpt-4o-mini",
    evaluate: bool = True,
):
    """Run the enhanced crux pipeline experiment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Enhanced Crux Pipeline Experiment ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {model}")
    print(f"Sample: {n_earnings} earnings + {n_non_earnings} non-earnings days")

    # Load sample stock-days
    print("\n=== Loading Sample Stock-Days ===")
    stock_days = load_sample_stock_days(n_earnings, n_non_earnings)
    print(f"Loaded {len(stock_days)} stock-days:")
    for sd in stock_days:
        is_earnings = "EARNINGS" if sd["is_earnings_day"] else "non-earnings"
        print(f"  {sd['ticker']} on {sd['date']} ({is_earnings})")

    # Generate enhanced cruxes
    print("\n=== Generating Enhanced Cruxes ===")
    all_enhanced = []

    for i, sd in enumerate(stock_days):
        is_earnings = "EARNINGS" if sd["is_earnings_day"] else "non-earnings"
        print(f"\n[{i+1}/{len(stock_days)}] {sd['ticker']} on {sd['date']} ({is_earnings})")

        cruxes = generate_enhanced_cruxes(
            ticker=sd["ticker"],
            date=sd["date"],
            company_name=sd["company_name"],
            sector=sd["sector"],
            context=sd["context"],
            n_cruxes=n_cruxes_per_stock,
            model=model,
        )

        print(f"  Generated {len(cruxes)} cruxes")

        ultimate = f"Will {sd['ticker']} close higher than it opened on {sd['date']}?"

        for j, crux in enumerate(cruxes):
            row = {
                "ticker": sd["ticker"],
                "date": sd["date"],
                "company_name": sd["company_name"],
                "sector": sd["sector"],
                "is_earnings_day": sd["is_earnings_day"],
                "crux_index": j,
                "question": crux.get("question", ""),
                "background_information": crux.get("background_information", ""),
                "resolution_criteria": crux.get("resolution_criteria", ""),
                "rationale": crux.get("rationale", ""),
                "ultimate_question": ultimate,
                "pipeline": "enhanced",
            }

            # Evaluate if requested
            if evaluate:
                eval_result = evaluate_crux(crux, ultimate, model)
                if "error" not in eval_result:
                    for dim in ["resolvability", "specificity", "research_quality", "relevance"]:
                        if dim in eval_result:
                            row[f"{dim}_score"] = eval_result[dim].get("score")
                            row[f"{dim}_reason"] = eval_result[dim].get("reason")
                    row["overall_score"] = eval_result.get("overall_score")

            all_enhanced.append(row)

            # Print sample
            if j == 0:
                print(f"  Sample: {crux.get('question', '')[:80]}...")

    enhanced_df = pd.DataFrame(all_enhanced)

    # Load baseline
    print("\n=== Loading Baseline Cruxes ===")
    baseline_df = load_baseline_cruxes(stock_days)
    print(f"Loaded {len(baseline_df)} baseline cruxes")

    # Compare
    print("\n=== Comparison ===")
    comparison = compare_pipelines(enhanced_df, baseline_df)
    print(json.dumps(comparison, indent=2, default=float))

    # Save results
    enhanced_df.to_parquet(OUTPUT_DIR / "enhanced_cruxes.parquet", index=False)
    enhanced_df.to_csv(OUTPUT_DIR / "enhanced_cruxes.csv", index=False)

    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=float)

    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "n_earnings": n_earnings,
            "n_non_earnings": n_non_earnings,
            "n_cruxes_per_stock": n_cruxes_per_stock,
            "n_stock_days": len(stock_days),
            "n_enhanced_cruxes": len(enhanced_df),
            "n_baseline_cruxes": len(baseline_df),
        }, f, indent=2)

    print(f"\nSaved results to {OUTPUT_DIR}")

    return enhanced_df, baseline_df, comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run enhanced crux pipeline experiment")
    parser.add_argument("--n-earnings", type=int, default=5, help="Number of earnings days to sample")
    parser.add_argument("--n-non-earnings", type=int, default=8, help="Number of non-earnings days to sample")
    parser.add_argument("--n-cruxes", type=int, default=5, help="Cruxes per stock-day")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    run_experiment(
        n_earnings=args.n_earnings,
        n_non_earnings=args.n_non_earnings,
        n_cruxes_per_stock=args.n_cruxes,
        model=args.model,
        evaluate=not args.no_eval,
    )
