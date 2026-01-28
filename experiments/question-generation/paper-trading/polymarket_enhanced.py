"""Polymarket Q4 Enhanced Crux Generation.

Generate baseline and enhanced cruxes for Polymarket Q4 ultimates with VOI scoring.
Enhanced cruxes use web search grounding for real scheduled events.

Usage:
    uv run python experiments/question-generation/paper-trading/polymarket_enhanced.py
    uv run python experiments/question-generation/paper-trading/polymarket_enhanced.py --n-cruxes 10 --n-ultimates 20

Output:
    results/q4_enhanced_cruxes.json (overwrites existing)

Validation:
    uv run python experiments/question-generation/paper-trading/validate_q4.py --results enhanced
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import requests
from dotenv import dotenv_values
from openai import OpenAI

# Load env
env = dotenv_values("/Users/elsehow/Projects/llm-forecasting/.env")

# Paths
PAPER_TRADING_DIR = Path(__file__).parent
RESULTS_DIR = PAPER_TRADING_DIR / "results"
EXISTING_CRUXES = RESULTS_DIR / "q4_enhanced_cruxes.json"

# API keys
SERP_API_KEY = env.get("SERP_API_KEY")
client = OpenAI(api_key=env.get("OPENAI_API_KEY"))


# =============================================================================
# Ultimate Loading
# =============================================================================

def load_existing_ultimates() -> list[dict]:
    """Load ultimates from existing q4_enhanced_cruxes.json."""
    if not EXISTING_CRUXES.exists():
        raise FileNotFoundError(
            f"Existing cruxes not found at {EXISTING_CRUXES}\n"
            "Need existing file to determine ultimates."
        )

    with open(EXISTING_CRUXES) as f:
        data = json.load(f)

    ultimates = []
    for pred in data.get("predictions", []):
        ultimates.append({
            "ultimate": pred["ultimate"],
            "condition_id": pred["condition_id"],
        })

    return ultimates


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

BASELINE_PROMPT = """Generate {n_cruxes} binary yes/no crux questions for this prediction.

Ultimate question: {ultimate}

Generate cruxes that are:
1. Specific to this question's topic
2. Resolvable before the ultimate resolves
3. Relevant to the outcome of the ultimate

Return JSON object: {{"cruxes": ["question1", "question2", ...]}}

Return only the question text, no additional formatting."""


def generate_baseline_cruxes(
    ultimate: str,
    n_cruxes: int = 10,
) -> list[str]:
    """Generate cruxes without web search. Returns list of strings."""
    prompt = BASELINE_PROMPT.format(
        ultimate=ultimate,
        n_cruxes=n_cruxes,
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

        # Ensure we return strings
        result = []
        for c in cruxes[:n_cruxes]:
            if isinstance(c, str):
                result.append(c)
            elif isinstance(c, dict):
                text = c.get("text") or c.get("question") or ""
                if text:
                    result.append(text)

        return result
    except Exception as e:
        print(f"  Baseline error: {e}")
        return []


# =============================================================================
# Enhanced Crux Generation (with web search)
# =============================================================================

ENHANCED_SYSTEM_PROMPT = """You are a forecasting assistant generating cruxes for prediction questions.

Use web search to find REAL SCHEDULED EVENTS that could affect the outcome. Only generate cruxes grounded in what you find.

Good cruxes:
- Based on real scheduled events (elections, meetings, product launches, deadlines)
- Specific to this question and timeframe
- Resolvable with clear criteria

Bad cruxes to avoid:
- Speculative events without evidence
- Generic cruxes that apply to any similar question"""

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
    ultimate: str,
    n_cruxes: int = 10,
    max_iterations: int = 10,
) -> tuple[list[dict], int]:
    """Generate cruxes with web search grounding.

    Returns (cruxes, n_searches_used).
    """
    user_prompt = f"""Generate {n_cruxes} cruxes for:

Ultimate question: {ultimate}

Search for scheduled events, deadlines, and relevant news, then generate grounded cruxes."""

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

def run_enhanced_generation(
    n_ultimates: int = None,  # None = all
    n_cruxes: int = 10,
    search_budget: int = 500,
):
    """Generate baseline and enhanced cruxes for Polymarket Q4 ultimates."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Polymarket Q4 Enhanced Crux Generation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Cruxes per ultimate: {n_cruxes}")

    # Load ultimates
    print("\nLoading ultimates from existing file...")
    ultimates = load_existing_ultimates()
    print(f"  Found {len(ultimates)} ultimates")

    if n_ultimates is not None and n_ultimates < len(ultimates):
        ultimates = ultimates[:n_ultimates]
        print(f"  Using {len(ultimates)} ultimates")

    # Generate cruxes
    total_searches = 0
    predictions = []

    for i, ult in enumerate(ultimates):
        print(f"\n[{i+1}/{len(ultimates)}] {ult['ultimate'][:60]}...")

        # Baseline cruxes (strings only)
        baseline_cruxes = generate_baseline_cruxes(
            ultimate=ult["ultimate"],
            n_cruxes=n_cruxes,
        )
        print(f"  Baseline: {len(baseline_cruxes)} cruxes")

        # Enhanced cruxes (with web search)
        enhanced_cruxes = []
        if total_searches < search_budget:
            enhanced_cruxes, n_searches = generate_enhanced_cruxes(
                ultimate=ult["ultimate"],
                n_cruxes=n_cruxes,
            )
            total_searches += n_searches
            print(f"  Enhanced: {len(enhanced_cruxes)} cruxes ({n_searches} searches, total: {total_searches})")
        else:
            print(f"  Enhanced: skipped (search budget exhausted)")

        # Compute VOI for enhanced cruxes
        for crux in enhanced_cruxes:
            compute_voi(crux, ult["ultimate"])

        # Store prediction
        predictions.append({
            "ultimate": ult["ultimate"],
            "condition_id": ult["condition_id"],
            "baseline_cruxes": baseline_cruxes,
            "enhanced_cruxes": enhanced_cruxes,
        })

        # Print samples
        if baseline_cruxes:
            print(f"  Sample baseline: {baseline_cruxes[0][:60]}...")
        if enhanced_cruxes:
            print(f"  Sample enhanced: {enhanced_cruxes[0].get('text', '')[:60]}...")

    # Save results
    output = {
        "metadata": {
            "experiment": "polymarket_q4_enhanced",
            "generated_at": datetime.now().isoformat(),
            "n_ultimates": len(ultimates),
            "n_cruxes_per_ultimate": n_cruxes,
            "total_searches": total_searches,
            "check_date": "2026-02-03",
        },
        "predictions": predictions,
    }

    output_file = RESULTS_DIR / "q4_enhanced_cruxes.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    print(f"Ultimates: {len(ultimates)}")
    print(f"Baseline cruxes: {sum(len(p['baseline_cruxes']) for p in predictions)}")
    print(f"Enhanced cruxes: {sum(len(p['enhanced_cruxes']) for p in predictions)}")
    print(f"Total searches: {total_searches}")
    print(f"\nSaved to: {output_file}")

    # VOI stats
    all_vois = []
    for pred in predictions:
        for c in pred.get("enhanced_cruxes", []):
            if "voi_linear" in c:
                all_vois.append(c["voi_linear"])

    if all_vois:
        print(f"\nEnhanced VOI: mean={sum(all_vois)/len(all_vois):.3f}, n={len(all_vois)}")

    print(f"\nValidation command:")
    print(f"  uv run python experiments/question-generation/paper-trading/validate_q4.py --results enhanced")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate enhanced cruxes for Polymarket Q4 ultimates")
    parser.add_argument("--n-ultimates", type=int, default=None, help="Number of ultimates (default: all)")
    parser.add_argument("--n-cruxes", type=int, default=10, help="Cruxes per ultimate (default: 10)")
    parser.add_argument("--search-budget", type=int, default=500, help="Max Serper searches (default: 500)")

    args = parser.parse_args()

    run_enhanced_generation(
        n_ultimates=args.n_ultimates,
        n_cruxes=args.n_cruxes,
        search_budget=args.search_budget,
    )
