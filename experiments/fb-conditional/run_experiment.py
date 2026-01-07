#!/usr/bin/env python3
"""Run the conditional forecasting experiment.

For each question pair, elicits:
- P(A) - unconditional probability of A
- P(A|B=1) - probability of A given B resolved YES
- P(A|B=0) - probability of A given B resolved NO

Compares agent forecasts to independence baseline.

Usage:
    uv run python experiments/fb-conditional/run_experiment.py
    uv run python experiments/fb-conditional/run_experiment.py --pairs pairs.json
    uv run python experiments/fb-conditional/run_experiment.py --model claude-sonnet-4-20250514
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from llm_forecasting.storage.sqlite import SQLiteStorage


async def elicit_forecast(agent, question_text: str, condition: str | None = None) -> float:
    """Elicit a probability forecast from an agent.

    Args:
        agent: Forecasting agent
        question_text: The question to forecast
        condition: Optional conditioning statement

    Returns:
        Probability estimate (0-1)
    """
    # TODO: Implement using llm_forecasting agents
    raise NotImplementedError("Agent elicitation not yet implemented")


async def run_pair_experiment(
    agent,
    storage: SQLiteStorage,
    pair: dict,
) -> dict:
    """Run experiment on a single question pair.

    Returns dict with:
    - p_a: P(A) unconditional
    - p_a_given_b1: P(A|B=YES)
    - p_a_given_b0: P(A|B=NO)
    - sensitivity: |P(A|B=1) - P(A|B=0)|
    - actual_b: Actual resolution of B
    - actual_a: Actual resolution of A
    """
    q_a = await storage.get_question(pair["source_a"], pair["id_a"])
    q_b = await storage.get_question(pair["source_b"], pair["id_b"])

    if not q_a or not q_b:
        return {"error": "Questions not found"}

    # Get resolutions
    res_a = await storage.get_resolution(pair["source_a"], pair["id_a"])
    res_b = await storage.get_resolution(pair["source_b"], pair["id_b"])

    if not res_a or not res_b:
        return {"error": "Resolutions not found"}

    # Elicit forecasts
    p_a = await elicit_forecast(agent, q_a.text)
    p_a_given_b1 = await elicit_forecast(
        agent,
        q_a.text,
        condition=f"Assume '{q_b.text}' resolved YES."
    )
    p_a_given_b0 = await elicit_forecast(
        agent,
        q_a.text,
        condition=f"Assume '{q_b.text}' resolved NO."
    )

    return {
        "pair_id": pair.get("id"),
        "category": pair.get("category"),
        "p_a": p_a,
        "p_a_given_b1": p_a_given_b1,
        "p_a_given_b0": p_a_given_b0,
        "sensitivity": abs(p_a_given_b1 - p_a_given_b0),
        "actual_a": res_a.value,
        "actual_b": res_b.value,
        # Conditional forecast given actual B
        "p_a_given_actual_b": p_a_given_b1 if res_b.value == 1.0 else p_a_given_b0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run conditional forecasting experiment")
    parser.add_argument(
        "--db",
        type=str,
        default="data/forecastbench.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="experiments/fb-conditional/pairs.json",
        help="Path to curated pairs file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/fb-conditional/results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for forecasting",
    )
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        print("Run generate_pairs.py first and curate the pairs.")
        return

    with open(pairs_path) as f:
        pairs_data = json.load(f)

    pairs = pairs_data.get("pairs", [])
    if not pairs:
        print("No pairs found in file. Add pairs to the 'pairs' array.")
        return

    print(f"Running experiment with {len(pairs)} pairs using {args.model}")
    print("TODO: Implement agent elicitation")

    # TODO: Initialize agent and run experiment
    # storage = SQLiteStorage(args.db)
    # results = []
    # for pair in pairs:
    #     result = await run_pair_experiment(agent, storage, pair)
    #     results.append(result)
    # await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
