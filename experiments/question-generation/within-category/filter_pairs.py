#!/usr/bin/env python3
"""
Filter Polymarket pairs by category.

Categorizes markets using keyword matching and filters pairs where
both markets belong to the same category.

Categories:
- Fed/Monetary: Fed rates, FOMC, monetary policy
- Politics: Elections, nominations, political events
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Category definitions with keyword patterns
CATEGORIES = {
    "crypto": {
        "description": "Bitcoin, Ethereum, crypto prices",
        "patterns": [
            r"\bBitcoin\b",
            r"\bBTC\b",
            r"\bEthereum\b",
            r"\bETH\b",
            r"\bSolana\b",
            r"\bSOL\b",
            r"\bcrypto",
        ],
        "exclude_patterns": []
    },
    "fed_monetary": {
        "description": "Fed rates, FOMC meetings, monetary policy",
        "patterns": [
            r"\bFed\b.*\b(rate|interest|bps|meeting)",
            r"\bFOMC\b",
            r"\binterest rate",
            r"\brate cut",
            r"\brate hike",
            r"\bbps\b.*meeting",
            r"monetary policy",
            r"Federal Reserve",
        ],
        "exclude_patterns": [
            r"Fed chair",
            r"Fed nominee",
            r"Fed president",
        ]
    },
    "politics": {
        "description": "Elections, nominations, political events",
        "patterns": [
            r"\belection\b",
            r"\bpresident",
            r"\bnominate",
            r"\bcandidate",
            r"\bvote\b",
            r"\bpoll\b",
            r"\bCongress",
            r"\bSenate",
            r"\bHouse\b.*\b(vote|pass|bill)",
            r"\bRepublican",
            r"\bDemocrat",
            r"\bTrump\b",
            r"\bBiden\b",
        ],
        "exclude_patterns": []
    },
    "sports": {
        "description": "Sports outcomes",
        "patterns": [
            r"\bwin on 2026\b",
            r"\bvs\.?\b",
            r"\bNFL\b",
            r"\bNBA\b",
            r"\bMLB\b",
            r"\bNHL\b",
            r"\bFC\b",
            r"Premier League",
            r"Champions League",
        ],
        "exclude_patterns": []
    }
}


def categorize_market(question: str) -> str | None:
    """
    Categorize a market by its question text.
    Returns category name or None if no match.
    """
    question_lower = question.lower()

    for cat_name, cat_def in CATEGORIES.items():
        # Check exclude patterns first
        excluded = False
        for pattern in cat_def.get("exclude_patterns", []):
            if re.search(pattern, question, re.IGNORECASE):
                excluded = True
                break

        if excluded:
            continue

        # Check include patterns
        for pattern in cat_def["patterns"]:
            if re.search(pattern, question, re.IGNORECASE):
                return cat_name

    return None


def main():
    print("=" * 70)
    print("WITHIN-CATEGORY PAIR FILTERING")
    print("=" * 70)

    # Load pairs
    print("\n[1/3] Loading pairs...")
    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)
    print(f"      Total pairs: {len(pairs)}")

    # Categorize all markets mentioned in pairs
    print("\n[2/3] Categorizing markets...")
    market_categories = {}
    category_counts = defaultdict(int)

    for pair in pairs:
        for market in [pair["market_a"], pair["market_b"]]:
            cond_id = market["condition_id"]
            if cond_id not in market_categories:
                cat = categorize_market(market["question"])
                market_categories[cond_id] = cat
                if cat:
                    category_counts[cat] += 1

    print(f"      Markets categorized:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"        {cat}: {count}")
    print(f"        uncategorized: {sum(1 for c in market_categories.values() if c is None)}")

    # Filter within-category pairs
    print("\n[3/3] Filtering within-category pairs...")
    within_category_pairs = defaultdict(list)

    for pair in pairs:
        cat_a = market_categories.get(pair["market_a"]["condition_id"])
        cat_b = market_categories.get(pair["market_b"]["condition_id"])

        # Only include if both markets are in the same category
        if cat_a and cat_b and cat_a == cat_b:
            # Skip pairs with NaN rho
            if pair.get("rho") is None or (isinstance(pair.get("rho"), float) and pair["rho"] != pair["rho"]):
                continue
            within_category_pairs[cat_a].append(pair)

    # Output results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cat_name, cat_pairs in within_category_pairs.items():
        print(f"\n{cat_name}:")
        print(f"  Within-category pairs: {len(cat_pairs)}")

        # Show rho distribution
        rhos = [abs(p["rho"]) for p in cat_pairs]
        if rhos:
            print(f"  |rho| mean: {sum(rhos)/len(rhos):.3f}")
            print(f"  |rho| range: [{min(rhos):.3f}, {max(rhos):.3f}]")

        # Show example pairs
        print(f"\n  Example pairs:")
        for i, pair in enumerate(cat_pairs[:3]):
            print(f"    {i+1}. rho={pair['rho']:.2f}")
            print(f"       A: {pair['market_a']['question'][:60]}...")
            print(f"       B: {pair['market_b']['question'][:60]}...")

        # Save to file
        output_path = OUTPUT_DIR / f"{cat_name}_pairs.json"
        with open(output_path, "w") as f:
            json.dump(cat_pairs, f, indent=2)
        print(f"\n  Saved to {output_path}")

    # Summary
    print("\n" + "-" * 70)
    total_within = sum(len(pairs) for pairs in within_category_pairs.values())
    print(f"Total within-category pairs: {total_within}")
    print(f"Primary target (fed_monetary): {len(within_category_pairs.get('fed_monetary', []))}")


if __name__ == "__main__":
    main()
