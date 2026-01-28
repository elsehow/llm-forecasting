#!/usr/bin/env python3
"""
Curate additional LLM-classified pairs for within-domain validation.

Expands on the 34 curated pairs from curate_nontrivial.py by:
1. Targeting specific semantic domains (fed_rates, election_2028, iran)
2. Using LLM classification to identify real relationships
3. Building larger samples for within-domain correlation analysis

Goal: Increase N for within-domain validation:
- fed_rates: 10 → 25+ pairs
- election_2028: 8 → 20+ pairs
- iran (new): 0 → 8+ pairs

Usage:
    uv run python experiments/question-generation/within-category/curate_semantic_pairs.py
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
from dotenv import load_dotenv
import litellm

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
VOI_DIR = Path(__file__).parent.parent / "voi-validation"
OUTPUT_DIR = Path(__file__).parent / "data"

# Thresholds
MIN_VOLUME = 50_000  # Lower than curate_nontrivial to get more pairs
RESOLUTION_THRESHOLD_HIGH = 0.90
RESOLUTION_THRESHOLD_LOW = 0.10
MIN_OTHER_PROB = 0.10
MAX_OTHER_PROB = 0.90

MODEL = "claude-sonnet-4-20250514"

# Semantic domain patterns - more specific than curate_nontrivial
SEMANTIC_DOMAINS = {
    "fed_rates": {
        "patterns": [
            r"\bfed\b.*(?:rate|interest|bps|cut|increase|decrease|meeting)",
            r"\b(?:interest rate|rate cut|rate hike)",
            r"(?:january|march|may|june|september|november|december) 202\d.*meeting",
            r"fed\s+(?:chair|chairman)",
            r"trump.*(?:nominate|nominee).*fed",
        ],
        "exclude": [],
        "description": "Federal Reserve monetary policy and rate decisions",
    },
    "election_2028": {
        "patterns": [
            r"2028.*(?:presidential|nomination|president|election)",
            r"(?:democratic|republican).*nomination.*2028",
            r"win.*2028.*(?:presidential|election|nomination)",
        ],
        "exclude": [],
        "description": "2028 US Presidential election",
    },
    "iran": {
        "patterns": [
            r"\biran\b",
            r"\bkhamenei\b",
            r"\btehran\b",
            r"iranian regime",
            r"(?:us|israel).*strike.*iran",
        ],
        "exclude": [],
        "description": "Iran geopolitical events",
    },
    "russia_ukraine": {
        "patterns": [
            r"\brussia\b.*\bukraine\b",
            r"\bukraine\b.*\brussia\b",
            r"ceasefire",
            r"peace.*(?:deal|treaty|agreement)",
            r"\bputin\b",
            r"\bzelensky\b",
        ],
        "exclude": [],
        "description": "Russia-Ukraine conflict",
    },
    "bitcoin": {
        "patterns": [
            r"\bbitcoin\b",
            r"\bbtc\b",
        ],
        "exclude": [
            r"\bxrp\b",
            r"\bethereum\b",
            r"\beth\b",
            r"\bsolana\b",
        ],
        "description": "Bitcoin price movements",
    },
}


def extract_domain(question: str) -> str | None:
    """Extract semantic domain from question. Returns None if no match."""
    question_lower = question.lower()

    for domain, config in SEMANTIC_DOMAINS.items():
        # Check excludes first
        excluded = False
        for exclude_pattern in config.get("exclude", []):
            if re.search(exclude_pattern, question_lower, re.IGNORECASE):
                excluded = True
                break
        if excluded:
            continue

        # Check includes
        for pattern in config["patterns"]:
            if re.search(pattern, question_lower, re.IGNORECASE):
                return domain

    return None


def load_data():
    """Load markets, price histories, and pairs."""
    with open(DATA_DIR / "markets.json") as f:
        markets = {m["condition_id"]: m for m in json.load(f)}

    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = data

    with open(DATA_DIR / "pairs.json") as f:
        pairs = json.load(f)

    # Load existing curated pairs to avoid duplicates
    existing_ids = set()
    curated_path = VOI_DIR / "curated_pairs_nontrivial.json"
    if curated_path.exists():
        with open(curated_path) as f:
            existing = json.load(f)
        for p in existing.get("curated_pairs", []):
            key = tuple(sorted([p["condition_id_a"], p["condition_id_b"]]))
            existing_ids.add(key)

    return markets, histories, pairs, existing_ids


def detect_resolution(history: dict) -> tuple[str, datetime, int, float] | None:
    """Returns (outcome, date, resolution_idx, price_before)."""
    candles = history.get("candles", [])
    if len(candles) < 3:
        return None

    final_price = candles[-1]["close"]
    if final_price > RESOLUTION_THRESHOLD_HIGH:
        outcome = "YES"
    elif final_price < RESOLUTION_THRESHOLD_LOW:
        outcome = "NO"
    else:
        return None

    threshold = RESOLUTION_THRESHOLD_HIGH if outcome == "YES" else RESOLUTION_THRESHOLD_LOW
    for i, candle in enumerate(candles):
        if (outcome == "YES" and candle["close"] >= threshold) or \
           (outcome == "NO" and candle["close"] <= threshold):
            ts = candle["timestamp"]
            resolution_date = datetime.fromtimestamp(ts, tz=timezone.utc)
            if resolution_date.date() >= MODEL_KNOWLEDGE_CUTOFF:
                # Get price before resolution
                if i >= 3:
                    price_before = np.mean([c["close"] for c in candles[max(0, i-3):i]])
                else:
                    price_before = candles[0]["close"] if candles else 0.5
                return outcome, resolution_date, i, price_before
            return None
    return None


def get_other_price_at_resolution(other_candles: list[dict], res_ts: int) -> float | None:
    """Get other market's price at resolution time."""
    for c in reversed(other_candles):
        if c["timestamp"] <= res_ts:
            return c["close"]
    return other_candles[0]["close"] if other_candles else None


CLASSIFICATION_PROMPT = """Classify this prediction market pair as having a REAL relationship or SPURIOUS correlation.

DOMAIN: {domain}
Question A: {question_a}
Question B: {question_b}
Observed ρ: {rho:.2f}

Categories:
1. same_event_different_timeline - Same event, different time horizons (e.g., "X by March" vs "X by June")
2. sequential_prerequisite - One is prerequisite for other (e.g., "win primary" → "win election")
3. mutually_exclusive - Same race/competition, only one can win (e.g., two candidates for same position)
4. causal_relationship - Clear causal mechanism (e.g., Fed rate decision → market response)
5. shared_driver - Both driven by same factor (e.g., economic conditions affect both)
6. spurious - No real relationship (coincidental correlation)
7. weak_or_unclear - Might have relationship but too indirect

For this to be a REAL relationship within the {domain} domain, there must be a clear, direct mechanism.

JSON only: {{"category": "...", "confidence": "high/medium/low", "rationale": "..."}}
"""


async def classify_pair(domain: str, q_a: str, q_b: str, rho: float) -> dict:
    """Classify pair using LLM."""
    prompt = CLASSIFICATION_PROMPT.format(
        domain=domain, question_a=q_a, question_b=q_b, rho=rho
    )
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content
        if "```" in text:
            text = text.split("```")[1].split("```")[0]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        return {"category": "error", "confidence": "none", "rationale": str(e)}


async def main():
    print("=" * 70)
    print("CURATE SEMANTIC DOMAIN PAIRS")
    print("=" * 70)
    print(f"\nTarget domains: {list(SEMANTIC_DOMAINS.keys())}")
    print(f"Resolution after: {MODEL_KNOWLEDGE_CUTOFF}")
    print(f"Other market probability: {MIN_OTHER_PROB} - {MAX_OTHER_PROB}")

    markets, histories, pairs, existing_ids = load_data()
    print(f"\nLoaded {len(markets)} markets, {len(histories)} histories, {len(pairs)} pairs")
    print(f"Existing curated pairs to exclude: {len(existing_ids)}")

    # Find candidates by domain
    domain_candidates = defaultdict(list)

    for pair in pairs:
        q_a = pair["market_a"]["question"]
        q_b = pair["market_b"]["question"]
        cond_a = pair["market_a"]["condition_id"]
        cond_b = pair["market_b"]["condition_id"]
        rho = pair["rho"]

        if np.isnan(rho):
            continue

        # Skip existing curated pairs
        key = tuple(sorted([cond_a, cond_b]))
        if key in existing_ids:
            continue

        # Volume filter
        market_a = markets.get(cond_a)
        market_b = markets.get(cond_b)
        if not market_a or not market_b:
            continue
        if market_a.get("volume_total", 0) < MIN_VOLUME or market_b.get("volume_total", 0) < MIN_VOLUME:
            continue

        # Get histories
        hist_a = histories.get(cond_a)
        hist_b = histories.get(cond_b)
        if not hist_a or not hist_b:
            continue

        # Both must be in SAME domain
        domain_a = extract_domain(q_a)
        domain_b = extract_domain(q_b)

        if domain_a is None or domain_b is None:
            continue
        if domain_a != domain_b:
            continue

        domain = domain_a

        # Check for resolution
        res_a = detect_resolution(hist_a)
        res_b = detect_resolution(hist_b)

        if not res_a and not res_b:
            continue

        # Determine which resolved
        if res_a:
            resolution_outcome, resolution_date, res_idx, resolved_price_before = res_a
            res_ts = hist_a["candles"][res_idx]["timestamp"]
            other_candles = hist_b.get("candles", [])
            other_price = get_other_price_at_resolution(other_candles, res_ts)
            resolved_q = q_a
            other_q = q_b
            resolved_cond = cond_a
            other_cond = cond_b
        else:
            resolution_outcome, resolution_date, res_idx, resolved_price_before = res_b
            res_ts = hist_b["candles"][res_idx]["timestamp"]
            other_candles = hist_a.get("candles", [])
            other_price = get_other_price_at_resolution(other_candles, res_ts)
            resolved_q = q_b
            other_q = q_a
            resolved_cond = cond_b
            other_cond = cond_a

        if other_price is None:
            continue

        # Key filter: other market must have non-trivial probability
        if not (MIN_OTHER_PROB <= other_price <= MAX_OTHER_PROB):
            continue

        domain_candidates[domain].append({
            "question_a": q_a,
            "question_b": q_b,
            "condition_id_a": cond_a,
            "condition_id_b": cond_b,
            "rho": rho,
            "domain": domain,
            "resolved_question": resolved_q,
            "other_question": other_q,
            "resolved_cond": resolved_cond,
            "other_cond": other_cond,
            "resolution_outcome": resolution_outcome,
            "resolution_date": resolution_date.isoformat(),
            "resolved_price_before": resolved_price_before,
            "other_price_at_resolution": other_price,
        })

    print("\n" + "-" * 70)
    print("CANDIDATES BY DOMAIN")
    print("-" * 70)

    for domain, candidates in sorted(domain_candidates.items()):
        print(f"\n{domain}: {len(candidates)} candidates")
        # Sort by |rho| to show strongest correlations
        candidates.sort(key=lambda p: -abs(p["rho"]))
        for c in candidates[:3]:
            print(f"  ρ={c['rho']:+.2f} | {c['question_a'][:40]}...")
            print(f"         | {c['question_b'][:40]}...")

    # Classify all candidates
    print("\n" + "=" * 70)
    print("LLM CLASSIFICATION")
    print("=" * 70)

    all_curated = defaultdict(list)
    real_categories = {
        "same_event_different_timeline",
        "sequential_prerequisite",
        "mutually_exclusive",
        "causal_relationship",
        "shared_driver",
    }

    for domain, candidates in domain_candidates.items():
        if not candidates:
            continue

        print(f"\nClassifying {len(candidates)} {domain} candidates...")

        batch_size = 10
        classifications = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            batch_results = await asyncio.gather(*[
                classify_pair(domain, c["question_a"], c["question_b"], c["rho"])
                for c in batch
            ])
            classifications.extend(batch_results)
            print(f"  Processed {min(i+batch_size, len(candidates))}/{len(candidates)}")

        # Filter to real relationships
        for cand, classif in zip(candidates, classifications):
            if classif.get("category") in real_categories:
                if classif.get("confidence") in ("high", "medium"):
                    cand["classification"] = classif
                    all_curated[domain].append(cand)

        print(f"  Real relationships: {len(all_curated[domain])}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = 0
    for domain in SEMANTIC_DOMAINS.keys():
        n = len(all_curated.get(domain, []))
        total += n
        desc = SEMANTIC_DOMAINS[domain]["description"]
        print(f"  {domain}: {n} pairs ({desc})")
    print(f"\n  TOTAL: {total} new curated pairs")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "experiment": "semantic_domain_curation",
            "generated_at": datetime.now().isoformat(),
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "domains": list(SEMANTIC_DOMAINS.keys()),
        },
        "summary": {domain: len(pairs) for domain, pairs in all_curated.items()},
        "pairs_by_domain": dict(all_curated),
    }

    output_path = OUTPUT_DIR / "semantic_domain_pairs.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Show curated pairs by domain
    for domain, pairs in all_curated.items():
        if not pairs:
            continue
        print(f"\n{'-' * 70}")
        print(f"{domain.upper()} CURATED PAIRS ({len(pairs)})")
        print(f"{'-' * 70}")
        for i, p in enumerate(pairs[:10]):
            c = p["classification"]
            print(f"\n{i+1}. [{c['category']}] ρ={p['rho']:+.2f}")
            print(f"   Resolved: {p['resolved_question'][:60]}...")
            print(f"   Other:    {p['other_question'][:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
