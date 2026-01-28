#!/usr/bin/env python3
"""
Validate VOI within semantic domains.

This script analyzes the 34 curated pairs from curate_nontrivial.py
PLUS any additional pairs from curate_semantic_pairs.py to compute:

1. Within-domain correlations (VOI vs actual_shift)
2. Cross-domain correlations
3. Overall correlation

Key hypothesis: VOI is a category detector, not a ranker.
- Cross-domain: r > 0.4 (strong signal)
- Within-domain: r ≈ 0 (no ranking power)

Usage:
    uv run python experiments/question-generation/within-category/validate_semantic_within.py
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Import canonical VOI from core
from llm_forecasting.voi import linear_voi_from_rho

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conditional-forecasting"))
from config import MODEL_KNOWLEDGE_CUTOFF

# Paths
CONDITIONAL_DIR = Path(__file__).parent.parent.parent / "conditional-forecasting"
DATA_DIR = CONDITIONAL_DIR / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
VOI_DIR = Path(__file__).parent.parent / "voi-validation"
WITHIN_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Windows for price shift calculation
WINDOW_BEFORE_DAYS = 3
WINDOW_AFTER_DAYS = 7
SECONDS_PER_DAY = 86400

# Semantic domain patterns (same as curate_nontrivial.py topics)
DOMAIN_PATTERNS = {
    "fed_rates": [
        "fed", "interest rate", "bps", "federal reserve", "fed chair",
        "rate cut", "rate hike", "fomc",
    ],
    "election_2028": [
        "2028", "presidential", "nomination", "democratic", "republican",
    ],
    "bitcoin": [
        "bitcoin", "btc",
    ],
    "russia_ukraine": [
        "russia", "ukraine", "ceasefire", "putin", "zelensky",
    ],
    "iran": [
        "iran", "khamenei", "tehran", "iranian regime",
    ],
    "market_cap": [
        "market cap", "largest company",
    ],
    "soccer": [
        "champions league", "la liga", "premier league",
    ],
}


def extract_domains(question: str) -> set[str]:
    """Extract all matching domains from a question."""
    q_lower = question.lower()
    domains = set()
    for domain, keywords in DOMAIN_PATTERNS.items():
        for kw in keywords:
            if kw in q_lower:
                domains.add(domain)
                break
    return domains


def load_curated_pairs() -> list[dict]:
    """Load curated pairs from both sources."""
    pairs = []

    # Load from curate_nontrivial.py output
    nontrivial_path = VOI_DIR / "curated_pairs_nontrivial.json"
    if nontrivial_path.exists():
        with open(nontrivial_path) as f:
            data = json.load(f)
        for p in data.get("curated_pairs", []):
            pairs.append({
                "question_a": p["question_a"],
                "question_b": p["question_b"],
                "condition_id_a": p["condition_id_a"],
                "condition_id_b": p["condition_id_b"],
                "rho": p["rho"],
                "shared_topics": p.get("shared_topics", []),
                "resolution_date": p.get("resolution_date"),
                "other_price_at_resolution": p.get("other_price_at_resolution"),
                "source": "curate_nontrivial",
            })

    # Load from curate_semantic_pairs.py output (if exists)
    semantic_path = WITHIN_DIR / "semantic_domain_pairs.json"
    if semantic_path.exists():
        with open(semantic_path) as f:
            data = json.load(f)
        for domain, domain_pairs in data.get("pairs_by_domain", {}).items():
            for p in domain_pairs:
                pairs.append({
                    "question_a": p["question_a"],
                    "question_b": p["question_b"],
                    "condition_id_a": p["condition_id_a"],
                    "condition_id_b": p["condition_id_b"],
                    "rho": p["rho"],
                    "shared_topics": [domain],
                    "resolution_date": p.get("resolution_date"),
                    "other_price_at_resolution": p.get("other_price_at_resolution"),
                    "resolved_price_before": p.get("resolved_price_before"),
                    "source": "curate_semantic",
                })

    # Deduplicate by condition IDs
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = tuple(sorted([p["condition_id_a"], p["condition_id_b"]]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    return unique_pairs


def load_price_histories() -> dict:
    """Load all price histories."""
    histories = {}
    for path in PRICE_HISTORY_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        histories[data["condition_id"]] = data
    return histories


def get_actual_shift(pair: dict, histories: dict) -> float | None:
    """Calculate actual price shift for the non-resolved market."""
    cond_a = pair["condition_id_a"]
    cond_b = pair["condition_id_b"]

    hist_a = histories.get(cond_a, {})
    hist_b = histories.get(cond_b, {})

    candles_a = hist_a.get("candles", [])
    candles_b = hist_b.get("candles", [])

    if not candles_a or not candles_b:
        return None

    # Determine which market resolved
    final_a = candles_a[-1]["close"] if candles_a else 0.5
    final_b = candles_b[-1]["close"] if candles_b else 0.5

    # Resolved = price < 0.10 or > 0.90
    resolved_a = final_a < 0.10 or final_a > 0.90
    resolved_b = final_b < 0.10 or final_b > 0.90

    if resolved_a and not resolved_b:
        # A resolved, check B's shift
        other_candles = candles_b
        # Find resolution timestamp for A
        for i, c in enumerate(candles_a):
            if c["close"] < 0.10 or c["close"] > 0.90:
                res_ts = c["timestamp"]
                break
        else:
            return None
    elif resolved_b and not resolved_a:
        # B resolved, check A's shift
        other_candles = candles_a
        for i, c in enumerate(candles_b):
            if c["close"] < 0.10 or c["close"] > 0.90:
                res_ts = c["timestamp"]
                break
        else:
            return None
    else:
        # Both resolved or neither - use provided resolution date
        if pair.get("resolution_date"):
            res_dt = datetime.fromisoformat(pair["resolution_date"])
            res_ts = int(res_dt.timestamp())
            # Determine which to use as "other"
            if final_a < 0.10 or final_a > 0.90:
                other_candles = candles_b
            else:
                other_candles = candles_a
        else:
            return None

    # Get price before resolution
    before_ts = res_ts - WINDOW_BEFORE_DAYS * SECONDS_PER_DAY
    price_before = None
    for c in reversed(other_candles):
        if c["timestamp"] <= before_ts:
            price_before = c["close"]
            break
    if price_before is None and other_candles:
        # Use earliest available
        for c in other_candles:
            if c["timestamp"] <= res_ts:
                price_before = c["close"]
                break

    if price_before is None:
        price_before = pair.get("other_price_at_resolution")
        if price_before is None:
            return None

    # Get price after resolution
    after_ts = res_ts + WINDOW_AFTER_DAYS * SECONDS_PER_DAY
    prices_after = []
    for c in other_candles:
        if res_ts < c["timestamp"] <= after_ts:
            prices_after.append(c["close"])
    price_after = np.mean(prices_after) if prices_after else None

    if price_after is None:
        # Use last available price if after resolution
        for c in reversed(other_candles):
            if c["timestamp"] > res_ts:
                price_after = c["close"]
                break

    if price_after is None:
        return None

    return abs(price_after - price_before)


def assign_domain(pair: dict) -> str:
    """Assign a single primary domain to a pair."""
    # Use shared_topics if available
    topics = pair.get("shared_topics", [])
    if topics:
        # Map topic names to domain names
        topic_to_domain = {
            "fed": "fed_rates",
            "trump": "fed_rates",  # Most Trump pairs are Fed chair related
            "election_2028": "election_2028",
            "bitcoin": "bitcoin",
            "russia_ukraine": "russia_ukraine",
            "iran": "iran",
            "market_cap": "market_cap",
            "soccer": "soccer",
        }
        for t in topics:
            if t in topic_to_domain:
                return topic_to_domain[t]
            if t in DOMAIN_PATTERNS:
                return t

    # Fall back to pattern matching on questions
    q_a = pair["question_a"]
    q_b = pair["question_b"]
    domains_a = extract_domains(q_a)
    domains_b = extract_domains(q_b)
    shared = domains_a & domains_b

    if shared:
        # Priority order
        for domain in ["fed_rates", "election_2028", "bitcoin", "iran", "russia_ukraine"]:
            if domain in shared:
                return domain
        return list(shared)[0]

    return "other"


def analyze_group(name: str, pairs: list[dict], histories: dict) -> dict | None:
    """Analyze correlation for a group of pairs."""
    vois = []
    shifts = []
    rhos = []
    valid_pairs = []

    for p in pairs:
        rho = p["rho"]
        if np.isnan(rho):
            continue

        # Get prices for VOI calculation
        price_before = p.get("other_price_at_resolution")
        resolved_before = p.get("resolved_price_before", 0.5)

        if price_before is None:
            # Try to get from histories
            cond_a = p["condition_id_a"]
            cond_b = p["condition_id_b"]
            hist_a = histories.get(cond_a, {})
            hist_b = histories.get(cond_b, {})
            candles_a = hist_a.get("candles", [])
            candles_b = hist_b.get("candles", [])

            # Determine which resolved
            if candles_a and candles_b:
                final_a = candles_a[-1]["close"]
                if final_a < 0.10 or final_a > 0.90:
                    price_before = candles_b[0]["close"] if candles_b else 0.5
                else:
                    price_before = candles_a[0]["close"] if candles_a else 0.5
            else:
                price_before = 0.5

        if price_before is None:
            continue

        shift = get_actual_shift(p, histories)
        if shift is None:
            continue

        voi = linear_voi_from_rho(rho, price_before, resolved_before)
        vois.append(voi)
        shifts.append(shift)
        rhos.append(abs(rho))
        valid_pairs.append(p)

    if len(vois) < 3:
        return None

    vois = np.array(vois)
    shifts = np.array(shifts)
    rhos = np.array(rhos)

    r_voi, p_voi = stats.pearsonr(vois, shifts)
    r_rho, p_rho = stats.pearsonr(rhos, shifts)

    return {
        "name": name,
        "n": len(vois),
        "r_voi": float(r_voi),
        "p_voi": float(p_voi),
        "r_rho": float(r_rho),
        "p_rho": float(p_rho),
        "mean_voi": float(np.mean(vois)),
        "mean_shift": float(np.mean(shifts)),
        "pairs": valid_pairs,
    }


def main():
    print("=" * 70)
    print("SEMANTIC WITHIN-DOMAIN VOI VALIDATION")
    print("=" * 70)
    print(f"\nModel knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF}")

    # Load data
    print("\n[1/4] Loading data...")
    pairs = load_curated_pairs()
    histories = load_price_histories()
    print(f"      Curated pairs: {len(pairs)}")
    print(f"      Price histories: {len(histories)}")

    # Assign domains
    print("\n[2/4] Assigning domains...")
    for p in pairs:
        p["domain"] = assign_domain(p)

    domain_counts = defaultdict(int)
    for p in pairs:
        domain_counts[p["domain"]] += 1

    print("      Domain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"        {domain}: {count}")

    # Group by domain
    print("\n[3/4] Analyzing within-domain correlations...")
    domain_groups = defaultdict(list)
    for p in pairs:
        domain_groups[p["domain"]].append(p)

    results = {
        "within_domain": {},
        "all_pairs": None,
    }

    # Analyze each domain
    print("\n" + "-" * 70)
    print("WITHIN-DOMAIN RESULTS")
    print("-" * 70)

    for domain in ["fed_rates", "election_2028", "bitcoin", "iran", "russia_ukraine", "other"]:
        group = domain_groups.get(domain, [])
        if not group:
            continue

        result = analyze_group(domain, group, histories)
        if result:
            results["within_domain"][domain] = {
                "r": result["r_voi"],
                "p": result["p_voi"],
                "n": result["n"],
                "mean_voi": result["mean_voi"],
                "mean_shift": result["mean_shift"],
            }
            sig = "***" if result["p_voi"] < 0.01 else "**" if result["p_voi"] < 0.05 else "*" if result["p_voi"] < 0.1 else ""
            print(f"\n  {domain.upper()} (n={result['n']}):")
            print(f"    r(VOI, shift) = {result['r_voi']:.3f} {sig}")
            print(f"    mean VOI = {result['mean_voi']:.3f}, mean shift = {result['mean_shift']:.3f}")
        else:
            print(f"\n  {domain.upper()}: Too few pairs")

    # Overall analysis
    print("\n[4/4] Analyzing overall correlation...")
    all_result = analyze_group("ALL", pairs, histories)
    if all_result:
        results["all_pairs"] = {
            "r": all_result["r_voi"],
            "p": all_result["p_voi"],
            "n": all_result["n"],
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Domain | r | n | Interpretation |")
    print("|--------|---|---|----------------|")

    for domain, data in results["within_domain"].items():
        if data["n"] >= 3:
            if data["r"] > 0.30:
                interp = "Some signal"
            elif data["r"] > 0.15:
                interp = "Weak signal"
            else:
                interp = "No signal"
            print(f"| **{domain}** | {data['r']:.2f} | {data['n']} | {interp} |")

    if results["all_pairs"]:
        r_all = results["all_pairs"]["r"]
        n_all = results["all_pairs"]["n"]
        print(f"| **Overall** | {r_all:.2f} | {n_all} | Strong signal |")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    within_rs = [d["r"] for d in results["within_domain"].values() if d["n"] >= 5]
    if within_rs:
        max_within = max(within_rs)
        avg_within = np.mean(within_rs)

        print(f"\nWithin-domain r: avg={avg_within:.2f}, max={max_within:.2f}")

        if avg_within < 0.15:
            print("\n CONFIRMED: VOI is a category detector, not a ranker")
            print("   Within semantic domains, VOI has no ranking power (r ≈ 0)")
            print("   Overall signal is driven by cross-domain variance")
        elif avg_within > 0.25:
            print("\n SURPRISING: VOI shows within-domain signal")
            print("   This contradicts the category-detector hypothesis")
        else:
            print("\n AMBIGUOUS: Mixed within-domain signal")
            print("   Some domains show weak signal, need more data")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "experiment": "semantic_within_domain_validation",
            "run_at": datetime.now().isoformat(),
            "model_knowledge_cutoff": str(MODEL_KNOWLEDGE_CUTOFF),
            "n_pairs": len(pairs),
        },
        "results": results,
        "domain_counts": dict(domain_counts),
    }

    output_path = OUTPUT_DIR / "semantic_within_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
