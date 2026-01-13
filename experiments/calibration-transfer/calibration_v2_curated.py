#!/usr/bin/env python3
"""Calibration Transfer Experiment v2: 45 Curated Pairs.

Tests whether calibration helps more on "strong" correlation pairs than "none" pairs.
Uses existing Two-Stage predictions from scaffolding experiments.

Usage:
    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/calibration-transfer/calibration_v2_curated.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Paths
PAIRS_PATH = Path("experiments/fb-conditional/pairs_filtered.json")
PREDICTIONS_PATH = Path("experiments/fb-conditional/scaffolding/two-stage/results_20260112_113429.json")
RESULTS_DIR = Path("experiments/calibration-transfer/results")


def load_data():
    """Load pairs and predictions."""
    with open(PAIRS_PATH) as f:
        pairs_data = json.load(f)

    with open(PREDICTIONS_PATH) as f:
        predictions_data = json.load(f)

    # Index predictions by pair_id
    predictions_by_id = {}
    for result in predictions_data["results"]:
        predictions_by_id[result["pair_id"]] = result

    return pairs_data["pairs"], predictions_by_id


def compute_ground_truth(pairs, predictions_by_id):
    """Compute ground truth P(A|B=actual) for each pair.

    For Two-Stage predictions, we have:
    - p_a_given_b1: LLM's P(A|B=YES)
    - p_a_given_b0: LLM's P(A|B=NO)

    Ground truth:
    - If B resolved YES (1): P_correct(A|B=1) = resolution_a (0 or 1)
    - If B resolved NO (0): P_correct(A|B=0) = resolution_a (0 or 1)
    """
    results = []

    for pair in pairs:
        pair_id = f"{pair['id_a']}_{pair['id_b']}"

        if pair_id not in predictions_by_id:
            print(f"Warning: No prediction for pair {pair_id[:50]}...")
            continue

        pred = predictions_by_id[pair_id]

        # Skip pairs with errors or missing stage2
        if "error" in pred or "stage2" not in pred:
            print(f"Warning: Skipping pair with error/missing stage2: {pair_id[:30]}...")
            continue

        category = pair["category"]

        resolution_a = pair["resolution_a"]
        resolution_b = pair["resolution_b"]

        # Get the relevant LLM prediction based on B's resolution
        stage2 = pred["stage2"]

        # Handle baseline method (independent pairs get p_a only)
        if stage2["method"] == "baseline":
            p_llm = stage2["p_a"]
            p_correct = resolution_a
        elif resolution_b == 1.0:
            p_llm = stage2.get("p_a_given_b1")
            p_correct = resolution_a
        else:
            p_llm = stage2.get("p_a_given_b0")
            p_correct = resolution_a

        # Skip if prediction is None
        if p_llm is None:
            print(f"Warning: Skipping pair with None prediction: {pair_id[:30]}...")
            continue

        error = p_llm - p_correct

        results.append({
            "pair_id": pair_id,
            "category": category,
            "resolution_a": resolution_a,
            "resolution_b": resolution_b,
            "p_llm": p_llm,
            "p_correct": p_correct,
            "error": error,
            "method": pred["stage2"]["method"],
            "classification": pred["stage1"]["classification"],
        })

    return results


def analyze_by_group(results):
    """Analyze errors by correlation group."""
    groups = {"strong": [], "weak": [], "none": []}

    for r in results:
        groups[r["category"]].append(r)

    analysis = {}
    for group_name, group_results in groups.items():
        if not group_results:
            continue

        errors = [r["error"] for r in group_results]
        p_llm = [r["p_llm"] for r in group_results]
        p_correct = [r["p_correct"] for r in group_results]

        # Count usable pairs (where we can compute ground truth)
        n_total = len(group_results)
        n_a1 = sum(1 for r in group_results if r["resolution_a"] == 1.0)
        n_b1 = sum(1 for r in group_results if r["resolution_b"] == 1.0)

        analysis[group_name] = {
            "n_total": n_total,
            "n_a1": n_a1,
            "n_b1": n_b1,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "median_error": float(np.median(errors)),
            "mean_p_llm": float(np.mean(p_llm)),
            "mean_p_correct": float(np.mean(p_correct)),
            "errors": errors,
            "p_llm": p_llm,
            "p_correct": p_correct,
        }

    return analysis


def train_calibration(results, train_group="strong"):
    """Train calibration on a specific group."""
    # Filter to training group
    train_data = [r for r in results if r["category"] == train_group]

    if len(train_data) < 3:
        return {"error": f"Not enough data in {train_group} group"}

    X = np.array([[r["p_llm"]] for r in train_data])
    y = np.array([r["p_correct"] for r in train_data])

    model = LinearRegression()
    model.fit(X, y)

    alpha = model.coef_[0]
    beta = model.intercept_

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "train_size": len(train_data),
        "train_group": train_group,
    }


def validate_by_group(results, calibration):
    """Validate calibration on each group."""
    if "error" in calibration:
        return {"error": calibration["error"]}

    alpha = calibration["alpha"]
    beta = calibration["beta"]

    groups = {"strong": [], "weak": [], "none": []}
    for r in results:
        groups[r["category"]].append(r)

    validation = {}
    for group_name, group_results in groups.items():
        if len(group_results) < 2:
            continue

        p_llm = np.array([r["p_llm"] for r in group_results])
        p_correct = np.array([r["p_correct"] for r in group_results])

        # Uncalibrated Brier
        brier_uncal = np.mean((p_llm - p_correct) ** 2)

        # Calibrated predictions
        p_calibrated = np.clip(alpha * p_llm + beta, 0, 1)
        brier_cal = np.mean((p_calibrated - p_correct) ** 2)

        improvement = brier_uncal - brier_cal

        # Bootstrap for significance
        n_bootstrap = 1000
        improvements = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(p_llm), len(p_llm), replace=True)
            brier_u = np.mean((p_llm[idx] - p_correct[idx]) ** 2)
            brier_c = np.mean((p_calibrated[idx] - p_correct[idx]) ** 2)
            improvements.append(brier_u - brier_c)

        improvements = np.array(improvements)
        p_value = float(np.mean(improvements <= 0))

        validation[group_name] = {
            "n": len(group_results),
            "brier_uncalibrated": float(brier_uncal),
            "brier_calibrated": float(brier_cal),
            "improvement": float(improvement),
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    return validation


def main():
    print("=" * 70)
    print("CALIBRATION TRANSFER v2: 45 CURATED PAIRS")
    print("=" * 70)

    # Load data
    print("\n" + "-" * 70)
    print("STEP 1: Load data")
    print("-" * 70)
    pairs, predictions = load_data()
    print(f"Loaded {len(pairs)} pairs")
    print(f"Loaded {len(predictions)} predictions")

    # Compute ground truth
    print("\n" + "-" * 70)
    print("STEP 2: Compute ground truth")
    print("-" * 70)
    results = compute_ground_truth(pairs, predictions)
    print(f"Computed ground truth for {len(results)} pairs")

    # Count by category
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"By category: {by_cat}")

    # Analyze by group
    print("\n" + "-" * 70)
    print("STEP 3: Analyze errors by group")
    print("-" * 70)
    analysis = analyze_by_group(results)

    for group_name, stats in analysis.items():
        print(f"\n{group_name.upper()} (n={stats['n_total']}):")
        print(f"  Resolution A=1: {stats['n_a1']}, B=1: {stats['n_b1']}")
        print(f"  Mean error: {stats['mean_error']:.3f}")
        print(f"  Std error: {stats['std_error']:.3f}")
        print(f"  Mean P(LLM): {stats['mean_p_llm']:.3f}")
        print(f"  Mean P(correct): {stats['mean_p_correct']:.3f}")

    # Train calibration on strong pairs
    print("\n" + "-" * 70)
    print("STEP 4: Train calibration on strong pairs")
    print("-" * 70)
    calibration = train_calibration(results, train_group="strong")

    if "error" in calibration:
        print(f"Error: {calibration['error']}")
    else:
        print(f"Learned: calibrated = {calibration['alpha']:.3f} * raw + {calibration['beta']:.3f}")
        print(f"Trained on {calibration['train_size']} pairs")

    # Validate by group
    print("\n" + "-" * 70)
    print("STEP 5: Validate by group")
    print("-" * 70)
    validation = validate_by_group(results, calibration)

    if "error" in validation:
        print(f"Error: {validation['error']}")
    else:
        print("\n┌──────────┬────┬──────────────┬────────────┬────────────┬─────────┐")
        print("│ Group    │  N │ Uncal Brier  │ Cal Brier  │ Improvement│ p-value │")
        print("├──────────┼────┼──────────────┼────────────┼────────────┼─────────┤")
        for group_name in ["strong", "weak", "none"]:
            if group_name in validation:
                v = validation[group_name]
                print(f"│ {group_name:8} │ {v['n']:2} │ {v['brier_uncalibrated']:.4f}       │ {v['brier_calibrated']:.4f}     │ {v['improvement']:+.4f}    │ {v['p_value']:.3f}   │")
        print("└──────────┴────┴──────────────┴────────────┴────────────┴─────────┘")

    # Determine verdict
    print("\n" + "-" * 70)
    print("STEP 6: Verdict")
    print("-" * 70)

    if "error" in validation:
        verdict = "FAILURE - Could not validate"
    else:
        strong_v = validation.get("strong", {})
        none_v = validation.get("none", {})

        strong_improves = strong_v.get("improvement", 0) > 0
        strong_sig = strong_v.get("significant", False)
        none_improves = none_v.get("improvement", 0) > 0

        strong_better = strong_v.get("improvement", 0) > none_v.get("improvement", 0)

        if strong_sig and strong_better:
            verdict = "STRONG SIGNAL - Calibration significantly helps on strong pairs more than none"
        elif strong_improves and strong_better:
            verdict = "WEAK SIGNAL - Calibration helps more on strong pairs (not significant)"
        elif strong_improves:
            verdict = "AMBIGUOUS - Calibration helps on strong but not more than none"
        else:
            verdict = "NO SIGNAL - Calibration doesn't help on strong pairs"

    print(f"\n>>> {verdict} <<<")

    # Save results
    output = {
        "analysis": {k: {**v, "errors": None, "p_llm": None, "p_correct": None}
                     for k, v in analysis.items()},
        "calibration": calibration,
        "validation": validation,
        "verdict": verdict,
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "n_pairs": len(results),
        }
    }

    with open(RESULTS_DIR / "calibration_v2_curated.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR / 'calibration_v2_curated.json'}")

    # Return results for further processing
    return {
        "analysis": analysis,
        "calibration": calibration,
        "validation": validation,
        "verdict": verdict,
    }


if __name__ == "__main__":
    main()
