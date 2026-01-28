"""Validate Russell 2000 forward-looking predictions.

Run any time after dates have resolved. Fetches stock prices automatically.

Usage:
    uv run python experiments/question-generation/paper-trading/validate_russell_forward.py

Output:
    - Prints correlation between VOI and |return| for resolved dates
    - Saves detailed results to results/russell_forward_validation.json
"""

import json
from datetime import datetime
from pathlib import Path

import yfinance as yf
import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"


def load_predictions() -> dict:
    """Load the forward-looking predictions."""
    with open(RESULTS_DIR / "russell_forward_cruxes.json") as f:
        return json.load(f)


def fetch_stock_return(ticker: str, date: str) -> dict | None:
    """Fetch open/close for a stock on a specific date.

    Returns dict with open, close, return, went_up, or None if not available.
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch a few days around the target to handle weekends/holidays
        hist = stock.history(start=date, period="5d")

        if hist.empty:
            return None

        # Get the first available day (should be target date if market was open)
        row = hist.iloc[0]
        actual_date = hist.index[0].strftime("%Y-%m-%d")

        # Check if we got the right date
        if actual_date != date:
            # Date might have been a holiday, skip
            return None

        open_price = row["Open"]
        close_price = row["Close"]
        ret = (close_price - open_price) / open_price

        return {
            "date": actual_date,
            "open": float(open_price),
            "close": float(close_price),
            "return": float(ret),
            "went_up": close_price > open_price,
        }
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def validate():
    """Run validation on all resolved dates."""
    print("=== Russell 2000 Forward-Looking Validation ===")
    print(f"Run time: {datetime.now().isoformat()}")

    data = load_predictions()
    predictions = data["predictions"]

    print(f"\nLoaded {len(predictions)} stock-days")
    print(f"Trading days: {data['metadata']['trading_days'][0]} to {data['metadata']['trading_days'][-1]}")

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Today: {today}")

    # Fetch actual returns for resolved dates
    results = []
    resolved_count = 0
    pending_count = 0

    for pred in predictions:
        ticker = pred["ticker"]
        date = pred["date"]

        # Check if date has passed
        if date >= today:
            pending_count += 1
            print(f"  {ticker} {date}: PENDING (future date)")
            continue

        # Fetch actual return
        actual = fetch_stock_return(ticker, date)

        if actual is None:
            print(f"  {ticker} {date}: NO DATA (holiday or error)")
            continue

        resolved_count += 1
        print(f"  {ticker} {date}: return={actual['return']:.4f} ({'UP' if actual['went_up'] else 'DOWN'})")

        # Collect VOI and return data
        result = {
            "ticker": ticker,
            "date": date,
            "actual_return": actual["return"],
            "abs_return": abs(actual["return"]),
            "went_up": actual["went_up"],
            "baseline_cruxes": [],
            "enhanced_cruxes": [],
        }

        for crux in pred.get("baseline_cruxes", []):
            if "voi_linear" in crux:
                result["baseline_cruxes"].append({
                    "text": crux.get("text", ""),
                    "voi": crux["voi_linear"],
                })

        for crux in pred.get("enhanced_cruxes", []):
            if "voi_linear" in crux:
                result["enhanced_cruxes"].append({
                    "text": crux.get("text", ""),
                    "voi": crux["voi_linear"],
                })

        results.append(result)

    print(f"\n=== Summary ===")
    print(f"Resolved: {resolved_count}")
    print(f"Pending: {pending_count}")

    if resolved_count < 3:
        print("\nNot enough resolved dates for correlation analysis.")
        print("Re-run after more dates have passed.")
        return

    # Compute correlations
    print("\n=== Correlation Analysis ===")

    # Aggregate VOI per stock-day (mean of cruxes)
    baseline_vois = []
    enhanced_vois = []
    abs_returns = []

    for r in results:
        abs_returns.append(r["abs_return"])

        if r["baseline_cruxes"]:
            baseline_vois.append(np.mean([c["voi"] for c in r["baseline_cruxes"]]))
        else:
            baseline_vois.append(np.nan)

        if r["enhanced_cruxes"]:
            enhanced_vois.append(np.mean([c["voi"] for c in r["enhanced_cruxes"]]))
        else:
            enhanced_vois.append(np.nan)

    abs_returns = np.array(abs_returns)
    baseline_vois = np.array(baseline_vois)
    enhanced_vois = np.array(enhanced_vois)

    # Baseline correlation
    baseline_mask = ~np.isnan(baseline_vois)
    if baseline_mask.sum() >= 3:
        r_baseline, p_baseline = stats.pearsonr(
            baseline_vois[baseline_mask],
            abs_returns[baseline_mask]
        )
        print(f"\nBaseline VOI vs |return|:")
        print(f"  r = {r_baseline:.3f}, p = {p_baseline:.3f}, n = {baseline_mask.sum()}")
    else:
        r_baseline, p_baseline = np.nan, np.nan
        print(f"\nBaseline: insufficient data (n={baseline_mask.sum()})")

    # Enhanced correlation
    enhanced_mask = ~np.isnan(enhanced_vois)
    if enhanced_mask.sum() >= 3:
        r_enhanced, p_enhanced = stats.pearsonr(
            enhanced_vois[enhanced_mask],
            abs_returns[enhanced_mask]
        )
        print(f"\nEnhanced VOI vs |return|:")
        print(f"  r = {r_enhanced:.3f}, p = {p_enhanced:.3f}, n = {enhanced_mask.sum()}")
    else:
        r_enhanced, p_enhanced = np.nan, np.nan
        print(f"\nEnhanced: insufficient data (n={enhanced_mask.sum()})")

    # Individual crux-level analysis
    print("\n=== Crux-Level Analysis ===")

    all_baseline = []
    all_enhanced = []

    for r in results:
        for c in r["baseline_cruxes"]:
            all_baseline.append({"voi": c["voi"], "abs_return": r["abs_return"]})
        for c in r["enhanced_cruxes"]:
            all_enhanced.append({"voi": c["voi"], "abs_return": r["abs_return"]})

    if len(all_baseline) >= 10:
        vois = [x["voi"] for x in all_baseline]
        rets = [x["abs_return"] for x in all_baseline]
        r_crux, p_crux = stats.pearsonr(vois, rets)
        print(f"Baseline (crux-level): r={r_crux:.3f}, p={p_crux:.3f}, n={len(all_baseline)}")

    if len(all_enhanced) >= 10:
        vois = [x["voi"] for x in all_enhanced]
        rets = [x["abs_return"] for x in all_enhanced]
        r_crux, p_crux = stats.pearsonr(vois, rets)
        print(f"Enhanced (crux-level): r={r_crux:.3f}, p={p_crux:.3f}, n={len(all_enhanced)}")

    # Interpretation
    print("\n=== Interpretation ===")
    if not np.isnan(r_baseline):
        if r_baseline > 0.3 and p_baseline < 0.05:
            print("STRONG RESULT: VOI predicts forward returns")
        elif r_baseline > 0.2 and p_baseline < 0.1:
            print("MODERATE RESULT: Directionally correct")
        elif r_baseline > 0:
            print("WEAK RESULT: Positive but not significant")
        else:
            print("NEGATIVE RESULT: VOI does not predict returns")

    # Save detailed results
    output = {
        "metadata": {
            "validated_at": datetime.now().isoformat(),
            "resolved_count": resolved_count,
            "pending_count": pending_count,
        },
        "correlations": {
            "baseline_stock_level": {"r": r_baseline, "p": p_baseline} if not np.isnan(r_baseline) else None,
            "enhanced_stock_level": {"r": r_enhanced, "p": p_enhanced} if not np.isnan(r_enhanced) else None,
        },
        "results": results,
    }

    output_file = RESULTS_DIR / "russell_forward_validation.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nSaved detailed results to: {output_file}")


if __name__ == "__main__":
    validate()
