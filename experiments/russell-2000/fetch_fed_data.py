"""Fetch FOMC meeting dates and outcomes (2015-2025).

Encodes outcomes as: +1 (raise), 0 (hold), -1 (cut)
Source: FRED API for Federal Funds Rate target upper bound.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"


def fetch_fed_funds_rate() -> pd.DataFrame:
    """Fetch Federal Funds Target Rate upper bound from FRED."""
    # DFEDTARU: Federal Funds Target Range - Upper Limit
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DFEDTARU",
        "api_key": "your_key_here",  # FRED requires API key
        "file_type": "json",
        "observation_start": "2015-01-01",
        "observation_end": "2025-12-31",
    }

    # Try without API key first (some series work)
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df
    except Exception:
        pass

    # Fallback: use manual FOMC data
    return None


def get_fomc_dates_and_outcomes() -> list[dict]:
    """Get FOMC meeting dates with rate decisions.

    Returns list of {date, outcome, rate_before, rate_after}.
    outcome: +1 (raise), 0 (hold), -1 (cut)
    """
    # Historical FOMC data 2015-2025
    # Source: Federal Reserve press releases
    # Format: (announcement_date, rate_change_bps)
    # Positive = hike, Negative = cut, Zero = hold

    fomc_decisions = [
        # 2015
        ("2015-01-28", 0),
        ("2015-03-18", 0),
        ("2015-04-29", 0),
        ("2015-06-17", 0),
        ("2015-07-29", 0),
        ("2015-09-17", 0),
        ("2015-10-28", 0),
        ("2015-12-16", 25),  # First hike since 2006

        # 2016
        ("2016-01-27", 0),
        ("2016-03-16", 0),
        ("2016-04-27", 0),
        ("2016-06-15", 0),
        ("2016-07-27", 0),
        ("2016-09-21", 0),
        ("2016-11-02", 0),
        ("2016-12-14", 25),

        # 2017
        ("2017-02-01", 0),
        ("2017-03-15", 25),
        ("2017-05-03", 0),
        ("2017-06-14", 25),
        ("2017-07-26", 0),
        ("2017-09-20", 0),
        ("2017-11-01", 0),
        ("2017-12-13", 25),

        # 2018
        ("2018-01-31", 0),
        ("2018-03-21", 25),
        ("2018-05-02", 0),
        ("2018-06-13", 25),
        ("2018-08-01", 0),
        ("2018-09-26", 25),
        ("2018-11-08", 0),
        ("2018-12-19", 25),

        # 2019
        ("2019-01-30", 0),
        ("2019-03-20", 0),
        ("2019-05-01", 0),
        ("2019-06-19", 0),
        ("2019-07-31", -25),  # First cut since 2008
        ("2019-09-18", -25),
        ("2019-10-30", -25),
        ("2019-12-11", 0),

        # 2020
        ("2020-01-29", 0),
        ("2020-03-03", -50),  # Emergency cut
        ("2020-03-15", -100),  # Emergency cut to zero
        ("2020-04-29", 0),
        ("2020-06-10", 0),
        ("2020-07-29", 0),
        ("2020-09-16", 0),
        ("2020-11-05", 0),
        ("2020-12-16", 0),

        # 2021
        ("2021-01-27", 0),
        ("2021-03-17", 0),
        ("2021-04-28", 0),
        ("2021-06-16", 0),
        ("2021-07-28", 0),
        ("2021-09-22", 0),
        ("2021-11-03", 0),
        ("2021-12-15", 0),

        # 2022
        ("2022-01-26", 0),
        ("2022-03-16", 25),  # First hike in cycle
        ("2022-05-04", 50),
        ("2022-06-15", 75),
        ("2022-07-27", 75),
        ("2022-09-21", 75),
        ("2022-11-02", 75),
        ("2022-12-14", 50),

        # 2023
        ("2023-02-01", 25),
        ("2023-03-22", 25),
        ("2023-05-03", 25),
        ("2023-06-14", 0),
        ("2023-07-26", 25),
        ("2023-09-20", 0),
        ("2023-11-01", 0),
        ("2023-12-13", 0),

        # 2024
        ("2024-01-31", 0),
        ("2024-03-20", 0),
        ("2024-05-01", 0),
        ("2024-06-12", 0),
        ("2024-07-31", 0),
        ("2024-09-18", -50),  # First cut in cycle
        ("2024-11-07", -25),
        ("2024-12-18", -25),

        # 2025 (through Jan)
        ("2025-01-29", 0),  # Expected hold
    ]

    results = []
    for date_str, change_bps in fomc_decisions:
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Convert to categorical outcome
        if change_bps > 0:
            outcome = 1  # Raise
        elif change_bps < 0:
            outcome = -1  # Cut
        else:
            outcome = 0  # Hold

        results.append({
            "date": date_str,
            "outcome": outcome,
            "change_bps": change_bps,
            "outcome_label": {1: "raise", 0: "hold", -1: "cut"}[outcome],
        })

    return results


def split_train_test(meetings: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into train (2015-2022) and test (2023-2025)."""
    train = [m for m in meetings if m["date"] < "2023-01-01"]
    test = [m for m in meetings if m["date"] >= "2023-01-01"]
    return train, test


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meetings = get_fomc_dates_and_outcomes()
    train, test = split_train_test(meetings)

    # Summary
    print(f"Total FOMC meetings: {len(meetings)}")
    print(f"Train (2015-2022): {len(train)} meetings")
    print(f"Test (2023-2025): {len(test)} meetings")

    # Count outcomes in train
    train_outcomes = [m["outcome"] for m in train]
    print(f"\nTrain outcome distribution:")
    print(f"  Raises: {train_outcomes.count(1)}")
    print(f"  Holds: {train_outcomes.count(0)}")
    print(f"  Cuts: {train_outcomes.count(-1)}")

    test_outcomes = [m["outcome"] for m in test]
    print(f"\nTest outcome distribution:")
    print(f"  Raises: {test_outcomes.count(1)}")
    print(f"  Holds: {test_outcomes.count(0)}")
    print(f"  Cuts: {test_outcomes.count(-1)}")

    # Save
    output = {
        "all_meetings": meetings,
        "train": train,
        "test": test,
        "metadata": {
            "train_period": "2015-01-01 to 2022-12-31",
            "test_period": "2023-01-01 to 2025-12-31",
            "outcome_encoding": "+1=raise, 0=hold, -1=cut",
            "generated_at": datetime.now().isoformat(),
        }
    }

    output_path = DATA_DIR / "fed_meetings.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
