"""Fetch stock price data for the experiment period."""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE

DATA_DIR = Path(__file__).parent / "data"


def fetch_stock_returns(tickers: list[str]) -> pd.DataFrame:
    """Fetch daily OHLC and compute intraday returns.

    Returns DataFrame with (ticker, date, open, close, return).
    return = (close - open) / open
    """
    # Add buffer for data completeness
    start_str = (START_DATE - timedelta(days=5)).strftime("%Y-%m-%d")
    end_str = (END_DATE + timedelta(days=1)).strftime("%Y-%m-%d")

    results = []

    print(f"Fetching {len(tickers)} tickers from {start_str} to {end_str}...")

    # Batch fetch
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"  Processing {i+1}-{min(i+chunk_size, len(tickers))} of {len(tickers)}...")

        try:
            data = yf.download(
                " ".join(chunk),
                start=start_str,
                end=end_str,
                progress=False,
                group_by="ticker",
            )

            for ticker in chunk:
                try:
                    if len(chunk) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    for date_idx in ticker_data.index:
                        date_str = date_idx.strftime("%Y-%m-%d")
                        date_obj = date_idx.date()

                        # Only include dates in our range
                        if not (START_DATE <= date_obj <= END_DATE):
                            continue

                        row = ticker_data.loc[date_idx]
                        open_price = row["Open"]
                        close_price = row["Close"]

                        if pd.notna(open_price) and pd.notna(close_price) and open_price > 0:
                            ret = (close_price - open_price) / open_price
                            results.append({
                                "ticker": ticker,
                                "date": date_str,
                                "open": float(open_price),
                                "close": float(close_price),
                                "return": float(ret),
                            })
                except Exception:
                    pass

        except Exception as e:
            print(f"  Chunk error: {e}")

    df = pd.DataFrame(results)
    print(f"\nFetched {len(df)} ticker-date observations")
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load stock universe
    universe_path = DATA_DIR / "stock_universe.json"
    if not universe_path.exists():
        print("Run fetch_earnings_calendar.py first!")
        return

    with open(universe_path) as f:
        universe = json.load(f)

    tickers = [s["ticker"] for s in universe]
    print(f"Fetching data for {len(tickers)} stocks")

    # Fetch returns
    returns_df = fetch_stock_returns(tickers)

    # Load earnings calendar to tag earnings days
    with open(DATA_DIR / "earnings_calendar.json") as f:
        earnings = json.load(f)

    earnings_lookup = {(e["ticker"], e["earnings_date"]): True for e in earnings}
    returns_df["is_earnings_day"] = returns_df.apply(
        lambda r: earnings_lookup.get((r["ticker"], r["date"]), False),
        axis=1
    )

    # Summary
    print(f"\n=== Stock Returns Summary ===")
    print(f"Total observations: {len(returns_df)}")
    print(f"Earnings day observations: {returns_df['is_earnings_day'].sum()}")
    print(f"Non-earnings day observations: {(~returns_df['is_earnings_day']).sum()}")

    print(f"\nReturn statistics:")
    print(f"  All days: mean={returns_df['return'].mean():.4f}, std={returns_df['return'].std():.4f}")

    earnings_returns = returns_df[returns_df["is_earnings_day"]]["return"]
    non_earnings_returns = returns_df[~returns_df["is_earnings_day"]]["return"]

    if len(earnings_returns) > 0:
        print(f"  Earnings days: mean={earnings_returns.mean():.4f}, std={earnings_returns.std():.4f}")
        print(f"  Earnings days |return|: mean={earnings_returns.abs().mean():.4f}")

    if len(non_earnings_returns) > 0:
        print(f"  Non-earnings days: mean={non_earnings_returns.mean():.4f}, std={non_earnings_returns.std():.4f}")
        print(f"  Non-earnings days |return|: mean={non_earnings_returns.abs().mean():.4f}")

    # Save
    returns_df.to_parquet(DATA_DIR / "stock_returns.parquet", index=False)
    print(f"\nSaved to {DATA_DIR / 'stock_returns.parquet'}")


if __name__ == "__main__":
    main()
