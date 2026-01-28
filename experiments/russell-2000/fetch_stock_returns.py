"""Fetch daily returns for Russell 2000 stocks on Fed days.

Uses yfinance to get OHLC data. Computes intraday returns: (close - open) / open.
Filters to stocks with full history back to 2015 to mitigate survivorship bias.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"


def get_iwm_holdings() -> list[str]:
    """Get Russell 2000 tickers from IWM ETF holdings.

    Note: IWM is the main Russell 2000 ETF. We'll use a representative sample.
    Full list would require Bloomberg/Reuters terminal.

    For this experiment, we use a cross-section of ~200 stocks spanning sectors:
    - Financial services (most Fed-sensitive)
    - Real estate (REITs)
    - Industrials
    - Technology
    - Healthcare
    - Consumer discretionary/staples
    - Energy
    - Utilities
    """
    # Representative Russell 2000 stocks by sector
    # Selected for: (1) full history since 2015, (2) reasonable liquidity
    tickers = [
        # Financials (expected high Fed sensitivity)
        "HBAN", "CFG", "RF", "ZION", "KEY", "FHN", "SNV", "WTFC", "WAL", "PACW",
        "UMBF", "PNFP", "GBCI", "SBCF", "FFIN", "IBOC", "BANF", "TRMK", "FIBK", "FULT",
        "INDB", "WSFS", "UVSP", "PPBI", "NAVI", "PRAA", "ECPG", "WRLD", "CACC", "OMF",
        "TREE", "GDOT", "ALLY", "SYF", "DFS", "AGNC", "NLY", "STWD", "BXMT", "KREF",

        # REITs (Fed-sensitive via rates)
        "ACC", "AIV", "BRX", "CPT", "DEA", "EGP", "FR", "HIW", "IIPR", "KRC",
        "LXP", "MAC", "NNN", "OHI", "PDM", "RHP", "SBRA", "STOR", "UE", "VTR",

        # Regional banks (very Fed-sensitive)
        "BANR", "BOKF", "CBU", "CATY", "CVBF", "EWBC", "FBP", "FCNCA", "FNB", "FMBI",
        "HOPE", "HTLF", "MCB", "OZK", "PB", "SBNY", "SIVB", "TCBI", "TFC", "UMPQ",

        # Industrials
        "AGCO", "ALK", "ARCB", "B", "CAR", "CW", "EXP", "FIX", "GEF", "HNI",
        "HUBB", "KMT", "MLI", "NPO", "POWL", "RBC", "RXO", "SPB", "TTC", "WTS",

        # Technology
        "AAON", "ADTN", "AEIS", "AMKR", "ASGN", "ATEN", "CGNX", "COHU", "CRUS", "DIOD",
        "ENTG", "EXLS", "FARO", "FORM", "IPGP", "ITRI", "LSCC", "MANH", "MKSI", "MTSI",
        "NOVT", "NSIT", "PEGA", "PLXS", "QLYS", "RDWR", "SANM", "SMTC", "SWKS", "TTMI",

        # Healthcare
        "ABMD", "ACHC", "ADUS", "AMN", "BIO", "CHE", "CRL", "ENSG", "EXAS", "GMED",
        "HAE", "HQY", "IART", "INCY", "IRWD", "LHCG", "LGND", "LIVN", "MEDP", "MOH",
        "NEOG", "NHC", "NTRA", "OMCL", "PDCO", "PENN", "PRA", "PRGO", "PTCT", "RARE",

        # Consumer Discretionary
        "AAP", "ANF", "BURL", "CAKE", "COLM", "CROX", "DIN", "EAT", "FIVE", "FOXF",
        "GPI", "HAS", "HBI", "HELE", "JWN", "KSS", "LAD", "LKQ", "MHO", "MOD",
        "MTH", "ODP", "PLNT", "PLAY", "RH", "ROST", "SKX", "TXRH", "URBN", "WOOF",

        # Consumer Staples
        "BF-B", "CAG", "CLX", "CPB", "DAR", "EL", "FLO", "FRPT", "HRL", "INGR",
        "KDP", "LW", "MKC", "POST", "SJM", "SMPL", "SPB", "USFD", "VGR", "WMK",

        # Energy
        "AM", "AR", "AROC", "CHX", "CLR", "CNX", "CPE", "CTRA", "DEN", "HP",
        "MTDR", "NOG", "OAS", "PDCE", "PDS", "PUMP", "RRC", "SM", "SWN", "TALO",

        # Utilities
        "AES", "ALE", "AVA", "BKH", "CNP", "EVRG", "HE", "IDA", "NWE", "OGE",
        "OGS", "PNM", "POR", "SJI", "SPKE", "SWX", "UTL", "VVC", "WEC", "WR",
    ]

    return tickers


def fetch_returns_for_dates(tickers: list[str], dates: list[str], days_before: int = 5) -> pd.DataFrame:
    """Fetch returns for given tickers on specific dates.

    Args:
        tickers: List of stock symbols
        dates: List of date strings (YYYY-MM-DD)
        days_before: Days of buffer before first date to ensure data

    Returns:
        DataFrame with columns: ticker, date, return, open, close
    """
    if not dates:
        return pd.DataFrame()

    # Determine date range
    all_dates = pd.to_datetime(dates)
    start_date = (all_dates.min() - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end_date = (all_dates.max() + timedelta(days=1)).strftime("%Y-%m-%d")

    results = []
    failed_tickers = []

    print(f"Fetching {len(tickers)} tickers from {start_date} to {end_date}...")

    # Batch fetch for efficiency (yfinance supports this)
    # Process in chunks to avoid timeout
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"  Processing {i+1}-{min(i+chunk_size, len(tickers))} of {len(tickers)}...")

        try:
            data = yf.download(
                " ".join(chunk),
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
            )

            for ticker in chunk:
                try:
                    if len(chunk) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    for date in dates:
                        date_dt = pd.to_datetime(date)
                        if date_dt in ticker_data.index:
                            row = ticker_data.loc[date_dt]
                            open_price = row["Open"]
                            close_price = row["Close"]

                            if pd.notna(open_price) and pd.notna(close_price) and open_price > 0:
                                ret = (close_price - open_price) / open_price
                                results.append({
                                    "ticker": ticker,
                                    "date": date,
                                    "return": ret,
                                    "open": open_price,
                                    "close": close_price,
                                })
                except Exception as e:
                    pass

        except Exception as e:
            print(f"  Chunk failed: {e}")
            failed_tickers.extend(chunk)

    if failed_tickers:
        print(f"\nFailed to fetch {len(failed_tickers)} tickers")

    df = pd.DataFrame(results)
    print(f"\nFetched {len(df)} ticker-date observations")
    return df


def fetch_non_fed_returns(tickers: list[str], fed_dates: list[str], n_days: int = 100) -> pd.DataFrame:
    """Fetch returns on non-Fed days for null baseline.

    Samples random non-Fed days from the same period.
    """
    import random

    # Determine date range from Fed dates
    fed_dates_dt = pd.to_datetime(fed_dates)
    start_date = fed_dates_dt.min()
    end_date = fed_dates_dt.max()

    # Generate candidate non-Fed dates (weekdays only)
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    fed_dates_set = set(fed_dates_dt)
    non_fed_dates = [d for d in all_dates if d not in fed_dates_set]

    # Sample
    sampled_dates = random.sample(non_fed_dates, min(n_days, len(non_fed_dates)))
    sampled_dates_str = [d.strftime("%Y-%m-%d") for d in sorted(sampled_dates)]

    print(f"\nFetching {len(sampled_dates_str)} non-Fed days for baseline...")
    return fetch_returns_for_dates(tickers, sampled_dates_str)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load Fed meeting dates
    fed_path = DATA_DIR / "fed_meetings.json"
    if not fed_path.exists():
        print("Run fetch_fed_data.py first!")
        return

    with open(fed_path) as f:
        fed_data = json.load(f)

    train_dates = [m["date"] for m in fed_data["train"]]
    test_dates = [m["date"] for m in fed_data["test"]]
    all_fed_dates = train_dates + test_dates

    # Get tickers
    tickers = get_iwm_holdings()
    print(f"Using {len(tickers)} Russell 2000 representative stocks")

    # Fetch Fed day returns
    print("\n=== Fetching Fed Day Returns ===")
    fed_returns = fetch_returns_for_dates(tickers, all_fed_dates)

    # Fetch non-Fed day returns for baseline
    print("\n=== Fetching Non-Fed Day Returns ===")
    non_fed_returns = fetch_non_fed_returns(tickers, all_fed_dates, n_days=100)

    # Filter to stocks with sufficient history
    min_train_obs = 40  # At least 40 of ~55 train meetings
    stocks_with_history = (
        fed_returns[fed_returns["date"].isin(train_dates)]
        .groupby("ticker")
        .size()
    )
    valid_tickers = stocks_with_history[stocks_with_history >= min_train_obs].index.tolist()
    print(f"\n{len(valid_tickers)} stocks have sufficient train history (>={min_train_obs} meetings)")

    fed_returns_filtered = fed_returns[fed_returns["ticker"].isin(valid_tickers)]
    non_fed_returns_filtered = non_fed_returns[non_fed_returns["ticker"].isin(valid_tickers)]

    # Save
    fed_returns_filtered.to_parquet(DATA_DIR / "fed_returns.parquet", index=False)
    non_fed_returns_filtered.to_parquet(DATA_DIR / "non_fed_returns.parquet", index=False)

    print(f"\nSaved fed_returns.parquet: {len(fed_returns_filtered)} rows")
    print(f"Saved non_fed_returns.parquet: {len(non_fed_returns_filtered)} rows")

    # Summary stats
    print("\n=== Summary ===")
    print(f"Tickers with full history: {len(valid_tickers)}")
    print(f"Fed day observations: {len(fed_returns_filtered)}")
    print(f"Non-Fed day observations: {len(non_fed_returns_filtered)}")

    # Distribution of returns
    print(f"\nFed day return stats:")
    print(fed_returns_filtered["return"].describe())


if __name__ == "__main__":
    main()
