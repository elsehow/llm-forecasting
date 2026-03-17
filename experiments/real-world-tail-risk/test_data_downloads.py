"""
Test data availability for LLM vs options-implied volatility study.
Downloads CBOE vol indices and Yahoo Finance underlying prices,
then reports summary tables.
"""

import requests
import yfinance as yf
import pandas as pd
import io
import sys
from datetime import datetime

# ── CBOE Vol Indices ──────────────────────────────────────────────────────────

CBOE_TICKERS = ["VIX", "OVX", "GVZ", "RVX", "VXN", "VXEEM", "VXTLT", "VXAPL", "VXAZN", "VXGOG"]
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"

START = "2025-09-01"
END = "2026-03-16"

print("=" * 90)
print("CBOE VOLATILITY INDEX DOWNLOADS")
print("=" * 90)

cboe_rows = []

for ticker in CBOE_TICKERS:
    url = CBOE_URL.format(ticker)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            cboe_rows.append({
                "Ticker": ticker,
                "Status": f"HTTP {resp.status_code}",
                "Date Range": "-",
                "Columns": "-",
                "Covers Sep25-Mar26": "-",
                "Latest Date": "-",
                "Latest Close": "-",
            })
            continue

        df = pd.read_csv(io.StringIO(resp.text))

        # Find the date column (varies by file)
        date_col = None
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
        if date_col is None:
            # Try first column
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        date_min = df[date_col].min()
        date_max = df[date_col].max()

        # Check columns
        cols = [c for c in df.columns if c != date_col]
        has_ohlc = all(
            any(k in c.lower() for c in df.columns)
            for k in ["open", "high", "low", "close"]
        )
        col_desc = "OHLC" if has_ohlc else ", ".join(cols[:5])

        # Check coverage
        mask = (df[date_col] >= START) & (df[date_col] <= END)
        study_df = df[mask]
        covers = len(study_df) > 0
        covers_start = study_df[date_col].min() if covers else None
        covers_end = study_df[date_col].max() if covers else None

        # Latest value
        close_col = None
        for c in df.columns:
            if "close" in c.lower():
                close_col = c
                break
        latest_val = df.iloc[-1][close_col] if close_col else df.iloc[-1][cols[0]] if cols else "-"
        latest_date = df.iloc[-1][date_col].strftime("%Y-%m-%d")

        # Recent value near 2026-03-14
        target = pd.Timestamp("2026-03-14")
        near = df.iloc[(df[date_col] - target).abs().argsort()[:1]]
        near_date = near[date_col].values[0]
        near_date_str = pd.Timestamp(near_date).strftime("%Y-%m-%d")
        val_col = close_col if close_col else cols[0]
        near_val = near[val_col].values[0]

        cboe_rows.append({
            "Ticker": ticker,
            "Status": "OK",
            "Date Range": f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}",
            "Columns": col_desc,
            "Covers Sep25-Mar26": f"Yes ({len(study_df)} rows)" if covers else "No",
            "Latest Date": near_date_str,
            "Latest Close": f"{near_val:.2f}",
        })

    except Exception as e:
        cboe_rows.append({
            "Ticker": ticker,
            "Status": f"ERROR: {e}",
            "Date Range": "-",
            "Columns": "-",
            "Covers Sep25-Mar26": "-",
            "Latest Date": "-",
            "Latest Close": "-",
        })

cboe_df = pd.DataFrame(cboe_rows)
print(cboe_df.to_string(index=False))
print()

# ── Yahoo Finance Underlying Prices ──────────────────────────────────────────

YF_TICKERS = {
    "^GSPC": "S&P 500",
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
    "^RUT": "Russell 2000",
    "^NDX": "Nasdaq-100",
    "EEM": "EM ETF",
    "TLT": "Treasury ETF",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Google",
}

print("=" * 90)
print("YAHOO FINANCE UNDERLYING PRICE DOWNLOADS")
print("=" * 90)

yf_rows = []

for ticker, label in YF_TICKERS.items():
    try:
        data = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)

        if data.empty:
            yf_rows.append({
                "Ticker": ticker,
                "Name": label,
                "Status": "EMPTY",
                "Trading Days": 0,
                "Date Range": "-",
                "Missing Days": "-",
                "Latest Date": "-",
                "Latest Close": "-",
            })
            continue

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        n_days = len(data)
        date_min = data.index.min().strftime("%Y-%m-%d")
        date_max = data.index.max().strftime("%Y-%m-%d")

        # Check for NaN gaps in Close
        close = data["Close"]
        n_nan = close.isna().sum()

        latest_close = close.dropna().iloc[-1]
        latest_date = close.dropna().index[-1].strftime("%Y-%m-%d")

        yf_rows.append({
            "Ticker": ticker,
            "Name": label,
            "Status": "OK",
            "Trading Days": n_days,
            "Date Range": f"{date_min} to {date_max}",
            "NaN Gaps": n_nan,
            "Latest Date": latest_date,
            "Latest Close": f"{latest_close:.2f}",
        })

    except Exception as e:
        yf_rows.append({
            "Ticker": ticker,
            "Name": label,
            "Status": f"ERROR: {e}",
            "Trading Days": "-",
            "Date Range": "-",
            "NaN Gaps": "-",
            "Latest Date": "-",
            "Latest Close": "-",
        })

yf_df = pd.DataFrame(yf_rows)
print(yf_df.to_string(index=False))
print()

# ── Summary ──────────────────────────────────────────────────────────────────

cboe_ok = sum(1 for r in cboe_rows if r["Status"] == "OK" and "Yes" in str(r.get("Covers Sep25-Mar26", "")))
cboe_fail = len(cboe_rows) - cboe_ok
yf_ok = sum(1 for r in yf_rows if r["Status"] == "OK")
yf_fail = len(yf_rows) - yf_ok

print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"CBOE vol indices:  {cboe_ok}/{len(CBOE_TICKERS)} available with Sep 2025 - Mar 2026 coverage")
if cboe_fail:
    failed = [r["Ticker"] for r in cboe_rows if r["Status"] != "OK" or "Yes" not in str(r.get("Covers Sep25-Mar26", ""))]
    print(f"  Failed/missing coverage: {', '.join(failed)}")
print(f"Yahoo prices:      {yf_ok}/{len(YF_TICKERS)} available")
if yf_fail:
    failed = [r["Ticker"] for r in yf_rows if r["Status"] != "OK"]
    print(f"  Failed: {', '.join(failed)}")
