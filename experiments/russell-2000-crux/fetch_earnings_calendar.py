"""Fetch earnings calendar for Russell 2000 stocks.

Gets earnings dates within the post-cutoff period (Nov 2025 - Jan 2026).
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE, PILOT_N_STOCKS

DATA_DIR = Path(__file__).parent / "data"


def get_russell_2000_tickers(n_stocks: int = 500) -> list[str]:
    """Get a diverse sample of Russell 2000 tickers.

    Args:
        n_stocks: Maximum number of tickers to return.

    Returns:
        List of ticker symbols, up to n_stocks.
    """
    # Expanded Russell 2000 stocks by sector
    # Focus on companies with regular quarterly earnings
    tickers = [
        # Technology (expanded)
        "AAON", "ADTN", "AEIS", "AMKR", "ASGN", "CGNX", "COHU", "CRUS", "DIOD",
        "ENTG", "EXLS", "FORM", "IPGP", "ITRI", "LSCC", "MANH", "MKSI", "MTSI",
        "NOVT", "NSIT", "PEGA", "PLXS", "QLYS", "SANM", "SMTC", "SWKS", "TTMI",
        "SLAB", "ICHR", "OSIS", "VICR", "VECO", "HLIT", "MLAB", "RMBS", "SITM",
        "ONTO", "NATI", "OLED", "POWI", "SGH", "VRRM", "VRNT", "CDNS", "SNPS",
        "CCMP", "IDCC", "AMBA", "PRFT", "AVNW", "VIAV", "DGII", "COMM", "NMRK",

        # Financials (expanded)
        "HBAN", "CFG", "RF", "ZION", "KEY", "FHN", "SNV", "WTFC", "WAL",
        "UMBF", "PNFP", "GBCI", "SBCF", "FFIN", "IBOC", "BANF", "TRMK", "FIBK",
        "CVBF", "CADE", "FULT", "NBTB", "ABCB", "FCNCA", "ONB", "UCBI", "SFNC",
        "NBHC", "INDB", "PPBI", "SBNY", "PACW", "OFG", "HTLF", "EGBN", "WSFS",
        "CATY", "WAFD", "CBSH", "SFBS", "BUSE", "FBNC", "TOWN", "SRCE", "NWBI",
        "SYBT", "GABC", "BHLB", "HOPE", "FRME", "QCRH", "CFFN", "DCOM", "RNST",

        # Healthcare (expanded)
        "ACHC", "ADUS", "AMN", "BIO", "CHE", "CRL", "ENSG", "EXAS", "GMED",
        "HAE", "HQY", "IART", "INCY", "IRWD", "LGND", "MEDP", "MOH", "PTCT",
        "NEOG", "NVRO", "OFIX", "PCRX", "PDCO", "PEN", "PRGO", "PZZA", "QDEL",
        "RGEN", "RXRX", "SEM", "SGRY", "STAA", "SUPN", "TCMD", "TNDM", "TREX",
        "USPH", "UTHR", "VAPO", "VIR", "XENE", "XRAY", "RARE", "RVNC", "SGEN",
        "SRRK", "ALKS", "ANIP", "ATEC", "AUPH", "BCRX", "BLFS", "BMRN", "CERS",

        # Industrials (expanded)
        "AGCO", "ALK", "ARCB", "CAR", "CW", "EXP", "FIX", "GEF", "HNI",
        "HUBB", "KMT", "MLI", "NPO", "POWL", "RBC", "SPB", "TTC", "WTS",
        "APOG", "ASTE", "ATI", "AWI", "AYI", "BBCP", "BC", "BWXT", "CBT",
        "CFX", "CSWI", "CXT", "DLX", "DNOW", "ENS", "ESAB", "FELE", "FLOW",
        "FSS", "GBX", "GGG", "GHC", "GTES", "GVA", "HI", "HR", "HWKN",
        "IIIN", "JBT", "KAI", "KMT", "KNF", "LAUR", "LCII", "LNN", "MAN",

        # Consumer Discretionary (expanded)
        "ANF", "BURL", "CAKE", "COLM", "CROX", "DIN", "EAT", "FIVE", "FOXF",
        "GPI", "HELE", "KSS", "LAD", "LKQ", "MHO", "PLNT", "TXRH", "URBN",
        "AAP", "AEO", "BBWI", "BKE", "BOOT", "CRI", "DBI", "DXLG", "FL",
        "GIII", "GPS", "GCO", "HIBB", "HVT", "JILL", "JWN", "KRUS", "LE",
        "LEG", "LEVI", "LZB", "MLKN", "MOV", "ODP", "PBH", "PRPL", "RL",
        "RH", "RVLV", "SCVL", "SKX", "SN", "SNBR", "SHOO", "TCS", "VSCO",

        # Consumer Staples
        "BF.B", "CAG", "CLX", "CPB", "FLO", "HRL", "INGR", "K", "KHC",
        "LW", "MKC", "POST", "SJM", "SPB", "UNFI", "USFD", "BGS", "DAR",
        "HAIN", "IPAR", "LANC", "LNDC", "MGPI", "NATR", "SENEA", "SMPL", "TR",
        "THS", "VITL", "WMK", "CENT", "CENTA", "CHEF", "PRMW", "FARM", "JJSF",

        # Energy (expanded)
        "AM", "AR", "AROC", "CNX", "CTRA", "HP", "MTDR", "NOG", "RRC", "SM",
        "AMRC", "ARCH", "BCEI", "BKR", "BRSP", "BTMD", "CIVI", "CLB", "CNK",
        "CPE", "CRGY", "CRK", "DINO", "DK", "DNOW", "DRQ", "ERF", "GEL",
        "GPOR", "HES", "HESM", "LBRT", "LEU", "LONE", "LPI", "MGY", "MRC",
        "MTDR", "MUR", "NEXT", "NFE", "OIS", "OVV", "PDCE", "PDS", "PTEN",

        # Utilities (expanded)
        "AES", "ALE", "AVA", "BKH", "CNP", "EVRG", "HE", "IDA", "NWE", "OGE",
        "PNM", "PNW", "POR", "RGCO", "SJW", "SR", "SWX", "UTLM", "UTL", "WEC",
        "YORW", "CPK", "MSEX", "AWK", "WTRG", "CWT", "ARTNA", "ARIS", "SJW",

        # Real Estate
        "AAT", "ACC", "ADC", "AGNC", "ALEX", "AMH", "ARI", "BDN", "BFS",
        "BRSP", "BRX", "BXMT", "CADE", "CAKE", "CBL", "CIO", "CLPR", "COLD",
        "CONE", "CORR", "CTO", "CTT", "CUZ", "DEA", "DEI", "DLR", "DOC",
        "EGP", "EPR", "EQR", "ESS", "ESRT", "FAF", "FPI", "FR", "FSP",

        # Materials
        "AXTA", "BCPC", "CC", "CF", "CMC", "CRS", "ECL", "EMN", "FMC",
        "GCP", "GPRE", "GRA", "GPRK", "GWW", "HUN", "IFF", "IOSP", "KOP",
        "KRA", "KWR", "LPX", "LYB", "MERC", "MLM", "MOS", "NGVT", "NUE",
        "OI", "OLN", "PKG", "PPG", "RPM", "RS", "SEE", "SLVM", "SMG",

        # Communication Services
        "CABO", "CCO", "CCTS", "CHK", "CHRD", "CMCSA", "CNXN", "COMP", "CRUS",
        "DISH", "DLTR", "DXC", "EBAY", "EGHT", "EXPE", "FANG", "FB", "FOXA",
        "GDDY", "GOOG", "GOOGL", "IAC", "IART", "INFO", "IPG", "LBRDA", "LBRDK",
        "LBTYA", "LBTYK", "LILAK", "LUMN", "MTCH", "NFLX", "NWS", "NWSA", "OMC",
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    return unique_tickers[:n_stocks]


def fetch_earnings_dates(tickers: list[str]) -> pd.DataFrame:
    """Fetch upcoming/recent earnings dates for tickers.

    Returns DataFrame with (ticker, earnings_date, company_name, sector).
    """
    results = []

    print(f"Fetching earnings info for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(tickers)}...")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get company info
            company_name = info.get("shortName", info.get("longName", ticker))
            sector = info.get("sector", "Unknown")

            # Try earnings_dates attribute (most reliable)
            try:
                ed_df = stock.earnings_dates
                if ed_df is not None and len(ed_df) > 0:
                    for idx in ed_df.index:
                        try:
                            ed_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                            if START_DATE <= ed_date <= END_DATE:
                                # Check if we already have this
                                existing = [r for r in results
                                           if r["ticker"] == ticker and r["earnings_date"] == ed_date.isoformat()]
                                if not existing:
                                    results.append({
                                        "ticker": ticker,
                                        "earnings_date": ed_date.isoformat(),
                                        "company_name": company_name,
                                        "sector": sector,
                                    })
                        except Exception:
                            pass
            except Exception:
                pass

            # Also try calendar dict (new yfinance API)
            try:
                calendar = stock.calendar
                if calendar and isinstance(calendar, dict):
                    # New API returns dict with 'Earnings Date' key
                    if "Earnings Date" in calendar:
                        ed_list = calendar["Earnings Date"]
                        if not isinstance(ed_list, list):
                            ed_list = [ed_list]
                        for ed in ed_list:
                            if ed and pd.notna(ed):
                                try:
                                    ed_date = pd.to_datetime(ed).date()
                                    if START_DATE <= ed_date <= END_DATE:
                                        existing = [r for r in results
                                                   if r["ticker"] == ticker and r["earnings_date"] == ed_date.isoformat()]
                                        if not existing:
                                            results.append({
                                                "ticker": ticker,
                                                "earnings_date": ed_date.isoformat(),
                                                "company_name": company_name,
                                                "sector": sector,
                                            })
                                except Exception:
                                    pass
            except Exception:
                pass

        except Exception as e:
            print(f"  Error for {ticker}: {e}")
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.drop_duplicates(subset=["ticker", "earnings_date"])

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch earnings calendar for Russell 2000 stocks")
    parser.add_argument(
        "--n_stocks",
        type=int,
        default=PILOT_N_STOCKS,
        help=f"Number of stocks to include (default: {PILOT_N_STOCKS})"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get tickers
    tickers = get_russell_2000_tickers(args.n_stocks)
    print(f"Using {len(tickers)} Russell 2000 tickers")

    # Fetch earnings dates
    earnings_df = fetch_earnings_dates(tickers)

    print(f"\n=== Earnings Calendar ===")
    print(f"Found {len(earnings_df)} earnings events in {START_DATE} to {END_DATE}")

    if len(earnings_df) > 0:
        print(f"\nBy sector:")
        print(earnings_df.groupby("sector").size())

        print(f"\nSample earnings events:")
        print(earnings_df.head(20).to_string(index=False))

        # Save
        earnings_df.to_json(DATA_DIR / "earnings_calendar.json", orient="records", indent=2)
        print(f"\nSaved to {DATA_DIR / 'earnings_calendar.json'}")

        # Also save the stock universe (all tickers with their info)
        universe = earnings_df[["ticker", "company_name", "sector"]].drop_duplicates()
        universe.to_json(DATA_DIR / "stock_universe.json", orient="records", indent=2)
        print(f"Saved {len(universe)} stocks to {DATA_DIR / 'stock_universe.json'}")
    else:
        print("\nNo earnings found in date range!")
        print("Note: yfinance may not have earnings dates for all stocks.")
        print("Consider adding manual earnings dates or expanding ticker list.")


if __name__ == "__main__":
    main()
