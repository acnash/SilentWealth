# =========================
# fetch_multi_stooq_combined.py
# Dependencies: pandas
#   pip install pandas
# Run: python fetch_multi_stooq_combined.py
# =========================

import time, random
from datetime import datetime, timedelta
import pandas as pd

TICKERS = ["GOOGL","META","NVDA","AMZN","QBTS","TSLA"]
BASE_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
OUTFILE = "ALL_PRICES.txt"
MIN_YEARS = 8

def stooq_symbol(ticker: str) -> str: return f"{ticker.lower()}.us"

def fetch_one(symbol: str, max_retries: int = 5) -> pd.DataFrame:
    attempt = 0
    url = BASE_URL.format(symbol=symbol)
    while True:
        attempt += 1
        try:
            df = pd.read_csv(url)
            if df is None or df.empty: raise RuntimeError("Stooq returned no data.")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            if "Adj Close" not in df.columns: df["Adj Close"] = df["Close"]
            cols = ["Date","Open","High","Low","Close","Adj Close","Volume"]
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception as e:
            if attempt >= max_retries: raise
            time.sleep(min(2 ** attempt, 10) + random.uniform(0, 0.5))

def slice_last_n_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty: return df
    cutoff = datetime.now() - timedelta(days=int(365.25 * years))
    return df[df["Date"] >= cutoff].copy()

def main() -> None:
    print(f"[{datetime.now().isoformat()}] Fetching daily prices from Stooq …")
    frames = []
    failures = []
    per_ticker_counts = {}
    per_ticker_span = {}
    for t in TICKERS:
        sym = stooq_symbol(t)
        try:
            df = fetch_one(sym, max_retries=5)
            df = slice_last_n_years(df, MIN_YEARS)
            df["Ticker"] = t  # ← ticker name added to each row
            df = df[["Date","Ticker","Open","High","Low","Close","Adj Close","Volume"]]
            frames.append(df)
            per_ticker_counts[t] = len(df)
            if len(df) > 0: per_ticker_span[t] = (df["Date"].min().date(), df["Date"].max().date())
            time.sleep(0.5 + random.uniform(0, 0.5))
        except Exception as e:
            print(f"⚠️  {t}: failed to fetch ({e})")
            failures.append(t)
    if not frames:
        print("No data retrieved. Aborting.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["Date","Ticker"]).reset_index(drop=True)
    all_df.to_csv(OUTFILE, sep="\t", index=False)
    total_rows = len(all_df)
    print(f"✅ Saved {total_rows:,} rows to {OUTFILE}")
    print("\n=== PER-TICKER ROW COUNTS ===")
    for t in sorted(per_ticker_counts):
        rows = per_ticker_counts[t]
        span = per_ticker_span.get(t, (None, None))
        print(f"{t}: {rows:,} rows  |  {span[0]} → {span[1]}")
    if failures:
        print("\nTickers with errors: " + ", ".join(failures))

if __name__ == "__main__":
    main()
