# =========================
# fetch_multi_stooq_combined.py
# Dependencies: pandas
#   pip install pandas
# Run: python fetch_multi_stooq_combined.py
# =========================

import time, random
from datetime import datetime, timedelta
from itertools import combinations
import pandas as pd
import urllib.parse

#TICKERS = ["QBTS"] #,"MSFT","JNJ","PG","KO","VZ","PEP","WMT","PFE"]
#TICKERS = ["%5Espx"]   # literal URL-encoded ^SPX S&P 500
TICKERS = ["%5Endx"] # NASDAQ 100


BASE_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
MIN_YEARS = 10                 # keep only the last N years
SLEEP_BETWEEN_REQUESTS = (0.5, 1.0)  # seconds (min, max) jitter

def stooq_symbol(ticker: str) -> str:
    t = ticker.strip()
    # If user already provided an encoded symbol like "%5Espx", trust it
    if "%" in t:
        return t.lower()
    # If user provided caret form like "^SPX", encode it
    if t.startswith("^"):
        return urllib.parse.quote(t.lower(), safe='')
    # Preserve dotted symbols like 'googl.us'
    if "." in t:
        return t.lower()
    # Default: assume US equity
    return f"{t.lower()}.us"

    #return f"{ticker.lower()}.us"

def fetch_one(symbol: str, max_retries: int = 5) -> pd.DataFrame:
    attempt = 0
    url = BASE_URL.format(symbol=symbol)
    while True:
        attempt += 1
        try:

            print("DEBUG: fetching url =", BASE_URL.format(symbol=symbol))

            df = pd.read_csv(url)
            if df is None or df.empty:
                raise RuntimeError("Stooq returned no data.")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]
            cols = ["Date","Open","High","Low","Close","Adj Close","Volume"]
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception as e:
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt, 10) + random.uniform(0, 0.5))

def slice_last_n_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = datetime.now() - timedelta(days=int(365.25 * years))
    return df[df["Date"] >= cutoff].copy()

def save_combo(df_map: dict, combo: tuple) -> int:
    frames = []
    for t in combo:
        df = df_map.get(t)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["Ticker"] = t
        tmp = tmp[["Date","Ticker","Open","High","Low","Close","Adj Close","Volume"]]
        frames.append(tmp)
    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True)
    if all_df.empty:
        return 0
    all_df = all_df.sort_values(["Date","Ticker"]).reset_index(drop=True)
    combo_id = "_".join(combo)  # safe filename component
    outfile = f"{combo_id}_PRICES.txt"
    all_df.to_csv(outfile, sep="\t", index=False)
    print(f"{outfile}")
    return len(all_df)

def main() -> None:
    print(f"[{datetime.now().isoformat()}] Fetching daily prices from Stooq …")
    cache = {}
    failures = []
    per_ticker_counts = {}
    per_ticker_span = {}

    # Fetch each ticker once (with caching)
    for t in TICKERS:
        sym = stooq_symbol(t)
        try:
            df = fetch_one(sym, max_retries=5)
            df = slice_last_n_years(df, MIN_YEARS)
            per_ticker_counts[t] = len(df)
            if len(df) > 0:
                per_ticker_span[t] = (df["Date"].min().date(), df["Date"].max().date())
            cache[t] = df
            time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))
        except Exception as e:
            print(f"⚠️  {t}: failed to fetch ({e})")
            failures.append(t)

    # Filter to successfully fetched tickers
    ok_tickers = sorted([t for t in TICKERS if t in cache])
    if not ok_tickers:
        print("No data retrieved. Aborting.")
        return

    # Iterate all non-empty combinations
    total_files = 0
    total_rows_written = 0
    print(f"\nBuilding combinations for {len(ok_tickers)} tickers ({', '.join(ok_tickers)}) …")
    for r in range(1, len(ok_tickers) + 1):
        for combo in combinations(ok_tickers, r):
            rows = save_combo(cache, combo)
            if rows > 0:
                total_files += 1
                total_rows_written += rows
                print(f"✅ Saved {rows:,} rows to {'_'.join(combo)}_PRICES.txt")
            else:
                print(f"— Skipped {'_'.join(combo)} (no rows)")

    # Summary
    print("\n=== PER-TICKER ROW COUNTS (after MIN_YEARS filter) ===")
    for t in ok_tickers:
        rows = per_ticker_counts.get(t, 0)
        span = per_ticker_span.get(t, (None, None))
        print(f"{t}: {rows:,} rows  |  {span[0]} → {span[1]}")
    if failures:
        print("\nTickers with errors: " + ", ".join(failures))
    print(f"\nFinished: wrote {total_files} files, {total_rows_written:,} total combo rows.")

if __name__ == "__main__":
    main()
