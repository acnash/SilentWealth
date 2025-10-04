# =========================
# fetch_meta_stooq.py
# Dependencies: pandas
#   pip install pandas
# Run: python fetch_meta_stooq.py
# =========================

import time, random
from datetime import datetime
import pandas as pd

STOOQ_URL = "https://stooq.com/q/d/l/?s=meta.us&i=d"  # META daily (US)
OUTFILE = "META_prices.txt"

def get_meta_stooq(max_retries=5):
    attempt = 0
    while True:
        attempt += 1
        try:
            df = pd.read_csv(STOOQ_URL)  # columns: Date,Open,High,Low,Close,Volume
            if df is None or df.empty: raise RuntimeError("Stooq returned no data.")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            # Align with yfinance-style columns by adding Adj Close = Close (Stooq has no adjustments)
            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]
            # Order columns if present
            cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception as e:
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt, 10) + random.uniform(0, 0.5))

def save_txt(df, path=OUTFILE):
    df.to_csv(path, sep="\t", index=True)
    print(f"✅ Saved {len(df):,} rows to {path}")

def main():
    print(f"[{datetime.now().isoformat()}] Fetching META daily prices from Stooq ...")
    df = get_meta_stooq()
    save_txt(df, OUTFILE)
    print(df.tail())

if __name__ == "__main__":
    main()
