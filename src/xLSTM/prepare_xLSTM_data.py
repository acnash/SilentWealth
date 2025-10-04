# ==============================================
# sp500_prep_xlstmts_fixed_with_plots.py
# Pipeline: load -> clean -> business-day align -> wavelet denoise -> normalize -> window -> splits (86/7/7)
# - Configurable ticker via DATA_TICKER (e.g. '^spx' or 'meta.us')
# - Configurable date window via START_DATE / END_DATE (YYYY-MM-DD)
# - No functions hard-code tickers or dates
# ==============================================

import os, re, io, urllib.parse, requests, math
import numpy as np
import pandas as pd
import pywt
from typing import Tuple, List
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

# ---------------------- USER CONFIG ----------------------
# Change this single variable to switch data:
# - S&P 500 (Stooq examples): '^spx' or '^gspc'
# - Meta Platforms (US listing on Stooq): 'meta.us'
DATA_TICKER = "^spx"       # <-- change to "meta.us" for META

# Date window you requested (YYYY-MM-DD). Set to None to disable filtering.
START_DATE = "2000-01-03"  # 3 Jan 2000
END_DATE   = "2023-12-29"  # 29 Dec 2023

# If True the script will re-download even if CSV exists
FORCE_DOWNLOAD = False
# --------------------------------------------------------

def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\-\.]+", "_", s)

INPUT_PATH = f"{_sanitize_filename(DATA_TICKER)}_prices.csv"

DATE_COL = "Date"
TARGET_COL = "Close"
FEATURE_COLS = ["Open","High","Low","Close","Adj Close","Volume"]
WINDOW_LENGTH = 60
HORIZON = 1
TRAIN_FRACTION = 0.86
VAL_FRACTION = 0.07
BUSINESS_DAY_METHOD = "time"
WAVELET = "db4"
WLEVEL = None
THRESH_MODE = "soft"
VERBOSE = True

# --- Plot config ---
PLOT_COLUMNS = ["Close"]
PLOT_DIR = "plots_raw_vs_denoised"
SHOW_PLOTS = True

def log(msg:str)->None:
    print(msg) if VERBOSE else None

# -------------------- downloader (generic, no hard-coded ticker) --------------------
def download_from_stooq(ticker: str, path: str, timeout: int = 30) -> pd.DataFrame:
    """
    Download daily data from Stooq for 'ticker' and save to CSV at 'path'.
    Keeps only the standard OHLCV/Adj Close columns that exist.
    """
    q = urllib.parse.quote(ticker, safe="")
    url = f"https://stooq.com/q/d/l/?s={q}&i=d"
    log(f"Downloading {ticker} from Stooq: {url}")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
    if "Date" in df.columns:
        df = df.set_index("Date").sort_index()
    df.index.name = "Date"
    # Ensure 'Adj Close' exists; Stooq usually provides 'Close'
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    cols_present = [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols_present]
    df.to_csv(path, index=True)
    log(f"Saved {len(df)} rows to {path}")
    return df

# -------------------- existing helpers (unchanged behaviour) --------------------
def read_any_table(path:str,date_col:str)->pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_csv(path,sep=None,engine="python") if ext in [".csv",".txt",".tsv",".data",""] else pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col],errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    if not isinstance(df.index,pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index,errors="coerce")
    df = df.sort_index()
    return df

def ensure_business_days(df:pd.DataFrame)->pd.DataFrame:
    bidx = pd.date_range(start=df.index.min(),end=df.index.max(),freq="B")
    df = df.reindex(bidx)
    if BUSINESS_DAY_METHOD == "ffill":
        df = df.ffill()
    elif BUSINESS_DAY_METHOD == "time":
        df = df.interpolate(method="time").ffill().bfill()
    else:
        df = df.interpolate(method="time").ffill().bfill()
    return df

def wavelet_denoise_1d(x:np.ndarray,wavelet:str=WAVELET,level:int=None,mode:str=THRESH_MODE)->np.ndarray:
    x = np.asarray(x,dtype=float)
    if level is None:
        level = pywt.dwt_max_level(len(x),pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x,wavelet,mode="periodization",level=level)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745 if len(coeffs[-1])>0 else 0.0
    uthresh = sigma*np.sqrt(2*np.log(len(x))) if len(x)>0 else 0.0
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c,uthresh,mode=mode) for c in coeffs[1:]]
    y = pywt.waverec(denoised_coeffs,wavelet,mode="periodization")
    return y[:len(x)]

def wavelet_denoise_df(df:pd.DataFrame,cols:List[str])->pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            series = out[c].astype(float).values
            level = WLEVEL if WLEVEL is not None else None
            out[c] = wavelet_denoise_1d(series,wavelet=WAVELET,level=level,mode=THRESH_MODE)
    return out

def darts_series_from_df(df:pd.DataFrame, cols:List[str])->TimeSeries:
    ts = TimeSeries.from_dataframe(df=df,value_cols=cols,freq="B")
    return ts

def split_series_by_fraction(ts:TimeSeries,train_frac:float,val_frac:float)->Tuple[TimeSeries,TimeSeries,TimeSeries]:
    n = len(ts)
    n_train = int(math.floor(n*train_frac))
    n_val = int(math.floor(n*val_frac))
    train = ts[:n_train]
    val = ts[n_train:n_train+n_val] if n_val>0 else ts[n_train:n_train]
    test = ts[n_train+n_val:]
    return train,val,test

def scale_with_darts(train_ts:TimeSeries, other_ts_list:List[TimeSeries])->Tuple[Scaler,List[TimeSeries]]:
    scaler = Scaler()
    scaler.fit(train_ts)
    scaled_train = scaler.transform(train_ts)
    scaled_others = [scaler.transform(t) for t in other_ts_list]
    return scaler,[scaled_train] + scaled_others

def ts_to_df(ts:TimeSeries)->pd.DataFrame:
    if hasattr(ts,"to_dataframe"): df = ts.to_dataframe()
    else: df = ts.pd_dataframe(copy=True)
    df.index.name = "Date"
    return df

def build_windows_numpy(df:pd.DataFrame, feature_cols:List[str], target_col:str, window:int, horizon:int)->Tuple[np.ndarray,np.ndarray]:
    cols = [c for c in feature_cols if c in df.columns]
    data = df[cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    X_list = []
    y_list = []
    for t in range(window, len(df)-horizon+1):
        X_list.append(data[t-window:t,:])
        y_list.append(target[t+horizon-1])
    X = np.stack(X_list,axis=0) if len(X_list)>0 else np.zeros((0,window,len(cols)),dtype=np.float32)
    y = np.array(y_list,dtype=np.float32) if len(y_list)>0 else np.zeros((0,),dtype=np.float32)
    return X,y

def plot_raw_vs_denoised(df_raw:pd.DataFrame, df_denoised:pd.DataFrame, cols:List[str], out_dir:str, show:bool=True)->None:
    os.makedirs(out_dir, exist_ok=True)
    for c in cols:
        if c not in df_raw.columns or c not in df_denoised.columns: continue
        fig = plt.figure(figsize=(11,5))
        plt.plot(df_raw.index, df_raw[c].values, label=f"{c} (raw)")
        plt.plot(df_denoised.index, df_denoised[c].values, label=f"{c} (denoised)")
        plt.title(f"{c}: raw vs denoised (wavelet={WAVELET}, mode={THRESH_MODE})")
        plt.xlabel("Date"); plt.ylabel(c); plt.legend(); plt.tight_layout()
        out_png = os.path.join(out_dir, f"{c.replace(' ','_').lower()}_raw_vs_denoised.png")
        plt.savefig(out_png, dpi=150)
        if show: plt.show()
        else: plt.close(fig)
        log(f"Saved plot: {out_png}")

def plot_adj_close_before_after(original_df:pd.DataFrame, denoised_df:pd.DataFrame, train_inv_df:pd.DataFrame, val_inv_df:pd.DataFrame, test_den_df:pd.DataFrame, out_dir:str, show:bool=True)->None:
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    if train_inv_df is not None and not train_inv_df.empty: parts.append(train_inv_df[[TARGET_COL]])
    if val_inv_df is not None and not val_inv_df.empty: parts.append(val_inv_df[[TARGET_COL]])
    if test_den_df is not None and not test_den_df.empty: parts.append(test_den_df[[TARGET_COL]])
    if len(parts)==0:
        log("No data to plot for adjusted-close before/after.")
        return
    combined_after = pd.concat(parts).sort_index()
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14,8), sharex=True)
    ax1.plot(original_df.index, original_df[TARGET_COL].values)
    ax1.set_title("Adjusted Close — before wavelet denoising"); ax1.set_xlabel("Date"); ax1.set_xticks([]); ax1.set_ylabel("Adjusted Close")
    ax2.plot(combined_after.index, combined_after[TARGET_COL].values)
    ax2.set_title("Adjusted Close — after wavelet denoising"); ax2.set_xlabel("Date"); ax2.set_xticks([]); ax2.set_ylabel("Adjusted Close")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "adj_close_before_after.png")
    plt.savefig(out_png, dpi=150)
    if show: plt.show()
    else: plt.close(fig)
    log(f"Saved plot: {out_png}")

# -------------------- main pipeline --------------------
def main()->None:
    log(f"DATA_TICKER={DATA_TICKER}; CSV={INPUT_PATH}; date window={START_DATE} -> {END_DATE}")
    # download if missing or forced
    if FORCE_DOWNLOAD or not os.path.exists(INPUT_PATH):
        try:
            download_from_stooq(DATA_TICKER, INPUT_PATH)
        except Exception as e:
            log(f"Download failed for ticker '{DATA_TICKER}': {e}")
            raise

    # Step 1: Load CSV
    log("Step 1: Load data")
    df = read_any_table(INPUT_PATH, DATE_COL)
    if df is None or df.empty:
        raise RuntimeError(f"No data read from {INPUT_PATH}")

    # Apply requested date window filtering (if provided)
    if START_DATE is not None or END_DATE is not None:
        start = pd.to_datetime(START_DATE) if START_DATE is not None else df.index.min()
        end = pd.to_datetime(END_DATE) if END_DATE is not None else df.index.max()
        df = df.loc[start:end]
        log(f"After date filter: rows={len(df)}, range={df.index.min().date() if len(df)>0 else 'N/A'} -> {df.index.max().date() if len(df)>0 else 'N/A'}")

    assert TARGET_COL in df.columns, f"Missing target column '{TARGET_COL}'"
    keep_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep_cols].copy()
    log(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Cleaning
    log("Step 2: Clean nulls")
    df = df.replace([np.inf,-np.inf],np.nan)
    df = df.dropna(how="all")
    df = df.interpolate(method="time").ffill().bfill()
    log(f"Nulls after clean: {df.isna().sum().sum()}")

    # Business-day align
    log("Step 3: Align to business days")
    df = ensure_business_days(df)
    log(f"Business-day rows: {len(df)} from {df.index.min().date()} to {df.index.max().date()}")

    # Denoise
    log("Step 4: Wavelet denoise each feature")
    df_denoised = wavelet_denoise_df(df,keep_cols)

    # Per-column raw vs denoised plots
    log("Plot: raw vs denoised (per-column)")
    plot_raw_vs_denoised(df, df_denoised, PLOT_COLUMNS, PLOT_DIR, SHOW_PLOTS)

    # Convert denoised to TimeSeries
    log("Step 5: Convert denoised DataFrame to Darts TimeSeries (business-day freq)")
    ts = darts_series_from_df(df_denoised,keep_cols)

    # Split
    log("Step 6: Train/Val/Test split (86/7/7)")
    train_ts,val_ts,test_ts = split_series_by_fraction(ts,TRAIN_FRACTION,VAL_FRACTION)
    log(f"Lengths -> train:{len(train_ts)}, val:{len(val_ts)}, test:{len(test_ts)}")

    # Scale (train & val only); test left denoised-only
    log("Step 7: Normalize with Scaler (fit on train only). NOTE: Test will NOT be normalized.")
    scaler, scaled_list = scale_with_darts(train_ts, [val_ts])
    train_scaled = scaled_list[0]
    val_scaled = scaled_list[1] if len(scaled_list)>1 else None

    train_df = ts_to_df(train_scaled)
    val_df = ts_to_df(val_scaled) if val_scaled is not None else pd.DataFrame(columns=keep_cols)
    test_df = ts_to_df(test_ts)

    train_df = train_df.reindex(columns=[c for c in keep_cols if c in train_df.columns])
    val_df = val_df.reindex(columns=[c for c in keep_cols if c in val_df.columns])
    test_df = test_df.reindex(columns=[c for c in keep_cols if c in test_df.columns])

    # Inverse-transform scaled train/val for plotting comparison
    train_inv_df = None
    val_inv_df = None
    try:
        train_inv_ts = scaler.inverse_transform(train_scaled)
        train_inv_df = ts_to_df(train_inv_ts)
    except Exception as e:
        log(f"Warning inverse-transform train: {e}")
    if val_scaled is not None:
        try:
            val_inv_ts = scaler.inverse_transform(val_scaled)
            val_inv_df = ts_to_df(val_inv_ts)
        except Exception as e:
            log(f"Warning inverse-transform val: {e}")
    test_den_df = test_df.copy()

    log("Plot: Adjusted Close before vs after denoising (stacked)")
    plot_adj_close_before_after(df, df_denoised, train_inv_df, val_inv_df, test_den_df, PLOT_DIR, SHOW_PLOTS)

    # Build windows
    log("Step 8: Build fixed-length windows")
    X_train,y_train = build_windows_numpy(train_df,keep_cols,TARGET_COL,WINDOW_LENGTH,HORIZON)
    X_val,y_val = build_windows_numpy(val_df,keep_cols,TARGET_COL,WINDOW_LENGTH,HORIZON)
    X_test,y_test = build_windows_numpy(test_df,keep_cols,TARGET_COL,WINDOW_LENGTH,HORIZON)
    log(f"Windows -> X_train:{X_train.shape}, y_train:{y_train.shape}, X_val:{X_val.shape}, y_val:{y_val.shape}, X_test:{X_test.shape}, y_test:{y_test.shape}")

    np.savez_compressed("xlstmts_dataset_snp500_daily.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, feature_cols=np.array(keep_cols), target_col=np.array([TARGET_COL]))

    train_df.to_csv("scaled_train_businessB_denoised.csv")
    val_df.to_csv("scaled_val_businessB_denoised.csv")
    test_df.to_csv("denoised_test_businessB.csv")

    log("Done. Saved:")
    log(" - xlstmts_dataset_snp500_daily.npz")
    log(" - scaled_train_businessB_denoised.csv")
    log(" - scaled_val_businessB_denoised.csv")
    log(" - denoised_test_businessB.csv")

if __name__ == "__main__":
    main()
