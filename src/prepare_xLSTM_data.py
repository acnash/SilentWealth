# ==============================================
# sp500_prep_xlstmts_fixed.py
# Pipeline: load -> clean -> business-day align -> wavelet denoise -> normalize -> window -> splits (86/7/7)
# Dependencies:
#   pip install pandas numpy pywavelets "u8darts[all]"
# ==============================================

# ==============================================
# prep_with_plots.py  — full pipeline + raw vs denoised plots
# Dependencies: pandas numpy pywavelets matplotlib darts scikit-learn
#   pip install pandas numpy pywavelets matplotlib "u8darts[all]"
# ==============================================

import os
import math
import numpy as np
import pandas as pd
import pywt
from typing import Tuple, List
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

INPUT_PATH = "META_prices.txt"
DATE_COL = "Date"
TARGET_COL = "Adj Close"
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
PLOT_COLUMNS = ["Adj Close"]  # e.g. ["Adj Close","Close","Volume"]
PLOT_DIR = "plots_raw_vs_denoised"
SHOW_PLOTS = True             # set False to skip showing (still saves PNGs)

def log(msg:str)->None: print(msg) if VERBOSE else None

def read_any_table(path:str,date_col:str)->pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_csv(path,sep=None,engine="python") if ext in [".csv",".txt",".tsv",".data",""] else pd.read_csv(path)
    if date_col in df.columns: df[date_col] = pd.to_datetime(df[date_col],errors="coerce")
    if date_col in df.columns: df = df.dropna(subset=[date_col]).set_index(date_col)
    if not isinstance(df.index,pd.DatetimeIndex): df.index = pd.to_datetime(df.index,errors="coerce")
    df = df.sort_index()
    return df

def ensure_business_days(df:pd.DataFrame)->pd.DataFrame:
    bidx = pd.date_range(start=df.index.min(),end=df.index.max(),freq="B")
    df = df.reindex(bidx)
    if BUSINESS_DAY_METHOD == "ffill": df = df.ffill()
    elif BUSINESS_DAY_METHOD == "time": df = df.interpolate(method="time").ffill().bfill()
    else: df = df.interpolate(method="time").ffill().bfill()
    return df

def wavelet_denoise_1d(x:np.ndarray,wavelet:str=WAVELET,level:int=None,mode:str=THRESH_MODE)->np.ndarray:
    x = np.asarray(x,dtype=float)
    if level is None: level = pywt.dwt_max_level(len(x),pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x,wavelet,mode="periodization",level=level)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745 if len(coeffs[-1])>0 else 0.0
    uthresh = sigma*np.sqrt(2*np.log(len(x))) if len(x)>0 else 0.0
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c,uthresh,mode=mode) for c in coeffs[1:]]
    y = pywt.waverec(denoised_coeffs,wavelet,mode="periodization")
    y = y[:len(x)]
    return y

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
        plt.xlabel("Date")
        plt.ylabel(c)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{c.replace(' ','_').lower()}_raw_vs_denoised.png")
        plt.savefig(out_png, dpi=150)
        if show: plt.show()
        else: plt.close(fig)
        log(f"Saved plot: {out_png}")

def main()->None:
    log("Step 1: Load data")
    df = read_any_table(INPUT_PATH,DATE_COL)
    assert TARGET_COL in df.columns, f"Missing target column '{TARGET_COL}'"
    keep_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep_cols].copy()
    log(f"Loaded {len(df)} rows from {INPUT_PATH}")
    log("Step 2: Clean nulls")
    df = df.replace([np.inf,-np.inf],np.nan)
    df = df.dropna(how="all")
    df = df.interpolate(method="time").ffill().bfill()
    log(f"Nulls after clean: {df.isna().sum().sum()}")
    log("Step 3: Align to business days")
    df = ensure_business_days(df)
    log(f"Business-day rows: {len(df)} from {df.index.min().date()} to {df.index.max().date()}")
    log("Step 4: Wavelet denoise each feature")
    df_denoised = wavelet_denoise_df(df,keep_cols)
    log("Plot: raw vs denoised")
    plot_raw_vs_denoised(df, df_denoised, PLOT_COLUMNS, PLOT_DIR, SHOW_PLOTS)
    log("Step 5: Convert to Darts TimeSeries")
    ts = darts_series_from_df(df_denoised,keep_cols)
    log("Step 6: Train/Val/Test split (86/7/7)")
    train_ts,val_ts,test_ts = split_series_by_fraction(ts,TRAIN_FRACTION,VAL_FRACTION)
    log(f"Lengths -> train:{len(train_ts)}, val:{len(val_ts)}, test:{len(test_ts)}")
    log("Step 7: Normalize with Scaler (fit on train only)")
    scaler,[train_scaled,val_scaled,test_scaled] = scale_with_darts(train_ts,[val_ts,test_ts])
    train_df = ts_to_df(train_scaled)
    val_df = ts_to_df(val_scaled)
    test_df = ts_to_df(test_scaled)
    if any(c not in train_df.columns for c in keep_cols): train_df = train_df.reindex(columns=[c for c in keep_cols if c in train_df.columns])
    if any(c not in val_df.columns for c in keep_cols): val_df = val_df.reindex(columns=[c for c in keep_cols if c in val_df.columns])
    if any(c not in test_df.columns for c in keep_cols): test_df = test_df.reindex(columns=[c for c in keep_cols if c in test_df.columns])
    log("Step 8: Build fixed-length windows")
    X_train,y_train = build_windows_numpy(train_df,keep_cols,TARGET_COL,WINDOW_LENGTH,HORIZON)
    X_val,y_val = build_windows_numpy(val_df,keep_cols,TARGET_COL,WINDOW_LENGTH,HORIZON)
    X_test,y_test = build_windows_numpy(test_df,keep_cols,TARGET_COL,WINDOW_LENGTH,HORIZON)
    log(f"Windows -> X_train:{X_train.shape}, y_train:{y_train.shape}, X_val:{X_val.shape}, y_val:{y_val.shape}, X_test:{X_test.shape}, y_test:{y_test.shape}")
    np.savez_compressed("xlstmts_dataset_snp500_daily.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, feature_cols=np.array(keep_cols), target_col=np.array([TARGET_COL]))
    train_df.to_csv("scaled_train_businessB_denoised.csv")
    val_df.to_csv("scaled_val_businessB_denoised.csv")
    test_df.to_csv("scaled_test_businessB_denoised.csv")
    log("Done. Saved:")
    log(" - xlstmts_dataset_snp500_daily.npz")
    log(" - scaled_train_businessB_denoised.csv")
    log(" - scaled_val_businessB_denoised.csv")
    log(" - scaled_test_businessB_denoised.csv")

if __name__ == "__main__":
    main()

