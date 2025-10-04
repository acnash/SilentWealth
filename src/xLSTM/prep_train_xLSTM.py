# ===========================================================
# combined_prep_and_tpe_train_xlstmts_multiticker_reporting.py
# Full pipeline: multi-ticker preprocessing + TPE tuning + verdict
# with publication-grade reporting for train/val/test.
#
# Key reporting artifacts (saved in ./artifacts):
#  - data_config.json                      # preprocessing + scaling description
#  - split_report.csv                      # per-ticker date ranges & counts per split
#  - scaling_params.csv                    # per-ticker scaler mins/maxes (train-denoised)
#  - window_counts_seqLEN.csv              # per-ticker window counts per split for chosen seq_len
#  - study_trials.csv                      # Optuna trials table (params + values)
#  - learning_curve_best.csv               # epoch-wise train/val loss for best trial
#  - predictions_val.csv                   # y_true, y_pred, y_naive, ticker (val)
#  - predictions_test.csv                  # y_true, y_pred, y_naive, ticker (test)
#  - metrics_val_overall.json              # overall metrics on val
#  - metrics_val_by_ticker.csv             # per-ticker metrics on val
#  - metrics_test_overall.json             # overall metrics on test
#  - metrics_test_by_ticker.csv            # per-ticker metrics on test
#  - xlstm_ts_best_state_dict_STABLE_PRICES.pt
#  - xlstm_ts_best_hparams_STABLE_PRICES.json
# ===========================================================

import os, math, json, warnings
import numpy as np
import pandas as pd
import pywt
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Any, Optional
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# Preprocessing config
# ----------------------
INPUT_PATH = "STABLE_PRICES.txt"         # unified file with multiple tickers
DATE_COL = "Date"
TICKER_COL = "Ticker"
TARGET_COL = "Close"
FEATURE_COLS = ["Open","High","Low","Close","Adj Close","Volume"]
HORIZON = 1
TRAIN_FRACTION = 0.86
VAL_FRACTION = 0.07
BUSINESS_DAY_METHOD = "time"             # {"time","ffill"}
WAVELET = "db4"
WLEVEL = None
THRESH_MODE = "soft"
VERBOSE = True

# ----------------------
# Model/training fixed config
# ----------------------
INPUT_SIZE = 1          # feed only TARGET_COL (univariate)
EMBED_DIM = 64
OUTPUT_SIZE = 1
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 200
PATIENCE = 30
CLIP_MAX_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# xLSTM block configs
MLSTM_CONV_K = 4
MLSTM_PROJ_SIZE = 2
MLSTM_HEADS = 2
SLSTM_CONV_K = 2
SLSTM_HEADS = 2
SLSTM_FF_FACTOR = 1.1

# ----------------------
# TPE search space
# ----------------------
SEQ_LEN_CHOICES = [60, 100, 150, 200, 256]
#LR_CHOICES = [1e-4, 5e-4, 1e-3]
BATCH_CHOICES = [8, 16, 32, 64]
EMBED_DIM_CHOICES = [32, 64, 128, 256, 384]
LR_RANGE = (1e-5, 3e-3)  # lower, upper

def sample_lr(trial):
    # log-uniform learning rate in [1e-5, 3e-3]
    return trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def log(msg: str) -> None:
    if VERBOSE: print(msg)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_any_table(path: str, date_col: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_csv(path, sep=None, engine="python") if ext in [".csv", ".txt", ".tsv", ".data", ""] else pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df

def ensure_business_days(df: pd.DataFrame) -> pd.DataFrame:
    bidx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    out = df.reindex(bidx)
    if BUSINESS_DAY_METHOD == "ffill":
        out = out.ffill()
    elif BUSINESS_DAY_METHOD == "time":
        out = out.interpolate(method="time").ffill().bfill()
    else:
        out = out.interpolate(method="time").ffill().bfill()
    return out

def wavelet_denoise_1d(x: np.ndarray, wavelet: str=WAVELET, level: Optional[int]=None, mode: str=THRESH_MODE) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if level is None: level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(len(x))) if len(x) > 0 else 0.0
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode=mode) for c in coeffs[1:]]
    y = pywt.waverec(denoised_coeffs, wavelet, mode="periodization")
    return y[:len(x)]

def wavelet_denoise_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns: out[c] = wavelet_denoise_1d(out[c].astype(float).values)
    return out

def darts_series_from_df(df: pd.DataFrame, cols: List[str]) -> TimeSeries:
    return TimeSeries.from_dataframe(df=df, value_cols=cols, freq="B")

def ts_to_df(ts: Optional[TimeSeries]) -> pd.DataFrame:
    if ts is None: return pd.DataFrame()
    df = ts.to_dataframe() if hasattr(ts, "to_dataframe") else ts.pd_dataframe(copy=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[-1] for c in df.columns]
    df.index.name = "Date"; return df

def split_indices(n: int, train_frac: float, val_frac: float) -> Tuple[int, int]:
    n_train = int(math.floor(n * train_frac)); n_val = int(math.floor(n * val_frac)); return n_train, n_val

def minmax_stats(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    s = pd.to_numeric(df[col], errors="coerce")
    return float(np.nanmin(s.values)), float(np.nanmax(s.values))

# -------------------------
# Window building (with labels for reporting)
# -------------------------
def build_windows_target_only(series: pd.Series, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray]:
    vals = series.values.astype("float32"); X_list, y_list = [], []; n = len(vals)
    for t in range(window, n - horizon + 1):
        X_list.append(vals[t - window:t]); y_list.append(vals[t + horizon - 1])
    X = np.stack(X_list, axis=0).astype("float32") if len(X_list) > 0 else np.zeros((0, window), dtype="float32")
    y = np.array(y_list, dtype="float32") if len(y_list) > 0 else np.zeros((0,), dtype="float32")
    X = X[..., None]; return X, y  # (N, T, 1), (N,)

def build_windows_from_multiticker(df: pd.DataFrame, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, tkrs = [], [], []
    for tkr, g in df.groupby(TICKER_COL, sort=False):
        g = g.sort_index()
        if TARGET_COL not in g.columns: continue
        s = g[TARGET_COL].astype("float32").dropna()
        if len(s) < window + horizon: continue
        X, y = build_windows_target_only(s, window=window, horizon=horizon)
        if X.shape[0] > 0:
            Xs.append(X); ys.append(y); tkrs.extend([tkr] * X.shape[0])
    if len(Xs) == 0: return np.zeros((0, window, 1), dtype="float32"), np.zeros((0,), dtype="float32"), np.array([])
    X_all = np.concatenate(Xs, axis=0); y_all = np.concatenate(ys, axis=0); tkrs_all = np.asarray(tkrs)
    return X_all, y_all, tkrs_all

# ------------------------------------------------------------
# Model definition (xLSTM-TS style)
# ------------------------------------------------------------
class Window1DDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and X.shape[-1] == 1
        if y.ndim == 1: y = y[:, None]
        self.X = X.astype("float32"); self.y = y.astype("float32")
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, idx: int): return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0, groups=channels, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1,2); x = nn.functional.pad(x, (self.pad, 0)); x = self.conv(x); return x.transpose(1,2)

class ProjectionBlock(nn.Module):
    def __init__(self, d: int, proj_size: int):
        super().__init__()
        hidden = max(1, d // proj_size)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, d: int, factor: float):
        super().__init__()
        hidden = max(1, int(math.ceil(d * factor)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

class mLSTMBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, heads: int, proj_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.proj = ProjectionBlock(d_model, proj_size)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm_in(x); x = x + self.conv(x)
        a, _ = self.attn(x, x, x, need_weights=False); x = x + a
        x, _ = self.lstm(x); x = x + self.proj(x); x = self.norm_out(x); return x + r

class sLSTMBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, heads: int, ff_factor: float):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.ff = FeedForward(d_model, ff_factor)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm_in(x); x = x + self.conv(x)
        a, _ = self.attn(x, x, x, need_weights=False); x = x + a
        x, _ = self.lstm(x); x = x + self.ff(x); x = self.norm_out(x); return x + r

class xLSTM_TS(nn.Module):
    def __init__(self, input_size: int, d_model: int, output_size: int):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.block1 = mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        self.block2 = sLSTMBlock(d_model=d_model, kernel=SLSTM_CONV_K, heads=SLSTM_HEADS, ff_factor=SLSTM_FF_FACTOR)
        self.block3 = mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x); x = self.block1(x); x = self.block2(x); x = self.block3(x); return self.head(x[:, -1, :])

# ------------------------------------------------------------
# Metrics & evaluation helpers
# ------------------------------------------------------------
def mse(y, p): return float(np.mean((p - y) ** 2))
def rmse(y, p): return float(np.sqrt(mse(y, p)))
def mae(y, p): return float(np.mean(np.abs(p - y)))
def mape(y, p, eps=1e-8):
    y_safe = np.where(np.abs(y) < eps, eps, np.abs(y)); return float(np.mean(np.abs((p - y) / y_safe))) * 100.0
def smape(y, p, eps=1e-8):
    num = np.abs(p - y); den = (np.abs(y) + np.abs(p) + eps) / 2.0; return float(np.mean(num / den)) * 100.0
def r2(y, p):
    y_bar = np.mean(y); ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y_bar) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def diebold_mariano(e_model: np.ndarray, e_naive: np.ndarray, h: int = 1) -> Dict[str, float]:
    """
    DM test (two-sided) using squared-error loss at horizon h.
    For h=1, HAC lag q = 0; for h>1, q = h-1 (simple Newey-West).
    """
    d = (e_model ** 2) - (e_naive ** 2)                         # loss differential
    n = len(d)
    if n < 3: return {"DM_stat": float("nan"), "p_value": float("nan")}
    q = max(0, h - 1)
    d_bar = np.mean(d)
    # Newey-West variance estimate
    gamma0 = np.var(d, ddof=1)
    var_hat = gamma0
    for lag in range(1, q + 1):
        cov = np.cov(d[:-lag], d[lag:], ddof=1)[0,1]
        w = 1.0 - lag / (q + 1)
        var_hat += 2.0 * w * cov
    dm_denom = np.sqrt(var_hat / n) if var_hat > 0 else np.nan
    dm_stat = d_bar / dm_denom if dm_denom and not np.isnan(dm_denom) else float("nan")
    # Approximate N(0,1) p-value (two-sided)
    try:
        from math import erf, sqrt
        def norm_cdf(z): return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        p = 2.0 * (1.0 - norm_cdf(abs(dm_stat)))
    except Exception:
        p = float("nan")
    return {"DM_stat": float(dm_stat), "p_value": float(p)}

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> Dict[str, float]:
    e_model = y_pred - y_true; e_naive = y_naive - y_true
    out = {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "sMAPE_pct": smape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    out.update(diebold_mariano(e_model, e_naive, h=HORIZON))
    return out

@torch.no_grad()
def evaluate_losses(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval(); total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb); loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0); count += xb.size(0)
    return total / max(1, count)

@torch.no_grad()
def predict_on_loader(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Return predictions and ground-truth in loader order."""
    model.eval(); preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE); yp = model(xb).detach().cpu().numpy().reshape(-1)
        yb_np = yb.numpy().reshape(-1)
        preds.append(yp); trues.append(yb_np)
    if len(preds) == 0: return np.array([]), np.array([])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------
def load_scaled_split_df(path: str) -> pd.DataFrame:
    assert os.path.exists(path), f"Missing file: {path}"
    df = pd.read_csv(path, sep=None, engine="python")
    assert DATE_COL in df.columns and TICKER_COL in df.columns and TARGET_COL in df.columns, "Missing required columns"
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL).sort_index()
    return df[[TICKER_COL, TARGET_COL]].copy()

def make_arrays(seq_len: int) -> Dict[str, Any]:
    train_df = load_scaled_split_df("scaled_train_businessB_denoised.csv")
    val_df   = load_scaled_split_df("scaled_val_businessB_denoised.csv")
    test_df  = load_scaled_split_df("scaled_test_businessB.csv")  # test is scaled but not denoised
    Xtr, ytr, ttr = build_windows_from_multiticker(train_df, window=seq_len, horizon=HORIZON)
    Xva, yva, tva = build_windows_from_multiticker(val_df,   window=seq_len, horizon=HORIZON)
    Xte, yte, tte = build_windows_from_multiticker(test_df,  window=seq_len, horizon=HORIZON)
    return {"Xtr":Xtr, "ytr":ytr, "ttr":ttr, "Xva":Xva, "yva":yva, "tva":tva, "Xte":Xte, "yte":yte, "tte":tte}

def make_loaders(arrs: Dict[str, Any], batch: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = Window1DDataset(arrs["Xtr"], arrs["ytr"])
    val_ds   = Window1DDataset(arrs["Xva"], arrs["yva"])
    test_ds  = Window1DDataset(arrs["Xte"], arrs["yte"])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train(); total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        pred = model(xb); loss = loss_fn(pred, yb); loss.backward()
        if CLIP_MAX_NORM and CLIP_MAX_NORM > 0: nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
        opt.step(); total += loss.item() * xb.size(0); count += xb.size(0)
    return total / max(1, count)

# ------------------------------------------------------------
# Preprocess end-to-end for multi-ticker (LEAK-SAFE DENOISING)
# ------------------------------------------------------------
def preprocess_and_save() -> None:
    ensure_dir("artifacts")
    log("Step 1: Load multi-ticker data")
    df_all = read_any_table(INPUT_PATH, DATE_COL)
    assert TICKER_COL in df_all.columns, f"Missing column '{TICKER_COL}'"
    missing = [c for c in FEATURE_COLS if c not in df_all.columns]
    if missing: log(f"Warning: missing columns in input: {missing}")
    keep_cols = [c for c in FEATURE_COLS if c in df_all.columns]
    df_all = df_all[[TICKER_COL] + keep_cols].copy()
    log(f"Loaded {len(df_all)} rows across {df_all[TICKER_COL].nunique()} tickers")

    # Reporting collectors
    split_rows = []
    scaling_rows = []

    train_frames, val_frames, test_frames = [], [], []

    for ticker, g in df_all.groupby(TICKER_COL, sort=False):
        log(f"\n--- Ticker: {ticker} ---")
        g_num = g.drop(columns=[TICKER_COL]).copy()
        raw_start, raw_end, raw_n = g_num.index.min(), g_num.index.max(), len(g_num)
        log("Clean nulls"); g_num = g_num.replace([np.inf, -np.inf], np.nan).dropna(how="all"); g_num = g_num.interpolate(method="time").ffill().bfill()
        log("Align to business days"); g_num = ensure_business_days(g_num)
        n_total = len(g_num); start, end = g_num.index.min(), g_num.index.max()

        # --- Split first to avoid denoising leakage ---
        n_train, n_val = split_indices(n_total, TRAIN_FRACTION, VAL_FRACTION)
        train_raw_df = g_num.iloc[:n_train].copy()
        val_raw_df   = g_num.iloc[n_train:n_train + n_val].copy()
        test_raw_df  = g_num.iloc[n_train + n_val:].copy()

        # --- Denoise ONLY train/val (split-local), keep test RAW ---
        log("Wavelet denoise (train & val ONLY; leak-safe)")
        train_den_df = wavelet_denoise_df(train_raw_df, keep_cols) if len(train_raw_df) else train_raw_df
        val_den_df   = wavelet_denoise_df(val_raw_df,   keep_cols) if len(val_raw_df)   else val_raw_df

        # Build TimeSeries from split-local data
        train_den_ts = darts_series_from_df(train_den_df, keep_cols) if len(train_den_df) else None
        val_den_ts   = darts_series_from_df(val_den_df, keep_cols)   if len(val_den_df)   else None
        test_raw_ts  = darts_series_from_df(test_raw_df, keep_cols)  if len(test_raw_df)  else None

        # --- Scaling: fit on denoised TRAIN (per ticker), apply to denoised VAL and RAW TEST ---
        log("Scale (fit on denoised train only). Test is RAW but scaled with the same scaler.")
        scaler = Scaler()  # MinMax(0,1)
        if train_den_ts is not None and len(train_den_ts) > 0:
            scaler.fit(train_den_ts)
            train_s = scaler.transform(train_den_ts)
            val_s   = scaler.transform(val_den_ts)  if val_den_ts  is not None and len(val_den_ts)  > 0 else None
            test_s  = scaler.transform(test_raw_ts) if test_raw_ts is not None and len(test_raw_ts) > 0 else None
            # Record scaling mins/maxes for TARGET_COL on train-denoised (pre-scale)
            if TARGET_COL in keep_cols and TARGET_COL in train_den_df.columns and len(train_den_df) > 0:
                vmin, vmax = minmax_stats(train_den_df, TARGET_COL)
                scaling_rows.append({"Ticker": ticker, "target_min_train_den": vmin, "target_max_train_den": vmax})
        else:
            train_s, val_s, test_s = None, None, None

        # Convert back to DataFrames for saving/aggregation
        train_df = ts_to_df(train_s)
        val_df   = ts_to_df(val_s)
        test_df  = ts_to_df(test_s)

        # Attach ticker column and append to master lists
        for df, bucket in [(train_df, train_frames), (val_df, val_frames), (test_df, test_frames)]:
            if df is None or df.empty: continue
            df = df.reset_index().rename(columns={"index": DATE_COL})
            df[TICKER_COL] = ticker
            bucket.append(df)

        # Split report row
        def safe_minmax(d: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
            if d is None or len(d) == 0: return None, None, 0
            return d.index.min(), d.index.max(), len(d)
        tr_s, tr_e, tr_n = safe_minmax(train_raw_df)
        va_s, va_e, va_n = safe_minmax(val_raw_df)
        te_s, te_e, te_n = safe_minmax(test_raw_df)
        split_rows.append({
            "Ticker": ticker,
            "raw_start": raw_start, "raw_end": raw_end, "raw_len": raw_n,
            "aligned_start": start, "aligned_end": end, "aligned_len": n_total,
            "train_start": tr_s, "train_end": tr_e, "train_len": tr_n,
            "val_start": va_s, "val_end": va_e, "val_len": va_n,
            "test_start": te_s, "test_end": te_e, "test_len": te_n
        })

        log(f"Lengths -> train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}")

    if len(train_frames) == 0: raise RuntimeError("No training data assembled.")
    train_all = pd.concat(train_frames).reset_index(drop=True).sort_values([DATE_COL, TICKER_COL])
    val_all   = pd.concat(val_frames)  .reset_index(drop=True).sort_values([DATE_COL, TICKER_COL]) if val_frames else pd.DataFrame(columns=[DATE_COL, TICKER_COL] + keep_cols)
    test_all  = pd.concat(test_frames) .reset_index(drop=True).sort_values([DATE_COL, TICKER_COL]) if test_frames else pd.DataFrame(columns=[DATE_COL, TICKER_COL] + keep_cols)

    # Save scaled splits:
    train_all.to_csv("scaled_train_businessB_denoised.csv", index=False)
    val_all.to_csv("scaled_val_businessB_denoised.csv", index=False)
    test_all.to_csv("scaled_test_businessB.csv", index=False)

    # Save reporting artifacts for preprocessing
    pd.DataFrame(split_rows).to_csv(os.path.join("artifacts", "split_report.csv"), index=False)
    pd.DataFrame(scaling_rows).to_csv(os.path.join("artifacts", "scaling_params.csv"), index=False)
    with open(os.path.join("artifacts", "data_config.json"), "w") as f:
        json.dump({
            "input_path": INPUT_PATH,
            "date_col": DATE_COL,
            "ticker_col": TICKER_COL,
            "target_col": TARGET_COL,
            "features": keep_cols,
            "horizon": HORIZON,
            "train_fraction": TRAIN_FRACTION,
            "val_fraction": VAL_FRACTION,
            "business_day_method": BUSINESS_DAY_METHOD,
            "wavelet": WAVELET,
            "wavelet_level": WLEVEL,
            "threshold_mode": THRESH_MODE,
            "denoise_scope": "split_level_train_and_val_only",
            "scaling": "per_ticker_minmax_[0,1]_fit_on_denoised_train; apply to denoised val and raw test"
        }, f, indent=2)

    log(f"Saved splits: train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")

# ------------------------------------------------------------
# Optuna TPE search
# ------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    seq_len   = trial.suggest_categorical("seq_len", SEQ_LEN_CHOICES)
    #lr        = trial.suggest_categorical("lr", LR_CHOICES)
    lr = sample_lr(trial)
    batch_sz  = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    embed_dim = trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES)

    arrs = make_arrays(seq_len)
    log(f"DEBUG: Xtr.shape={arrs['Xtr'].shape}, Xva.shape={arrs['Xva'].shape}, Xte.shape={arrs['Xte'].shape}")
    if arrs['Xtr'].shape[0] == 0 or arrs['Xva'].shape[0] == 0: raise optuna.TrialPruned()

    train_loader, val_loader, _ = make_loaders(arrs, batch_sz)

    model = xLSTM_TS(input_size=INPUT_SIZE, d_model=embed_dim, output_size=OUTPUT_SIZE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=False, min_lr=1e-8)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None; epochs_no_improve = 0
    curve = []  # record learning curve for reporting

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        va_loss = evaluate_losses(model, val_loader, loss_fn)
        curve.append({"epoch": epoch, "train_loss": float(tr_loss), "val_loss": float(va_loss)})
        trial.report(va_loss, step=epoch)
        scheduler.step(va_loss)
        if trial.should_prune(): raise optuna.TrialPruned()
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE: break

    if best_state is not None:
        trial.set_user_attr("best_state", best_state)
    trial.set_user_attr("learning_curve", curve)
    trial.set_user_attr("seq_len", seq_len)
    trial.set_user_attr("batch_size_final", batch_sz)
    trial.set_user_attr("embed_dim", embed_dim)
    return float(best_val)

def export_study(study: optuna.Study, out_csv: str) -> None:
    try:
        df = study.trials_dataframe(attrs=("number","value","params","user_attrs","state"))
        df.to_csv(out_csv, index=False)
    except Exception:
        # Fallback: minimal export
        rows = []
        for t in study.trials:
            rows.append({
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                **{f"param_{k}": v for k, v in t.params.items()}
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)

def window_count_report(arrs: Dict[str, Any], seq_len: int, out_csv: str) -> None:
    rows = []
    for split in ["ttr","tva","tte"]:
        if split not in arrs: continue
        tkrs = arrs[split]
        if tkrs is None or len(tkrs) == 0: continue
        cnt = Counter(tkrs)
        for tkr, nwin in cnt.items():
            rows.append({"split": split, "ticker": tkr, "seq_len": seq_len, "num_windows": int(nwin)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def predict_and_report(model: nn.Module, loader: DataLoader, X: np.ndarray, y: np.ndarray, tickers: np.ndarray, out_prefix: str) -> Dict[str, Any]:
    y_pred, y_true = predict_on_loader(model, loader)
    if len(y_pred) != len(y_true) or len(y_true) != len(tickers):
        # Align to min length to be safe
        n = min(len(y_pred), len(y_true), len(tickers))
        y_pred, y_true, tickers = y_pred[:n], y_true[:n], tickers[:n]
    y_naive = X[:, -1, 0][:len(y_true)]
    # overall metrics
    overall = metrics_dict(y_true, y_pred, y_naive)
    with open(os.path.join("artifacts", f"metrics_{out_prefix}_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    # per-ticker metrics
    rows = []
    for tkr in sorted(set(tickers.tolist())):
        idx = (tickers == tkr)
        m = metrics_dict(y_true[idx], y_pred[idx], y_naive[idx])
        m_row = {"Ticker": tkr}; m_row.update(m); rows.append(m_row)
    by_ticker_df = pd.DataFrame(rows)
    by_ticker_df.to_csv(os.path.join("artifacts", f"metrics_{out_prefix}_by_ticker.csv"), index=False)
    # save predictions
    pred_df = pd.DataFrame({
        "Ticker": tickers,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_naive": y_naive,
        "err_model": y_pred - y_true,
        "err_naive": y_naive - y_true
    })
    pred_df.to_csv(os.path.join("artifacts", f"predictions_{out_prefix}.csv"), index=False)
    return {"overall": overall, "by_ticker": by_ticker_df}

def verdict_from_metrics(test_mse: float, baseline_mse: float) -> str:
    rmse_val = float(np.sqrt(test_mse)) if math.isfinite(test_mse) else float("inf")
    imp = (baseline_mse - test_mse) / baseline_mse if (math.isfinite(baseline_mse) and baseline_mse > 0) else 0.0
    if imp >= 0.25 and rmse_val <= 0.03: return "EXCELLENT: large improvement over naïve baseline with very low error."
    elif imp >= 0.10 and rmse_val <= 0.05: return "GOOD: clear improvement over baseline with low error."
    elif imp > 0.0 or rmse_val <= 0.07: return "FAIR: modest improvement or acceptable error; may need more tuning/data."
    else: return "POOR: does not beat baseline meaningfully; consider revising features/model/tuning."

def train_eval_best(study: optuna.Study) -> Dict[str, Any]:
    ensure_dir("artifacts")
    export_study(study, os.path.join("artifacts", "study_trials.csv"))

    best_trial = study.best_trial
    hp = {
        "seq_len": best_trial.params["seq_len"],
        "lr": best_trial.params["lr"],
        "batch_size": best_trial.params["batch_size"],
        "embed_dim": best_trial.params.get("embed_dim", EMBED_DIM),
        "optimizer": "Adam",
        "loss": "MSE",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "clip_max_norm": CLIP_MAX_NORM
    }

    # Save learning curve of the best trial
    curve = best_trial.user_attrs.get("learning_curve", [])
    if curve:
        pd.DataFrame(curve).to_csv(os.path.join("artifacts", "learning_curve_best.csv"), index=False)

    arrs = make_arrays(hp["seq_len"])
    # window count reporting for the chosen seq_len
    window_count_report(arrs, hp["seq_len"], os.path.join("artifacts", f"window_counts_seq{hp['seq_len']}.csv"))

    train_loader, val_loader, test_loader = make_loaders(arrs, hp["batch_size"])

    model = xLSTM_TS(input_size=INPUT_SIZE, d_model=hp["embed_dim"], output_size=OUTPUT_SIZE).to(DEVICE)
    best_state = best_trial.user_attrs.get("best_state", None)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    loss_fn = nn.MSELoss()

    # Re-evaluate losses
    val_mse  = evaluate_losses(model, val_loader, loss_fn)
    test_mse = evaluate_losses(model, test_loader, loss_fn)

    # Baseline (persistence) on test (and val for reporting)
    Xva, yva, tva = arrs["Xva"], arrs["yva"], arrs["tva"]
    Xte, yte, tte = arrs["Xte"], arrs["yte"], arrs["tte"]
    naive_val = Xva[:, -1, 0] if Xva.shape[0] > 0 else np.array([])
    naive_test = Xte[:, -1, 0] if Xte.shape[0] > 0 else np.array([])
    baseline_mse_val = float(np.mean((naive_val - yva) ** 2)) if len(naive_val) else float("nan")
    baseline_mse_test = float(np.mean((naive_test - yte) ** 2)) if len(naive_test) else float("nan")

    # Rich prediction reports (overall + per ticker) for val and test
    _val_report  = predict_and_report(model, val_loader, Xva, yva, tva, out_prefix="val")
    _test_report = predict_and_report(model, test_loader, Xte, yte, tte, out_prefix="test")

    # Save model + hyperparams
    torch.save(model.state_dict(), os.path.join("artifacts", "xlstm_ts_best_state_dict_STABLE_PRICES.pt"))
    with open(os.path.join("artifacts", "xlstm_ts_best_hparams_STABLE_PRICES.json"), "w") as f:
        json.dump({
            "best_params": hp,
            "best_val_mse_during_tuning": study.best_value,
            "val_mse": val_mse,
            "val_baseline_mse": baseline_mse_val,
            "test_mse": test_mse,
            "baseline_test_mse": baseline_mse_test
        }, f, indent=2)

    verdict = verdict_from_metrics(test_mse, baseline_mse_test)

    print("\n=== BEST TRIAL SUMMARY ===")
    print(f"Best params: {hp}")
    print(f"Best val MSE during tuning: {study.best_value:.6f}")
    print(f"Re-evaluated Val MSE: {val_mse:.6f}  | Baseline Val MSE: {baseline_mse_val:.6f}")
    print(f"Test MSE (best model): {test_mse:.6f} | Baseline Test MSE: {baseline_mse_test:.6f}")
    print(f"Verdict: {verdict}")
    print("Saved artifacts in ./artifacts")

    return {"hp": hp, "val_mse": val_mse, "test_mse": test_mse, "baseline_mse": baseline_mse_test, "verdict": verdict}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    preprocess_and_save()
    sampler = TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print("Starting TPE search over {seq_len, lr, batch_size, embed_dim} ...")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    _ = train_eval_best(study)

if __name__ == "__main__":
    main()
