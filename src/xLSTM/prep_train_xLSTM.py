# ===========================================================
# train_all_combos_xlstmts.py
# Iterate over training-data files (e.g., xLSTM/GOOGL_PRICES.txt),
# train xLSTM-TS per file, and save models, reports, and graphs per run.
#
# NEW:
#  - CLI switch to run on a single file or an entire directory
#  - Tunable architecture: number of layers + block pattern
#  - Docstrings for all functions/classes
# ===========================================================

import os, math, json, warnings, glob, argparse
import numpy as np
import pandas as pd
import pywt
from collections import Counter
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

# Plotting (non-interactive)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Darts scaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

# Torch / Optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# Global config
# ----------------------
INPUT_DIR = "xLSTM"                                 # directory containing *_PRICES.txt combo files
DATE_COL = "Date"
TICKER_COL = "Ticker"
TARGET_COL = "Close"
FEATURE_COLS = ["Open","High","Low","Close","Adj Close","Volume"]
HORIZON = 1
TRAIN_FRACTION = 0.86
VAL_FRACTION = 0.07
BUSINESS_DAY_METHOD = "time"                        # {"time","ffill"}
WAVELET = "db4"
WLEVEL = None
THRESH_MODE = "soft"
VERBOSE = True

INPUT_SIZE = 1
OUTPUT_SIZE = 1
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 200
PATIENCE = 30
CLIP_MAX_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Block hyperparams (shared)
MLSTM_CONV_K = 4
MLSTM_PROJ_SIZE = 2
MLSTM_HEADS = 2
SLSTM_CONV_K = 2
SLSTM_HEADS = 2
SLSTM_FF_FACTOR = 1.1

# Search spaces
SEQ_LEN_CHOICES = [60, 100, 150, 200, 256]
BATCH_CHOICES = [8, 16, 32, 64]
EMBED_DIM_CHOICES = [64, 128, 192, 256, 384, 512]
LR_RANGE = (1e-5, 3e-3)
# NEW: architecture choices
NUM_LAYERS_MIN, NUM_LAYERS_MAX = 2, 6
ARCH_STYLE_CHOICES = ["alternating_ms", "all_m", "all_s"]

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# runtime context for per-run paths
CURRENT = {
    "artifact_dir": None,
    "train_path": None,
    "val_path": None,
    "test_path": None,
    "run_id": None,
    "input_path": None
}

# ----------------------
# Utility & IO
# ----------------------
def log(msg: str) -> None:
    """Print a message if VERBOSE is enabled."""
    if VERBOSE:
        print(msg)

def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> None:
    """Create directory `p` if it doesn't exist."""
    os.makedirs(p, exist_ok=True)

def read_any_table(path: str, date_col: str) -> pd.DataFrame:
    """
    Load a delimited text/CSV file, parse `date_col` to DatetimeIndex,
    and return a time-sorted DataFrame.
    """
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_csv(path, sep=None, engine="python") if ext in [".csv", ".txt", ".tsv", ".data", ""] else pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df

def ensure_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a continuous business-day frequency (Mon–Fri) between
    the first and last timestamps and fill gaps:
      - If BUSINESS_DAY_METHOD == "time": time-based interpolate, then ffill/bfill
      - If BUSINESS_DAY_METHOD == "ffill": forward-fill only
      - Else: time interpolate + ffill/bfill
    Note: holidays are treated as business days and may be filled.
    """
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
    """
    Wavelet denoising of a 1D array using universal thresholding and 'periodization' mode.
    Returns a length-preserving denoised array.
    """
    x = np.asarray(x, dtype=float)
    if level is None:
        level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(len(x))) if len(x) > 0 else 0.0
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode=mode) for c in coeffs[1:]]
    y = pywt.waverec(denoised_coeffs, wavelet, mode="periodization")
    return y[:len(x)]

def wavelet_denoise_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Apply wavelet denoising column-wise to `cols` in a DataFrame.
    Columns not present are skipped.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = wavelet_denoise_1d(out[c].astype(float).values)
    return out

def darts_series_from_df(df: pd.DataFrame, cols: List[str]) -> TimeSeries:
    """Create a Darts TimeSeries from `df` using `cols` as components (freq='B')."""
    return TimeSeries.from_dataframe(df=df, value_cols=cols, freq="B")

def ts_to_df(ts: Optional[TimeSeries]) -> pd.DataFrame:
    """
    Convert a Darts TimeSeries back to a pandas DataFrame.
    Flattens MultiIndex columns and names index 'Date'.
    """
    if ts is None:
        return pd.DataFrame()
    df = ts.to_dataframe() if hasattr(ts, "to_dataframe") else ts.pd_dataframe(copy=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df.index.name = "Date"
    return df

def split_indices(n: int, train_frac: float, val_frac: float) -> Tuple[int, int]:
    """Compute integer counts for train and val splits from total length `n`."""
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    return n_train, n_val

def minmax_stats(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    """Return (min, max) for column `col`, ignoring non-numeric values."""
    s = pd.to_numeric(df[col], errors="coerce")
    return float(np.nanmin(s.values)), float(np.nanmax(s.values))

# -------------------------
# Windowing
# -------------------------
def build_windows_target_only(series: pd.Series, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows on a single Series for one-step-ahead targets.
    Returns:
      X: (num_samples, window, 1)
      y: (num_samples,)
    """
    vals = series.values.astype("float32")
    X_list, y_list = [], []
    n = len(vals)
    for t in range(window, n - horizon + 1):
        X_list.append(vals[t - window:t])
        y_list.append(vals[t + horizon - 1])
    X = np.stack(X_list, axis=0).astype("float32") if len(X_list) > 0 else np.zeros((0, window), dtype="float32")
    y = np.array(y_list, dtype="float32") if len(y_list) > 0 else np.zeros((0,), dtype="float32")
    X = X[..., None]
    return X, y

def build_windows_from_multiticker(df: pd.DataFrame, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build windows per ticker, then concatenate:
      - sequences are per-ticker, contiguous, time-ordered
      - returns (X_all, y_all, tickers_array)
    """
    Xs, ys, tkrs = [], [], []
    for tkr, g in df.groupby(TICKER_COL, sort=False):
        g = g.sort_index()
        if TARGET_COL not in g.columns:
            continue
        s = g[TARGET_COL].astype("float32").dropna()
        if len(s) < window + horizon:
            continue
        X, y = build_windows_target_only(s, window=window, horizon=horizon)
        if X.shape[0] > 0:
            Xs.append(X)
            ys.append(y)
            tkrs.extend([tkr] * X.shape[0])
    if len(Xs) == 0:
        return np.zeros((0, window, 1), dtype="float32"), np.zeros((0,), dtype="float32"), np.array([])
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    tkrs_all = np.asarray(tkrs)
    return X_all, y_all, tkrs_all

# ------------------------------------------------------------
# Model (xLSTM-TS) — dynamic architecture
# ------------------------------------------------------------
class Window1DDataset(Dataset):
    """Simple dataset wrapping (X, y) arrays for 1D windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and X.shape[-1] == 1
        if y.ndim == 1:
            y = y[:, None]
        self.X = X.astype("float32")
        self.y = y.astype("float32")
    def __len__(self) -> int:
        return self.X.shape[0]
    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class CausalConv1d(nn.Module):
    """Depthwise 1D convolution with causal padding (left pad only)."""
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0, groups=channels, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

class ProjectionBlock(nn.Module):
    """Two-layer MLP projection used inside mLSTM block residual path."""
    def __init__(self, d: int, proj_size: int):
        super().__init__()
        hidden = max(1, d // proj_size)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FeedForward(nn.Module):
    """Two-layer MLP feed-forward used inside sLSTM block residual path."""
    def __init__(self, d: int, factor: float):
        super().__init__()
        hidden = max(1, int(math.ceil(d * factor)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class mLSTMBlock(nn.Module):
    """
    'm' block: LN → causal depthwise Conv → MHA → LSTM → Projection MLP + residual norms.
    """
    def __init__(self, d_model: int, kernel: int, heads: int, proj_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.proj = ProjectionBlock(d_model, proj_size)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm_in(x)
        x = x + self.conv(x)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = x + a
        x, _ = self.lstm(x)
        x = x + self.proj(x)
        x = self.norm_out(x)
        return x + r

class sLSTMBlock(nn.Module):
    """
    's' block: LN → causal depthwise Conv → MHA → LSTM → FeedForward MLP + residual norms.
    """
    def __init__(self, d_model: int, kernel: int, heads: int, ff_factor: float):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.ff = FeedForward(d_model, ff_factor)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm_in(x)
        x = x + self.conv(x)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = x + a
        x, _ = self.lstm(x)
        x = x + self.ff(x)
        x = self.norm_out(x)
        return x + r

class xLSTM_TS(nn.Module):
    """
    Extended LSTM time-series model with dynamic architecture.
    Args:
      input_size: feature dimension per timestep (1 for univariate)
      d_model: embedding / model width
      output_size: prediction size (1)
      num_layers: total number of stacked blocks
      arch_style: one of {"alternating_ms","all_m","all_s"}
    """
    def __init__(self, input_size: int, d_model: int, output_size: int,
                 num_layers: int = 3, arch_style: str = "alternating_ms"):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert arch_style in {"alternating_ms", "all_m", "all_s"}
        self.embed = nn.Linear(input_size, d_model)
        blocks: List[nn.Module] = []
        for i in range(num_layers):
            if arch_style == "all_m" or (arch_style == "alternating_ms" and i % 2 == 0):
                blocks.append(mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE))
            else:
                blocks.append(sLSTMBlock(d_model=d_model, kernel=SLSTM_CONV_K, heads=SLSTM_HEADS, ff_factor=SLSTM_FF_FACTOR))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: embed → stacked blocks → predict last timestep."""
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, -1, :])

# ------------------------------------------------------------
# Metrics & evaluation helpers
# ------------------------------------------------------------
def mse(y, p):
    """Mean Squared Error."""
    return float(np.mean((p - y) ** 2))

def rmse(y, p):
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y, p)))

def mae(y, p):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(p - y)))

def mape(y, p, eps=1e-8):
    """Mean Absolute Percentage Error (in %)."""
    y_safe = np.where(np.abs(y) < eps, eps, np.abs(y))
    return float(np.mean(np.abs((p - y) / y_safe))) * 100.0

def smape(y, p, eps=1e-8):
    """Symmetric MAPE (in %)."""
    num = np.abs(p - y)
    den = (np.abs(y) + np.abs(p) + eps) / 2.0
    return float(np.mean(num / den)) * 100.0

def r2(y, p):
    """Coefficient of determination R^2."""
    y_bar = np.mean(y)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def diebold_mariano(e_model: np.ndarray, e_naive: np.ndarray, h: int = 1) -> Dict[str, float]:
    """
    Diebold–Mariano test (two-sided) comparing squared-error loss at horizon h.
    Returns statistic and approx p-value (normal approximation).
    """
    d = (e_model ** 2) - (e_naive ** 2)
    n = len(d)
    if n < 3:
        return {"DM_stat": float("nan"), "p_value": float("nan")}
    q = max(0, h - 1)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    var_hat = gamma0
    for lag in range(1, q + 1):
        cov = np.cov(d[:-lag], d[lag:], ddof=1)[0, 1]
        w = 1.0 - lag / (q + 1)
        var_hat += 2.0 * w * cov
    dm_denom = np.sqrt(var_hat / n) if var_hat > 0 else np.nan
    dm_stat = d_bar / dm_denom if dm_denom and not np.isnan(dm_denom) else float("nan")
    try:
        from math import erf
        def norm_cdf(z): return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        p = 2.0 * (1.0 - norm_cdf(abs(dm_stat)))
    except Exception:
        p = float("nan")
    return {"DM_stat": float(dm_stat), "p_value": float(p)}

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> Dict[str, float]:
    """
    Compute a suite of regression metrics and the DM test vs persistence baseline.
    """
    e_model = y_pred - y_true
    e_naive = y_naive - y_true
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
    """
    Evaluate average loss over all batches in `loader` (AMP-friendly).
    """
    model.eval()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            pred = model(xb)
            loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(1, count)

@torch.no_grad()
def predict_on_loader(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass on `loader` and collect predictions and targets.
    Returns (y_pred, y_true) aligned in batch order.
    """
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yp = model(xb).detach().cpu().numpy().reshape(-1)
        yb_np = yb.numpy().reshape(-1)
        preds.append(yp)
        trues.append(yb_np)
    if len(preds) == 0:
        return np.array([]), np.array([])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def load_scaled_split_df(path: str) -> pd.DataFrame:
    """
    Load a split CSV (scaled_*), parse Date to index, and return [Ticker, TARGET_COL].
    """
    assert os.path.exists(path), f"Missing file: {path}"
    df = pd.read_csv(path, sep=None, engine="python")
    assert DATE_COL in df.columns and TICKER_COL in df.columns and TARGET_COL in df.columns, "Missing required columns"
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL).sort_index()
    return df[[TICKER_COL, TARGET_COL]].copy()

def make_arrays(seq_len: int) -> Dict[str, Any]:
    """
    Build (X, y, ticker) arrays for train/val/test using the current run's split files.
    """
    train_df = load_scaled_split_df(CURRENT["train_path"])
    val_df   = load_scaled_split_df(CURRENT["val_path"])
    test_df  = load_scaled_split_df(CURRENT["test_path"])
    Xtr, ytr, ttr = build_windows_from_multiticker(train_df, window=seq_len, horizon=HORIZON)
    Xva, yva, tva = build_windows_from_multiticker(val_df,   window=seq_len, horizon=HORIZON)
    Xte, yte, tte = build_windows_from_multiticker(test_df,  window=seq_len, horizon=HORIZON)
    return {"Xtr":Xtr, "ytr":ytr, "ttr":ttr, "Xva":Xva, "yva":yva, "tva":tva, "Xte":Xte, "yte":yte, "tte":tte}

def make_loaders(arrs: Dict[str, Any], batch: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap arrays in Dataset/DataLoader objects.
    Train loader shuffles windows; val/test do not.
    """
    train_ds = Window1DDataset(arrs["Xtr"], arrs["ytr"])
    val_ds   = Window1DDataset(arrs["Xva"], arrs["yva"])
    test_ds  = Window1DDataset(arrs["Xte"], arrs["yte"])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

# ------------------------------------------------------------
# Training (AMP-enabled)
# ------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module, scaler: amp.GradScaler) -> float:
    """
    One epoch of training with AMP, gradient clipping, and AdamW.
    Returns average training loss.
    """
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            pred = model(xb)
            loss = loss_fn(pred, yb)
        scaler.scale(loss).backward()
        if CLIP_MAX_NORM and CLIP_MAX_NORM > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
        scaler.step(opt)
        scaler.update()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(1, count)

# ------------------------------------------------------------
# Preprocess per run (leak-safe denoising)
# ------------------------------------------------------------
def preprocess_and_save(input_path: str, artifact_dir: str) -> None:
    """
    Preprocess a multi-ticker input file into train/val/test CSVs with:
      - per-ticker chronological split (86/7/7)
      - denoise train/val only (wavelet), test remains raw
      - per-ticker MinMax scaling fit on denoised train; applied to denoised val and raw test
    Saves split files and reporting artifacts under `artifact_dir`.
    """
    ensure_dir(artifact_dir)
    log(f"[{CURRENT['run_id']}] Load data: {input_path}")
    df_all = read_any_table(input_path, DATE_COL)
    assert TICKER_COL in df_all.columns, f"Missing column '{TICKER_COL}'"
    keep_cols = [c for c in FEATURE_COLS if c in df_all.columns]
    df_all = df_all[[TICKER_COL] + keep_cols].copy()
    log(f"Rows: {len(df_all)} | Tickers: {df_all[TICKER_COL].nunique()}")

    split_rows, scaling_rows = [], []
    train_frames, val_frames, test_frames = [], [], []

    for ticker, g in df_all.groupby(TICKER_COL, sort=False):
        g_num = g.drop(columns=[TICKER_COL]).copy()
        raw_start, raw_end, raw_n = g_num.index.min(), g_num.index.max(), len(g_num)
        g_num = g_num.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        g_num = g_num.interpolate(method="time").ffill().bfill()
        g_num = ensure_business_days(g_num)
        n_total = len(g_num); start, end = g_num.index.min(), g_num.index.max()

        n_train, n_val = split_indices(n_total, TRAIN_FRACTION, VAL_FRACTION)
        train_raw_df = g_num.iloc[:n_train].copy()
        val_raw_df   = g_num.iloc[n_train:n_train + n_val].copy()
        test_raw_df  = g_num.iloc[n_train + n_val:].copy()

        train_den_df = wavelet_denoise_df(train_raw_df, keep_cols) if len(train_raw_df) else train_raw_df
        val_den_df   = wavelet_denoise_df(val_raw_df,   keep_cols) if len(val_raw_df)   else val_raw_df

        train_den_ts = darts_series_from_df(train_den_df, keep_cols) if len(train_den_df) else None
        val_den_ts   = darts_series_from_df(val_den_df,   keep_cols) if len(val_den_df)  else None
        test_raw_ts  = darts_series_from_df(test_raw_df,  keep_cols) if len(test_raw_df) else None

        scaler = Scaler()
        if train_den_ts is not None and len(train_den_ts) > 0:
            scaler.fit(train_den_ts)
            train_s = scaler.transform(train_den_ts)
            val_s   = scaler.transform(val_den_ts)  if val_den_ts  is not None and len(val_den_ts)  > 0 else None
            test_s  = scaler.transform(test_raw_ts) if test_raw_ts is not None and len(test_raw_ts) > 0 else None
            if TARGET_COL in keep_cols and TARGET_COL in train_den_df.columns and len(train_den_df) > 0:
                vmin, vmax = minmax_stats(train_den_df, TARGET_COL)
                scaling_rows.append({"Ticker": ticker, "target_min_train_den": vmin, "target_max_train_den": vmax})
        else:
            train_s, val_s, test_s = None, None, None

        train_df = ts_to_df(train_s)
        val_df   = ts_to_df(val_s)
        test_df  = ts_to_df(test_s)

        for df, bucket in [(train_df, train_frames), (val_df, val_frames), (test_df, test_frames)]:
            if df is None or df.empty:
                continue
            df = df.reset_index().rename(columns={"index": DATE_COL})
            df[TICKER_COL] = ticker
            bucket.append(df)

        def safe_minmax(d: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
            if d is None or len(d) == 0:
                return None, None, 0
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

    if len(train_frames) == 0:
        raise RuntimeError("No training data assembled.")
    train_all = pd.concat(train_frames).reset_index(drop=True).sort_values([DATE_COL, TICKER_COL])
    val_all   = pd.concat(val_frames).reset_index(drop=True).sort_values([DATE_COL, TICKER_COL]) if val_frames else pd.DataFrame(columns=[DATE_COL, TICKER_COL] + keep_cols)
    test_all  = pd.concat(test_frames).reset_index(drop=True).sort_values([DATE_COL, TICKER_COL]) if test_frames else pd.DataFrame(columns=[DATE_COL, TICKER_COL] + keep_cols)

    train_path = os.path.join(artifact_dir, "scaled_train_businessB_denoised.csv")
    val_path   = os.path.join(artifact_dir, "scaled_val_businessB_denoised.csv")
    test_path  = os.path.join(artifact_dir, "scaled_test_businessB.csv")

    train_all.to_csv(train_path, index=False)
    val_all.to_csv(val_path, index=False)
    test_all.to_csv(test_path, index=False)

    pd.DataFrame(split_rows).to_csv(os.path.join(artifact_dir, "split_report.csv"), index=False)
    pd.DataFrame(scaling_rows).to_csv(os.path.join(artifact_dir, "scaling_params.csv"), index=False)
    with open(os.path.join(artifact_dir, "data_config.json"), "w") as f:
        json.dump({
            "input_path": input_path,
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
            "scaling": "per_ticker_minmax_[0,1]_fit_on_denoised_train; apply to denoised val and raw test",
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    CURRENT["train_path"] = train_path
    CURRENT["val_path"] = val_path
    CURRENT["test_path"] = test_path

# ------------------------------------------------------------
# Optuna objective and study utils
# ------------------------------------------------------------
def sample_lr(trial: optuna.Trial) -> float:
    """Suggest a log-uniform learning rate."""
    return trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective:
      - samples seq_len, lr, batch_size, embed_dim
      - NEW: samples num_layers and arch_style (model architecture)
      - trains with early stopping & pruning
      - returns best validation MSE
    """
    seq_len   = trial.suggest_categorical("seq_len", SEQ_LEN_CHOICES)
    lr        = sample_lr(trial)
    batch_sz  = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    embed_dim = trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES)
    num_layers = trial.suggest_int("num_layers", NUM_LAYERS_MIN, NUM_LAYERS_MAX)
    arch_style = trial.suggest_categorical("arch_style", ARCH_STYLE_CHOICES)

    # sanity: embed_dim divisible by heads used in attention
    assert embed_dim % MLSTM_HEADS == 0 and embed_dim % SLSTM_HEADS == 0, "embed_dim must be divisible by num_heads"

    arrs = make_arrays(seq_len)
    if arrs['Xtr'].shape[0] == 0 or arrs['Xva'].shape[0] == 0:
        raise optuna.TrialPruned()
    train_loader, val_loader, _ = make_loaders(arrs, batch_sz)

    model = xLSTM_TS(
        input_size=INPUT_SIZE,
        d_model=embed_dim,
        output_size=OUTPUT_SIZE,
        num_layers=num_layers,
        arch_style=arch_style
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=False, min_lr=1e-8)
    loss_fn = nn.MSELoss()
    scaler = amp.GradScaler(enabled=USE_AMP)

    best_val = float("inf"); best_state = None; epochs_no_improve = 0
    curve = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, scaler)
        va_loss = evaluate_losses(model, val_loader, loss_fn)
        curve.append({"epoch": epoch, "train_loss": float(tr_loss), "val_loss": float(va_loss)})
        trial.report(va_loss, step=epoch); scheduler.step(va_loss)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    if best_state is not None:
        trial.set_user_attr("best_state", best_state)
    trial.set_user_attr("learning_curve", curve)
    trial.set_user_attr("seq_len", seq_len)
    trial.set_user_attr("batch_size", batch_sz)
    trial.set_user_attr("embed_dim", embed_dim)
    trial.set_user_attr("num_layers", num_layers)
    trial.set_user_attr("arch_style", arch_style)
    return float(best_val)

def export_study(study: optuna.Study, out_csv: str) -> None:
    """
    Export Optuna trials to CSV; fallback to minimal export on failure.
    """
    try:
        df = study.trials_dataframe(attrs=("number","value","params","user_attrs","state"))
        df.to_csv(out_csv, index=False)
    except Exception:
        rows = []
        for t in study.trials:
            rows.append({"number": t.number, "value": t.value, "state": str(t.state), **{f"param_{k}": v for k, v in t.params.items()}})
        pd.DataFrame(rows).to_csv(out_csv, index=False)

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def plot_learning_curve(curve: List[Dict[str, float]], out_png: str, title: str) -> None:
    """Save train/val learning curves as a PNG."""
    if not curve:
        return
    df = pd.DataFrame(curve)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str) -> None:
    """Plot parity (y_true vs y_pred) and save as PNG."""
    if len(y_true) == 0:
        return
    lims = [float(np.min([y_true.min(), y_pred.min()])), float(np.max([y_true.max(), y_pred.max()]))]
    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    plt.plot(lims, lims, linewidth=1)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_residual_hist(errors: np.ndarray, out_png: str, title: str) -> None:
    """Plot histogram of prediction errors and save as PNG."""
    if len(errors) == 0:
        return
    plt.figure()
    plt.hist(errors, bins=40)
    plt.xlabel("Prediction Error"); plt.ylabel("Count"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

# ------------------------------------------------------------
# Prediction reporting
# ------------------------------------------------------------
def predict_and_report(model: nn.Module, loader: DataLoader, X: np.ndarray, y: np.ndarray, tickers: np.ndarray, out_prefix: str) -> Dict[str, Any]:
    """
    Generate predictions on `loader`, compute overall & per-ticker metrics,
    and save CSVs with predictions and metrics.
    """
    y_pred, y_true = predict_on_loader(model, loader)
    if len(y_pred) != len(y_true) or len(y_true) != len(tickers):
        n = min(len(y_pred), len(y_true), len(tickers))
        y_pred, y_true, tickers = y_pred[:n], y_true[:n], tickers[:n]
    y_naive = X[:, -1, 0][:len(y_true)]
    overall = metrics_dict(y_true, y_pred, y_naive)
    with open(os.path.join(CURRENT["artifact_dir"], f"metrics_{out_prefix}_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    rows = []
    for tkr in sorted(set(tickers.tolist())):
        idx = (tickers == tkr)
        m = metrics_dict(y_true[idx], y_pred[idx], y_naive[idx])
        m_row = {"Ticker": tkr}
        m_row.update(m)
        rows.append(m_row)
    by_ticker_df = pd.DataFrame(rows)
    by_ticker_df.to_csv(os.path.join(CURRENT["artifact_dir"], f"metrics_{out_prefix}_by_ticker.csv"), index=False)
    pred_df = pd.DataFrame({
        "Ticker": tickers, "y_true": y_true, "y_pred": y_pred,
        "y_naive": y_naive, "err_model": y_pred - y_true, "err_naive": y_naive - y_true
    })
    pred_df.to_csv(os.path.join(CURRENT["artifact_dir"], f"predictions_{out_prefix}.csv"), index=False)
    return {"overall": overall, "by_ticker": by_ticker_df, "y_true": y_true, "y_pred": y_pred, "y_naive": y_naive}

def verdict_from_metrics(test_mse: float, baseline_mse: float) -> str:
    """
    Heuristic textual verdict vs baseline persistence based on MSE and RMSE.
    """
    rmse_val = float(np.sqrt(test_mse)) if math.isfinite(test_mse) else float("inf")
    imp = (baseline_mse - test_mse) / baseline_mse if (math.isfinite(baseline_mse) and baseline_mse > 0) else 0.0
    if imp >= 0.25 and rmse_val <= 0.03: return "EXCELLENT: large improvement over naïve baseline with very low error."
    elif imp >= 0.10 and rmse_val <= 0.05: return "GOOD: clear improvement over baseline with low error."
    elif imp > 0.0 or rmse_val <= 0.07: return "FAIR: modest improvement or acceptable error; may need more tuning/data."
    else: return "POOR: does not beat baseline meaningfully; consider revising features/model/tuning."

def train_eval_best(study: optuna.Study) -> Dict[str, Any]:
    """
    Restore best-trial weights, evaluate on val/test, save predictions/metrics/plots,
    and persist the model & hyperparameters under the run's artifact dir.
    """
    ensure_dir(CURRENT["artifact_dir"])
    export_study(study, os.path.join(CURRENT["artifact_dir"], "study_trials.csv"))

    best_trial = study.best_trial
    hp = {
        "seq_len": best_trial.params["seq_len"],
        "lr": best_trial.params["lr"],
        "batch_size": best_trial.params["batch_size"],
        "embed_dim": best_trial.params.get("embed_dim", EMBED_DIM_CHOICES[0]),
        "num_layers": best_trial.params.get("num_layers", 3),
        "arch_style": best_trial.params.get("arch_style", "alternating_ms"),
        "optimizer": "AdamW",
        "loss": "MSE",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "clip_max_norm": CLIP_MAX_NORM
    }

    curve = best_trial.user_attrs.get("learning_curve", [])
    if curve:
        pd.DataFrame(curve).to_csv(os.path.join(CURRENT["artifact_dir"], "learning_curve_best.csv"), index=False)
        plot_learning_curve(curve, os.path.join(CURRENT["artifact_dir"], "learning_curve_best.png"), f"{CURRENT['run_id']} — Learning Curve")

    arrs = make_arrays(hp["seq_len"])
    # window counts by ticker
    window_counts_path = os.path.join(CURRENT["artifact_dir"], f"window_counts_seq{hp['seq_len']}.csv")
    rows = []
    for split_key, tkrs in [("train", arrs["ttr"]), ("val", arrs["tva"]), ("test", arrs["tte"])]:
        if tkrs is not None and len(tkrs) > 0:
            cnt = Counter(tkrs)
            for tkr, nwin in cnt.items():
                rows.append({"split": split_key, "ticker": tkr, "seq_len": hp["seq_len"], "num_windows": int(nwin)})
    pd.DataFrame(rows).to_csv(window_counts_path, index=False)

    train_loader, val_loader, test_loader = make_loaders(arrs, hp["batch_size"])

    model = xLSTM_TS(
        input_size=INPUT_SIZE,
        d_model=hp["embed_dim"],
        output_size=OUTPUT_SIZE,
        num_layers=hp["num_layers"],
        arch_style=hp["arch_style"]
    ).to(DEVICE)

    best_state = best_trial.user_attrs.get("best_state", None)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    loss_fn = nn.MSELoss()

    val_mse  = evaluate_losses(model, val_loader, loss_fn)
    test_mse = evaluate_losses(model, test_loader, loss_fn)

    Xva, yva, tva = arrs["Xva"], arrs["yva"], arrs["tva"]
    Xte, yte, tte = arrs["Xte"], arrs["yte"], arrs["tte"]
    naive_val = Xva[:, -1, 0] if Xva.shape[0] > 0 else np.array([])
    naive_test = Xte[:, -1, 0] if Xte.shape[0] > 0 else np.array([])
    baseline_mse_val = float(np.mean((naive_val - yva) ** 2)) if len(naive_val) else float("nan")
    baseline_mse_test = float(np.mean((naive_test - yte) ** 2)) if len(naive_test) else float("nan")

    val_report  = predict_and_report(model, val_loader, Xva, yva, tva, out_prefix="val")
    test_report = predict_and_report(model, test_loader, Xte, yte, tte, out_prefix="test")

    # Graphs: parity and residuals for val/test
    plot_parity(val_report["y_true"],  val_report["y_pred"],  os.path.join(CURRENT["artifact_dir"], "parity_val.png"),  f"{CURRENT['run_id']} — Parity (Validation)")
    plot_parity(test_report["y_true"], test_report["y_pred"], os.path.join(CURRENT["artifact_dir"], "parity_test.png"), f"{CURRENT['run_id']} — Parity (Test)")
    plot_residual_hist(val_report["y_pred"] - val_report["y_true"],  os.path.join(CURRENT["artifact_dir"], "residuals_val.png"),  f"{CURRENT['run_id']} — Residuals (Validation)")
    plot_residual_hist(test_report["y_pred"] - test_report["y_true"], os.path.join(CURRENT["artifact_dir"], "residuals_test.png"), f"{CURRENT['run_id']} — Residuals (Test)")

    # Save model + hyperparams (per run id)
    model_path = os.path.join(CURRENT["artifact_dir"], f"xlstm_ts_best_state_dict_{CURRENT['run_id']}.pt")
    hparam_path = os.path.join(CURRENT["artifact_dir"], f"xlstm_ts_best_hparams_{CURRENT['run_id']}.json")
    torch.save(model.state_dict(), model_path)
    with open(hparam_path, "w") as f:
        json.dump({
            "run_id": CURRENT["run_id"],
            "input_path": CURRENT["input_path"],
            "best_params": hp,
            "best_val_mse_during_tuning": study.best_value,
            "val_mse": val_mse,
            "val_baseline_mse": baseline_mse_val,
            "test_mse": test_mse,
            "baseline_test_mse": baseline_mse_test
        }, f, indent=2)

    verdict = verdict_from_metrics(test_mse, baseline_mse_test)
    print(f"\n=== {CURRENT['run_id']} — BEST TRIAL SUMMARY ===")
    print(f"Best params: {hp}")
    print(f"Best val MSE during tuning: {study.best_value:.6f}")
    print(f"Re-evaluated Val MSE: {val_mse:.6f}  | Baseline Val MSE: {baseline_mse_val:.6f}")
    print(f"Test MSE (best model): {test_mse:.6f} | Baseline Test MSE: {baseline_mse_test:.6f}")
    print(f"Verdict: {verdict}")
    print(f"Artifacts saved in: {CURRENT['artifact_dir']}")
    return {"hp": hp, "val_mse": val_mse, "test_mse": test_mse, "baseline_mse": baseline_mse_test, "verdict": verdict}

# ------------------------------------------------------------
# Per-file runner
# ------------------------------------------------------------
def run_pipeline_for_file(input_path: str) -> None:
    """
    End-to-end pipeline for one input file:
      preprocess → TPE tuning → evaluate best → save artifacts.
    """
    run_id = os.path.splitext(os.path.basename(input_path))[0].replace("_PRICES","")
    artifact_dir = os.path.join("artifacts", run_id)
    ensure_dir(artifact_dir)
    CURRENT.update({"artifact_dir": artifact_dir, "run_id": run_id, "input_path": input_path})

    set_seed(SEED)
    preprocess_and_save(input_path, artifact_dir)

    sampler = TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print(f"\n[{run_id}] TPE search over {{seq_len, lr, batch_size, embed_dim, num_layers, arch_style}} …")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    _ = train_eval_best(study)

# ------------------------------------------------------------
# CLI & Main
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments controlling run mode and locations.
    Modes:
      - single: run once on --input_file
      - multi:  iterate all *_PRICES.txt in --input_dir
    """
    p = argparse.ArgumentParser(description="Train xLSTM-TS on single or multiple datasets.")
    p.add_argument("--run_mode", choices=["single", "multi"], default="multi",
                   help="Run once on a file ('single') or over a directory ('multi').")
    p.add_argument("--input_file", type=str, default=None,
                   help="Path to a single *_PRICES.txt file (used when --run_mode single).")
    p.add_argument("--input_dir", type=str, default=INPUT_DIR,
                   help="Directory containing *_PRICES.txt files (used when --run_mode multi).")
    p.add_argument("--n_trials", type=int, default=20, help="Optuna trials per run.")
    return p.parse_args()

def main() -> None:
    """
    Entry point: dispatch to single-file or multi-file training based on CLI args.
    """
    args = parse_args()
    if args.run_mode == "single":
        if not args.input_file or not os.path.exists(args.input_file):
            print("Please provide a valid --input_file pointing to a *_PRICES.txt file.")
            return
        # Run a single file
        try:
            run_pipeline_for_file(args.input_file)
        except Exception as e:
            print(f"❌ Run failed for {args.input_file}: {e}")
    else:
        # Multi-file: iterate directory
        files = sorted(glob.glob(os.path.join(args.input_dir, "*_PRICES.txt")))
        if not files:
            print(f"No files found in {args.input_dir} matching *_PRICES.txt")
            return
        print(f"Discovered {len(files)} files. Beginning training runs …")
        for fp in files:
            try:
                run_pipeline_for_file(fp)
            except Exception as e:
                print(f"❌ Run failed for {fp}: {e}")

if __name__ == "__main__":
    main()
