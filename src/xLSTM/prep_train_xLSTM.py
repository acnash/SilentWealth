# ===========================================================
# prep_train_xLSTM.py
# Train xLSTM-TS on either a single dataset file or iterate
# over a directory of *_PRICES.txt files. Saves artifacts,
# metrics, and graphs per run. Supports Optuna TPE tuning
# and tunable model architecture (num_layers, arch_style).
# ===========================================================

import os, math, json, warnings, glob, argparse
import numpy as np
import pandas as pd
import pywt
from collections import Counter
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from contextlib import nullcontext

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
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# Global config (edit as needed)
# ----------------------
INPUT_DIR = "xLSTM"                                 # directory containing *_PRICES.txt combo files
DATE_COL = "Date"
TICKER_COL = "Ticker"
TARGET_COL = "Close"
FEATURE_COLS = ["Open","High","Low","Close","Adj Close","Volume"]
HORIZON = 1                                         # day-ahead prediction
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

# Default block hyperparams (some can be tuned too if you like)
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
ARCH_STYLE_CHOICES = ["alternating_ms", "all_m", "all_s", "ms_first", "sm_first"]
NUM_LAYERS_CHOICES = [2, 3, 4, 6]                   # tune model depth
LR_RANGE = (1e-5, 3e-3)

# ----------------------
# AMP config (version-robust)
# ----------------------
AMP_ENABLED = torch.cuda.is_available()
AMP_DTYPE   = torch.bfloat16 if (AMP_ENABLED and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

def make_grad_scaler(enabled: bool):
    """
    Create a GradScaler compatible with the user's torch version.
    Falls back to legacy torch.cuda.amp if needed. On CPU, returns
    a dummy scaler that no-ops but matches the API used below.
    """
    if torch.cuda.is_available() and enabled:
        try:
            # Newer API: positional device string (no keyword)
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            # Older API: use legacy CUDA AMP
            from torch.cuda.amp import GradScaler as CudaGradScaler
            return CudaGradScaler(enabled=True)
    # Dummy scaler for CPU / disabled AMP
    class _DummyScaler:
        def scale(self, loss): return loss
        def unscale_(self, opt): pass
        def step(self, opt): opt.step()
        def update(self): pass
    return _DummyScaler()

def autocast_ctx():
    """
    Return an autocast context that works across torch versions.
    On CPU or when disabled, returns a nullcontext.
    """
    if torch.cuda.is_available() and AMP_ENABLED:
        try:
            return torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE)
        except TypeError:
            from torch.cuda.amp import autocast as legacy_autocast
            return legacy_autocast(enabled=True, dtype=AMP_DTYPE)
    return nullcontext()

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
    """Print a message if VERBOSE is True."""
    if VERBOSE:
        print(msg)

def set_seed(seed: int) -> None:
    """Set numpy and torch RNG seeds for reproducibility."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> None:
    """Create directory p if it doesn't exist."""
    os.makedirs(p, exist_ok=True)

def read_any_table(path: str, date_col: str) -> pd.DataFrame:
    """
    Read a delimited file (txt/csv/tsv) into a DataFrame, parse date index,
    and sort by time.
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
    Reindex to business days between the min/max dates (freq='B').
    Then fill values using the configured method:
      - 'time': time interpolation + ffill/bfill
      - 'ffill': forward fill only
      - other: default to time interpolation + ffill/bfill
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
    Wavelet denoise a 1D signal with universal thresholding.
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
    """Apply wavelet denoise to selected columns of a DataFrame."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = wavelet_denoise_1d(out[c].astype(float).values)
    return out

def darts_series_from_df(df: pd.DataFrame, cols: List[str]) -> TimeSeries:
    """Convert DataFrame to Darts TimeSeries with business-day frequency."""
    return TimeSeries.from_dataframe(df=df, value_cols=cols, freq="B")

def ts_to_df(ts: Optional[TimeSeries]) -> pd.DataFrame:
    """Convert Darts TimeSeries back to a flat DataFrame with 'Date' index name."""
    if ts is None: return pd.DataFrame()
    df = ts.to_dataframe() if hasattr(ts, "to_dataframe") else ts.pd_dataframe(copy=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df.index.name = "Date"
    return df

def split_indices(n: int, train_frac: float, val_frac: float) -> Tuple[int, int]:
    """Compute absolute counts for train and val given total n and fractions."""
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    return n_train, n_val

def minmax_stats(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    """Return (min, max) of a numeric column (nan-safe)."""
    s = pd.to_numeric(df[col], errors="coerce")
    return float(np.nanmin(s.values)), float(np.nanmax(s.values))

# -------------------------
# Windowing
# -------------------------
def build_windows_target_only(series: pd.Series, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows over a single univariate series.
    Returns:
      X: (N, window, 1)
      y: (N,)  <-- now returns the *residual* target: (future_value - last_input_value)
    """
    vals = series.values.astype("float32")
    X_list, y_list = [], []
    n = len(vals)
    for t in range(window, n - horizon + 1):
        X_list.append(vals[t - window:t])
        # last input in the window is vals[t-1]; target is vals[t + horizon - 1]
        y_list.append(vals[t + horizon - 1] - vals[t - 1])
    X = np.stack(X_list, axis=0).astype("float32") if len(X_list) > 0 else np.zeros((0, window), dtype="float32")
    y = np.array(y_list, dtype="float32") if len(y_list) > 0 else np.zeros((0,), dtype="float32")
    X = X[..., None]
    return X, y

def build_windows_from_multiticker(df: pd.DataFrame, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build windows per ticker and concatenate.
    Ensures each window comes from a single ticker (no timeline mixing).
    Returns:
      X_all: (N, window, 1)
      y_all: (N,)
      tkrs_all: (N,) ticker labels per window
    """
    Xs, ys, tkrs = [], [], []
    for tkr, g in df.groupby(TICKER_COL, sort=False):
        g = g.sort_index()
        if TARGET_COL not in g.columns: continue
        s = g[TARGET_COL].astype("float32").dropna()
        if len(s) < window + horizon: continue
        X, y = build_windows_target_only(s, window=window, horizon=horizon)
        if X.shape[0] > 0:
            Xs.append(X); ys.append(y); tkrs.extend([tkr] * X.shape[0])
    if len(Xs) == 0:
        return np.zeros((0, window, 1), dtype="float32"), np.zeros((0,), dtype="float32"), np.array([])
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    tkrs_all = np.asarray(tkrs)
    return X_all, y_all, tkrs_all

# ------------------------------------------------------------
# Model (xLSTM-TS) — tunable architecture
# ------------------------------------------------------------
class Window1DDataset(Dataset):
    """Torch dataset for 1D windowed sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and X.shape[-1] == 1
        if y.ndim == 1: y = y[:, None]
        self.X = X.astype("float32")
        self.y = y.astype("float32")
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, idx: int): return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class CausalConv1d(nn.Module):
    """Depthwise causal 1D conv with kernel padding."""
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
    """Simple bottleneck projection MLP."""
    def __init__(self, d: int, proj_size: int):
        super().__init__()
        hidden = max(1, d // proj_size)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FeedForward(nn.Module):
    """Position-wise feed-forward with expansion factor."""
    def __init__(self, d: int, factor: float):
        super().__init__()
        hidden = max(1, int(math.ceil(d * factor)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class mLSTMBlock(nn.Module):
    """Mixed block: conv + MHA + LSTM + projection."""
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
    """Simplified block: conv + MHA + LSTM + feed-forward."""
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
    Extended LSTM time-series model with tunable depth and block layout.

    Args:
      input_size: number of input features (1 for univariate)
      d_model: embedding/hidden size
      output_size: number of outputs (1 for next-step regression)
      num_layers: number of stacked blocks
      arch_style: one of {'alternating_ms','all_m','all_s','ms_first','sm_first'}
    """
    def __init__(self, input_size: int, d_model: int, output_size: int, num_layers: int, arch_style: str):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        blocks: List[nn.Module] = []

        def m(): return mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        def s(): return sLSTMBlock(d_model=d_model, kernel=SLSTM_CONV_K, heads=SLSTM_HEADS, ff_factor=SLSTM_FF_FACTOR)

        if arch_style == "all_m":
            blocks = [m() for _ in range(num_layers)]
        elif arch_style == "all_s":
            blocks = [s() for _ in range(num_layers)]
        elif arch_style == "ms_first":
            # first half m, second half s
            k = num_layers // 2
            blocks = [m() for _ in range(k)] + [s() for _ in range(num_layers - k)]
        elif arch_style == "sm_first":
            # first half s, second half m
            k = num_layers // 2
            blocks = [s() for _ in range(k)] + [m() for _ in range(num_layers - k)]
        else:  # "alternating_ms"
            for i in range(num_layers):
                blocks.append(m() if i % 2 == 0 else s())

        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, -1, :])

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
    Diebold-Mariano test (two-sided) for predictive accuracy using squared error.
    Returns test statistic and an approximate p-value under N(0,1).
    """
    d = (e_model ** 2) - (e_naive ** 2)
    n = len(d)
    if n < 3: return {"DM_stat": float("nan"), "p_value": float("nan")}
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
    """Bundle standard regression metrics + DM test vs naive."""
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
    """
    Evaluate average loss over a DataLoader.
    Uses AMP autocast context when available.
    """
    model.eval(); total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with autocast_ctx():
            pred = model(xb); loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0); count += xb.size(0)
    return total / max(1, count)

@torch.no_grad()
def predict_on_loader(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict for an entire DataLoader; returns (preds, trues).
    """
    model.eval(); preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        with autocast_ctx():
            yp = model(xb).detach().cpu().numpy().reshape(-1)
        yb_np = yb.numpy().reshape(-1)
        preds.append(yp); trues.append(yb_np)
    if len(preds) == 0: return np.array([]), np.array([])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def load_scaled_split_df(path: str) -> pd.DataFrame:
    """Load a preprocessed split CSV and return (Ticker, TARGET_COL) indexed by Date."""
    assert os.path.exists(path), f"Missing file: {path}"
    df = pd.read_csv(path, sep=None, engine="python")
    assert DATE_COL in df.columns and TICKER_COL in df.columns and TARGET_COL in df.columns, "Missing required columns"
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL).sort_index()
    return df[[TICKER_COL, TARGET_COL]].copy()

def make_arrays(seq_len: int) -> Dict[str, Any]:
    """
    Build windowed arrays for train/val/test from current split CSVs.
    Returns dict with X/y and tickers per split.
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
    Wrap window arrays into PyTorch DataLoaders.
    Train loader shuffles; val/test do not.
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
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: "torch.amp.GradScaler",  # quoted for Py<=3.10 forward refs
) -> float:
    """
    Train the model for one epoch over 'loader' using AMP when available.
    Returns the average training loss.
    """
    model.train(); total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            pred = model(xb); loss = loss_fn(pred, yb)
        scaler.scale(loss).backward()
        if CLIP_MAX_NORM and CLIP_MAX_NORM > 0:
            try:
                scaler.unscale_(opt)
            except AttributeError:
                # dummy scaler has no unscale_
                pass
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
        scaler.step(opt); scaler.update()
        total += loss.item() * xb.size(0); count += xb.size(0)
    return total / max(1, count)

# ------------------------------------------------------------
# Preprocess per run (leak-safe denoising)
# ------------------------------------------------------------
def preprocess_and_save(input_path: str, artifact_dir: str) -> None:
    """
    Read a combined prices file, align to business days, split per ticker into
    train/val/test (chronological), denoise train/val only, and scale per ticker
    using MinMax fitted on (denoised) train. Save split CSVs and reports.
    """
    ensure_dir(artifact_dir)
    run_id = CURRENT['run_id']
    log(f"[{run_id}] Load data: {input_path}")
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
            if df is None or df.empty: continue
            df = df.reset_index().rename(columns={"index": DATE_COL})
            df[TICKER_COL] = ticker
            bucket.append(df)

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

    if len(train_frames) == 0: raise RuntimeError("No training data assembled.")
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
    CURRENT["val_path"]   = val_path
    CURRENT["test_path"]  = test_path

# ------------------------------------------------------------
# Optuna objective and study utils
# ------------------------------------------------------------
def sample_lr(trial: optuna.Trial) -> float:
    """Suggest a log-uniform learning rate within LR_RANGE."""
    return trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective:
      - samples seq_len, lr, batch_size, embed_dim, num_layers, arch_style
      - trains with early stopping (patience)
      - returns best validation MSE
    """
    seq_len   = trial.suggest_categorical("seq_len", SEQ_LEN_CHOICES)
    lr        = sample_lr(trial)
    batch_sz  = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    embed_dim = trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES)
    num_layers = trial.suggest_categorical("num_layers", NUM_LAYERS_CHOICES)
    arch_style = trial.suggest_categorical("arch_style", ARCH_STYLE_CHOICES)

    # attention divisibility checks if you tune heads
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
        arch_style=arch_style,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=False, min_lr=1e-8)
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    scaler = make_grad_scaler(AMP_ENABLED)

    best_val = float("inf"); best_state = None; epochs_no_improve = 0
    curve = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, scaler)
        va_loss = evaluate_losses(model, val_loader, loss_fn)
        curve.append({"epoch": epoch, "train_loss": float(tr_loss), "val_loss": float(va_loss)})
        trial.report(va_loss, step=epoch); scheduler.step(va_loss)
        if trial.should_prune(): raise optuna.TrialPruned()
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE: break

    if best_state is not None: trial.set_user_attr("best_state", best_state)
    trial.set_user_attr("learning_curve", curve)
    trial.set_user_attr("seq_len", seq_len)
    trial.set_user_attr("batch_size", batch_sz)
    trial.set_user_attr("embed_dim", embed_dim)
    trial.set_user_attr("num_layers", num_layers)
    trial.set_user_attr("arch_style", arch_style)
    return float(best_val)

def export_study(study: optuna.Study, out_csv: str) -> None:
    """Export Optuna trials to CSV, with a robust fallback."""
    try:
        df = study.trials_dataframe(attrs=("number","value","params","user_attrs","state"))
        df.to_csv(out_csv, index=False)
    except Exception:
        rows = []
        for t in study.trials:
            rows.append({
                "number": t.number, "value": t.value, "state": str(t.state),
                **{f"param_{k}": v for k, v in t.params.items()}
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def plot_learning_curve(curve: List[Dict[str, float]], out_png: str, title: str) -> None:
    """Save a train-vs-val loss curve PNG."""
    if not curve: return
    df = pd.DataFrame(curve)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str) -> None:
    """Save a parity (y_true vs y_pred) scatter plot."""
    if len(y_true) == 0: return
    lims = [float(np.min([y_true.min(), y_pred.min()])), float(np.max([y_true.max(), y_pred.max()]))]
    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    plt.plot(lims, lims, linewidth=1)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_residual_hist(errors: np.ndarray, out_png: str, title: str) -> None:
    """Save a histogram of prediction errors."""
    if len(errors) == 0: return
    plt.figure()
    plt.hist(errors, bins=40)
    plt.xlabel("Prediction Error"); plt.ylabel("Count"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

# ------------------------------------------------------------
# Prediction reporting
# ------------------------------------------------------------
def predict_and_report(model: nn.Module, loader: DataLoader, X: np.ndarray, y: np.ndarray, tickers: np.ndarray, out_prefix: str) -> Dict[str, Any]:
    """
    Run predictions for a split and write:
      - overall metrics JSON
      - per-ticker metrics CSV
      - predictions CSV

    Important: `y` coming from the dataset is the *residual* (y_true - last_input).
    This function reconstructs absolute values for reporting:
      y_true_abs = y + last_input
      y_pred_abs = y_pred_res + last_input
    """
    # model outputs and dataset y are residuals
    y_pred_res, y_true_res = predict_on_loader(model, loader)
    if len(y_pred_res) != len(y_true_res) or len(y_true_res) != len(tickers):
        n = min(len(y_pred_res), len(y_true_res), len(tickers))
        y_pred_res, y_true_res, tickers = y_pred_res[:n], y_true_res[:n], tickers[:n]

    # reconstruct absolute values using last value from X windows
    last_vals = X[:, -1, 0][:len(y_true_res)]
    y_true_abs = y_true_res + last_vals
    y_pred_abs = y_pred_res + last_vals
    y_naive = last_vals  # naive prediction is the last value

    # overall metrics are computed on absolute values vs naive
    overall = metrics_dict(y_true_abs, y_pred_abs, y_naive)
    # add baseline MSE explicitly for convenience
    overall["baseline_MSE"] = float(np.mean((y_naive - y_true_abs) ** 2)) if len(y_naive) else float("nan")

    with open(os.path.join(CURRENT["artifact_dir"], f"metrics_{out_prefix}_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)

    # per-ticker metrics (absolute)
    rows = []
    unique_tkrs = sorted(set(tickers.tolist()))
    for tkr in unique_tkrs:
        idx = (tickers == tkr)
        if idx.sum() == 0: continue
        m = metrics_dict(y_true_abs[idx], y_pred_abs[idx], y_naive[idx])
        m_row = {"Ticker": tkr}
        m_row.update(m)
        rows.append(m_row)
    by_ticker_df = pd.DataFrame(rows)
    by_ticker_df.to_csv(os.path.join(CURRENT["artifact_dir"], f"metrics_{out_prefix}_by_ticker.csv"), index=False)

    # prediction CSV: include residuals and absolute values and naive
    pred_df = pd.DataFrame({
        "Ticker": tickers[:len(y_true_res)],
        "y_true_abs": y_true_abs,
        "y_pred_abs": y_pred_abs,
        "y_pred_res": y_pred_res,
        "y_true_res": y_true_res,
        "y_naive": y_naive,
        "err_model": y_pred_abs - y_true_abs,
        "err_naive": y_naive - y_true_abs
    })
    pred_df.to_csv(os.path.join(CURRENT["artifact_dir"], f"predictions_{out_prefix}.csv"), index=False)

    return {
        "overall": overall,
        "by_ticker": by_ticker_df,
        "y_true_res": y_true_res,
        "y_pred_res": y_pred_res,
        "y_true_abs": y_true_abs,
        "y_pred_abs": y_pred_abs,
        "y_naive": y_naive
    }


def verdict_from_metrics(test_mse: float, baseline_mse: float) -> str:
    """Make a simple textual verdict vs naive baseline."""
    rmse_val = float(np.sqrt(test_mse)) if math.isfinite(test_mse) else float("inf")
    imp = (baseline_mse - test_mse) / baseline_mse if (math.isfinite(baseline_mse) and baseline_mse > 0) else 0.0
    if imp >= 0.25 and rmse_val <= 0.03: return "EXCELLENT: large improvement over naïve baseline with very low error."
    elif imp >= 0.10 and rmse_val <= 0.05: return "GOOD: clear improvement over baseline with low error."
    elif imp > 0.0 or rmse_val <= 0.07: return "FAIR: modest improvement or acceptable error; may need more tuning/data."
    else: return "POOR: does not beat baseline meaningfully; consider revising features/model/tuning."

def train_eval_best(study: optuna.Study) -> Dict[str, Any]:
    """
    Rebuild the best model from the study, re-evaluate val/test,
    write artifacts/graphs, and return a summary dict.
    """
    ensure_dir(CURRENT["artifact_dir"])
    export_study(study, os.path.join(CURRENT["artifact_dir"], "study_trials.csv"))

    best_trial = study.best_trial
    hp = {
        "seq_len": best_trial.params["seq_len"],
        "lr": best_trial.params["lr"],
        "batch_size": best_trial.params["batch_size"],
        "embed_dim": best_trial.params.get("embed_dim", EMBED_DIM_CHOICES[0]),
        "num_layers": best_trial.params.get("num_layers", NUM_LAYERS_CHOICES[0]),
        "arch_style": best_trial.params.get("arch_style", ARCH_STYLE_CHOICES[0]),
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
        arch_style=hp["arch_style"],
    ).to(DEVICE)

    best_state = best_trial.user_attrs.get("best_state", None)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # keep the loss_fn definition (used elsewhere / for consistency)
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    # --- Replace the old evaluate_losses + manual baseline block with these calls ---
    # predict_and_report knows the model outputs residuals and reconstructs absolute values
    val_report = predict_and_report(model, val_loader, arrs["Xva"], arrs["yva"], arrs["tva"], out_prefix="val")
    test_report = predict_and_report(model, test_loader, arrs["Xte"], arrs["yte"], arrs["tte"], out_prefix="test")

    # read absolute MSEs (and baseline MSE) from the reports produced above
    val_mse = float(val_report["overall"].get("MSE", float("nan")))
    test_mse = float(test_report["overall"].get("MSE", float("nan")))

    baseline_mse_val = float(val_report["overall"].get("baseline_MSE", float("nan")))
    baseline_mse_test = float(test_report["overall"].get("baseline_MSE", float("nan")))

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
def run_pipeline_for_file(input_path: str, n_trials: int, multivariate: bool=False) -> None:
    """
    Run the full pipeline (preprocess -> Optuna -> evaluate) for a single input file.
    Artifacts are written to ./artifacts/<RUN_ID>/.
    """
    run_id = os.path.splitext(os.path.basename(input_path))[0].replace("_PRICES","")
    artifact_dir = os.path.join("artifacts", run_id)
    ensure_dir(artifact_dir)
    CURRENT.update({"artifact_dir": artifact_dir, "run_id": run_id, "input_path": input_path})

    set_seed(SEED)
    preprocess_and_save(input_path, artifact_dir)

    # Choose sampler; multivariate can help but is marked experimental in Optuna
    if multivariate:
        from optuna.exceptions import ExperimentalWarning
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        sampler = TPESampler(seed=SEED, multivariate=True)
    else:
        sampler = TPESampler(seed=SEED)

    print(f"\n[{run_id}] TPE search over {{seq_len, lr, batch_size, embed_dim, num_layers, arch_style}} …")
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    _ = train_eval_best(study)

# ------------------------------------------------------------
# Main: single-file or directory mode (user switch)
# ------------------------------------------------------------
def main() -> None:
    """
    Entry point. Two run modes:
      - single: train once on --input_file
      - dir   : iterate all *_PRICES.txt in --input_dir
    """
    p = argparse.ArgumentParser()
    p.add_argument("--run_mode", choices=["single","dir"], default="dir", help="Run once on a file or iterate a directory.")
    p.add_argument("--input_file", type=str, default="", help="Path to a single *_PRICES.txt file.")
    p.add_argument("--input_dir", type=str, default=INPUT_DIR, help="Directory with *_PRICES.txt files.")
    p.add_argument("--n_trials", type=int, default=20, help="Optuna trials per run.")
    p.add_argument("--multivariate", action="store_true", help="Use Optuna multivariate TPE (experimental).")
    args = p.parse_args()

    if args.run_mode == "single":
        if not args.input_file:
            print("Please pass --input_file for single mode.")
            return
        if not os.path.exists(args.input_file):
            print(f"Input file not found: {args.input_file}")
            return
        try:
            run_pipeline_for_file(args.input_file, n_trials=args.n_trials, multivariate=args.multivariate)
        except Exception as e:
            print(f"❌ Run failed for {args.input_file}: {e}")
    else:
        input_dir = args.input_dir or INPUT_DIR
        files = sorted(glob.glob(os.path.join(input_dir, "*_PRICES.txt")))
        if not files:
            print(f"No files found in {input_dir} matching *_PRICES.txt")
            return
        print(f"Discovered {len(files)} files. Beginning training runs …")
        for fp in files:
            try:
                run_pipeline_for_file(fp, n_trials=args.n_trials, multivariate=args.multivariate)
            except Exception as e:
                print(f"❌ Run failed for {fp}: {e}")

if __name__ == "__main__":
    main()