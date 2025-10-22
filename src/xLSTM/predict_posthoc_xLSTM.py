#!/usr/bin/env python3
# predict_posthoc_xLSTM.py
# End-to-end inference for one trained run (derived from price data).
# Loads artifacts from C:\Users\Anthony\PycharmProjects\SilentWealth\src\artifacts\<model_data>\,
# runs horizon-day predictions for a single ticker,
# and saves per-ticker CSVs and "last N days" plots with direction markers.
#
# INPUT DATA MODES (single-ticker enforced):
#   1) --source file  --data_file <path>   : load prices from a tab/CSV file (your original mode).
#      -> CLAMP to the TEST window from split_report.csv (as trained).
#   2) --source stooq --ticker_name <TICKER>: fetch daily bars from Stooq (direct CSV; no pandas-datareader).
#      -> BYPASS the TEST window clamp; use the full fetched date range.
#      -> FUTURE TARGETS (this version): create exactly --future_steps FUTURE dates that are the
#         next business days after the last known date, BUT each future point is the HORIZON-ahead
#         target for the last --future_steps historical prediction days.
#         Example (horizon=4): last hist dates = 18,19,20,21  →  future targets plotted at 22,23,24,25
#         from prediction days 18→22, 19→23, 20→24, 21→25. We do NOT space predictions 4 days apart.
#
# PLOTTING (customised):
# - X-axis dates: smaller font, rotated 45°, with extra bottom margin to avoid clipping.
# - Actual series drawn in BLACK.
# - Predicted (model) markers are PURPLE circles.
# - RSI-adjusted predictions are green triangles with the numeric RSI above them (no "RSI " prefix).
# - NO arrows of any kind.
# - ALWAYS show the LAST DATE label, and always show the LAST FIVE DATES in reduced font.
# - If the angle between predicted price at day t and t+3 exceeds +35°,
#   draw a RED line between those two purple markers (no arrows).
#
# RSI ADJUSTMENT:
# - RSI used for a target at index t_target is from the prediction day (last input day) at (t_target - horizon).
# - Nonlinear multiplier that strengthens the move as RSI goes further outside [30, 70]:
#     strength = 0 if 30 <= RSI <= 70
#              = ((|RSI-50|-20)/30)^2 otherwise  (ranges 0..1)
#     multiplier = 1 + alpha * strength
#   Use --rsi_alpha to tune intensity (default 0.5).

import os
import json
import math
import argparse
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
import urllib.parse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Predict horizon-day Close for one trained combo using its artifacts.")
    # Data sources (one of data_file or (source=stooq & ticker_name))
    p.add_argument("--data_file", default=None, help="Path to the raw price file (TSV/CSV). If omitted, use --source stooq with --ticker_name.")
    p.add_argument("--source", choices=["file", "stooq"], default="file", help="Data source: 'file' (default) or 'stooq' (requires --ticker_name).")
    p.add_argument("--ticker_name", default=None, help="Ticker to fetch when --source stooq (e.g., JNJ).")
    # Artifacts
    p.add_argument("--model_data", required=True, help="Artifacts subfolder name, e.g. JNJ_1D_Optuna_50_trials")
    p.add_argument("--artifacts_root", default="artifacts", help="(Reserved; not used for absolute paths.)")
    p.add_argument("--out_subdir", default="inference", help="Subdirectory under the artifacts folder to write predictions (default: inference).")
    # Inference/setup
    p.add_argument("--plots_last_n", type=int, default=30, help="Show only the last N target days in per-ticker plots (default: 30).")
    p.add_argument("--seq_len", type=int, default=None, help="Override seq_len for inference (defaults to the saved best seq_len).")
    p.add_argument("--horizon", type=int, default=None, help="Override horizon (days ahead) for inference. If not provided, reads from data_config.json or defaults to 1.")
    p.add_argument("--rsi_period", type=int, default=14, help="RSI period (used to compute rsi_prev for each target). Default 14.")
    p.add_argument("--rsi_alpha", type=float, default=0.5, help="Strength of RSI adjustment: multiplier = 1 + alpha*g(RSI). Default 0.5.")
    p.add_argument("--future_steps", type=int, default=4, help="(stooq mode only) Number of FUTURE targets to plot (default: 4).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device (default auto).")
    # Compatibility flags (kept; not used for selection anymore)
    p.add_argument("--scope", choices=["all", "single"], default="all", help="(Compatibility; ignored after single-ticker enforcement.)")
    p.add_argument("--ticker", type=str, default=None, help="(Compatibility; ignored after single-ticker enforcement.)")
    p.add_argument("--plot_type", choices=["full", "simple"], default="full", help="Plot style (kept for compatibility).")
    return p.parse_args()

# -------------------------
# Constants (must match training)
# -------------------------
DATE_COL = "Date"
TARGET_COL = "Close"
TICKER_COL_CANDIDATES = ["Ticker", "Company", "Symbol", "Name"]

MLSTM_CONV_K = 4
MLSTM_HEADS = 2
MLSTM_PROJ_SIZE = 2
SLSTM_CONV_K = 2
SLSTM_HEADS = 2
SLSTM_FF_FACTOR = 1.1

# -------------------------
# Model blocks (matching training implementation)
# -------------------------
class CausalConv1d(nn.Module):
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
    def __init__(self, d: int, proj_size: int):
        super().__init__()
        hidden = max(1, d // proj_size)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, d: int, factor: float):
        super().__init__()
        hidden = max(1, int(math.ceil(d * factor)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class mLSTMBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, heads: int, proj_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        a_heads = max(1, heads)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=a_heads, batch_first=True)
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
    def __init__(self, d_model: int, kernel: int, heads: int, ff_factor: float):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        a_heads = max(1, heads)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=a_heads, batch_first=True)
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
    def __init__(self, input_size: int, d_model: int, output_size: int, num_layers: int, arch_style: str):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        blocks = []
        def m(): return mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        def s(): return sLSTMBlock(d_model=d_model, kernel=SLSTM_CONV_K, heads=SLSTM_HEADS, ff_factor=SLSTM_FF_FACTOR)
        if arch_style == "all_m":
            blocks = [m() for _ in range(num_layers)]
        elif arch_style == "all_s":
            blocks = [s() for _ in range(num_layers)]
        elif arch_style == "ms_first":
            k = num_layers // 2
            blocks = [m() for _ in range(k)] + [s() for _ in range(num_layers - k)]
        elif arch_style == "sm_first":
            k = num_layers // 2
            blocks = [s() for _ in range(k)] + [m() for _ in range(num_layers - k)]
        else:
            for i in range(num_layers):
                blocks.append(m() if i % 2 == 0 else s())
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, -1, :])

# -------------------------
# Data loading helpers
# -------------------------
def find_ticker_col(df: pd.DataFrame) -> str:
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No ticker/company column found. Tried: {TICKER_COL_CANDIDATES}")

def load_prices_from_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if DATE_COL not in df.columns:
        raise ValueError(f"Input must contain a '{DATE_COL}' column")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Input must contain a '{TARGET_COL}' column")
    tkr_col = find_ticker_col(df)
    keep_cols = [c for c in ["Date","Ticker","Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if "Ticker" not in keep_cols:
        keep_cols.insert(1, tkr_col)
    df = df[keep_cols].copy()
    df.rename(columns={tkr_col: "Ticker"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
    return df

# --- Stooq direct fetch (no pandas-datareader required) ---
STOOQ_BASE_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

def stooq_symbol(ticker: str) -> str:
    """
    Build a Stooq symbol from user input:
      - URL-encoded caret indices are supported (e.g., '%5Espx'); returned lower-case.
      - Plain caret form like '^SPX' is URL-encoded.
      - Dotted symbols (e.g., 'googl.us') are preserved.
      - Otherwise, assume US equity and append '.us'.
    """
    t = ticker.strip()
    if "%" in t:
        return t.lower()
    if t.startswith("^"):
        return urllib.parse.quote(t.lower(), safe="")
    if "." in t:
        return t.lower()
    return f"{t.lower()}.us"

def load_prices_from_stooq(ticker: str, days: int = 1000) -> pd.DataFrame:
    symbol = stooq_symbol(ticker)
    url = STOOQ_BASE_URL.format(symbol=symbol)
    df = pd.read_csv(url)
    if df is None or df.empty:
        raise RuntimeError(f"Stooq returned no data for '{ticker}' ({symbol}).")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    cols = ["Date","Open","High","Low","Close","Adj Close","Volume"]
    df = df[[c for c in cols if c in df.columns]]
    if days is not None and days > 0:
        df = df.tail(days).reset_index(drop=True)
    df["Ticker"] = str(ticker).upper()
    df = df[["Date","Ticker","Open","High","Low","Close","Adj Close","Volume"]]
    return df

# -------------------------
# Misc helpers
# -------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_hparams(hp_path: str) -> dict:
    hp_all = load_json(hp_path)
    return hp_all.get("best_params", hp_all)

def scale_minmax(x: np.ndarray, vmin: float, vmax: float, eps: float = 1e-12) -> np.ndarray:
    rng = max(eps, float(vmax - vmin))
    return (x - vmin) / rng

def inverse_minmax(x_s: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return x_s * (vmax - vmin) + vmin

def build_windows_target_only(s: np.ndarray, window: int, horizon: int = 1):
    """
    Build sliding windows for inference.
    Returns:
      X: (N, window, 1) scaled last-window inputs
      y_res: (N,) scaled residual targets = scaled_target - last_input_scaled
      idx_arr: (N,) indices of the target day in the original series
    Notes:
      - last input in the window is at index t-1
      - target index = t + horizon - 1
    """
    X_list, y_list, idx_list = [], [], []
    n = len(s)
    for t in range(window, n - horizon + 1):
        target_idx = t + horizon - 1
        last_input_idx = t - 1
        X_list.append(s[t - window:t])
        y_res = s[target_idx] - s[last_input_idx]
        y_list.append(y_res)
        idx_list.append(target_idx)
    if not X_list:
        return np.zeros((0, window, 1), dtype="float32"), np.zeros((0,), dtype="float32"), np.array([], dtype=int)
    X = np.stack(X_list, axis=0).astype("float32")[..., None]
    y = np.array(y_list, dtype="float32")
    idx_arr = np.array(idx_list, dtype=int)
    return X, y, idx_arr

@torch.no_grad()
def predict_batches(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        yp = model(xb).detach().cpu().numpy().reshape(-1)
        out.append(yp)
    return np.concatenate(out, axis=0) if out else np.array([])

def compute_direction_markers(raw_close_all: np.ndarray, target_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mark direction correctness per target.
    Compare target day price vs last input day (prediction day): last_input_idx = target_idx - horizon.
    """
    assert len(target_indices) == len(y_true) == len(y_pred)
    dirs_true, dirs_pred = [], []
    for k, t in enumerate(target_indices):
        last_input_idx = t - horizon
        if last_input_idx < 0:
            dirs_true.append(0); dirs_pred.append(0); continue
        prev = raw_close_all[last_input_idx]
        delta_t = raw_close_all[t] - prev
        delta_p = y_pred[k] - prev
        sgn = lambda x: (-1 if x < 0 else (1 if x > 0 else 0))
        dirs_true.append(sgn(delta_t)); dirs_pred.append(sgn(delta_p))
    dirs_true = np.array(dirs_true, dtype=int); dirs_pred = np.array(dirs_pred, dtype=int)
    correct = dirs_true == dirs_pred; wrong = ~correct
    return correct, wrong

# -------------------------
# RSI helpers
# -------------------------
def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder-smoothing RSI. rsi[i] corresponds to the RSI at day i (window ending at i).
    """
    close = np.asarray(close, dtype=float); n = len(close)
    if n == 0 or period < 1:
        return np.full(n, np.nan)
    delta = np.diff(close, prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0); losses = np.where(delta < 0, -delta, 0.0)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= period:
        return rsi
    avg_gain = np.nanmean(gains[1:period+1]); avg_loss = np.nanmean(losses[1:period+1])
    if avg_loss == 0.0:
        rsi_val = 100.0
    else:
        rs = avg_gain / (avg_loss if avg_loss > 0 else 1e-12)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
    rsi[period] = rsi_val
    for i in range(period + 1, n):
        g = gains[i]; l = losses[i]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / (avg_loss if avg_loss > 0 else 1e-12)
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def rsi_multiplier_nonlinear(rsi_value: float, alpha: float) -> float:
    """
    Nonlinear scaling that increases adjustment strength the further RSI moves outside [30, 70]:
    - Let dist = |RSI - 50|.
    - If dist <= 20 (i.e., 30 <= RSI <= 70): strength = 0  -> multiplier = 1.0  (no adjustment in the neutral band)
    - Else: strength = ((dist - 20.0) / 30.0) ** 2  (quadratic ramp 0..1 across [30,0]∪[70,100])
    - Multiplier = 1 + alpha * strength  (>= 1).
    """
    if np.isnan(rsi_value):
        return 1.0
    dist = abs(rsi_value - 50.0)
    if dist <= 20.0:
        return 1.0
    strength = ((dist - 20.0) / 30.0) ** 2  # 0..1
    return 1.0 + alpha * strength

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # Auto-detect stooq if user supplied ticker_name but omitted data_file
    if args.source == "file" and not args.data_file and args.ticker_name:
        print("[info] No --data_file provided but --ticker_name is set; switching --source to 'stooq'.")
        args.source = "stooq"

    # Sanity on input mode
    if args.source == "file":
        if not args.data_file:
            raise ValueError("When --source file, you must provide --data_file.")
    else:
        if not args.ticker_name:
            raise ValueError("When --source stooq, you must provide --ticker_name.")

    # Resolve artifacts directory from --model_data (base path unchanged)
    abs_artifacts_base = r"C:\Users\Anthony\PycharmProjects\SilentWealth\src\artifacts"
    artifact_dir = os.path.join(abs_artifacts_base, args.model_data)

    # Resolve device
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    print(f"[model_data={args.model_data}] Using artifacts dir: {artifact_dir}")
    print(f"Using device: {DEVICE}")

    # Resolve files (basenames unchanged)
    split_report_path   = os.path.join(artifact_dir, "split_report.csv")
    scaling_params_path = os.path.join(artifact_dir, "scaling_params.csv")
    hparams_path        = os.path.join(artifact_dir, "xlstm_ts_best_hparams_JNJ.json")
    model_state_path    = os.path.join(artifact_dir, "xlstm_ts_best_state_dict_JNJ.pt")
    data_config_path    = os.path.join(artifact_dir, "data_config.json")

    # Output dirs (under artifacts subfolder)
    out_dir   = os.path.join(artifact_dir, args.out_subdir)
    plots_dir = os.path.join(out_dir, "plots")
    csv_dir   = os.path.join(out_dir, "per_ticker_csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Load horizon
    horizon = 1
    if os.path.exists(data_config_path):
        try:
            data_conf = load_json(data_config_path)
            horizon = int(data_conf.get("horizon", horizon))
        except Exception:
            print(f"[WARN] Could not read horizon from {data_config_path}; defaulting to 1")
    if args.horizon is not None:
        horizon = int(args.horizon)
    print(f"Horizon for inference: {horizon} day(s) ahead")

    # Load hparams, build model
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Missing hparams: {hparams_path}")
    hp = load_hparams(hparams_path)
    embed_dim = int(hp.get("embed_dim", 64))
    num_layers = int(hp.get("num_layers", 3))
    arch_style = str(hp.get("arch_style", "alternating_ms"))
    seq_len = int(args.seq_len) if args.seq_len is not None else int(hp.get("seq_len", 100))
    model = xLSTM_TS(input_size=1, d_model=embed_dim, output_size=1, num_layers=num_layers, arch_style=arch_style).to(DEVICE)

    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Missing model state: {model_state_path}")

    # Robust torch.load (supports newer weights_only param, falls back if unsupported)
    def _safe_torch_load(path: str, device: str):
        try:
            return torch.load(path, map_location=device, weights_only=True)  # PyTorch >= 2.4 (experimental)
        except TypeError:
            return torch.load(path, map_location=device)

    state = _safe_torch_load(model_state_path, DEVICE)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    print("Model loaded and ready.")

    # Load scaling params and split report
    if not os.path.exists(scaling_params_path):
        raise FileNotFoundError(f"Missing scaling params: {scaling_params_path}")
    sp = pd.read_csv(scaling_params_path)
    req_sp = {"Ticker", "target_min_train_den", "target_max_train_den"}
    if not req_sp.issubset(sp.columns):
        raise ValueError(f"{os.path.basename(scaling_params_path)} must contain columns: {sorted(req_sp)}")

    test_ranges = {}
    if os.path.exists(split_report_path):
        sr = pd.read_csv(
            split_report_path,
            parse_dates=["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]
        )
        req_sr = {"Ticker", "test_start", "test_end"}
        if not req_sr.issubset(sr.columns):
            raise ValueError(f"{os.path.basename(split_report_path)} must contain columns: {sorted(req_sr)}")
        test_ranges = {row["Ticker"]: (row["test_start"], row["test_end"]) for _, row in sr.iterrows()}
    else:
        print(f"[WARN] split_report.csv not found at {split_report_path}. Proceeding without test window metadata.")

    # -------- Load price data (single-ticker enforced) --------
    if args.source == "file":
        df_raw = load_prices_from_file(args.data_file)
        ticker_list = df_raw["Ticker"].unique().tolist()
        if len(ticker_list) != 1:
            raise ValueError(f"Input file must contain exactly one ticker, but found: {ticker_list}")
        tkr_single = ticker_list[0]

        # CLAMP to TEST window (required for file mode)
        if tkr_single not in test_ranges:
            raise RuntimeError(f"Ticker '{tkr_single}' not found in artifacts split_report.csv for this model run.")
        ts, te = test_ranges[tkr_single]
        df_use = df_raw[(df_raw["Ticker"] == tkr_single) & (df_raw["Date"] >= ts) & (df_raw["Date"] <= te)].copy()
        if df_use.empty:
            raise RuntimeError("No rows found in the raw data for the saved TEST window. Check the ticker and test range.")
        clamp_info = f"[file mode] Using TEST window clamp: {ts.date()} → {te.date()}"

    else:  # stooq mode
        df_raw = load_prices_from_stooq(args.ticker_name, days=1000)
        tkr_single = df_raw["Ticker"].iloc[0]
        # BYPASS the clamp: use full fetched range
        df_use = df_raw[df_raw["Ticker"] == tkr_single].copy()
        # Optional: if test window exists, just log it for reference
        if tkr_single in test_ranges:
            ts, te = test_ranges[tkr_single]
            print(f"[stooq mode] Bypassing TEST window clamp (available: {ts.date()} → {te.date()}); using full fetched range.")
        else:
            print("[stooq mode] No TEST window metadata found for this ticker; using full fetched range.")
        clamp_info = "[stooq mode] No clamp applied."

    print(f"Ticker: {tkr_single}. Rows selected: {len(df_use)}. {clamp_info}")

    # Core columns only
    df_test = df_use[["Date","Ticker","Close"]].sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Per-ticker inference (single ticker)
    tkr = tkr_single
    raw_close = pd.to_numeric(df_test["Close"], errors="coerce").values.astype("float32")
    dates = df_test["Date"].values

    # RSI on raw closes
    rsi_series = compute_rsi(raw_close, period=int(args.rsi_period))

    # Scaling params for this ticker
    row = sp[sp["Ticker"] == tkr]
    if row.empty:
        raise RuntimeError(f"Missing scaling params for '{tkr}'.")
    vmin = float(row["target_min_train_den"].iloc[0])
    vmax = float(row["target_max_train_den"].iloc[0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise RuntimeError(f"Bad scaling range for '{tkr}' (min={vmin}, max={vmax}).")

    # Scale test close using TRAIN min/max
    scaled_close = scale_minmax(raw_close, vmin, vmax)

    # Sliding windows over selected segment (historical targets only)
    X, y_true_res_scaled, idxs = build_windows_target_only(scaled_close, window=seq_len, horizon=horizon)
    if X.shape[0] == 0:
        raise RuntimeError(f"Not enough rows for '{tkr}' (need at least {seq_len + horizon}).")

    # Predict residuals (scaled) -> reconstruct absolute (scaled) -> inverse-scale
    y_pred_res_scaled = predict_batches(model, X, DEVICE, batch_size=256)
    last_vals_scaled = X[:, -1, 0]
    y_pred_abs_scaled = y_pred_res_scaled + last_vals_scaled
    y_true_abs_scaled = y_true_res_scaled + last_vals_scaled
    y_pred = inverse_minmax(y_pred_abs_scaled, vmin, vmax)
    y_true = inverse_minmax(y_true_abs_scaled, vmin, vmax)
    last_vals_raw = inverse_minmax(last_vals_scaled, vmin, vmax)
    y_naive = last_vals_raw

    # RSI for the prediction day (last input day): rsi_prev = rsi_series[target_idx - horizon]
    rsi_prev_list = []
    for t in idxs:
        pred_day_idx = t - horizon
        if pred_day_idx < 0 or pred_day_idx >= len(rsi_series):
            rsi_prev_list.append(np.nan)
        else:
            rsi_prev_list.append(rsi_series[pred_day_idx])
    rsi_prev_arr = np.array(rsi_prev_list, dtype=float)

    # --- Nonlinear RSI adjustment (historical targets) ---
    alpha = float(args.rsi_alpha)
    pred_res_raw = y_pred - last_vals_raw  # residual in raw units
    mult_arr = np.array([rsi_multiplier_nonlinear(rv, alpha) for rv in rsi_prev_arr], dtype=float)
    pred_res_adj_raw = pred_res_raw * mult_arr
    y_pred_rsi_adj = last_vals_raw + pred_res_adj_raw

    target_dates = pd.to_datetime(dates[idxs])

    # Errors for historical targets
    err_model = y_pred - y_true
    err_naive = y_naive - y_true
    err_model_rsi_adj = y_pred_rsi_adj - y_true

    # Historical rows
    out_df_hist = pd.DataFrame({
        "Date": target_dates,
        "Ticker": tkr,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_rsi_adj": y_pred_rsi_adj,
        "y_naive": y_naive,
        "y_true_res": y_true_res_scaled,
        "y_pred_res": y_pred_res_scaled,
        "err_model": err_model,
        "err_model_rsi_adj": err_model_rsi_adj,
        "err_naive": err_naive,
        "rsi_prev": rsi_prev_arr,
        "is_future": False
    }).sort_values("Date")

    # -------- FUTURE TARGETS (stooq mode only) --------
    # Use the last K historical PREDICTION days (consecutive business days),
    # and for each, predict its HORIZON-ahead target date (consecutive business days right after the last known date).
    out_df_future = pd.DataFrame(columns=out_df_hist.columns)
    if args.source == "stooq" and args.future_steps > 0:
        n = len(scaled_close)
        # choose prediction-day indices: last K days (chronological)
        k = int(args.future_steps)
        start_idx = max(0, n - k)
        pred_day_indices = list(range(start_idx, n))  # chronological last K indices
        # ensure each has enough history for the seq_len window
        pred_day_indices = [P for P in pred_day_indices if P - (seq_len - 1) >= 0]
        if len(pred_day_indices) < k:
            print(f"[WARN] Only {len(pred_day_indices)} of the last {k} days have enough history for future targets.")

        fut_dates: List[pd.Timestamp] = []
        fut_pred_raw_list: List[float] = []
        fut_pred_adj_list: List[float] = []
        fut_last_val_raw_list: List[float] = []
        fut_rsi_prev_list: List[float] = []
        fut_pred_res_scaled_list: List[float] = []

        for P in pred_day_indices:
            # window ending at prediction day P
            window_scaled = scaled_close[P - (seq_len - 1): P + 1]
            X_last = window_scaled.reshape(1, seq_len, 1).astype("float32")

            y_pred_res_scaled_step = predict_batches(model, X_last, DEVICE, batch_size=1)[0]
            last_val_scaled_step = X_last[0, -1, 0]
            y_pred_abs_scaled_step = y_pred_res_scaled_step + last_val_scaled_step

            # Convert to raw
            y_pred_raw_step = inverse_minmax(np.array([y_pred_abs_scaled_step]), vmin, vmax)[0]
            last_val_raw_step = inverse_minmax(np.array([last_val_scaled_step]), vmin, vmax)[0]

            # RSI prev = RSI at prediction day P
            rprev = rsi_series[P] if P < len(rsi_series) else np.nan
            mult_step = rsi_multiplier_nonlinear(rprev, alpha)
            y_pred_adj_raw_step = last_val_raw_step + (y_pred_raw_step - last_val_raw_step) * mult_step

            # FUTURE DATE = business-day horizon ahead of the prediction day P
            pred_day_date = pd.to_datetime(df_test["Date"].iloc[P])
            future_target_date = pred_day_date + BDay(horizon)

            fut_dates.append(pd.to_datetime(future_target_date))
            fut_pred_raw_list.append(y_pred_raw_step)
            fut_pred_adj_list.append(y_pred_adj_raw_step)
            fut_last_val_raw_list.append(last_val_raw_step)
            fut_rsi_prev_list.append(rprev)
            fut_pred_res_scaled_list.append(y_pred_res_scaled_step)

        if fut_dates:
            out_df_future = pd.DataFrame({
                "Date": fut_dates,
                "Ticker": tkr,
                "y_true": [np.nan] * len(fut_dates),
                "y_pred": fut_pred_raw_list,
                "y_pred_rsi_adj": fut_pred_adj_list,
                "y_naive": fut_last_val_raw_list,
                "y_true_res": [np.nan] * len(fut_dates),
                "y_pred_res": fut_pred_res_scaled_list,
                "err_model": [np.nan] * len(fut_dates),
                "err_model_rsi_adj": [np.nan] * len(fut_dates),
                "err_naive": [np.nan] * len(fut_dates),
                "rsi_prev": fut_rsi_prev_list,
                "is_future": True
            }).sort_values("Date")

    # Combine historical + future (note: future dates may interleave after last known)
    out_df = pd.concat([out_df_hist, out_df_future], ignore_index=True).sort_values("Date")
    out_csv = os.path.join(csv_dir, f"{tkr}_predictions_seq{seq_len}_h{horizon}.csv")
    out_df.to_csv(out_csv, index=False)

    # Direction markers only for historical rows (where y_true is known)
    correct_mask = np.array([], dtype=bool)
    if not out_df_hist.empty:
        # recompute using original arrays and indices
        correct_mask, _ = compute_direction_markers(
            raw_close_all=raw_close, target_indices=idxs, y_true=out_df_hist["y_true"].values,
            y_pred=out_df_hist["y_pred"].values, horizon=horizon
        )

    # Plot only the last N targets (over combined)
    n_total = len(out_df)
    n_plot = min(max(1, int(args.plots_last_n)), n_total)
    df_plot = out_df.tail(n_plot).copy()

    x_dates = df_plot["Date"].values
    y_actual = df_plot["y_true"].values
    y_predpl = df_plot["y_pred"].values
    y_pred_rsi_pl = df_plot["y_pred_rsi_adj"].values
    rsi_plot = df_plot["rsi_prev"].values
    is_future_plot = df_plot["is_future"].values

    # Determine which of the last N are historical to place correctness ticks
    hist_in_plot_mask = (~is_future_plot) & (~np.isnan(y_actual))
    num_hist_in_plot = int(np.sum(hist_in_plot_mask))
    correct_last = np.array([], dtype=bool)
    if num_hist_in_plot > 0 and correct_mask.size >= num_hist_in_plot:
        correct_last = correct_mask[-num_hist_in_plot:]

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    # BLACK actual line (only draw through known actuals)
    if np.any(~np.isnan(y_actual)):
        ax.plot(
            x_dates[~np.isnan(y_actual)],
            y_actual[~np.isnan(y_actual)],
            label="Actual Close",
            linewidth=2,
            color="black",
            marker="|",
            markersize=6,
            markeredgewidth=1.2,
            zorder=2
        )

    # Model predictions (PURPLE markers) — includes historical and future
    ax.scatter(x_dates, y_predpl, label="Predicted Close (model)", s=36, zorder=4, marker="o", color="purple")

    # RSI-adjusted predictions (green triangle)
    ax.scatter(x_dates, y_pred_rsi_pl, label="Predicted Close (RSI-adjusted)", s=50, zorder=5, marker="^", color="green")

    # Direction correctness ticks on historical points only
    if args.plot_type == "full" and num_hist_in_plot > 0:
        hist_indices = np.where(hist_in_plot_mask)[0]
        good_hist_positions = hist_indices[correct_last]
        if len(good_hist_positions) > 0:
            ax.scatter(
                x_dates[good_hist_positions],
                y_predpl[good_hist_positions],
                marker=r'$\checkmark$',
                s=120,
                color='green',
                linewidths=0.0,
                label="Correct direction (vs prediction day)"
            )

    # RSI value above RSI-adjusted prediction (no 'RSI ' prefix)
    for xd, yp_adj, rsi_val in zip(x_dates, y_pred_rsi_pl, rsi_plot):
        if not np.isnan(rsi_val):
            ax.annotate(
                f"{rsi_val:.1f}",
                xy=(xd, yp_adj),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black"
            )

    # Steep 3-day rise detection & annotation (NO arrows; red line only) — operate on y_predpl (which includes future)
    for i in range(0, len(x_dates) - 3):
        dy = y_predpl[i + 3] - y_predpl[i]
        theta = math.degrees(math.atan2(dy, 3.0))
        if theta > 35.0:
            ax.plot([x_dates[i], x_dates[i + 3]], [y_predpl[i], y_predpl[i + 3]],
                    color='red', linewidth=1.5, zorder=3)

    # Axes labels & title
    ax.set_title(f"{tkr} — {horizon}-Day Ahead Targets (seq_len={seq_len}) — Last {n_plot} points (incl. future)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    # Date formatting: rotate, smaller font
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, fontsize=8)

    # Ensure x-limits include the last date
    ax.set_xlim(left=x_dates[0], right=x_dates[-1])

    # --- ALWAYS SHOW the last five dates (in reduced font) ---
    last_five = list(pd.to_datetime(x_dates[-5:]))
    last_five_nums = [mdates.date2num(d) for d in last_five]

    current_ticks = list(ax.get_xticks())
    merged = current_ticks + last_five_nums
    merged_sorted = sorted(merged)
    deduped = []
    tol = 0.25  # in days
    for tpos in merged_sorted:
        if not deduped or abs(tpos - deduped[-1]) > tol:
            deduped.append(tpos)
    ax.set_xticks(deduped)

    # After setting the ticks, adjust font size for the last five labels (reduced font)
    ax.figure.canvas.draw_idle()
    tick_positions = list(ax.get_xticks())
    tick_labels = ax.get_xticklabels()

    def is_last_five(pos):
        return any(abs(pos - lf) < tol for lf in last_five_nums)

    for pos, lbl in zip(tick_positions, tick_labels):
        if is_last_five(pos):
            lbl.set_fontsize(7)  # reduced font for last five
        else:
            lbl.set_fontsize(8)  # default small font for others

    # Layout: avoid clipping bottom labels
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    out_png = os.path.join(plots_dir, f"{tkr}_actual_vs_pred_seq{seq_len}_h{horizon}_last{n_plot}.png")
    plt.savefig(out_png, dpi=160)
    plt.close()

    # Combined outputs and quick metrics (historical only for metrics)
    combo_hist = out_df[out_df["is_future"] == False].copy()
    combo_hist.to_csv(os.path.join(out_dir, f"all_tickers_predictions_seq{seq_len}_h{horizon}.csv"), index=False)
    if not combo_hist.empty:
        y = combo_hist["y_true"].values; p = combo_hist["y_pred"].values; p_adj = combo_hist["y_pred_rsi_adj"].values
        mae = float(np.mean(np.abs(p - y))); mae_adj = float(np.mean(np.abs(p_adj - y)))
        rmse = float(np.sqrt(np.mean((p - y) ** 2))); rmse_adj = float(np.sqrt(np.mean((p_adj - y) ** 2)))
        pd.DataFrame([{"Ticker": tkr, "N": len(combo_hist), "MAE": mae, "RMSE": rmse, "MAE_rsi_adj": mae_adj, "RMSE_rsi_adj": rmse_adj}]
                    ).to_csv(os.path.join(out_dir, f"summary_metrics_by_ticker_seq{seq_len}_h{horizon}.csv"), index=False)

    print("\nInference complete.")
    print(f"Artifacts dir:        {artifact_dir}")
    print(f"Outputs written to:   {out_dir}")
    print(f"Per-ticker CSVs:      {os.path.join(out_dir, 'per_ticker_csv')}")
    print(f"Per-ticker plots:     {os.path.join(out_dir, 'plots')}")
    print(f"Combined predictions: {os.path.join(out_dir, f'all_tickers_predictions_seq{seq_len}_h{horizon}.csv')}")
    print(f"Summary metrics:      {os.path.join(out_dir, f'summary_metrics_by_ticker_seq{seq_len}_h{horizon}.csv')}")
    if args.source == "stooq" and args.future_steps > 0:
        print(f"Future targets created: {len(out_df_future)} (dates = last K hist days + {horizon} business day(s)).")

if __name__ == "__main__":
    main()
