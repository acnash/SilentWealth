#!/usr/bin/env python3
# predict_xlstmts_from_combo.py
# End-to-end inference for one trained run (derived from <COMBO>_PRICES.txt).
# Loads artifacts from C:\Users\Anthony\PycharmProjects\SilentWealth\src\artifacts\<model_data>\,
# runs horizon-day predictions on the TEST window per ticker (single ticker enforced),
# and saves per-ticker CSVs and "last N days" plots with direction markers.
#
# NOTE (updated):
# - The RSI used for a target at index t_target is taken from the day the prediction
#   would have been made, i.e., the last input day at index (t_target - horizon).
# - Direction markers compare target vs the last input day (t_target - horizon), for any horizon >= 1.
# - Plot styling: smaller, rotated dates; actual = black; no dotted connectors; RSI labels above RSI-adjusted points.
# - RSI adjustment uses a nonlinear scaling that gets stronger as RSI moves further outside [30, 70].
# - Plot now draws a faint blue arrow from the prediction day (true line) to the predicted (blue) marker.
# - Extra bottom margin added so rotated date labels are not clipped.

import os
import json
import math
import argparse
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Predict horizon-day Close for one trained combo using its artifacts.")
    p.add_argument("--data_file", required=True, help="Path to the raw <COMBO>_PRICES.txt used for that training run.")
    p.add_argument("--model_data", required=True, help="Artifacts subfolder name, e.g. JNJ_1D_Optuna_50_trials")
    p.add_argument("--artifacts_root", default="artifacts", help="(Reserved; not used for absolute paths.)")
    p.add_argument("--out_subdir", default="inference", help="Subdirectory under the artifacts folder to write predictions (default: inference).")
    p.add_argument("--plots_last_n", type=int, default=30, help="Show only the last N target days in per-ticker plots (default: 30).")
    p.add_argument("--seq_len", type=int, default=None, help="Override seq_len for inference (defaults to the saved best seq_len).")
    p.add_argument("--horizon", type=int, default=None, help="Override horizon (days ahead) for inference. If not provided, reads from data_config.json or defaults to 1.")
    p.add_argument("--rsi_period", type=int, default=14, help="RSI period (used to compute rsi_prev for each target). Default 14.")
    p.add_argument("--rsi_alpha", type=float, default=0.5, help="Strength of RSI adjustment: multiplier = 1 + alpha*g(RSI). Default 0.5.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device (default auto).")
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
        x = x.transpose(1, 2); x = nn.functional.pad(x, (self.pad, 0)); x = self.conv(x); return x.transpose(1, 2)

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
        self.norm_in = nn.LayerNorm(d_model); self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.proj = ProjectionBlock(d_model, proj_size); self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm_in(x); x = x + self.conv(x); a, _ = self.attn(x, x, x, need_weights=False)
        x = x + a; x, _ = self.lstm(x); x = x + self.proj(x); x = self.norm_out(x); return x + r

class sLSTMBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, heads: int, ff_factor: float):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model); self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.ff = FeedForward(d_model, ff_factor); self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm_in(x); x = x + self.conv(x); a, _ = self.attn(x, x, x, need_weights=False)
        x = x + a; x, _ = self.lstm(x); x = x + self.ff(x); x = self.norm_out(x); return x + r

class xLSTM_TS(nn.Module):
    def __init__(self, input_size: int, d_model: int, output_size: int, num_layers: int, arch_style: str):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model); blocks = []
        def m(): return mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        def s(): return sLSTMBlock(d_model=d_model, kernel=SLSTM_CONV_K, heads=SLSTM_HEADS, ff_factor=SLSTM_FF_FACTOR)
        if arch_style == "all_m": blocks = [m() for _ in range(num_layers)]
        elif arch_style == "all_s": blocks = [s() for _ in range(num_layers)]
        elif arch_style == "ms_first":
            k = num_layers // 2; blocks = [m() for _ in range(k)] + [s() for _ in range(num_layers - k)]
        elif arch_style == "sm_first":
            k = num_layers // 2; blocks = [s() for _ in range(k)] + [m() for _ in range(num_layers - k)]
        else:
            for i in range(num_layers): blocks.append(m() if i % 2 == 0 else s())
        self.blocks = nn.ModuleList(blocks); self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for blk in self.blocks: x = blk(x)
        return self.head(x[:, -1, :])

# -------------------------
# Helpers
# -------------------------
def find_ticker_col(df: pd.DataFrame) -> str:
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns: return c
    raise ValueError(f"No ticker/company column found. Tried: {TICKER_COL_CANDIDATES}")

def load_json(path: str) -> dict:
    with open(path, "r") as f: return json.load(f)

def load_hparams(hp_path: str) -> dict:
    hp_all = load_json(hp_path); return hp_all.get("best_params", hp_all)

def scale_minmax(x: np.ndarray, vmin: float, vmax: float, eps: float = 1e-12) -> np.ndarray:
    rng = max(eps, float(vmax - vmin)); return (x - vmin) / rng

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
    model.eval(); out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        yp = model(xb).detach().cpu().numpy().reshape(-1); out.append(yp)
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
    if n == 0 or period < 1: return np.full(n, np.nan)
    delta = np.diff(close, prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0); losses = np.where(delta < 0, -delta, 0.0)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= period: return rsi
    avg_gain = np.nanmean(gains[1:period+1]); avg_loss = np.nanmean(losses[1:period+1])
    if avg_loss == 0.0: rsi_val = 100.0
    else:
        rs = avg_gain / (avg_loss if avg_loss > 0 else 1e-12)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
    rsi[period] = rsi_val
    for i in range(period + 1, n):
        g = gains[i]; l = losses[i]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0.0: rsi[i] = 100.0
        else:
            rs = avg_gain / (avg_loss if avg_loss > 0 else 1e-12)
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def rsi_multiplier_nonlinear(rsi_value: float, alpha: float) -> float:
    """
    Nonlinear scaling that increases adjustment strength the further RSI moves outside [30, 70]:
    - Let dist = |RSI - 50|.
    - If dist <= 20 (i.e., 30 <= RSI <= 70): strength = 0  -> multiplier = 1.0  (no adjustment in the neutral band)
    - Else: strength = ((dist - 20) / 30)^2  (quadratic ramp from 0 to 1 across [30,0]∪[70,100])
    - Multiplier = 1 + alpha * strength  (>= 1). This boosts the magnitude of the predicted move
      regardless of direction, with a stronger boost near extreme RSI (close to 0 or 100).
    """
    if np.isnan(rsi_value): return 1.0
    dist = abs(rsi_value - 50.0)
    if dist <= 20.0: return 1.0
    strength = ((dist - 20.0) / 30.0) ** 2  # 0..1 as RSI moves from 30→0 or 70→100
    return 1.0 + alpha * strength

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # Resolve artifacts directory from --model_data (base path unchanged)
    abs_artifacts_base = r"C:\Users\Anthony\PycharmProjects\SilentWealth\src\artifacts"
    artifact_dir = os.path.join(abs_artifacts_base, args.model_data)

    # Resolve device
    if args.device == "auto": DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else: DEVICE = args.device
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
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(csv_dir, exist_ok=True)

    # Load horizon
    horizon = 1
    if os.path.exists(data_config_path):
        try:
            data_conf = load_json(data_config_path); horizon = int(data_conf.get("horizon", horizon))
        except Exception:
            print(f"[WARN] Could not read horizon from {data_config_path}; defaulting to 1")
    if args.horizon is not None: horizon = int(args.horizon)
    print(f"Horizon for inference: {horizon} day(s) ahead")

    # Load hparams, build model
    if not os.path.exists(hparams_path): raise FileNotFoundError(f"Missing hparams: {hparams_path}")
    hp = load_hparams(hparams_path)
    embed_dim = int(hp.get("embed_dim", 64)); num_layers = int(hp.get("num_layers", 3))
    arch_style = str(hp.get("arch_style", "alternating_ms"))
    seq_len = int(args.seq_len) if args.seq_len is not None else int(hp.get("seq_len", 100))
    model = xLSTM_TS(input_size=1, d_model=embed_dim, output_size=1, num_layers=num_layers, arch_style=arch_style).to(DEVICE)
    if not os.path.exists(model_state_path): raise FileNotFoundError(f"Missing model state: {model_state_path}")
    state = torch.load(model_state_path, map_location=DEVICE); model.load_state_dict(state, strict=True)
    print("Model loaded and ready.")

    # Load scaling params and split report
    if not os.path.exists(scaling_params_path): raise FileNotFoundError(f"Missing scaling params: {scaling_params_path}")
    sp = pd.read_csv(scaling_params_path)
    req_sp = {"Ticker", "target_min_train_den", "target_max_train_den"}
    if not req_sp.issubset(sp.columns): raise ValueError(f"{os.path.basename(scaling_params_path)} must contain columns: {sorted(req_sp)}")

    if not os.path.exists(split_report_path): raise FileNotFoundError(f"Missing split report: {split_report_path}")
    sr = pd.read_csv(split_report_path, parse_dates=["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"])
    req_sr = {"Ticker", "test_start", "test_end"}
    if not req_sr.issubset(sr.columns): raise ValueError(f"{os.path.basename(slit_report_path)} must contain columns: {sorted(req_sr)}")
    test_ranges = {row["Ticker"]: (row["test_start"], row["test_end"]) for _, row in sr.iterrows()}

    # Read the raw file (single ticker expected)
    df = pd.read_csv(args.data_file, sep=None, engine="python")
    if DATE_COL not in df.columns: raise ValueError(f"Input must contain a '{DATE_COL}' column")
    if TARGET_COL not in df.columns: raise ValueError(f"Input must contain a '{TARGET_COL}' column")
    tkr_col = find_ticker_col(df)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([tkr_col, DATE_COL]).reset_index(drop=True)

    # Restrict to TEST windows per ticker
    pieces = []
    for tkr, g in df.groupby(tkr_col, sort=False):
        ts, te = test_ranges.get(tkr, (None, None))
        if ts is None or pd.isna(ts) or pd.isna(te): continue
        g2 = g[(g[DATE_COL] >= ts) & (g[DATE_COL] <= te)].copy()
        if not g2.empty: pieces.append(g2)

    df_test = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0]
    if df_test.empty: raise RuntimeError("No rows found in raw data for the saved test windows. Check ticker names and artifacts/split_report.csv for this run.")

    df_test = df_test[[DATE_COL, tkr_col, TARGET_COL]].copy().sort_values([tkr_col, DATE_COL])

    # Enforce exactly one ticker
    tickers = df_test[tkr_col].unique().tolist()
    if len(tickers) != 1: raise ValueError(f"Input file must contain exactly one ticker, but found: {tickers}")
    tkr_single = tickers[0]; print(f"Single-ticker input confirmed: {tkr_single}")
    tickers = [tkr_single]

    # Per-ticker inference (single ticker)
    all_dfs = []
    for tkr in tickers:
        g = df_test[df_test[tkr_col] == tkr].sort_values(DATE_COL).reset_index(drop=True)
        raw_close = pd.to_numeric(g[TARGET_COL], errors="coerce").values.astype("float32")
        dates = g[DATE_COL].values

        # RSI on raw closes
        rsi_series = compute_rsi(raw_close, period=int(args.rsi_period))

        # Scaling params for this ticker
        row = sp[sp["Ticker"] == tkr]
        if row.empty: print(f"[WARN] Missing scaling params for '{tkr}'. Skipping."); continue
        vmin = float(row["target_min_train_den"].iloc[0]); vmax = float(row["target_max_train_den"].iloc[0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            print(f"[WARN] Bad scaling range for '{tkr}' (min={vmin}, max={vmax}). Skipping."); continue

        # Scale test close using TRAIN min/max
        scaled_close = scale_minmax(raw_close, vmin, vmax)

        # Sliding windows over TEST segment
        X, y_true_res_scaled, idxs = build_windows_target_only(scaled_close, window=seq_len, horizon=horizon)
        if X.shape[0] == 0:
            print(f"[WARN] Not enough test rows for '{tkr}' (need at least {seq_len + horizon}). Skipping."); continue

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
            if pred_day_idx < 0 or pred_day_idx >= len(rsi_series): rsi_prev_list.append(np.nan)
            else: rsi_prev_list.append(rsi_series[pred_day_idx])
        rsi_prev_arr = np.array(rsi_prev_list, dtype=float)

        # --- Nonlinear RSI adjustment ---
        alpha = float(args.rsi_alpha)
        pred_res_raw = y_pred - last_vals_raw  # residual in raw units
        mult_arr = np.array([rsi_multiplier_nonlinear(rv, alpha) for rv in rsi_prev_arr], dtype=float)
        pred_res_adj_raw = pred_res_raw * mult_arr
        y_pred_rsi_adj = last_vals_raw + pred_res_adj_raw

        target_dates = pd.to_datetime(dates[idxs])

        # Errors
        err_model = y_pred - y_true
        err_naive = y_naive - y_true
        err_model_rsi_adj = y_pred_rsi_adj - y_true

        # Save per-ticker CSV
        out_df = pd.DataFrame({
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
            "rsi_prev": rsi_prev_arr
        }).sort_values("Date")
        out_csv = os.path.join(csv_dir, f"{tkr}_predictions_seq{seq_len}_h{horizon}.csv")
        out_df.to_csv(out_csv, index=False)

        # Direction markers vs last input day (prediction day)
        correct_mask, _ = compute_direction_markers(
            raw_close_all=raw_close, target_indices=idxs, y_true=y_true, y_pred=y_pred, horizon=horizon
        )

        # Plot only the last N targets
        n_total = len(out_df)
        n_plot = min(max(1, int(args.plots_last_n)), n_total)
        df_plot = out_df.tail(n_plot).copy()
        correct_last = correct_mask[-n_plot:]

        x_dates = df_plot["Date"].values
        y_actual = df_plot["y_true"].values
        y_predpl = df_plot["y_pred"].values
        y_pred_rsi_pl = df_plot["y_pred_rsi_adj"].values
        rsi_plot = df_plot["rsi_prev"].values

        # Indices of these last targets to map back to prediction days (for arrows)
        idxs_last = idxs[-n_plot:]

        fig = plt.figure(figsize=(10, 4))

        # BLACK actual line
        plt.plot(
            x_dates,
            y_actual,
            label="Actual Close",
            linewidth=2,
            color="black",
            marker="|",
            markersize=6,
            markeredgewidth=1.2,
            zorder=2
        )

        # Model predictions (blue markers)
        plt.scatter(x_dates, y_predpl, label="Predicted Close (model)", s=36, zorder=4, marker="o")

        # RSI-adjusted predictions (green triangle)
        plt.scatter(x_dates, y_pred_rsi_pl, label="Predicted Close (RSI-adjusted)", s=50, zorder=5, marker="^")

        # Very faint blue arrow from prediction day (on true line) to the predicted (blue) marker
        for j, t_idx in enumerate(idxs_last):
            pred_day_idx = t_idx - horizon
            if pred_day_idx >= 0:
                x0 = dates[pred_day_idx]         # date of prediction day
                y0 = raw_close[pred_day_idx]     # true close on prediction day (on black line)
                x1 = x_dates[j]                  # target date
                y1 = y_predpl[j]                 # predicted close (blue marker)
                plt.annotate(
                    '',
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', linewidth=0.8, alpha=0.2, color='tab:blue')
                )

        # No dotted connectors

        # Direction correctness ticks (optional, kept)
        if args.plot_type == "full":
            if np.any(correct_last):
                plt.scatter(
                    x_dates[correct_last],
                    y_predpl[correct_last],
                    marker=r'$\checkmark$',
                    s=120,
                    color='green',
                    linewidths=0.0,
                    label="Correct direction (vs prediction day)"
                )

        # RSI value above RSI-adjusted prediction (prefix removed)
        for xd, yp_adj, rsi_val in zip(x_dates, y_pred_rsi_pl, rsi_plot):
            if not np.isnan(rsi_val):
                plt.annotate(
                    f"{rsi_val:.1f}",
                    xy=(xd, yp_adj),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black"
                )

        plt.title(f"{tkr} — {horizon}-Day Ahead Close (seq_len={seq_len}) — Last {n_plot} targets")
        plt.xlabel("Date"); plt.ylabel("Price")
        plt.legend(); plt.tight_layout()

        # Rotate dates and reduce font size; then pad bottom so labels aren't clipped
        plt.xticks(rotation=45, fontsize=8)
        fig.subplots_adjust(bottom=0.22)

        out_png = os.path.join(plots_dir, f"{tkr}_actual_vs_pred_seq{seq_len}_h{horizon}_last{n_plot}.png")
        plt.savefig(out_png, dpi=160); plt.close()

        all_dfs.append(out_df)

    # Combined outputs and quick metrics
    if all_dfs:
        combo = pd.concat(all_dfs, ignore_index=True).sort_values(["Ticker", "Date"])
        combo.to_csv(os.path.join(out_dir, f"all_tickers_predictions_seq{seq_len}_h{horizon}.csv"), index=False)
        rows = []
        for tkr in combo["Ticker"].unique():
            c = combo[combo["Ticker"] == tkr]
            y = c["y_true"].values; p = c["y_pred"].values; p_adj = c["y_pred_rsi_adj"].values
            mae = float(np.mean(np.abs(p - y))); mae_adj = float(np.mean(np.abs(p_adj - y)))
            rmse = float(np.sqrt(np.mean((p - y) ** 2))); rmse_adj = float(np.sqrt(np.mean((p_adj - y) ** 2)))
            rows.append({"Ticker": tkr, "N": len(c), "MAE": mae, "RMSE": rmse, "MAE_rsi_adj": mae_adj, "RMSE_rsi_adj": rmse_adj})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"summary_metrics_by_ticker_seq{seq_len}_h{horizon}.csv"), index=False)

    print("\nInference complete.")
    print(f"Artifacts dir:        {artifact_dir}")
    print(f"Outputs written to:   {out_dir}")
    print(f"Per-ticker CSVs:      {os.path.join(out_dir, 'per_ticker_csv')}")
    print(f"Per-ticker plots:     {os.path.join(out_dir, 'plots')}")
    print(f"Combined predictions: {os.path.join(out_dir, f'all_tickers_predictions_seq{seq_len}_h{horizon}.csv')}")
    print(f"Summary metrics:      {os.path.join(out_dir, f'summary_metrics_by_ticker_seq{seq_len}_h{horizon}.csv')}")

if __name__ == "__main__":
    main()
