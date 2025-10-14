# predict_xlstmts_from_combo.py
# End-to-end inference for one trained run (derived from <COMBO>_PRICES.txt).
# Loads artifacts from artifacts/<run_id>/, runs next-day predictions on the TEST
# window per ticker (sliding windows using actual history), computes RSI from the raw input data,
# applies a post-hoc RSI-aware shrink to the predicted move, and saves per-ticker CSVs and "last N days"
# plots with direction markers. Both base and RSI-adjusted predictions are output.
# Optional production mode: pass --production_file; uses last seq_len actuals to predict a single next-day value
# and writes a short text report (no plot).
#
# New:
#  - --scope {test,all} to predict only test targets or across full series
#  - --plot_type {simple,complex} to choose simple black/red line plots or the detailed plots

import os, json, math, argparse, warnings
import numpy as np
import pandas as pd

# non-interactive plotting
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
    p = argparse.ArgumentParser(description="Predict next-day Close for one trained combo using its artifacts.")
    p.add_argument("--data_file", required=True, help="Path to the raw <COMBO>_PRICES.txt used for that training run.")
    p.add_argument("--artifacts_root", default="artifacts", help="Root directory containing run artifacts (default: artifacts).")
    p.add_argument("--out_subdir", default="inference", help="Subdirectory under the run artifacts to write predictions (default: inference).")
    p.add_argument("--plots_last_n", type=int, default=30, help="Show only the last N target days in per-ticker plots (default: 30).")
    p.add_argument("--seq_len", type=int, default=None, help="Override seq_len for inference (defaults to the saved best seq_len).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device (default auto).")
    p.add_argument("--production_file", type=str, default=None,
                   help="(Optional) CSV file with most-recent dates/closes for production single-step prediction. If provided, script will run production predictions (no plots) and exit.")
    p.add_argument("--scope", choices=["test","all"], default="test",
                   help="'test' (default): current behavior, predicts only targets in saved test ranges. 'all': predict across full series and break down into train/val/test for reporting.")
    p.add_argument("--plot_type", choices=["simple","complex"], default="complex",
                   help="'complex' (default): full detailed plot. 'simple': thin black line for actual and thin red line for base predictions; keeps shaded train/val/test areas.")
    return p.parse_args()

# -------------------------
# Constants (column names)
# -------------------------
DATE_COL = "Date"
TARGET_COL = "Close"
TICKER_COL_CANDIDATES = ["Ticker", "Company", "Symbol", "Name"]

# -------------------------
# RSI configuration (edit here if desired)
# -------------------------
RSI_PERIOD = 14     # classic Wilder period
RSI_HI = 70.0       # overbought threshold
RSI_LO = 30.0       # oversold threshold
RSI_ALPHA = 0.50     # shrink/amplify strength in [0,1]; 0.5 = 50% effect at max pressure

# -------------------------
# Model blocks (updated to accept dropout)
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
    def __init__(self, d: int, proj_size: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, d // max(1, int(proj_size)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, d: int, factor: float, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, int(math.ceil(d * float(factor))))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class mLSTMBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, heads: int, proj_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        heads = int(heads) if d_model % int(max(1, heads)) == 0 else 1
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.proj = ProjectionBlock(d_model, proj_size, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm_in(x)
        x = x + self.conv(x)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.attn_dropout(a)
        x, _ = self.lstm(x)
        x = x + self.lstm_dropout(self.proj(x))
        x = self.norm_out(x)
        return x + r

class sLSTMBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, heads: int, ff_factor: float, dropout: float = 0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        heads = int(heads) if d_model % int(max(1, heads)) == 0 else 1
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.ff = FeedForward(d_model, ff_factor, dropout=dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm_in(x)
        x = x + self.conv(x)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.attn_dropout(a)
        x, _ = self.lstm(x)
        x = x + self.ff_dropout(self.ff(x))
        x = self.norm_out(x)
        return x + r

class xLSTM_TS_Flexible(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        output_size: int,
        num_layers: int = 3,
        arch_style: str = "alternating_ms",
        mlstm_conv_k: int = 4,
        mlstm_heads: int = 2,
        mlstm_proj_size: int = 2,
        slstm_conv_k: int = 2,
        slstm_heads: int = 2,
        slstm_ff_factor: float = 1.1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        def make_m():
            return mLSTMBlock(d_model, kernel=int(mlstm_conv_k), heads=int(mlstm_heads), proj_size=int(mlstm_proj_size), dropout=float(dropout))
        def make_s():
            return sLSTMBlock(d_model, kernel=int(slstm_conv_k), heads=int(slstm_heads), ff_factor=float(slstm_ff_factor), dropout=float(dropout))
        style = (arch_style or "alternating_ms").lower()
        blocks = []
        if style in ("all_m", "m_only"):
            blocks = [make_m() for _ in range(num_layers)]
        elif style in ("all_s", "s_only"):
            blocks = [make_s() for _ in range(num_layers)]
        elif style in ("ms_first",):
            k = num_layers // 2
            blocks = [make_m() for _ in range(k)] + [make_s() for _ in range(num_layers - k)]
        elif style in ("sm_first",):
            k = num_layers // 2
            blocks = [make_s() for _ in range(k)] + [make_m() for _ in range(num_layers - k)]
        elif style in ("msm", "sms"):
            pattern = list(style)
            if num_layers != len(pattern):
                pat = (pattern * ((num_layers + len(pattern) - 1) // len(pattern)))[:num_layers]
            else:
                pat = pattern
            for ch in pat:
                blocks.append(make_m() if ch == "m" else make_s())
        else:
            for i in range(num_layers):
                blocks.append(make_m() if i % 2 == 0 else make_s())
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(float(dropout)), nn.Linear(d_model, output_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, -1, :])

# -------------------------
# Small helpers
# -------------------------
def find_ticker_col(df: pd.DataFrame) -> str:
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No ticker/company column found. Tried: {TICKER_COL_CANDIDATES}")

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
    X_list, y_list, idx_list = [], [], []
    n = len(s)
    for t in range(window, n - horizon + 1):
        X_list.append(s[t - window:t])
        y_list.append(s[t + horizon - 1])
        idx_list.append(t)
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

def compute_direction_markers(raw_close_all: np.ndarray, target_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    assert len(target_indices) == len(y_true) == len(y_pred)
    dirs_true, dirs_pred = [], []
    for k, t in enumerate(target_indices):
        if t - 1 < 0:
            dirs_true.append(0); dirs_pred.append(0); continue
        prev = raw_close_all[t - 1]
        delta_t = raw_close_all[t] - prev
        delta_p = y_pred[k] - prev
        sgn = lambda x: (-1 if x < 0 else (1 if x > 0 else 0))
        dirs_true.append(sgn(delta_t)); dirs_pred.append(sgn(delta_p))
    dirs_true = np.array(dirs_true, dtype=int); dirs_pred = np.array(dirs_pred, dtype=int)
    return dirs_true == dirs_pred, dirs_true != dirs_pred

# -------------------------
# RSI computation & adjusted shrink/amplify
# -------------------------
def compute_rsi_wilder(prices: np.ndarray, period: int = 14) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    n = prices.shape[0]
    rsi = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return rsi
    deltas = np.diff(prices)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        gain = gains[i - 1]; loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def rsi_shrink_toward_prev(prev_close: float, base_pred: float, rsi_value: float,
                           hi: float = RSI_HI, lo: float = RSI_LO, alpha: float = RSI_ALPHA) -> (float, float):
    """
    New behavior:
      - if rsi >= hi:
          * if model predicts UP (base_pred > prev_close): shrink toward prev (s < 1)
          * if model predicts DOWN (base_pred < prev_close): amplify down (s > 1)
      - if rsi <= lo:
          * if model predicts UP: amplify up (s > 1)
          * if model predicts DOWN: move up toward prev (shrink magnitude, s < 1)
      - otherwise: no change (s = 1)
    Returns (adjusted_prediction, s_used).
    """
    if not np.isfinite(rsi_value):
        return base_pred, 1.0

    delta = base_pred - prev_close  # signed predicted move
    # compute pressure ∈ [0,1] on extremeness
    if rsi_value >= hi:
        pressure = min(1.0, (rsi_value - hi) / (100.0 - hi))
        if delta > 0:
            # predicted up but overbought -> shrink toward prev
            s = max(0.0, 1.0 - alpha * pressure)
        elif delta < 0:
            # predicted down while overbought -> amplify down
            s = 1.0 + alpha * pressure
        else:
            s = 1.0
    elif rsi_value <= lo:
        pressure = min(1.0, (lo - rsi_value) / lo)
        if delta > 0:
            # predicted up while oversold -> amplify up
            s = 1.0 + alpha * pressure
        elif delta < 0:
            # predicted down while oversold -> push up toward prev (reduce magnitude)
            s = max(0.0, 1.0 - alpha * pressure)
        else:
            s = 1.0
    else:
        return base_pred, 1.0

    adj = prev_close + s * delta
    return float(adj), float(s)

# -------------------------
# Small metric helpers
# -------------------------
def mae(a, b): return float(np.mean(np.abs(a - b))) if len(a) else float("nan")
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # derive run_id and artifact dir from the data file name
    run_id = os.path.splitext(os.path.basename(args.data_file))[0].replace("_PRICES", "")
    artifact_dir = os.path.join(args.artifacts_root, run_id)

    # resolve device
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    print(f"[run_id={run_id}] Using device: {DEVICE}")

    # keep your hard-coded artifact paths unchanged
    split_report_path = "C:\\Users\\Anthony\\PycharmProjects\\SilentWealth\\src\\artifacts\\SPX500\\split_report.csv"
    scaling_params_path = "C:\\Users\\Anthony\\PycharmProjects\\SilentWealth\\src\\artifacts\\SPX500\\scaling_params.csv"
    hparams_path = f"C:\\Users\\Anthony\\PycharmProjects\\SilentWealth\\src\\artifacts\\SPX500\\xlstm_ts_best_hparams_{run_id}.json"
    model_state_path = f"C:\\Users\\Anthony\\PycharmProjects\\SilentWealth\\src\\artifacts\\SPX500\\xlstm_ts_best_state_dict_{run_id}.pt"

    out_dir = os.path.join(artifact_dir, args.out_subdir)
    plots_dir = os.path.join(out_dir, "plots")
    csv_dir = os.path.join(out_dir, "per_ticker_csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # load hparams and model
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Missing hparams: {hparams_path}")
    hp = load_hparams(hparams_path)
    embed_dim = int(hp.get("embed_dim", 64))
    num_layers = int(hp.get("num_layers", 3))
    arch_style = str(hp.get("arch_style", "alternating_ms"))
    mlstm_conv_k    = int(hp.get("mlstm_conv_k", 4))
    mlstm_heads     = int(hp.get("mlstm_heads", 2))
    mlstm_proj_size = int(hp.get("mlstm_proj_size", 2))
    slstm_conv_k    = int(hp.get("slstm_conv_k", 2))
    slstm_heads     = int(hp.get("slstm_heads", 2))
    slstm_ff_factor = float(hp.get("slstm_ff_factor", 1.1))
    dropout = float(hp.get("dropout", 0.0))
    seq_len = int(args.seq_len) if args.seq_len is not None else int(hp.get("seq_len", 100))

    model = xLSTM_TS_Flexible(
        input_size=1,
        d_model=embed_dim,
        output_size=1,
        num_layers=num_layers,
        arch_style=arch_style,
        mlstm_conv_k=mlstm_conv_k,
        mlstm_heads=mlstm_heads,
        mlstm_proj_size=mlstm_proj_size,
        slstm_conv_k=slstm_conv_k,
        slstm_heads=slstm_heads,
        slstm_ff_factor=slstm_ff_factor,
        dropout=dropout,
    ).to(DEVICE)

    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Missing model state: {model_state_path}")
    state = torch.load(model_state_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    print("Model loaded and ready.")

    # load scaling params and split report
    if not os.path.exists(scaling_params_path):
        raise FileNotFoundError(f"Missing scaling params: {scaling_params_path}")
    sp = pd.read_csv(scaling_params_path)
    req_sp = {"Ticker", "target_min_train_den", "target_max_train_den"}
    if not req_sp.issubset(sp.columns):
        raise ValueError(f"{os.path.basename(scaling_params_path)} must contain columns: {sorted(req_sp)}")

    if not os.path.exists(split_report_path):
        raise FileNotFoundError(f"Missing split report: {split_report_path}")
    # parse train/val/test dates so we can draw vertical lines & split targets
    sr = pd.read_csv(split_report_path,
                     parse_dates=["raw_start","raw_end","aligned_start","aligned_end",
                                  "train_start","train_end","val_start","val_end","test_start","test_end"])
    req_sr = {"Ticker", "train_start","train_end", "val_start","val_end", "test_start", "test_end"}
    if not req_sr.issubset(sr.columns):
        raise ValueError(f"{os.path.basename(split_report_path)} must contain columns: {sorted(req_sr)}")
    # maps
    test_ranges = {row["Ticker"]: (row["test_start"], row["test_end"]) for _, row in sr.iterrows()}
    split_map = {row["Ticker"]: {"train_start": row["train_start"], "train_end": row["train_end"],
                                 "val_start": row["val_start"], "val_end": row["val_end"],
                                 "test_start": row["test_start"], "test_end": row["test_end"]} for _, row in sr.iterrows()}

    # read the raw multi-ticker file (used for test-mode and 'all' mode)
    df = pd.read_csv(args.data_file, sep=None, engine="python")
    if DATE_COL not in df.columns:
        raise ValueError(f"Input must contain a '{DATE_COL}' column")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Input must contain a '{TARGET_COL}' column")
    tkr_col = find_ticker_col(df)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([tkr_col, DATE_COL]).reset_index(drop=True)

    # If production_file provided, keep the same behaviour (unchanged)
    if args.production_file:
        prod_path = args.production_file
        if not os.path.exists(prod_path):
            raise FileNotFoundError(f"Missing production file: {prod_path}")
        dfp = pd.read_csv(prod_path, sep=None, engine="python")
        if DATE_COL not in dfp.columns or TARGET_COL not in dfp.columns:
            raise ValueError("Production CSV must have 'Date' and 'Close' columns")
        prod_tkr_col = find_ticker_col(dfp) if any(c in dfp.columns for c in TICKER_COL_CANDIDATES) else None
        dfp[DATE_COL] = pd.to_datetime(dfp[DATE_COL], errors="coerce")
        dfp = dfp.dropna(subset=[DATE_COL]).sort_values([prod_tkr_col] if prod_tkr_col else [DATE_COL]).reset_index(drop=True)

        out_lines = []
        prod_tickers = dfp[prod_tkr_col].unique().tolist() if prod_tkr_col else [None]
        for tkr in prod_tickers:
            if tkr is None:
                g_prod = dfp.copy()
                tkr_name = "PROD"
            else:
                g_prod = dfp[dfp[prod_tkr_col] == tkr].sort_values(DATE_COL).reset_index(drop=True)
                tkr_name = str(tkr)
            if len(g_prod) < seq_len:
                out_lines.append(f"{tkr_name}: NOT_ENOUGH_HISTORY ({len(g_prod)} < seq_len={seq_len})")
                continue
            raw_prod_close = pd.to_numeric(g_prod[TARGET_COL], errors="coerce").values.astype("float32")
            if tkr is None:
                out_lines.append(f"{tkr_name}: No ticker in prod file and scaling params require ticker. Skipping.")
                continue
            row = sp[sp["Ticker"] == tkr_name]
            if row.empty:
                out_lines.append(f"{tkr_name}: Missing scaling params (cannot scale). Skipping.")
                continue
            vmin = float(row["target_min_train_den"].iloc[0]); vmax = float(row["target_max_train_den"].iloc[0])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                out_lines.append(f"{tkr_name}: Bad scaling params (min={vmin}, max={vmax}). Skipping.")
                continue
            last_window = raw_prod_close[-seq_len:]
            scaled_window = scale_minmax(last_window, vmin, vmax)[None, ...]  # shape (1, seq_len)
            X_in = scaled_window.astype("float32")[..., None]  # (1, seq_len, 1)
            y_pred_scaled = predict_batches(model, X_in, DEVICE, batch_size=1).reshape(-1)
            y_pred_base = inverse_minmax(y_pred_scaled, vmin, vmax)[0]
            prev_close = float(raw_prod_close[-1])
            rsi_prod = compute_rsi_wilder(raw_prod_close, period=RSI_PERIOD)
            rsi_prev = rsi_prod[-1] if len(rsi_prod) > 0 else float("nan")
            y_pred_rsi, s_used = rsi_shrink_toward_prev(prev_close, y_pred_base, rsi_prev, hi=RSI_HI, lo=RSI_LO, alpha=RSI_ALPHA)
            dir_base = "UP" if (y_pred_base - prev_close) > 0 else ("DOWN" if (y_pred_base - prev_close) < 0 else "UNCHANGED")
            dir_rsi  = "UP" if (y_pred_rsi - prev_close) > 0 else ("DOWN" if (y_pred_rsi - prev_close) < 0 else "UNCHANGED")
            line = (f"{tkr_name}: prev_close={prev_close:.6f} | pred_base={y_pred_base:.6f} ({dir_base}) | "
                    f"pred_rsi={y_pred_rsi:.6f} ({dir_rsi}) | rsi_prev={np.nan if not np.isfinite(rsi_prev) else rsi_prev:.2f} | shrink={s_used:.3f}")
            out_lines.append(line)

        prod_out_path = os.path.join(out_dir, f"{run_id}_production_prediction.txt")
        with open(prod_out_path, "w") as f:
            for L in out_lines:
                f.write(L + "\n")
        print("\n".join(out_lines))
        print(f"\nProduction predictions written to: {prod_out_path}")
        return  # done (no test-mode plotting)

    # -------------------------
    # TEST or ALL MODE: sliding-window predictions using actual history only
    # -------------------------
    all_dfs = []
    tickers = sorted(df[tkr_col].unique().tolist())
    # filter tickers to those present in split report
    tickers = [t for t in tickers if t in split_map]
    print(f"Tickers with split info: {len(tickers)} -> {tickers}")

    for tkr in tickers:
        # full raw series for this ticker
        g_full = df[df[tkr_col] == tkr].sort_values(DATE_COL).reset_index(drop=True)
        if g_full.empty:
            print(f"[WARN] No rows for ticker '{tkr}'. Skipping.")
            continue
        full_dates = pd.to_datetime(g_full[DATE_COL]).values
        raw_full_close = pd.to_numeric(g_full[TARGET_COL], errors="coerce").values.astype("float32")

        # scaling params (must match training ticker)
        row = sp[sp["Ticker"] == tkr]
        if row.empty:
            print(f"[WARN] Missing scaling params for '{tkr}'. Skipping.")
            continue
        vmin = float(row["target_min_train_den"].iloc[0]); vmax = float(row["target_max_train_den"].iloc[0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            print(f"[WARN] Bad scaling range for '{tkr}' (min={vmin}, max={vmax}). Skipping.")
            continue

        # compute RSI over full series
        rsi_full = compute_rsi_wilder(raw_full_close, period=RSI_PERIOD)

        # scale the full series (using train min/max)
        scaled_full_close = scale_minmax(raw_full_close, vmin, vmax)

        # build sliding windows across the full series (actual history only)
        X_full, y_full_scaled, idxs_full = build_windows_target_only(scaled_full_close, window=seq_len, horizon=1)
        if X_full.shape[0] == 0:
            print(f"[WARN] Not enough full-series rows for '{tkr}' to build windows (need at least {seq_len + 1}). Skipping.")
            continue

        # Determine which windows to predict depending on scope
        if args.scope == "test":
            # current behavior: only windows whose target date falls into saved test window
            ts, te = test_ranges.get(tkr, (None, None))
            if ts is None or pd.isna(ts) or pd.isna(te):
                print(f"[WARN] Missing test range for '{tkr}'. Skipping.")
                continue
            ts = pd.to_datetime(ts); te = pd.to_datetime(te)
            target_dates_full = pd.to_datetime(full_dates[idxs_full])
            mask = (target_dates_full >= ts) & (target_dates_full <= te)
            if not np.any(mask):
                print(f"[WARN] No windows in test range for '{tkr}' (seq_len={seq_len}). Skipping.")
                continue
            sel_idxs = np.nonzero(mask)[0]
        else:
            # 'all' scope: use all windows built across the full series
            sel_idxs = np.arange(len(idxs_full))

        # selection
        X_sel = X_full[sel_idxs]
        y_true_sel_scaled = y_full_scaled[sel_idxs]
        idxs_sel_full = idxs_full[sel_idxs]  # indices into full series
        target_dates_sel = pd.to_datetime(full_dates[idxs_sel_full])

        # predict (scaled) and invert
        y_pred_scaled = predict_batches(model, X_sel, DEVICE, batch_size=256)
        y_pred_base = inverse_minmax(y_pred_scaled, vmin, vmax)
        y_true = inverse_minmax(y_true_sel_scaled, vmin, vmax)

        # determine which split each target date belongs to (train/val/test/out)
        split_info = split_map.get(tkr)
        ts_train = pd.to_datetime(split_info["train_start"]); te_train = pd.to_datetime(split_info["train_end"])
        ts_val = pd.to_datetime(split_info["val_start"]); te_val = pd.to_datetime(split_info["val_end"])
        ts_test = pd.to_datetime(split_info["test_start"]); te_test = pd.to_datetime(split_info["test_end"])

        def assign_split(dt):
            if pd.isna(dt):
                return "out"
            if (dt >= ts_train) and (dt <= te_train):
                return "train"
            if (dt >= ts_val) and (dt <= te_val):
                return "val"
            if (dt >= ts_test) and (dt <= te_test):
                return "test"
            return "out"

        splits = np.array([assign_split(d) for d in target_dates_sel])

        # apply RSI shrink per target (uses prev actual)
        y_pred_rsi = np.empty_like(y_pred_base)
        shrink_used = np.ones_like(y_pred_base)
        for k in range(len(y_pred_base)):
            fi = int(idxs_sel_full[k])
            if fi - 1 < 0 or fi - 1 >= len(rsi_full):
                y_pred_rsi[k] = y_pred_base[k]; shrink_used[k] = 1.0; continue
            prev_close = raw_full_close[fi - 1]
            rsi_val = rsi_full[fi - 1]
            adj, s_used = rsi_shrink_toward_prev(prev_close, y_pred_base[k], rsi_val, hi=RSI_HI, lo=RSI_LO, alpha=RSI_ALPHA)
            y_pred_rsi[k] = adj; shrink_used[k] = s_used

        # write per-ticker CSV (aligned to target dates)
        rsi_prev_series = []
        for fi in idxs_sel_full:
            if fi - 1 >= 0 and fi - 1 < len(rsi_full):
                rsi_prev_series.append(rsi_full[int(fi) - 1])
            else:
                rsi_prev_series.append(np.nan)
        rsi_prev_series = np.array(rsi_prev_series, dtype=float)

        out_df = pd.DataFrame({
            "Date": target_dates_sel,
            "Ticker": tkr,
            "split": splits,
            "y_true_close": y_true,
            "y_pred_close_base": y_pred_base,
            "y_pred_close_rsi": y_pred_rsi,
            "rsi_at_prev": rsi_prev_series,
            "shrink_factor": shrink_used,
            "target_idx_in_full": idxs_sel_full
        }).sort_values("Date")
        out_csv = os.path.join(csv_dir, f"{tkr}_predictions_seq{seq_len}_scope_{args.scope}.csv")
        out_df.to_csv(out_csv, index=False)

        # compute per-split metrics for this ticker
        metrics_rows = []
        for sp_name in ("train","val","test","out"):
            mask_sp = out_df["split"].values == sp_name
            if mask_sp.sum() == 0:
                continue
            y_sp = out_df.loc[mask_sp, "y_true_close"].values
            p_sp = out_df.loc[mask_sp, "y_pred_close_base"].values
            p_sp_rsi = out_df.loc[mask_sp, "y_pred_close_rsi"].values
            metrics_rows.append({
                "Ticker": tkr, "split": sp_name, "N": len(y_sp),
                "MAE_base": mae(y_sp, p_sp), "RMSE_base": rmse(y_sp, p_sp),
                "MAE_rsi": mae(y_sp, p_sp_rsi), "RMSE_rsi": rmse(y_sp, p_sp_rsi)
            })
        # save ticker-level split metrics
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(out_dir, f"{tkr}_metrics_by_split_seq{seq_len}_scope_{args.scope}.csv"), index=False)

        # direction markers computed relative to full-series previous actual close (for base predictions)
        correct_mask, _wrong_mask = compute_direction_markers(raw_close_all=raw_full_close,
                                                              target_indices=idxs_sel_full,
                                                              y_true=y_true,
                                                              y_pred=y_pred_base)

        # plotting: choose window to display
        n_total = len(out_df)
        n_plot = min(max(1, int(args.plots_last_n)), n_total)
        if args.scope == "all" and args.plots_last_n >= n_total:
            df_plot = out_df.copy()
        else:
            df_plot = out_df.tail(n_plot).copy()

        # compute correct mask aligned to df_plot
        # idxs_sel_full is increasing so we can map target_idx -> position via searchsorted or where
        pos_map = {int(idx): i for i, idx in enumerate(idxs_sel_full)}
        correct_mask_plot = np.array([ correct_mask[pos_map[int(ti)]] if int(ti) in pos_map else False for ti in df_plot["target_idx_in_full"].values ])

        x_dates = pd.to_datetime(df_plot["Date"].values)
        y_actual = df_plot["y_true_close"].values
        y_pred_base_last = df_plot["y_pred_close_base"].values
        y_pred_rsi_last  = df_plot["y_pred_close_rsi"].values
        rsi_prev_last    = df_plot["rsi_at_prev"].values
        splits_last      = df_plot["split"].values

        plt.figure(figsize=(12, 4))

        if args.plot_type == "simple":
            # Simple: thin black line for actual, thin red line for base predictions.
            plt.plot(x_dates, y_actual, label="Actual Close", linewidth=1.0, color="black", zorder=1)
            plt.plot(x_dates, y_pred_base_last, label="Predicted Close (base)", linewidth=1.0, color="red", zorder=2)
        else:
            # Complex: original detailed plotting
            plt.plot(x_dates, y_actual, label="Actual Close", linewidth=2, color="orange", marker="|", markersize=8, markeredgewidth=1.5, zorder=2)
            plt.scatter(x_dates, y_pred_base_last, label="Predicted Close (base)", s=24, zorder=3)
            plt.scatter(x_dates, y_pred_rsi_last, label="Predicted Close (+RSI)", s=28, marker="^", alpha=0.9, zorder=4)
            if len(x_dates) > 1:
                for i in range(1, len(x_dates)):
                    plt.plot([x_dates[i - 1], x_dates[i]], [y_actual[i - 1], y_pred_base_last[i]], linestyle=":", linewidth=1.0, alpha=0.5)

            if np.any(correct_mask_plot):
                plt.scatter(x_dates[correct_mask_plot], y_pred_base_last[correct_mask_plot], marker=r'$\checkmark$', s=120, color='green', linewidths=0.0, label="Correct direction (base)")

        # draw vertical red lines for train_end and val_end and shaded regions for train/val/test (full series)
        split_info = split_map.get(tkr, {})
        train_start = pd.to_datetime(split_info.get("train_start"))
        train_end = pd.to_datetime(split_info.get("train_end"))
        val_start = pd.to_datetime(split_info.get("val_start"))
        val_end = pd.to_datetime(split_info.get("val_end"))
        test_start = pd.to_datetime(split_info.get("test_start"))
        test_end = pd.to_datetime(split_info.get("test_end"))

        try:
            x0 = pd.to_datetime(x_dates[0]); x1 = pd.to_datetime(x_dates[-1])
            def intersects(a,b,x0,x1): return (a <= x1) and (b >= x0)
            if pd.notna(train_start) and pd.notna(train_end) and intersects(train_start, train_end, x0, x1):
                s0 = max(train_start, x0); s1 = min(train_end, x1)
                plt.axvspan(s0, s1, alpha=0.2, color="grey", label="TRAIN region")
                plt.axvline(train_end, color="red", linestyle="--", linewidth=1.0)
            if pd.notna(val_start) and pd.notna(val_end) and intersects(val_start, val_end, x0, x1):
                s0 = max(val_start, x0); s1 = min(val_end, x1)
                plt.axvspan(s0, s1, alpha=0.2, color="skyblue", label="VAL region")
                plt.axvline(val_end, color="red", linestyle="--", linewidth=1.0)
            if pd.notna(test_start) and pd.notna(test_end) and intersects(test_start, test_end, x0, x1):
                s0 = max(test_start, x0); s1 = min(test_end, x1)
                plt.axvspan(s0, s1, alpha=0.2, color="lightgreen", label="TEST region")
        except Exception:
            pass

        # additional labels for complex plot: RSI below points
        if args.plot_type == "complex":
            y_min = float(np.min([y_actual.min(), y_pred_base_last.min(), y_pred_rsi_last.min()]))
            y_max = float(np.max([y_actual.max(), y_pred_base_last.max(), y_pred_rsi_last.max()]))
        else:
            # simple: use actual and base only
            y_min = float(np.min([y_actual.min(), y_pred_base_last.min()]))
            y_max = float(np.max([y_actual.max(), y_pred_base_last.max()]))
        y_rng = max(1e-9, y_max - y_min)
        label_y = y_min - 0.08 * y_rng
        plt.ylim(y_min - 0.14 * y_rng, y_max + 0.04 * y_rng)

        if args.plot_type == "complex":
            for xi, rsi_val in zip(x_dates, rsi_prev_last):
                if np.isfinite(rsi_val):
                    plt.text(xi, label_y, f"{rsi_val:.0f}", ha="center", va="top", fontsize=6, rotation=45, alpha=0.9)

        plt.title(f"{tkr} — Next-Day Close (seq_len={seq_len}) — Showing last {len(df_plot)} targets (scope={args.scope})")
        plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(loc="upper left", fontsize=8); plt.tight_layout()
        out_png = os.path.join(plots_dir, f"{tkr}_actual_vs_pred_seq{seq_len}_scope_{args.scope}_last{len(df_plot)}_plot_{args.plot_type}.png")
        plt.savefig(out_png, dpi=160); plt.close()

        all_dfs.append(out_df)

    # combined CSV & summary metrics
    if all_dfs:
        combo = pd.concat(all_dfs, ignore_index=True).sort_values(["Ticker", "Date"])
        combo.to_csv(os.path.join(out_dir, f"all_tickers_predictions_seq{seq_len}_scope_{args.scope}.csv"), index=False)

        # overall per-ticker per-split metrics
        rows = []
        for tkr in combo["Ticker"].unique():
            c = combo[combo["Ticker"] == tkr]
            for spn in ("train","val","test","out"):
                cc = c[c["split"] == spn]
                if len(cc) == 0: continue
                y = cc["y_true_close"].values
                p_base = cc["y_pred_close_base"].values
                p_rsi  = cc["y_pred_close_rsi"].values
                rows.append({"Ticker": tkr, "split": spn, "N": len(cc),
                             "MAE_base": mae(y, p_base), "RMSE_base": rmse(y, p_base),
                             "MAE_rsi": mae(y, p_rsi), "RMSE_rsi": rmse(y, p_rsi)})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"summary_metrics_by_ticker_and_split_seq{seq_len}_scope_{args.scope}.csv"), index=False)

    print("\nInference complete.")
    print(f"Run artifacts:        {artifact_dir}")
    print(f"Outputs written to:   {out_dir}")
    print(f"Per-ticker CSVs:      {os.path.join(out_dir, 'per_ticker_csv')}")
    print(f"Per-ticker plots:     {os.path.join(out_dir, 'plots')}")
    print(f"Combined predictions: {os.path.join(out_dir, f'all_tickers_predictions_seq{seq_len}_scope_{args.scope}.csv')}")
    print(f"Summary metrics:      {os.path.join(out_dir, f'summary_metrics_by_ticker_and_split_seq{seq_len}_scope_{args.scope}.csv')}")

if __name__ == "__main__":
    main()
