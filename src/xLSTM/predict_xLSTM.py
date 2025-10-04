# predict_xlstmts_from_STABLE_PRICES_plot30_dirmarkers.py
# End-to-end inference for multi-ticker next-day Close from raw STABLE_PRICES.txt
# Plots only the last 30 target days and overlays direction-accuracy markers on the ACTUAL line.

import os, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Configuration
# -------------------------
RAW_PATH = "STABLE_PRICES.txt"                          # raw multi-ticker file used to build splits
DATE_COL = "Date"
TARGET_COL = "Close"
# If your company/ticker column isn't exactly "Ticker", the script will try the candidates below
TICKER_COL_CANDIDATES = ["Ticker", "Company", "Symbol", "Name"]

SEQ_LEN = 100
HORIZON = 1

SPLIT_REPORT_PATH   = os.path.join("artifacts", "split_report.csv")
SCALING_PARAMS_PATH = os.path.join("artifacts", "scaling_params.csv")  # per-ticker min/max fit on TRAIN (denoised)
MODEL_STATE_PATH    = os.path.join("artifacts", "xlstm_ts_best_state_dict_STABLE_PRICES.pt")
HPARAMS_PATH        = os.path.join("artifacts", "xlstm_ts_best_hparams_STABLE_PRICES.json")

OUT_DIR   = os.path.join("artifacts", "inference_from_raw")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CSV_DIR   = os.path.join(OUT_DIR, "per_ticker_csv")

# Plotting controls
PLOT_LAST_N_DAYS = 30  # <-- only final 30 target days

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# -------------------------
# Model definition (must match training)
# -------------------------
MLSTM_CONV_K = 4
MLSTM_HEADS = 2
MLSTM_PROJ_SIZE = 2
SLSTM_CONV_K = 2
SLSTM_HEADS = 2
SLSTM_FF_FACTOR = 1.1

class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0, groups=channels, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1,2)
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(1,2)

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
    def __init__(self, input_size: int, d_model: int, output_size: int):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.block1 = mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        self.block2 = sLSTMBlock(d_model=d_model, kernel=SLSTM_CONV_K, heads=SLSTM_HEADS, ff_factor=SLSTM_FF_FACTOR)
        self.block3 = mLSTMBlock(d_model=d_model, kernel=MLSTM_CONV_K, heads=MLSTM_HEADS, proj_size=MLSTM_PROJ_SIZE)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x[:, -1, :])

# -------------------------
# Helpers
# -------------------------
def find_ticker_col(df: pd.DataFrame) -> str:
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No ticker/company column found. Tried: {TICKER_COL_CANDIDATES}")

def load_hparams(path: str) -> dict:
    with open(path, "r") as f:
        hp_all = json.load(f)
    hp = hp_all.get("best_params", hp_all)
    return hp

def load_scaling_params(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scaling params at {path}")
    sp = pd.read_csv(path)
    req = {"Ticker", "target_min_train_den", "target_max_train_den"}
    if not req.issubset(sp.columns):
        raise ValueError(f"scaling_params.csv must contain columns: {sorted(req)}")
    return sp

def load_split_report(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split report at {path}")
    sr = pd.read_csv(path, parse_dates=["train_start","train_end","val_start","val_end","test_start","test_end"])
    req = {"Ticker","test_start","test_end"}
    if not req.issubset(sr.columns):
        raise ValueError(f"split_report.csv must contain columns: {sorted(req)}")
    return sr

def scale_minmax(x: np.ndarray, vmin: float, vmax: float, eps: float=1e-12) -> np.ndarray:
    rng = max(eps, float(vmax - vmin))
    return (x - vmin) / rng

def inverse_minmax(x_s: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return x_s * (vmax - vmin) + vmin

def build_windows_target_only(s: np.ndarray, window: int, horizon: int=1):
    X_list, y_list, idx_list = [], [], []
    n = len(s)
    for t in range(window, n - horizon + 1):
        X_list.append(s[t - window:t])
        y_list.append(s[t + horizon - 1])
        idx_list.append(t)  # y index in the original series
    if not X_list:
        return np.zeros((0, window, 1), dtype="float32"), np.zeros((0,), dtype="float32"), np.array([], dtype=int)
    X = np.stack(X_list, axis=0).astype("float32")[..., None]
    y = np.array(y_list, dtype="float32")
    idx_arr = np.array(idx_list, dtype=int)
    return X, y, idx_arr

@torch.no_grad()
def predict_batches(model: nn.Module, X: np.ndarray, batch_size: int=256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(DEVICE)
        yp = model(xb).detach().cpu().numpy().reshape(-1)
        out.append(yp)
    return np.concatenate(out, axis=0) if out else np.array([])

def compute_direction_markers(raw_close_all: np.ndarray, target_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """
    For each target index t (corresponding to y_true[k] and y_pred[k]),
    compare direction vs the previous day:
      delta_true = raw_close[t] - raw_close[t-1]
      delta_pred = y_pred[k]  - raw_close[t-1]
    Return boolean masks for correct (same sign) and wrong (opposite sign).
    Treat both zeros as correct; if one is zero and the other nonzero, mark as wrong.
    """
    assert len(target_indices) == len(y_true) == len(y_pred)
    dirs_true = []
    dirs_pred = []
    for k, t in enumerate(target_indices):
        if t - 1 < 0:
            # Shouldn't happen for valid targets, but guard anyway.
            dirs_true.append(0); dirs_pred.append(0); continue
        prev = raw_close_all[t - 1]
        delta_t = raw_close_all[t] - prev
        delta_p = y_pred[k] - prev
        # sign mapping: -1, 0, +1
        def sgn(x):
            return -1 if x < 0 else (1 if x > 0 else 0)
        dirs_true.append(sgn(delta_t))
        dirs_pred.append(sgn(delta_p))
    dirs_true = np.array(dirs_true, dtype=int)
    dirs_pred = np.array(dirs_pred, dtype=int)
    correct = dirs_true == dirs_pred
    wrong = ~correct
    return correct, wrong

# -------------------------
# Load assets
# -------------------------
print("Loading hparams and model…")
hp = load_hparams(HPARAMS_PATH)
embed_dim = int(hp.get("embed_dim", 64))
model = xLSTM_TS(input_size=1, d_model=embed_dim, output_size=1).to(DEVICE)
state = torch.load(MODEL_STATE_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
print("Model loaded on", DEVICE)

print("Loading scaling params and split report…")
sp = load_scaling_params(SCALING_PARAMS_PATH)
sr = load_split_report(SPLIT_REPORT_PATH)
test_ranges = {row["Ticker"]: (row["test_start"], row["test_end"]) for _, row in sr.iterrows()}

print("Reading raw multi-ticker file…")
df = pd.read_csv(RAW_PATH, sep=None, engine="python")
if DATE_COL not in df.columns:
    raise ValueError(f"Input file must contain a '{DATE_COL}' column")
if TARGET_COL not in df.columns:
    raise ValueError(f"Input file must contain a '{TARGET_COL}' column (raw Close)")
tkr_col = find_ticker_col(df)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).sort_values([tkr_col, DATE_COL]).reset_index(drop=True)

# -------------------------
# Restrict to TEST window per ticker
# -------------------------
pieces = []
for tkr, g in df.groupby(tkr_col, sort=False):
    ts, te = test_ranges.get(tkr, (None, None))
    if ts is None or pd.isna(ts) or pd.isna(te):
        continue
    g2 = g[(g[DATE_COL] >= ts) & (g[DATE_COL] <= te)].copy()
    if not g2.empty:
        pieces.append(g2)
df_test = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0]
if df_test.empty:
    raise RuntimeError("No rows found for test windows. Check artifacts/split_report.csv and ticker names.")

df_test = df_test[[DATE_COL, tkr_col, TARGET_COL]].copy().sort_values([tkr_col, DATE_COL])

# -------------------------
# Per-ticker inference
# -------------------------
all_dfs = []
tickers = df_test[tkr_col].unique().tolist()
print(f"Found {len(tickers)} tickers with test data.")

for tkr in tickers:
    g = df_test[df_test[tkr_col] == tkr].sort_values(DATE_COL).reset_index(drop=True)
    raw_close = pd.to_numeric(g[TARGET_COL], errors="coerce").values.astype("float32")
    dates = g[DATE_COL].values

    # scaling params for this ticker
    row = sp[sp["Ticker"] == tkr]
    if row.empty:
        print(f"[WARN] Missing scaling params for '{tkr}'. Skipping.")
        continue
    vmin = float(row["target_min_train_den"].iloc[0])
    vmax = float(row["target_max_train_den"].iloc[0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        print(f"[WARN] Bad scaling range for '{tkr}' (min={vmin}, max={vmax}). Skipping.")
        continue

    # scale raw test close using TRAIN min/max (to match training-time scaler)
    scaled_close = scale_minmax(raw_close, vmin, vmax)

    # build sliding windows over the TEST segment only
    X, y_true_scaled, idxs = build_windows_target_only(scaled_close, window=SEQ_LEN, horizon=HORIZON)
    if X.shape[0] == 0:
        print(f"[WARN] Not enough test rows for '{tkr}' (need at least {SEQ_LEN + HORIZON}). Skipping.")
        continue

    # predict in scaled space, then inverse-transform back to raw prices
    y_pred_scaled = predict_batches(model, X, batch_size=256)
    y_pred = inverse_minmax(y_pred_scaled, vmin, vmax)
    y_true = inverse_minmax(y_true_scaled, vmin, vmax)
    target_dates = pd.to_datetime(dates[idxs])

    # Save per-ticker CSV (full test)
    out_df = pd.DataFrame({
        "Date": target_dates,
        "Ticker": tkr,
        "y_true_close": y_true,
        "y_pred_close": y_pred
    }).sort_values("Date")
    out_csv = os.path.join(CSV_DIR, f"{tkr}_predictions_seq{SEQ_LEN}.csv")
    out_df.to_csv(out_csv, index=False)

    # ---- Direction markers computed against previous day's ACTUAL close ----
    correct_mask, wrong_mask = compute_direction_markers(raw_close_all=raw_close, target_indices=idxs, y_true=y_true, y_pred=y_pred)

    # ---- Plot only the final N target days ----
    n_total = len(out_df)
    n_plot = min(PLOT_LAST_N_DAYS, n_total)
    df_plot = out_df.tail(n_plot).copy()

    # Slice masks to last n_plot elements
    correct_last = correct_mask[-n_plot:]
    wrong_last   = wrong_mask[-n_plot:]

    # Build marker series aligned with df_plot
    x_dates = df_plot["Date"].values
    y_actual = df_plot["y_true_close"].values
    y_predpl = df_plot["y_pred_close"].values

    # For marker positions, use actual y values
    plt.figure(figsize=(10, 4))
    # Lines
    plt.plot(x_dates, y_actual, label="Actual")
    plt.plot(x_dates, y_predpl, label="Predicted")

    # Overlays on ACTUAL line
    if np.any(correct_last):
        plt.scatter(x_dates[correct_last], y_actual[correct_last], marker='o', color='green', s=36, label="Correct direction")
    if np.any(wrong_last):
        plt.scatter(x_dates[wrong_last], y_actual[wrong_last], marker='x', color='red', s=48, label="Wrong direction")

    plt.title(f"{tkr} — Next-Day Close (seq_len={SEQ_LEN}) — Last {n_plot} days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, f"{tkr}_actual_vs_pred_seq{SEQ_LEN}_last{n_plot}.png")
    plt.savefig(out_png, dpi=160)
    plt.close()

    all_dfs.append(out_df)

# -------------------------
# Combined outputs and quick metrics
# -------------------------
if all_dfs:
    combo = pd.concat(all_dfs, ignore_index=True).sort_values(["Ticker","Date"])
    combo.to_csv(os.path.join(OUT_DIR, f"all_tickers_predictions_seq{SEQ_LEN}.csv"), index=False)

    # simple MAE/RMSE per ticker
    rows = []
    for tkr in combo["Ticker"].unique():
        c = combo[combo["Ticker"] == tkr]
        y = c["y_true_close"].values
        p = c["y_pred_close"].values
        mae = float(np.mean(np.abs(p - y)))
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        rows.append({"Ticker": tkr, "N": len(c), "MAE": mae, "RMSE": rmse})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"summary_metrics_by_ticker_seq{SEQ_LEN}.csv"), index=False)

print("\nInference complete.")
print(f"Per-ticker CSVs: {CSV_DIR}")
print(f"Per-ticker plots (last {PLOT_LAST_N_DAYS} days): {PLOTS_DIR}")
print(f"Combined predictions CSV: {os.path.join(OUT_DIR, f'all_tickers_predictions_seq{SEQ_LEN}.csv')}")
print(f"Summary metrics CSV: {os.path.join(OUT_DIR, f'summary_metrics_by_ticker_seq{SEQ_LEN}.csv')}")
