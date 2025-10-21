# predict_xlstmts_from_combo.py
# End-to-end inference for one trained run (derived from <COMBO>_PRICES.txt).
# Loads artifacts from artifacts/<run_id>/, runs horizon-day predictions on the TEST
# window per ticker, and saves per-ticker CSVs and "last N days" plots with direction markers.

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
    p = argparse.ArgumentParser(description="Predict horizon-day Close for one trained combo using its artifacts.")
    p.add_argument("--data_file", required=True, help="Path to the raw <COMBO>_PRICES.txt used for that training run.")
    p.add_argument("--artifacts_root", default="artifacts", help="Root directory containing run artifacts (default: artifacts).")
    p.add_argument("--out_subdir", default="inference", help="Subdirectory under the run artifacts to write predictions (default: inference).")
    p.add_argument("--plots_last_n", type=int, default=30, help="Show only the last N target days in per-ticker plots (default: 30).")
    p.add_argument("--seq_len", type=int, default=None, help="Override seq_len for inference (defaults to the saved best seq_len).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device (default auto).")
    return p.parse_args()

# -------------------------
# Constants (must match training)
# -------------------------
DATE_COL = "Date"
TARGET_COL = "Close"
TICKER_COL_CANDIDATES = ["Ticker", "Company", "Symbol", "Name"]

# xLSTM block hyperparams used during training (non-tunable here)
MLSTM_CONV_K = 4
MLSTM_HEADS = 2
MLSTM_PROJ_SIZE = 2
SLSTM_CONV_K = 2
SLSTM_HEADS = 2
SLSTM_FF_FACTOR = 1.1

# -------------------------
# Model blocks
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

class xLSTM_TS_Flexible(nn.Module):
    """
    Flexible head that reconstructs the exact architecture used in training
    based on saved hyperparameters: embed_dim, num_layers, arch_style.
    """
    def __init__(self, input_size: int, d_model: int, output_size: int, num_layers: int = 3, arch_style: str = "alternating_ms"):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        blocks = []
        def make_block(kind: str):
            if kind == "m":
                return mLSTMBlock(d_model, MLSTM_CONV_K, MLSTM_HEADS, MLSTM_PROJ_SIZE)
            else:
                return sLSTMBlock(d_model, SLSTM_CONV_K, SLSTM_HEADS, SLSTM_FF_FACTOR)
        style = (arch_style or "alternating_ms").lower()
        if style in ("all_m", "m_only"):
            blocks = [make_block("m") for _ in range(num_layers)]
        elif style in ("all_s", "s_only"):
            blocks = [make_block("s") for _ in range(num_layers)]
        elif style in ("msm", "sms"):
            pattern = list(style)
            if num_layers != len(pattern):
                pat = (pattern * ((num_layers + len(pattern) - 1) // len(pattern)))[:num_layers]
            else:
                pat = pattern
            blocks = [make_block(ch) for ch in pat]
        else:
            pat = [("m" if i % 2 == 0 else "s") for i in range(num_layers)]
            blocks = [make_block(ch) for ch in pat]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
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
    """
    Build sliding windows for inference.

    Returns:
      X: (N, window, 1) scaled last-window inputs
      y_res: (N,) scaled residual targets = scaled_target - last_input_scaled
      idx_arr: (N,) target indices in the original series (index of the target day)
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

def compute_direction_markers(raw_close_all: np.ndarray, target_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """
    True/Pred direction are computed relative to the previous ACTUAL close.
    Returns boolean arrays marking prediction-direction correctness per target.

    NOTE: 'target_indices' must be indices of the target day in raw_close_all.
    We use raw_close_all[target_idx - 1] as the "previous actual close" (same as training-time RSI baseline).
    """
    assert len(target_indices) == len(y_true) == len(y_pred)
    dirs_true, dirs_pred = [], []
    for k, t in enumerate(target_indices):
        if t - 1 < 0:
            dirs_true.append(0)
            dirs_pred.append(0)
            continue
        prev = raw_close_all[t - 1]  # day-before-target actual close (unchanged semantics)
        delta_t = raw_close_all[t] - prev
        delta_p = y_pred[k] - prev
        sgn = lambda x: (-1 if x < 0 else (1 if x > 0 else 0))
        dirs_true.append(sgn(delta_t))
        dirs_pred.append(sgn(delta_p))
    dirs_true = np.array(dirs_true, dtype=int)
    dirs_pred = np.array(dirs_pred, dtype=int)
    correct = dirs_true == dirs_pred
    wrong = ~correct
    return correct, wrong

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

    # resolve file paths inside this run's artifacts
    split_report_path = os.path.join(artifact_dir, "split_report.csv")
    scaling_params_path = os.path.join(artifact_dir, "scaling_params.csv")
    hparams_path = os.path.join(artifact_dir, f"xlstm_ts_best_hparams_{run_id}.json")
    model_state_path = os.path.join(artifact_dir, f"xlstm_ts_best_state_dict_{run_id}.pt")
    data_config_path = os.path.join(artifact_dir, "data_config.json")

    # out dirs
    out_dir = os.path.join(artifact_dir, args.out_subdir)
    plots_dir = os.path.join(out_dir, "plots")
    csv_dir = os.path.join(out_dir, "per_ticker_csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # load horizon from data_config if available (fallback to 1)
    horizon = 1
    if os.path.exists(data_config_path):
        try:
            data_conf = load_json(data_config_path)
            horizon = int(data_conf.get("horizon", horizon))
        except Exception:
            print(f"[WARN] Could not read horizon from {data_config_path}; defaulting to 1")
    print(f"Horizon for inference: {horizon} day(s) ahead")

    # ---- load hparams & build model that matches training ----
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Missing hparams: {hparams_path}")
    hp = load_hparams(hparams_path)
    embed_dim = int(hp.get("embed_dim", 64))
    num_layers = int(hp.get("num_layers", 3))
    arch_style = str(hp.get("arch_style", "alternating_ms"))
    seq_len = int(args.seq_len) if args.seq_len is not None else int(hp.get("seq_len", 100))

    model = xLSTM_TS_Flexible(input_size=1, d_model=embed_dim, output_size=1, num_layers=num_layers, arch_style=arch_style).to(DEVICE)

    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Missing model state: {model_state_path}")
    state = torch.load(model_state_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    print("Model loaded and ready.")

    # ---- load scaling params and split report ----
    if not os.path.exists(scaling_params_path):
        raise FileNotFoundError(f"Missing scaling params: {scaling_params_path}")
    sp = pd.read_csv(scaling_params_path)

    req_sp = {"Ticker", "target_min_train_den", "target_max_train_den"}
    if not req_sp.issubset(sp.columns):
        raise ValueError(f"{os.path.basename(scaling_params_path)} must contain columns: {sorted(req_sp)}")

    if not os.path.exists(split_report_path):
        raise FileNotFoundError(f"Missing split report: {split_report_path}")
    sr = pd.read_csv(split_report_path, parse_dates=["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"])
    req_sr = {"Ticker", "test_start", "test_end"}
    if not req_sr.issubset(sr.columns):
        raise ValueError(f"{os.path.basename(split_report_path)} must contain columns: {sorted(req_sr)}")
    test_ranges = {row["Ticker"]: (row["test_start"], row["test_end"]) for _, row in sr.iterrows()}

    # ---- read the raw multi-ticker file ----
    df = pd.read_csv(args.data_file, sep=None, engine="python")
    if DATE_COL not in df.columns:
        raise ValueError(f"Input must contain a '{DATE_COL}' column")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Input must contain a '{TARGET_COL}' column")
    tkr_col = find_ticker_col(df)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([tkr_col, DATE_COL]).reset_index(drop=True)

    # ---- restrict to the saved TEST windows per ticker ----
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
        raise RuntimeError("No rows found in raw data for the saved test windows. Check ticker names and artifacts/split_report.csv for this run.")

    df_test = df_test[[DATE_COL, tkr_col, TARGET_COL]].copy().sort_values([tkr_col, DATE_COL])

    # ---- per-ticker inference on test segment only ----
    all_dfs = []
    tickers = df_test[tkr_col].unique().tolist()
    print(f"Tickers in TEST window: {len(tickers)} -> {tickers}")

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

        # sliding windows over TEST segment — returns residual targets (scaled)
        X, y_true_res_scaled, idxs = build_windows_target_only(scaled_close, window=seq_len, horizon=horizon)
        if X.shape[0] == 0:
            print(f"[WARN] Not enough test rows for '{tkr}' (need at least {seq_len + horizon}). Skipping.")
            continue

        # predict residuals in scaled space → reconstruct absolute scaled → inverse back to raw prices
        y_pred_res_scaled = predict_batches(model, X, DEVICE, batch_size=256)
        # last input scaled values (matching training reconstruction)
        last_vals_scaled = X[:, -1, 0]
        y_pred_abs_scaled = y_pred_res_scaled + last_vals_scaled
        y_true_abs_scaled = y_true_res_scaled + last_vals_scaled

        y_pred = inverse_minmax(y_pred_abs_scaled, vmin, vmax)
        y_true = inverse_minmax(y_true_abs_scaled, vmin, vmax)

        target_dates = pd.to_datetime(dates[idxs])

        # Save per-ticker CSV
        out_df = pd.DataFrame({"Date": target_dates, "Ticker": tkr, "y_true_close": y_true, "y_pred_close": y_pred}).sort_values("Date")
        out_csv = os.path.join(csv_dir, f"{tkr}_predictions_seq{seq_len}_h{horizon}.csv")
        out_df.to_csv(out_csv, index=False)

        # Direction markers vs previous actual close (previous = day-before-target)
        correct_mask, wrong_mask = compute_direction_markers(raw_close_all=raw_close, target_indices=idxs, y_true=y_true, y_pred=y_pred)

        # Plot only the last N targets
        n_total = len(out_df)
        n_plot = min(max(1, int(args.plots_last_n)), n_total)
        df_plot = out_df.tail(n_plot).copy()
        # align direction masks to the last N targets
        correct_last = correct_mask[-n_plot:]

        x_dates = df_plot["Date"].values
        y_actual = df_plot["y_true_close"].values
        y_predpl = df_plot["y_pred_close"].values

        plt.figure(figsize=(10, 4))
        # Actual: orange line with small vertical dash at each data point
        plt.plot(
            x_dates,
            y_actual,
            label="Actual Close",
            linewidth=2,
            color="orange",
            marker="|",
            markersize=8,
            markeredgewidth=1.5,
            zorder=2
        )
        # Predicted: POINTS ONLY (no lines)
        plt.scatter(x_dates, y_predpl, label="Predicted Close", s=28, zorder=3)

        # Faint dotted connectors from previous ACTUAL to current PREDICTED (skip the first)
        if len(x_dates) > 1:
            for i in range(1, len(x_dates)):
                plt.plot(
                    [x_dates[i - 1], x_dates[i]],
                    [y_actual[i - 1], y_predpl[i]],
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.5
                )

        # Green tick on predicted points when direction matches actual vs previous ACTUAL close
        if np.any(correct_last):
            plt.scatter(
                x_dates[correct_last],
                y_predpl[correct_last],
                marker=r'$\checkmark$',
                s=140,
                color='green',
                linewidths=0.0,
                label="Correct direction"
            )

        plt.title(f"{tkr} — {horizon}-Day Ahead Close (seq_len={seq_len}) — Last {n_plot} targets")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(plots_dir, f"{tkr}_actual_vs_pred_seq{seq_len}_h{horizon}_last{n_plot}.png")
        plt.savefig(out_png, dpi=160)
        plt.close()

        all_dfs.append(out_df)

    # ---- Combined outputs and quick metrics ----
    if all_dfs:
        combo = pd.concat(all_dfs, ignore_index=True).sort_values(["Ticker", "Date"])
        combo.to_csv(os.path.join(out_dir, f"all_tickers_predictions_seq{seq_len}_h{horizon}.csv"), index=False)
        rows = []
        for tkr in combo["Ticker"].unique():
            c = combo[combo["Ticker"] == tkr]
            y = c["y_true_close"].values
            p = c["y_pred_close"].values
            mae = float(np.mean(np.abs(p - y)))
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            rows.append({"Ticker": tkr, "N": len(c), "MAE": mae, "RMSE": rmse})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"summary_metrics_by_ticker_seq{seq_len}_h{horizon}.csv"), index=False)

    print("\nInference complete.")
    print(f"Run artifacts:        {artifact_dir}")
    print(f"Outputs written to:   {out_dir}")
    print(f"Per-ticker CSVs:      {os.path.join(out_dir, 'per_ticker_csv')}")
    print(f"Per-ticker plots:     {os.path.join(out_dir, 'plots')}")
    print(f"Combined predictions: {os.path.join(out_dir, f'all_tickers_predictions_seq{seq_len}_h{horizon}.csv')}")
    print(f"Summary metrics:      {os.path.join(out_dir, f'summary_metrics_by_ticker_seq{seq_len}_h{horizon}.csv')}")

if __name__ == "__main__":
    main()
