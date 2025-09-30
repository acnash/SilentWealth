# ===========================================================
# combined_prep_and_tpe_train_xlstmts_multiticker.py
# Full pipeline: multi-ticker preprocessing + TPE tuning + verdict
# Changes:
#  - Training & validation: denoised -> normalized (Scaler fitted on train only)
#  - Test: NOT denoised, but normalized (scaled using the train scaler)
#  - Removed plotting code
# ===========================================================

import os, math, json
import numpy as np
import pandas as pd
import pywt
from typing import Tuple, List, Dict, Any
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.samplers import TPESampler

# ----------------------
# Preprocessing config
# ----------------------
INPUT_PATH = "STABLE_PRICES.txt"         # ← unified file with multiple tickers
DATE_COL = "Date"
TICKER_COL = "Ticker"
TARGET_COL = "Adj Close"
FEATURE_COLS = ["Open","High","Low","Close","Adj Close","Volume"]
HORIZON = 1
TRAIN_FRACTION = 0.86
VAL_FRACTION = 0.07
BUSINESS_DAY_METHOD = "time"
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
SEQ_LEN_CHOICES = [100, 150, 200]
LR_CHOICES = [1e-4, 5e-4, 1e-4]
BATCH_CHOICES = [8, 16, 32]
EMBED_DIM_CHOICES = [32, 64, 128, 256]

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def log(msg: str) -> None:
    if VERBOSE: print(msg)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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

def wavelet_denoise_1d(x: np.ndarray, wavelet: str=WAVELET, level: int=None, mode: str=THRESH_MODE) -> np.ndarray:
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

def split_series_by_fraction(ts: TimeSeries, train_frac: float, val_frac: float) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    n = len(ts); n_train = int(math.floor(n * train_frac)); n_val = int(math.floor(n * val_frac))
    train = ts[:n_train]; val = ts[n_train:n_train + n_val] if n_val > 0 else ts[n_train:n_train]
    test = ts[n_train + n_val:]; return train, val, test

def scale_with_darts(train_ts: TimeSeries, other_ts_list: List[TimeSeries]) -> Tuple[Scaler, List[TimeSeries]]:
    scaler = Scaler()  # MinMax(0,1)
    scaler.fit(train_ts)
    scaled_train = scaler.transform(train_ts)
    scaled_others = [scaler.transform(t) for t in other_ts_list]
    return scaler, [scaled_train] + scaled_others

def ts_to_df(ts: TimeSeries) -> pd.DataFrame:
    df = ts.to_dataframe() if hasattr(ts, "to_dataframe") else ts.pd_dataframe(copy=True)
    # Darts may add a component column level; flatten column names back to originals
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[-1] for c in df.columns]
    df.index.name = "Date"; return df

def build_windows_target_only(series: pd.Series, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray]:
    vals = series.values.astype("float32"); X_list, y_list = [], []; n = len(vals)
    for t in range(window, n - horizon + 1):
        X_list.append(vals[t - window:t]); y_list.append(vals[t + horizon - 1])
    X = np.stack(X_list, axis=0).astype("float32") if len(X_list) > 0 else np.zeros((0, window), dtype="float32")
    y = np.array(y_list, dtype="float32") if len(y_list) > 0 else np.zeros((0,), dtype="float32")
    X = X[..., None]; return X, y  # (N, T, 1), (N,)

def build_windows_from_multiticker(df: pd.DataFrame, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for tkr, g in df.groupby(TICKER_COL, sort=False):
        g = g.sort_index()
        if TARGET_COL not in g.columns: continue
        s = g[TARGET_COL].astype("float32").dropna()
        if len(s) < window + horizon: continue
        X, y = build_windows_target_only(s, window=window, horizon=horizon)
        if X.shape[0] > 0:
            Xs.append(X); ys.append(y)
    if len(Xs) == 0: return np.zeros((0, window, 1), dtype="float32"), np.zeros((0,), dtype="float32")
    X_all = np.concatenate(Xs, axis=0); y_all = np.concatenate(ys, axis=0)
    return X_all, y_all

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
# Training helpers
# ------------------------------------------------------------
def load_scaled_split_df(path: str) -> pd.DataFrame:
    assert os.path.exists(path), f"Missing file: {path}"
    df = pd.read_csv(path, sep=None, engine="python")
    assert DATE_COL in df.columns and TICKER_COL in df.columns and TARGET_COL in df.columns, "Missing required columns"
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL).sort_index()
    return df[[TICKER_COL, TARGET_COL]].copy()

def make_arrays(seq_len: int) -> Dict[str, np.ndarray]:
    train_df = load_scaled_split_df("scaled_train_businessB_denoised.csv")
    val_df   = load_scaled_split_df("scaled_val_businessB_denoised.csv")
    test_df  = load_scaled_split_df("scaled_test_businessB.csv")  # test is scaled but not denoised
    Xtr, ytr = build_windows_from_multiticker(train_df, window=seq_len, horizon=1)
    Xva, yva = build_windows_from_multiticker(val_df,   window=seq_len, horizon=1)
    Xte, yte = build_windows_from_multiticker(test_df,  window=seq_len, horizon=1)
    return {"Xtr":Xtr, "ytr":ytr, "Xva":Xva, "yva":yva, "Xte":Xte, "yte":yte}

def make_loaders(arrs: Dict[str, np.ndarray], batch: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval(); total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb); loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0); count += xb.size(0)
    return total / max(1, count)

# ------------------------------------------------------------
# Preprocess end-to-end for multi-ticker
# ------------------------------------------------------------
def preprocess_and_save() -> None:
    log("Step 1: Load multi-ticker data")
    df_all = read_any_table(INPUT_PATH, DATE_COL)
    assert TICKER_COL in df_all.columns, f"Missing column '{TICKER_COL}'"
    missing = [c for c in FEATURE_COLS if c not in df_all.columns]
    if missing: log(f"Warning: missing columns in input: {missing}")
    keep_cols = [c for c in FEATURE_COLS if c in df_all.columns]
    df_all = df_all[[TICKER_COL] + keep_cols].copy()
    log(f"Loaded {len(df_all)} rows across {df_all[TICKER_COL].nunique()} tickers")

    train_frames, val_frames, test_frames = [], [], []

    for ticker, g in df_all.groupby(TICKER_COL, sort=False):
        log(f"\n--- Ticker: {ticker} ---")
        g_num = g.drop(columns=[TICKER_COL]).copy()
        log("Clean nulls"); g_num = g_num.replace([np.inf, -np.inf], np.nan).dropna(how="all"); g_num = g_num.interpolate(method="time").ffill().bfill()
        log("Align to business days"); g_num = ensure_business_days(g_num)

        # DENOISE for train & val — keep the RAW for test (per your instruction)
        log("Wavelet denoise (for train & val)")
        g_den = wavelet_denoise_df(g_num, keep_cols)

        # Build Darts time series for both denoised and raw versions (same index/length)
        ts_den = darts_series_from_df(g_den, keep_cols)
        ts_raw = darts_series_from_df(g_num, keep_cols)

        # Split both (positional split -> same split points)
        log("Split 86/7/7")
        train_den, val_den, test_den_from_den = split_series_by_fraction(ts_den, TRAIN_FRACTION, VAL_FRACTION)
        train_raw, val_raw, test_raw = split_series_by_fraction(ts_raw, TRAIN_FRACTION, VAL_FRACTION)

        # Fit scaler on denoised train and transform: train_den, val_den, AND test_raw (test is scaled but not denoised)
        log("Scale (fit on denoised train only). Test will be SCALED but NOT denoised.")
        scaler, scaled_list = scale_with_darts(train_den, [val_den, test_raw])
        # scaled_list = [scaled_train, scaled_val, scaled_test_raw_scaled]
        train_s = scaled_list[0]
        val_s   = scaled_list[1] if len(scaled_list) > 1 else None
        test_s  = scaled_list[2] if len(scaled_list) > 2 else None

        # Convert back to DataFrames for saving/aggregation
        train_df = ts_to_df(train_s)
        val_df   = ts_to_df(val_s) if val_s is not None else pd.DataFrame(columns=keep_cols)
        test_df  = ts_to_df(test_s) if test_s is not None else pd.DataFrame(columns=keep_cols)

        # Attach ticker column and append to master lists
        for df, bucket in [(train_df, train_frames), (val_df, val_frames), (test_df, test_frames)]:
            if df is None or df.empty:
                # still keep an empty placeholder (skip appending)
                continue
            df = df.reset_index().rename(columns={"index": DATE_COL})
            df[TICKER_COL] = ticker
            bucket.append(df)

        log(f"Lengths -> train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}")

    if len(train_frames) == 0: raise RuntimeError("No training data assembled.")
    train_all = pd.concat(train_frames).reset_index(drop=True).sort_values([DATE_COL, TICKER_COL])
    val_all   = pd.concat(val_frames)  .reset_index(drop=True).sort_values([DATE_COL, TICKER_COL]) if val_frames else pd.DataFrame(columns=[DATE_COL, TICKER_COL] + keep_cols)
    test_all  = pd.concat(test_frames) .reset_index(drop=True).sort_values([DATE_COL, TICKER_COL]) if test_frames else pd.DataFrame(columns=[DATE_COL, TICKER_COL] + keep_cols)

    # Save scaled splits:
    # - train/val are denoised then scaled
    # - test is RAW (not denoised) but SCALED using the train scaler
    train_all.to_csv("scaled_train_businessB_denoised.csv", index=False)
    val_all.to_csv("scaled_val_businessB_denoised.csv", index=False)
    test_all.to_csv("scaled_test_businessB.csv", index=False)

    log(f"Saved splits: train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")

# ------------------------------------------------------------
# Optuna TPE search
# ------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    seq_len   = trial.suggest_categorical("seq_len", SEQ_LEN_CHOICES)
    lr        = trial.suggest_categorical("lr", LR_CHOICES)
    batch_sz  = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    embed_dim = trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES)   # NEW

    arrs = make_arrays(seq_len)

    # DIAGNOSTIC (optional)
    log(f"DEBUG: Xtr.shape={arrs['Xtr'].shape}, Xva.shape={arrs['Xva'].shape}, Xte.shape={arrs['Xte'].shape}")

    if arrs['Xtr'].shape[0] == 0 or arrs['Xva'].shape[0] == 0:
        log("No training or validation windows available for this seq_len -> pruning trial.")
        raise optuna.TrialPruned()

    train_loader, val_loader, _ = make_loaders(arrs, batch_sz)

    # construct model with sampled embedding dimension
    model = xLSTM_TS(input_size=INPUT_SIZE, d_model=embed_dim, output_size=OUTPUT_SIZE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=False, min_lr=1e-8)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None; epochs_no_improve = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        va_loss = evaluate(model, val_loader, loss_fn)
        trial.report(va_loss, step=epoch)

        # step scheduler by validation metric (if you want scheduling)
        scheduler.step(va_loss)

        if trial.should_prune(): raise optuna.TrialPruned()

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
    return float(best_val)


def train_eval_best(study: optuna.Study) -> Dict[str, Any]:
    best_trial = study.best_trial
    hp = {
        "seq_len": best_trial.params["seq_len"],
        "lr": best_trial.params["lr"],
        "batch_size": best_trial.params["batch_size"],
        "embed_dim": best_trial.params.get("embed_dim", EMBED_DIM),  # NEW: pick best embed_dim (fallback to global)
        "optimizer": "Adam",
        "loss": "MSE",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "clip_max_norm": CLIP_MAX_NORM
    }

    arrs = make_arrays(hp["seq_len"])
    train_loader, val_loader, test_loader = make_loaders(arrs, hp["batch_size"])

    model = xLSTM_TS(input_size=INPUT_SIZE, d_model=hp["embed_dim"], output_size=OUTPUT_SIZE).to(DEVICE)
    best_state = best_trial.user_attrs.get("best_state", None)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    loss_fn = nn.MSELoss()

    val_mse  = evaluate(model, val_loader, loss_fn)
    test_mse = evaluate(model, test_loader, loss_fn)

    Xte, yte = arrs["Xte"], arrs["yte"]
    if Xte.shape[0] > 0: naive_pred = Xte[:, -1, 0]; baseline_mse = float(np.mean((naive_pred - yte) ** 2))
    else: baseline_mse = float("nan")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("artifacts", "xlstm_ts_best_state_dict_STABLE_PRICES.pt"))
    with open(os.path.join("artifacts", "xlstm_ts_best_hparams_STABLE_PRICES.json"), "w") as f:
        json.dump({"best_params": hp, "best_val_mse": study.best_value, "val_mse": val_mse, "test_mse": test_mse, "baseline_test_mse": baseline_mse}, f, indent=2)

    verdict = verdict_from_metrics(test_mse, baseline_mse)

    print("\n=== BEST TRIAL SUMMARY ===")
    print(f"Best params: {hp}")
    print(f"Best val MSE during tuning: {study.best_value:.6f}")
    print(f"Re-evaluated Val MSE: {val_mse:.6f}")
    print(f"Test MSE (best model): {test_mse:.6f}")
    print(f"Baseline (persistence) Test MSE: {baseline_mse:.6f}")
    print(f"Verdict: {verdict}")
    print("Saved:")
    print(" - artifacts/xlstm_ts_best_state_dict_STABLE_PRICES.pt")
    print(" - artifacts/xlstm_ts_best_hparams_STABLE_PRICES.json")

    return {"hp": hp, "val_mse": val_mse, "test_mse": test_mse, "baseline_mse": baseline_mse, "verdict": verdict}

def verdict_from_metrics(test_mse: float, baseline_mse: float) -> str:
    rmse = float(np.sqrt(test_mse)) if math.isfinite(test_mse) else float("inf")
    imp = (baseline_mse - test_mse) / baseline_mse if (math.isfinite(baseline_mse) and baseline_mse > 0) else 0.0
    if imp >= 0.25 and rmse <= 0.03: return "EXCELLENT: large improvement over naïve baseline with very low error."
    elif imp >= 0.10 and rmse <= 0.05: return "GOOD: clear improvement over baseline with low error."
    elif imp > 0.0 or rmse <= 0.07: return "FAIR: modest improvement or acceptable error; may need more tuning/data."
    else: return "POOR: does not beat baseline meaningfully; consider revising features/model/tuning."

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    preprocess_and_save()
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print("Starting TPE search over {seq_len, lr, batch_size} ...")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    _ = train_eval_best(study)

if __name__ == "__main__":
    main()
