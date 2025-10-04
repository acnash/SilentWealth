# ===========================================================
# train_xlstm_ts_from_scaled_csv.py
# Loads scaled CSV splits -> windows of len 150 (context=150)
# Trains an xLSTM-TS-style model per your configuration
# Dependencies: pandas numpy torch
#   pip install pandas numpy torch
# ===========================================================
# ===========================================================
# train_xlstmts_hparam_search.py
# Hyperparameter tuning with early stopping for xLSTM-TS
# Keeps: sequence length=150, model architecture, data loading from your CSVs
# Saves best model and reports best configuration and metrics
# Dependencies: numpy pandas torch
#   pip install numpy pandas torch
# ===========================================================

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------
# Fixed data + model config
# ----------------------
TRAIN_CSV = "scaled_train_businessB_denoised.csv"
VAL_CSV = "scaled_val_businessB_denoised.csv"
TEST_CSV = "scaled_test_businessB_denoised.csv"
TARGET_COL = "Adj Close"
INPUT_SIZE = 1
EMBED_DIM = 64
OUTPUT_SIZE = 1
SEQ_LEN = 150
CONTEXT_LEN = 150
WEIGHT_DECAY = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# xLSTM block configs (unchanged)
MLSTM_CONV_K = 4
MLSTM_PROJ_SIZE = 2
MLSTM_HEADS = 2
SLSTM_CONV_K = 2
SLSTM_HEADS = 2
SLSTM_FF_FACTOR = 1.1

# ----------------------
# Hyperparameter search space
# Adjust lists as desired
# ----------------------
SEARCH_SPACE = {
    "lr": [1e-4, 5e-4, 1e-3],
    "batch_size": [16, 32],
    "max_epochs": [100, 200],
    "patience": [20, 30],
    "clip_max_norm": [0.5, 1.0],
    "optimizer": ["Adam", "AdamW"],  # choose between Adam or AdamW
    "loss": ["MSE"],                 # fixed, but kept here for completeness
}

# --------------
# Utilities
# --------------
def set_seed(seed:int)->None:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_scaled_split(path:str, target_col:str)->pd.Series:
    assert os.path.exists(path), f"Missing file: {path}"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    assert target_col in df.columns, f"Target column '{target_col}' not found in {path} (have: {list(df.columns)})"
    s = df[target_col].astype("float32").copy()
    s = s.dropna()
    return s

def build_windows_1d(series:pd.Series, window:int, horizon:int=1)->Tuple[np.ndarray,np.ndarray]:
    x_list = []
    y_list = []
    vals = series.values
    n = len(vals)
    for t in range(window, n - horizon + 1):
        x_list.append(vals[t - window:t])
        y_list.append(vals[t + horizon - 1])
    X = np.stack(x_list, axis=0).astype("float32") if len(x_list) > 0 else np.zeros((0, window), dtype="float32")
    y = np.array(y_list, dtype="float32") if len(y_list) > 0 else np.zeros((0,), dtype="float32")
    X = X[..., None]
    return X, y

class Window1DDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray):
        assert X.ndim == 3 and X.shape[-1] == 1
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[-1] == 1)
        self.X = X.astype("float32")
        self.y = y[:, None].astype("float32") if y.ndim == 1 else y.astype("float32")
    def __len__(self)->int:
        return self.X.shape[0]
    def __getitem__(self, idx:int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class CausalConv1d(nn.Module):
    def __init__(self, channels:int, kernel_size:int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0, groups=channels, bias=True)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.transpose(1,2)
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = x.transpose(1,2)
        return x

class ProjectionBlock(nn.Module):
    def __init__(self, d:int, proj_size:int):
        super().__init__()
        hidden = max(1, d // proj_size)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, d:int, factor:float):
        super().__init__()
        hidden = max(1, int(math.ceil(d * factor)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.net(x)

class mLSTMBlock(nn.Module):
    def __init__(self, d_model:int, kernel:int, heads:int, proj_size:int):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.proj = ProjectionBlock(d_model, proj_size)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x:torch.Tensor)->torch.Tensor:
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
    def __init__(self, d_model:int, kernel:int, heads:int, ff_factor:float):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.ff = FeedForward(d_model, ff_factor)
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x:torch.Tensor)->torch.Tensor:
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
    def __init__(self, input_size:int, d_model:int, output_size:int, mlstm_k:int, mlstm_heads:int, mlstm_proj:int, slstm_k:int, slstm_heads:int, slstm_ff:float):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.block1 = mLSTMBlock(d_model=d_model, kernel=mlstm_k, heads=mlstm_heads, proj_size=mlstm_proj)
        self.block2 = sLSTMBlock(d_model=d_model, kernel=slstm_k, heads=slstm_heads, ff_factor=slstm_ff)
        self.block3 = mLSTMBlock(d_model=d_model, kernel=mlstm_k, heads=mlstm_heads, proj_size=mlstm_proj)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.embed(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x[:, -1, :])
        return x

def make_arrays(seq_len:int)->Dict[str, np.ndarray]:
    train_s = load_scaled_split(TRAIN_CSV, TARGET_COL)
    val_s = load_scaled_split(VAL_CSV, TARGET_COL)
    test_s = load_scaled_split(TEST_CSV, TARGET_COL)
    Xtr, ytr = build_windows_1d(train_s, window=seq_len, horizon=1)
    Xva, yva = build_windows_1d(val_s, window=seq_len, horizon=1)
    Xte, yte = build_windows_1d(test_s, window=seq_len, horizon=1)
    return {"Xtr":Xtr, "ytr":ytr, "Xva":Xva, "yva":yva, "Xte":Xte, "yte":yte}

def make_loaders_from_arrays(arrs:Dict[str,np.ndarray], batch_size:int)->Tuple[DataLoader,DataLoader,DataLoader]:
    train_ds = Window1DDataset(arrs["Xtr"], arrs["ytr"])
    val_ds = Window1DDataset(arrs["Xva"], arrs["yva"])
    test_ds = Window1DDataset(arrs["Xte"], arrs["yte"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def make_optimizer(name:str, params, lr:float, weight_decay:float):
    name = name.lower()
    if name == "adam": return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw": return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")

def make_loss(name:str):
    name = name.lower()
    if name in ["mse","mseLoss".lower()]: return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}")

def train_one_epoch(model:nn.Module, loader:DataLoader, optimizer:torch.optim.Optimizer, loss_fn:nn.Module, device:str, clip_max_norm:float)->Tuple[float,int]:
    model.train()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        if clip_max_norm is not None and clip_max_norm > 0.0: nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(1, count), count

@torch.no_grad()
def evaluate(model:nn.Module, loader:DataLoader, loss_fn:nn.Module, device:str)->Tuple[float,int]:
    model.eval()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(1, count), count

def run_trial(hp:Dict[str,Any], arrs:Dict[str,np.ndarray])->Dict[str,Any]:
    batch_size = hp["batch_size"]
    lr = hp["lr"]
    max_epochs = hp["max_epochs"]
    patience = hp["patience"]
    clip_max_norm = hp["clip_max_norm"]
    optimizer_name = hp["optimizer"]
    loss_name = hp["loss"]
    train_loader, val_loader, test_loader = make_loaders_from_arrays(arrs, batch_size)
    model = xLSTM_TS(input_size=INPUT_SIZE, d_model=EMBED_DIM, output_size=OUTPUT_SIZE, mlstm_k=MLSTM_CONV_K, mlstm_heads=MLSTM_HEADS, mlstm_proj=MLSTM_PROJ_SIZE, slstm_k=SLSTM_CONV_K, slstm_heads=SLSTM_HEADS, slstm_ff=SLSTM_FF_FACTOR).to(DEVICE)
    opt = make_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = make_loss(loss_name)
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        tr_loss, _ = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE, clip_max_norm)
        va_loss, _ = evaluate(model, val_loader, loss_fn, DEVICE)
        improved = va_loss < best_val - 1e-8
        if improved:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        history.append((epoch, float(tr_loss), float(va_loss)))
        print(f"[Trial lr={lr} bs={batch_size} maxep={max_epochs} pat={patience} clip={clip_max_norm} opt={optimizer_name}] Epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f} | {'*' if improved else ''}")
        if epochs_no_improve >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    test_loss, _ = evaluate(model, test_loader, loss_fn, DEVICE)
    return {"hp":hp, "val_loss":float(best_val), "test_loss":float(test_loss), "state":best_state, "history":history}

def product_dict(d:Dict[str,List[Any]])->List[Dict[str,Any]]:
    keys = list(d.keys())
    grids = []
    def rec(i:int, cur:Dict[str,Any]):
        if i == len(keys): grids.append(cur.copy()); return
        k = keys[i]
        for v in d[k]:
            cur[k] = v
            rec(i+1, cur)
    rec(0, {})
    return grids

def main()->None:
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Building arrays (once) ...")
    arrs = make_arrays(SEQ_LEN)
    trials = product_dict(SEARCH_SPACE)
    print(f"Total trials: {len(trials)}")
    best = None
    os.makedirs("artifacts", exist_ok=True)
    for i, hp in enumerate(trials, start=1):
        print(f"\n=== Trial {i}/{len(trials)} | hp={hp} ===")
        result = run_trial(hp, arrs)
        if best is None or result["val_loss"] < best["val_loss"]:
            best = result
            torch.save(best["state"], os.path.join("artifacts", "xlstm_ts_best_state_dict.pt"))
            with open(os.path.join("artifacts", "xlstm_ts_best_hparams.txt"), "w") as f:
                f.write(str(best["hp"]) + "\n")
                f.write(f"best_val_loss={best['val_loss']}\n")
                f.write(f"test_loss={best['test_loss']}\n")
        print(f"Trial done | val_loss={result['val_loss']:.6f} | test_loss={result['test_loss']:.6f}")
    if best is None:
        print("No successful trials.")
        return
    print("\n=== BEST TRIAL SUMMARY ===")
    print(f"Best hparams: {best['hp']}")
    print(f"Best val_loss: {best['val_loss']:.6f}")
    print(f"Test loss (best model): {best['test_loss']:.6f}")
    print("Saved:")
    print(" - artifacts/xlstm_ts_best_state_dict.pt")
    print(" - artifacts/xlstm_ts_best_hparams.txt")

if __name__ == "__main__":
    main()
