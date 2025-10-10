# xLSTM-TS Training Pipeline — Call Flow & Function Glossary

This project trains an extended LSTM time-series model (xLSTM-TS) to predict the next-day (H=1) close price from sliding windows of daily stock data. It supports single-file or batch (directory) runs, leak-safe preprocessing, Optuna TPE tuning, and per-run artifacts.

## End-to-end flow (one training run)

1. **main()**
   - Discovers input files (e.g., `*_PRICES.txt`) and decides single-file vs. batch mode.
   - Calls `run_pipeline_for_file(...)` per chosen file.

2. **run_pipeline_for_file(input_path)**
   - Sets run ID and artifact directory; seeds RNGs via `set_seed(...)`.
   - Calls `preprocess_and_save(...)` to create time-ordered splits with leak-safe denoising and per-ticker scaling.
   - Creates an Optuna study and runs `study.optimize(objective, n_trials=...)`.
   - Calls `train_eval_best(study)` to finalize metrics, predictions, and plots.

3. **preprocess_and_save(input_path, artifact_dir)**
   - Reads table: `read_any_table(...)`.
   - Cleans/interpolates and aligns to business days: `ensure_business_days(...)`.
   - Splits by time using `split_indices(...)` (train → val → test).
   - Denoises train/val only: `wavelet_denoise_df(...)`.
   - Converts to Darts: `darts_series_from_df(...)`.
   - Fits scaler on **train only** and transforms val/test: `Scaler()`.
   - Back to DataFrame: `ts_to_df(...)`.
   - Saves split CSVs and reports.

4. **objective(trial)** (Optuna, per trial)
   - Samples hyperparameters (seq_len, lr, batch_size, embed_dim, etc.).
   - Builds arrays: `make_arrays(seq_len)` → uses `load_scaled_split_df(...)`, `build_windows_from_multiticker(...)`, `build_windows_target_only(...)`.
   - Builds loaders: `make_loaders(...)`.
   - Instantiates model: `xLSTM_TS(...)`.
   - Trains/validates: `train_one_epoch(...)`, `evaluate_losses(...)` with early stopping.
   - Stores best weights and learning curve in `trial.user_attrs`.
   - Returns best validation loss.

5. **train_eval_best(study)**
   - Exports trials: `export_study(...)`. Plots curve: `plot_learning_curve(...)`.
   - Rebuilds arrays/loaders; loads best weights.
   - Computes val/test losses: `evaluate_losses(...)`.
   - Generates predictions and metrics: `predict_and_report(...)` (uses `predict_on_loader(...)` and `metrics_dict(...)`).
   - Plots parity and residual histograms: `plot_parity(...)`, `plot_residual_hist(...)`.
   - Saves model state dict, hyperparams JSON, and prints verdict via `verdict_from_metrics(...)`.

## Windowing and split semantics

- **Per-ticker windows**: `build_windows_from_multiticker(...)` builds windows within each ticker.  
- **No timeline mixing**: Each input window (e.g., 100 days) is chronological and from a single ticker.  
- **Time-ordered splits**: `split_indices(...)` makes contiguous train → val → test slices.

## Function & class one-liners

### Utilities & I/O
- `log(msg)`: Conditional print when `VERBOSE=True`.
- `set_seed(seed)`: Seed NumPy and Torch (incl. CUDA).
- `ensure_dir(path)`: Create directory if needed.
- `read_any_table(path, date_col)`: Read CSV/TSV/TXT, parse datetime index, sort.
- `ensure_business_days(df)`: Reindex to business days and fill (`"time"` interpolation + ffill/bfill by default).
- `split_indices(n, train_frac, val_frac)`: Compute train/val cut points (test is the remainder).
- `minmax_stats(df, col)`: Min/max for a column (scaling report).

### Denoising
- `wavelet_denoise_1d(x, wavelet, level, mode)`: Wavelet threshold denoise a 1D array.
- `wavelet_denoise_df(df, cols)`: Apply denoise to selected columns.

### Darts conversions
- `darts_series_from_df(df, cols)`: Build `TimeSeries` at business-day freq.
- `ts_to_df(ts)`: Convert `TimeSeries` back to flat DataFrame.

### Windowing
- `build_windows_target_only(series, window, horizon)`: Make `(N, window, 1)` inputs and `(N,)` targets from one series.
- `build_windows_from_multiticker(df, window, horizon)`: Per-ticker windows, concatenated; also returns per-window tickers.

### Datasets/Loaders
- `Window1DDataset(X, y)`: Dataset for `(N, T, 1)` → `(N, 1)`.
- `make_loaders(arrs, batch)`: Train/val/test `DataLoader`s.
- `load_scaled_split_df(path)`: Read split CSV to `(Date-indexed)` DataFrame with `_
