# xLSTM-Timeseries

See src/xLSTM for code. Calculates the next-day-ahead price for an asset. Use in conjunction with strong fundamentals and technical analysis to judge when to trade.

## Running order

- `multi_retrieve_ib_dataset.py` - builds historical one-day price datasets based on ticker names e.g. NVDA, GOOGL etc.
- `prep_train_xLSTM.py` - trains the xLSTM model using historical price data. Generates train, validate, and test data sets.
- `predict_xLSTM.py` - given the trained model and test data, retrieves the nest day price for each company.

## xLSTM-TS model methodology

# Methods

* **Data source & schema**

  * Input file: `STABLE_PRICES.txt` (multi-ticker table with columns: `Date`, `Ticker`, OHLCV; target is `Close`).
  * Flexible CSV/TXT parsing with automatic delimiter detection; `Date` coerced to `datetime` and sorted.

* **Business-day alignment & cleaning**

  * Per-ticker time series aligned to a business-day index.
  * Missing values handled by time interpolation → forward-fill → back-fill.
  * All infinities and all-NaN rows removed before alignment.

* **Leak-safe split & denoising**

  * Temporal split per ticker: **Train 86%**, **Validation 7%**, **Test (remainder)**.
  * **Wavelet denoising** (Daubechies-4, soft threshold) applied **only to train & validation** segments; **test remains raw**.
  * Denoising is split-local to prevent look-ahead leakage.

* **Scaling (per ticker)**

  * Darts `Scaler` Min–Max ([0,1]) **fit on denoised train only**.
  * The same scaler is applied to **denoised validation** and **raw test** segments.
  * Per-ticker train-denoised **min/max for `Close`** recorded in `artifacts/scaling_params.csv`.

* **Windowing (supervised samples)**

  * Univariate modeling of `Close` only: inputs (X) are shape ((N, \text{seq_len}, 1)).
  * Horizon **H=1** (next-day close).
  * Windows built per ticker within each split; counts reported in `artifacts/window_counts_seqLEN.csv`.

* **Model: xLSTM-TS block**

  * Input linear embedding → **mLSTM block** → **sLSTM block** → **mLSTM block** → layer norm + linear head.
  * Each block stacks: LayerNorm → causal depthwise Conv1d (kernels: m=4, s=2) → Multi-Head Attention (2 heads) → LSTM → projection/FF sublayer → LayerNorm with residuals.
  * Tunable embedding dimension; default shown as 64.

* **Training objective & optimization**

  * Loss: **MSE** (regression on scaled target).
  * Optimizer: **Adam**; learning rate scheduled by **ReduceLROnPlateau** (factor 0.5, patience 10, floor (1e{-8})).
  * **Gradient clipping**: max-norm 1.0.
  * **Early stopping** via validation loss with **patience 30**, up to **200 epochs**.
  * Determinism: global seed (42); GPU used if available.

* **Hyperparameter search (Optuna TPE)**

  * Search space:

    * `seq_len` ∈ {60, 100, 150, 200, 256}
    * `batch_size` ∈ {8, 16, 32, 64}
    * `embed_dim` ∈ {32, 64, 128, 256, 384}
    * `lr` ~ log-uniform in ([1e{-5}, 3e{-3}])
  * Sampler: **TPESampler** (multivariate, seeded).
  * Best trial’s state dict and learning curve saved; full trials exported to `artifacts/study_trials.csv`.

* **Evaluation & baselines**

  * Splits evaluated with DataLoaders (no shuffling except train).
  * **Naïve persistence baseline**: ( \hat{y}*t = x*{t-1} ) (last observed close).
  * Metrics (overall + per ticker): **MSE, RMSE, MAE, MAPE, sMAPE, (R^2)**.
  * **Diebold–Mariano test** (squared-error loss, (h{=}1)) with Newey–West variance for model vs naïve.
  * Predictions & errors exported for **validation** and **test**:

    * `artifacts/predictions_val.csv`, `artifacts/predictions_test.csv`
    * `artifacts/metrics_val_overall.json`, `artifacts/metrics_test_overall.json`
    * `artifacts/metrics_val_by_ticker.csv`, `artifacts/metrics_test_by_ticker.csv`

* **Artifacts & reproducibility**

  * Preprocessing report: `artifacts/data_config.json`, `artifacts/split_report.csv`, `artifacts/scaling_params.csv`.
  * Scaled splits: `scaled_train_businessB_denoised.csv`, `scaled_val_businessB_denoised.csv`, `scaled_test_businessB.csv`.
  * Best model: `artifacts/xlstm_ts_best_state_dict_STABLE_PRICES.pt` and `artifacts/xlstm_ts_best_hparams_STABLE_PRICES.json`.
  * Learning curve for best trial: `artifacts/learning_curve_best.csv`.

* **Runtime details**

  * Batch loading via PyTorch `DataLoader` (train shuffled; val/test not shuffled).
  * Device-agnostic execution (CPU/GPU).
  * Verbose logging toggle with `VERBOSE=True`.


# SilentWealth

Draft project. Day trades bitcoin using an Interactive Broker account and the Interactive Broker Gateway tool.

## Arguments
- `--ticker_name` the trade ticker name e.g. `BP.`, `LLOY`, `SOXS`, `SOXL`, `BTC`. Always required.
- `--exchange` the name of the exchange, which also dictates the trading hours. Options are, `LSE`, `ARCA`, and `NASDAQ` (not applicable for BTC).
- `--quantity` the number of stocks to trade (not applicable for BTC).
- `--frame_size` trade windows in minutes. Typically, set this to `1` or `5`. Always required.
- `--account` Set to `paper` or `live`. Always required.
- `--stop_loss` 0 to 1 as a fraction from the stock value e.g., 0.01 sets a -1% stop loss below the stock value. Set to 0 to switch off.
- `--take_profit` a take profit set from the buy position as a percentage of the investment. Default is 0.02 e.g., 2% of the investment value. Set to 0 to switch off.
- `--limit_order` When set the code tries to make a buy at the lower quartile of the bid-ask range using a limit order. At the moment this does not support stop loss or take profit (do not set either if you wish to place a Limit Order). Warning: If the code fails to fill the order at the requested price within 4 seconds, a market order is made..

### Adjusting BUY/SELL conditions:
- `--ema_short` short-range exponential moving average (default: 9). Optional.
- `--ema_medium` medium-range exponential moving average (default: 20). Optional.
- `--ema_long` long-range exponential moving average (default: 200). Set to 0 to exclude from BUY conditions.
- `--vwap` volume-weighted average price (default: 9). Set to 0 to exclude from BUY conditions.
- `--rsi_period` units to compute the RSI (default: 14). Set to 0 to exclude from BUY conditions.
- `--anchor_distance` Pins a buy order within a distance from the short crossing up and over the medium. The longer this value, the more likely the BUY signal will happen shortly before a SELL signal.

## Examples
Use the following minimal examples and options provided above to execute the software.

### London Stock Exchange
#### Attempting to make a limit order with default BUY conditions
`--ticker_name LLOY --exchange LSE --quantity 100 --frame_size 1 --account paper --limit_order`

#### Attempting to make a limit order with default BUY conditions and anchored within 4 events of the BUY signal
`--ticker_name LLOY --exchange LSE --quantity 100 --frame_size 1 --account paper --limit_order --anchor_distance 4`

#### Turing off RSI, VWAP, and long EMA BUY conditions
`--ticker_name LLOY --exchange LSE --quantity 100 --frame_size 1 --account paper --limit_order --vwap 0 --rsi_period 0 --ema_long 0`

### NASDAQ
`--ticker_name PLTR --exchange NASDAQ --quantity 100 --frame_size 1 --account paper --limit_order`

### Bitcoin
`--ticker_name BTC --frame_size 1 --account paper --stop_loss_percent 0.01 --dollar_amount 1000`

### ARCA ETF
`--ticker_name SOXS --exchange ARCA --quantity 100 --frame_size 1 --account paper --limit_order`

# Benchmarking

## BTC
### 1 min windows over 10 days
- Short: 12
- Medium: 30
- Long: 100
- RSI duration: 14
- RSI top: 80
- RSI bottom: 40
- VWAP: True
- ATR duration: 12
- Initial investment: $1000
- Commission pot: $200

- **TOTAL PROFIT/LOSS:** $
- **COMMISSION:** $
- **NET PROFIT AFTER COMMISSION:** $59.724179172215855


