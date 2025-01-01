# SilentWealth
- --
## Options
- `--ticker_name` the trade ticker name e.g. `BP.`, `LLOY`, `SOXS`, `SOXL`, `BTC`.
- `--exchange` the name of the exchange, which also dictates the trading hours. Options are, `LSE`, `ARCA`, and `NASDAQ` (not applicable for BTC).
- `--quantity` the number of stocks to trade (not applicable for BTC).
- `--frame_size` trade windows in minutes. Typically, set this to `1` or `5`.
- `--account` Set to `paper` or `live`. 
- `--stop_loss_percent` 0 to 1 as a fraction from the stock value e.g., 0.01 sets a -1% stop loss below the stock value. 

### Adjusting BUY/SELL conditions:

- `--ema_short` short-range exponential moving average (default: 9)
- `--ema_medium` medium-range exponential moving average (default: 20)
- `--ema_long` long-range exponential moving average (default: 200). Set to 0 to exclude from BUY conditions.
- `--vwap` volume-weighted average price (default: 9). Set to 0 to exclude from BUY conditions.
- `--rsi_period` units to compute the RSI. Default is 14. Set to 0 to turn off.
- `--take_profit` a take profit set from the buy position as a percentage of the investment. Set to 0 to switch off. Default is 0.02 e.g., 2% of the investment value.

## Examples

Use the following examples and options (provided above) to execute the software.

### London Stock Exchange
`--ticker_name LLOY --exchange LSE --quantity 100 --frame_size 1 --account paper --stop_loss_percent 0.01`

### NASDAQ
`--ticker_name PLTR --exchange NASDAQ --quantity 100 --frame_size 1 --account paper --stop_loss_percent 0.01`

### Bitcoin
`--ticker_name BTC --frame_size 1 --account paper --stop_loss_percent 0.01 --dollar_amount 1000`

### ARCA ETF
`--ticker_name SOXS --exchange ARCA --quantity 100 --frame_size 1 --account paper --stop_loss_percent 0.01`