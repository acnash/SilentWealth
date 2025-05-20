# SilentWealth
- --

## To-do list
1. Move all input arguments into a YAML file format. 
2. Write Python code to pull down historical data. 
3. Run canalysis over historical data with a measure of loss/gain.
4. Write a volume indicator/trigger.
5. Write a stochastic oscillator. 
- --

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
- Short: 9
- Medium: 20
- Long: 100
- RSI duration: 18
- RSI top: 80
- RSI bottom: 40
- VWAP: True
- Initial investment: $1000
- Commission pot: $200

**TOTAL PROFIT/LOSS:** $87.36492801411373

**COMMISSION:** $7.768246400705687

**NET PROFIT AFTER COMMISSION:** $79.59668161340804 
