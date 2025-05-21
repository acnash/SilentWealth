import os

import pandas as pd
import csv

def run_bencharmk(file_path, short, medium, long, rsi_duration, rsi_top, rsi_bottom, use_vwap, atr_period, investment, com_pot, file):
    df = pd.read_csv(file_path)

    df[f"{short}_day_EMA"] = df['close'].ewm(span=short, adjust=False, min_periods=short).mean()
    df[f"{medium}_day_EMA"] = df['close'].ewm(span=medium, adjust=False, min_periods=medium).mean()
    df[f"{long}_day_EMA"] = df['close'].ewm(span=long, adjust=False, min_periods=long).mean()

    # Calculate the technical analysis indicators
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # calculate RSI
    avg_gain = gain.rolling(window=rsi_duration, min_periods=rsi_duration).mean()
    avg_loss = loss.rolling(window=rsi_duration, min_periods=rsi_duration).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    df = df[(df[f"{long}_day_EMA"] > 0) & (df["RSI"].notna())].reset_index(drop=True)

    # calculate VWAP
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap_numerator = (typical_price * df["volume"]).cumsum()
    vwap_denominator = df["volume"].cumsum()
    df["VWAP"] = vwap_numerator / vwap_denominator

    # calculate the Average True Range
    df['previous_close'] = df['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_pc'] = (df['high'] - df['previous_close']).abs()
    df['low_pc'] = (df['low'] - df['previous_close']).abs()
    df['tr'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)

    if atr_period:
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        vol_threshold = df['atr'].mean()  # or a quantile
        df['volatile_enough'] = df['atr'] > vol_threshold
    else:
        # always set True so this executes
        df['volatile_enough'] = True

    # Set up loop variables
    seen = False
    btc_bought = 0.0
    close_profit_list = []
    commission = 0.0
    initial_investment = investment
    first_trade = True

    # Loop through each row of the price data (stock/coin history)
    writer = csv.writer(file)
    for _, row in df.iterrows():
        if commission >= com_pot:
            print("Ran out of commission.")
            break

        ema_short = row[f"{short}_day_EMA"]
        ema_medium = row[f"{medium}_day_EMA"]
        ema_long = row[f"{long}_day_EMA"]
        rsi_val = row["RSI"]
        close = row["close"]
        volatile = row["volatile_enough"]

        if use_vwap:
            vwap = row["VWAP"]
        else:
            vwap = 0

        # this prevent RSI impacting decisions.
        if not rsi_duration:
            if ema_short > ema_medium and not seen and close > vwap and ema_short > ema_long and volatile:
                # avoid buying unless I've seen an initial Sale. This avoids buying halfway up a trend.
                if first_trade:
                    continue

                btc_bought = initial_investment / close
                seen = True
            elif ema_short <= ema_medium:
                first_trade = False
                if seen:
                    cash_out = btc_bought * close
                    profit = cash_out - initial_investment
                    # adjust the initial investment after each sale (include the buy and sell commission)
                    initial_investment = initial_investment + (profit * 0.95)

                    close_profit_list.append(profit)
                    # this prevents a sale unless a BUY action occurred (seen == True)
                    seen = False
                    commission += 3.40 + (profit * 0.05)
        else:
            if ema_short > ema_medium and not seen and rsi_bottom < rsi_val < rsi_top and close > vwap and ema_short > ema_long and volatile:
                # avoid buying unless I've seen an initial Sale. This avoids buying halfway up a trend.
                if first_trade:
                    continue

                btc_bought = initial_investment / close
                seen = True
            elif ema_short <= ema_medium and rsi_val > rsi_top:
                first_trade = False
                if seen:
                    cash_out = btc_bought * close
                    profit = cash_out - initial_investment
                    # adjust the initial investment after each sale (include the buy and sell commission)
                    initial_investment = initial_investment + (profit * 0.95)

                    close_profit_list.append(profit)
                    # this prevents a sale unless a BUY action occurred (seen == True)
                    seen = False
                    commission += 3.40 + (profit * 0.05)

    # Report the settings
    total_profit = sum(close_profit_list)
    net_profit = total_profit - commission

    print("TOTAL PROFIT/LOSS:", total_profit)
    print("COMMISSION:", commission)
    print("NET PROFIT AFTER COMMISSION:", net_profit, "\n")

    writer.writerow([short, medium, long, rsi_duration, rsi_top, rsi_bottom, use_vwap, atr_period, investment, com_pot,
                     total_profit, commission, net_profit])

    return net_profit


def run_screen(output_file: str):
    file = open(output_file, "a")

    write_header = not os.path.exists(output_file) or os.stat(output_file).st_size == 0
    if write_header:
        writer = csv.writer(file)
        writer.writerow(["short", "medium", "long", "rsi_duration", "rsi_top", "rsi_bottom", "use_vwap", "atr_period",
                         "investment", "com_pot", "total_profit", "commission", "net_profit"])

    short_list = [9, 12, 14, 18]
    medium_list = [20, 30, 40, 50]
    long_list = [100, 150, 200]
    rsi_list = [9, 12, 14, 18, None]
    rsi_top_list = [70, 75, 80]
    rsi_bottom_list = [40, 45, 50]
    atr_list = [9, 12, 14, 18, None]
    vwap_list = [True, False, None]

    best_profit = 0
    best_short = 0
    best_medium = 0
    best_long = 0
    best_rsi_duration = 0
    best_rsi_top = 0
    best_rsi_bottom = 0
    best_vwap = None
    best_atr_duration = 0

    for short_entry in short_list:
        for medium_entry in medium_list:
            for long_entry in long_list:
                for rsi_entry in rsi_list:
                    for rsi_top_entry in rsi_top_list:
                        for rsi_bottom_entry in rsi_bottom_list:
                            for atr_entry in atr_list:
                                for vwap_entry in vwap_list:
                                    print("---------------------------------------------")
                                    print(f"short: {short_entry}")
                                    print(f"medium: {medium_entry}")
                                    print(f"long: {long_entry}")
                                    print(f"rsi_duration: {rsi_entry}")
                                    print(f"rsi_top: {rsi_top_entry}")
                                    print(f"rsi_bottom: {rsi_bottom_entry}")
                                    print(f"vwap: {vwap_entry}")
                                    print(f"atr_duration: {atr_entry}")
                                    profit = run_bencharmk("../temp/history_frame_size_1.dat",
                                                          short_entry,
                                                          medium_entry,
                                                          long_entry,
                                                          rsi_entry,
                                                          rsi_top_entry,
                                                          rsi_bottom_entry,
                                                          vwap_entry,
                                                          atr_entry,
                                                          1000.0,
                                                          200.0,
                                                           file)
                                    if profit > best_profit:
                                        best_profit = profit
                                        best_long = long_entry
                                        best_medium = medium_entry
                                        best_short = short_entry
                                        best_rsi_duration = rsi_entry
                                        best_rsi_top = rsi_top_entry
                                        best_rsi_bottom = rsi_bottom_entry
                                        best_atr_duration = atr_entry
                                        best_vwap = vwap_entry

    file.close()

    print("===============================================")
    print("BEST PARAMETERS")
    print(f"Investment: {1000.0}")
    print(f"Net profit: {best_profit}")
    print(f"short: {best_short}")
    print(f"medium: {best_medium}")
    print(f"long: {best_long}")
    print(f"rsi_duration: {best_rsi_duration}")
    print(f"rsi_top: {best_rsi_top}")
    print(f"rsi_bottom: {best_rsi_bottom}")
    print(f"vwap: {best_vwap}")
    print(f"atr_duration: {best_atr_duration}")

run_screen("../temp/BTC_1_min.txt")
#profit = run_bencharmk("../temp/history_frame_size_1.dat",
#                      14,
#                      20,
#                      100,
#                      18,
#                      75,
#                      40,
#                      True,
#                      14,
#                      1000.0,
#                      200.0)
