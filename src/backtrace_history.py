import pandas as pd


def run_bencharm(file_path, short, medium, rsi_duration, rsi_top, rsi_bottom, use_vwap, investment):
    # 1. Load both files
    df = pd.read_csv(file_path)  # Replace with actual path

    df[f"{short}_day_EMA"] = df['close'].ewm(span=short, adjust=False, min_periods=short).mean()
    df[f"{medium}_day_EMA"] = df['close'].ewm(span=medium, adjust=False, min_periods=medium).mean()
    #df[f"{long}_day_EMA"] = df['close'].ewm(span=long, adjust=False, min_periods=long).mean()

    # --- 1. Calculate RSI over 14-day period using "close" prices ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=rsi_duration, min_periods=rsi_duration).mean()
    avg_loss = loss.rolling(window=rsi_duration, min_periods=rsi_duration).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi

    # --- 2. Start from first row where {long}_day_EMA > 0 and RSI is not NaN ---
    df = df[(df[f"{medium}_day_EMA"] > 0) & (df["RSI"].notna())].reset_index(drop=True)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap_numerator = (typical_price * df["volume"]).cumsum()
    vwap_denominator = df["volume"].cumsum()
    df["VWAP"] = vwap_numerator / vwap_denominator

    # --- 3. Initialize ---
    seen = False
    btc_bought = 0.0
    close_profit_list = []
    commission = 0.0
    initial_investment = investment

    # --- 4. Loop with investment logic ---
    for _, row in df.iterrows():
        ema_short = row[f"{short}_day_EMA"]
        ema_medium = row[f"{medium}_day_EMA"]
        #ema_long = row[f"{long}_day_EMA"]
        rsi_val = row["RSI"]
        close = row["close"]

        if use_vwap:
            vwap = row["VWAP"]
        else:
            vwap = 0

        if ema_short > ema_medium and not seen and rsi_bottom < rsi_val < rsi_top and close > vwap:
            btc_bought = initial_investment / close
            #print(f"BUY @ {close:.2f} → BTC bought: {btc_bought:.6f}")
            seen = True
            commission += 1.80

        elif ema_short <= ema_medium and seen and rsi_val > rsi_top:
            cash_out = btc_bought * close
            profit = cash_out - initial_investment
            # adjust the initial investment after each sale (include the buy and sell commission)
            initial_investment = initial_investment + profit + 3.40
            #print(f"SELL @ {close:.2f} → Cash: {cash_out:.2f}, PROFIT: {profit:.2f}")
            close_profit_list.append(profit)
            seen = False
            commission += 1.80



    # --- 5. Final Report ---
    total_profit = sum(close_profit_list)
    net_profit = total_profit - commission

    print("TOTAL PROFIT/LOSS:", total_profit)
    print("COMMISSION:", commission)
    print("NET PROFIT AFTER COMMISSION:", net_profit, "\n")

    return net_profit


short_list = [9, 12, 14, 18]
medium_list = [20, 30, 40, 50, 60]
long_list = [100, 150, 200]
rsi_list = [9, 12, 14, 18]
rsi_top_list = [70, 75, 80]
rsi_bottom_list = [40, 45, 50, 55]
vwap_list = [True, False]

best_profit = 0
best_short = 0
best_medium = 0
best_long = 0
best_rsi_duration = 0
best_rsi_top = 0
best_rsi_bottom = 0
best_vwap = None

for short_entry in short_list:
    for medium_entry in medium_list:
        for long_entry in long_list:
            for rsi_entry in rsi_list:
                for rsi_top_entry in rsi_top_list:
                    for rsi_bottom_entry in rsi_bottom_list:
                        for vwap_entry in vwap_list:
                            print("---------------------------------------------")
                            print(f"short: {short_entry}")
                            print(f"medium: {medium_entry}")
                            print(f"rsi_duration: {rsi_entry}")
                            print(f"rsi_top: {rsi_top_entry}")
                            print(f"rsi_bottom: {rsi_bottom_entry}")
                            print(f"vwap: {vwap_entry}")
                            profit = run_bencharm("../temp/history_frame_size_1.dat",
                                         short_entry,
                                         medium_entry,
                                         long_entry,
                                         rsi_entry,
                                         rsi_top_entry,
                                         rsi_bottom_entry,
                                         vwap_entry,
                                         1000.0)
                            if profit > best_profit:
                                best_profit = profit
                                best_long = long_entry
                                best_medium = medium_entry
                                best_short = short_entry
                                best_rsi_duration = rsi_entry
                                best_rsi_top = rsi_top_entry
                                best_rsi_bottom = rsi_bottom_entry
                                best_vwap = vwap_entry

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
