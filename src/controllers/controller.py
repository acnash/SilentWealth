import os.path
import csv
from abc import ABC, abstractmethod
from datetime import datetime
import random
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, LimitOrder, StopLimitOrder
from matplotlib import pyplot as plt

from src.monitor_tools.RSI import RSI
from src.monitor_tools.exp_moving_average import ExpMovingAverage


class Controller(ABC):
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"

    def __init__(self):
        getcontext().prec = 12
        self.holding_stock = None
        self.commission_pot = None

    def _connect_to_ib(self, port):
        client_id = random.randint(1, 9999)
        ib = IB()
        ib.connect('127.0.0.1', port, clientId=client_id)

        return ib

    def _sell_market_order(self, ib, symbol, quantity, contract):
        if not self.holding_stock:
            return
        else:
            print(f"...Selling {quantity} of {symbol}")
            sell_order = MarketOrder(Controller.SELL, quantity)
            trade = ib.placeOrder(contract, sell_order)
            self.holding_stock = False

    def _sell_market_crypto_order(self, ib, contract, ticker_name):
        if self.holding_stock:
            try:
                positions = ib.positions()
                # Find the BTC position
                btc_position = next((pos for pos in positions if pos.contract.symbol == ticker_name), None)

                if btc_position:
                    btc_quantity = btc_position.position  # Quantity of BTC you hold
                    btc_quantity = Decimal(abs(btc_quantity)).quantize(Decimal("0.00000001"))
                    print(f"You have {btc_quantity} crypto.")
                else:
                    print("No crypto position found.")
                    return

                # Create a market order to sell all BTC
                if btc_position and btc_quantity > 0:
                    sell_order = MarketOrder(action=Controller.SELL, totalQuantity=float(btc_quantity))
                    sell_order.tif = "IOC"

                    # Place the order
                    trade = ib.placeOrder(contract, sell_order)
                    print(f"Sell order placed: {trade}")
                    self.holding_stock = False
                else:
                    print("No crypto to sell or invalid position.")
            except:
                print(f"***ERROR***: failed to sell crypto.")

    def _place_market_crypto_order(self, ib, contract, dollar_amount, ticker_name):
        if not self.holding_stock:
            try:
                #share_ticker = ib.reqMktData(contract, '', False, False)
                #ib.sleep(2)
                #btc_price = share_ticker.last if share_ticker.last > 0 else share_ticker.bid
                #btc_quantity = dollar_amount / btc_price
                #btc_quantity = round(btc_quantity, 8)

                order = MarketOrder(Controller.BUY, 0)
                order.cashQty = dollar_amount
                order.tif = "IOC"

                trade = ib.placeOrder(contract, order)
                print(f"Buy order placed: {trade}")
                self.holding_stock = True
            except:
                print(f"***ERROR***: failed to buy crypto at dollar_amount: {dollar_amount}")

    def _place_market_order(self, ib,
                            contract,
                            quantity,
                            symbol):
        if not self.holding_stock:

            buy_order = MarketOrder(Controller.BUY,
                                    quantity)  # 'BUY' indicates the action and 10 is the quantity of shares
            print(f"Placing market order of {quantity} of {symbol} shares.")
            trade = ib.placeOrder(contract, buy_order)

            # Wait for the order to fill
            while not trade.isDone():
                ib.waitOnUpdate()

            if trade.fills:
                self.holding_stock = True
                fill_price = trade.fills[0].execution.price
                print(f"Trade filled at {fill_price} price per share.")
            else:
                print("ERROR: trade not filled. Please check the Trader Workstation.")
                print(f"{trade}")
                return

    def _scheduled_task(self, ib,
                        contract,
                        ticker_name,
                        quantity,
                        frame_size,
                        unit_type,
                        dollar_amount,
                        commission_pot,
                        start_time,
                        stop_time,
                        ema_short,
                        ema_medium,
                        ema_long,
                        vwap,
                        rsi,
                        rsi_top,
                        rsi_bottom,
                        atr_period,
                        output_data,
                        test_mode,
                        test_data,
                        bootstrap):

        if not self.commission_pot:
            self.commission_pot = commission_pot

        current_time = datetime.now().time()

        start = datetime.strptime(start_time, "%H:%M").time() if start_time else None
        stop = datetime.strptime(stop_time, "%H:%M").time() if stop_time else None

        if test_mode:
            if bootstrap:
                bootstrap_total_profit = []
                bootstrap_total_commission = []
                bootstrap_net_profit = []
                bootstrap_running_amount = []
                bootstrap_samples = bootstrap.build_samples()

                for sample_df in bootstrap_samples:
                    total_profit, total_commission, running_amount = self._silent_wealth_start(None,
                                                                               None,
                                                                               frame_size,
                                                                               ticker_name,
                                                                               unit_type,
                                                                               ema_short,
                                                                               ema_medium,
                                                                               ema_long,
                                                                               vwap,
                                                                               rsi,
                                                                               rsi_top,
                                                                               rsi_bottom,
                                                                               atr_period,
                                                                               output_data,
                                                                               test_mode,
                                                                               sample_df)
                    print(f"===============RESULT===============")
                    #print(f"Gross profit: ${total_profit}")
                    print(f"Total commission: ${total_commission}")
                    #print(f"Net profit: ${total_profit - total_commission}")
                    print(f"Running amount: ${running_amount}")
                    bootstrap_total_profit.append(total_profit)
                    bootstrap_total_commission.append(total_commission)
                    bootstrap_net_profit.append(total_profit - total_commission)
                    bootstrap_running_amount.append(running_amount)

                # output the averages
                #mean_gross_profit = np.mean(bootstrap_total_profit)
                mean_commission = np.mean(bootstrap_total_commission)
                min_commission = np.min(bootstrap_total_commission)
                max_commission = np.max(bootstrap_total_commission)
                stdev_commission = np.std(bootstrap_total_commission)
                mean_running_amount = np.mean(bootstrap_running_amount)
                min_running_amount = np.min(bootstrap_running_amount)
                max_running_amount = np.max(bootstrap_running_amount)
                stdev_running_amount = np.std(bootstrap_running_amount)

                #mean_total_profit = np.mean(bootstrap_net_profit)

                if not os.path.exists("../temp/bs_results.txt"):
                    with open("../temp/bs_results.txt", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["ema_short", "ema_medium", "ema_long", "rsi", "rsi_top", "rsi_bottom", "atr",
                                         "mean_commission", "min_commission", "max_commission", "stdev_commission",
                                         "mean_running_amount", "min_running_amount", "max_running_amount", "stdev_running_amount"])
                        writer.writerow(
                            [ema_short, ema_medium, ema_long, rsi, rsi_top, rsi_bottom, atr_period,
                             mean_commission, min_commission, max_commission, stdev_commission,
                             mean_running_amount, min_running_amount, max_running_amount, stdev_running_amount])
                else:
                    with open("../temp/bs_results.txt", "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [ema_short, ema_medium, ema_long, rsi, rsi_top, rsi_bottom, atr_period,
                             mean_commission, min_commission, max_commission, stdev_commission,
                             mean_running_amount, min_running_amount, max_running_amount, stdev_running_amount])

                #lists = [bootstrap_total_profit, bootstrap_total_commission, bootstrap_net_profit]
                #plot_names = ["Gross Profit", "Total Commission", "Net Profit"]
                ## Create a 1x3 subplot
                #fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

                #for i, data in enumerate(lists):
                #    mean_val = np.mean(data)

                #    # Plot histogram
                #    axes[i].hist(data, bins=30, color='skyblue', edgecolor='black')
                #    axes[i].axvline(mean_val, color='red', linestyle='dashed', linewidth=2)

                #   # Add annotation
                #    axes[i].text(mean_val, max(axes[i].get_ylim()) * 0.9,
                #                 f'Mean: ${mean_val:.2f}', color='red', ha='center')

                #    # Titles and labels
                #    axes[i].set_title(f'{plot_names[i]}')
                #    axes[i].set_xlabel('Value $')
                #    if i == 0:
                #        axes[i].set_ylabel('Frequency')

                #plt.tight_layout()
                #plt.show()

            else:
                total_profit, total_commission, running_amount = self._silent_wealth_start(None,
                                                                           None,
                                                                           frame_size,
                                                                           unit_type,
                                                                           ticker_name,
                                                                           ema_short,
                                                                           ema_medium,
                                                                           ema_long,
                                                                           vwap,
                                                                           rsi,
                                                                           rsi_top,
                                                                           rsi_bottom,
                                                                           atr_period,
                                                                           output_data,
                                                                           test_mode,
                                                                           test_data)
                print(f"===============RESULT===============")
                print(f"Gross profit: ${total_profit}")
                print(f"Total commission: ${total_commission}")
                print(f"Net profit: ${total_profit - total_commission}")
                print(f"Running amount: ${running_amount}")
        else:
            if (start and stop and start <= current_time <= stop) or (not start and not stop):
                action = self._silent_wealth_start(ib=ib,
                                                   contract=contract,
                                                   frame_size=frame_size,
                                                   unit_type=unit_type,
                                                   ticker_name=ticker_name,
                                                   ema_short=ema_short,
                                                   ema_medium=ema_medium,
                                                   ema_long=ema_long,
                                                   vwap=vwap,
                                                   rsi=rsi,
                                                   rsi_top=rsi_top,
                                                   rsi_bottom=rsi_bottom,
                                                   atr_period=atr_period,
                                                   output_data=output_data,
                                                   test_mode=test_mode,
                                                   test_data=None)

                if action == Controller.HOLD:
                    #print(f"...holding {ticker_name}.")
                    pass
                elif action == Controller.SELL:
                    if self.holding_stock:
                        print(f"...selling {ticker_name}.")
                        self.commission_pot = self.commission_pot - 3.4
                        if self.commission_pot <= 0:
                            print("Ran out of commission. Finishing.")
                            exit()

                        if ticker_name == "BTC" or ticker_name == "SOL" or ticker_name == "ETH":
                            self._sell_market_crypto_order(ib, contract, ticker_name)
                        else:
                            self._sell_market_order(ib, ticker_name, quantity)
                        self.holding_stock = False
                elif action == Controller.BUY:
                    if self.holding_stock:
                        #print("...holding. Holding stock/crypto.")
                        pass
                    else:
                        print(f"...buying {ticker_name}.")
                        if ticker_name == "BTC" or ticker_name == "SOL" or ticker_name == "ETH":
                            self._place_market_crypto_order(ib, contract, dollar_amount, ticker_name)
                        else:
                            self._place_market_order(ib, contract, quantity, ticker_name)
                        self.holding_stock = True
            else:
                # outside of trading hours for regular stock
                close_down_trades = False
                if datetime.strptime(stop_time, "%H:%M").time() < current_time <= datetime.strptime(close_time,
                                                                                                    "%H:%M").time():
                    close_down_trades = True

                if close_down_trades:
                    positions = ib.positions()
                    position = next((p for p in positions if p.contract.symbol == ticker_name), None)
                    if position:
                        quantity_to_sell = position.position
                        print(f"Market nearing a close. Selling outstanding {quantity_to_sell} in {ticker_name}")
                        # Create a market order to sell all shares
                        sell_order = MarketOrder(Controller.SELL, quantity_to_sell)
                        # Place the order to sell the shares
                        ib.placeOrder(contract, sell_order)
                        exit()

    def __silent_wealth_start_test(self, ib,
                                   contract,
                                   frame_size,
                                   unit_type,
                                   ticker_name,
                                   ema_short,
                                   ema_medium,
                                   ema_long,
                                   vwap,
                                   rsi,
                                   rsi_top,
                                   rsi_bottom,
                                   atr_period,
                                   output_data,
                                   test_data):
        test_hold_security = False
        if isinstance(test_data, pd.DataFrame):
            test_df = test_data
        else:
            test_df = pd.read_csv(test_data)
        ema = ExpMovingAverage(ib, contract, frame_size, unit_type, ticker_name, output_data)
        df = ema.calculate_exp_moving_average([ema_short, ema_medium, ema_long], test_df)

        # calculate the Average True Range
        if atr_period > 0:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - prev_close).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=atr_period, adjust=False).mean()
            df[f"ATR_{atr_period}"] = atr

        if rsi > 0:
            rsi_obj = RSI(df)
            df = rsi_obj.calculate_rsi(rsi)

        df = df.dropna()

        # loop over the data frame for close information and logic
        # action logic to buy etc is in here and not returned to the calling function
        running_amount = 3000
        total_commission = 0
        bought_btc = 0
        total_profit = 0
        earlier_atr = 0  # always fudge the first ATR value
        for _, row in df.iterrows():
            close = row["close"]
            ema_short_value = row[f"{ema_short}_day_EMA"]
            ema_medium_value = row[f"{ema_medium}_day_EMA"]
            ema_long_value = row[f"{ema_long}_day_EMA"]
            if rsi > 0:
                rsi_value = row["rsi"]
            else:
                rsi_value = 0

            if atr_period > 0:
                if row[f"ATR_{atr_period}"] > earlier_atr:
                    atr_value = True
                else:
                    atr_value = False
                earlier_atr = row[f"ATR_{atr_period}"]
            else:
                atr_value = False

            if ema_short_value <= ema_medium_value:
                # print(
                #    f"......{ema_short_value} <= {ema_medium_value} -- ema_short_value <= ema_medium_value --> SELL")
                action = Controller.SELL
            elif close < ema_medium:
                # print(f"......{close} < {ema_medium_value} -- close < ema_medium_value --> SELL")
                action = Controller.SELL
            elif ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value:
                if rsi_value > 0:
                    if rsi_bottom < rsi_value <= rsi_top:
                        # print(
                        #    f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top --> BUY")
                        action = Controller.BUY
                    else:
                        # print(
                        #    f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top --> HOLD")
                        action = Controller.HOLD
                else:  # when no RSI involved
                    # print(
                    #    f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> BUY")
                    action = Controller.BUY
            else:
                # print(
                #    f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> HOLD")
                action = Controller.HOLD

            if action == Controller.HOLD:
                # print("Holding")
                pass
            elif action == Controller.BUY:
                if not test_hold_security:
                    # print("Buying")
                    bought_btc = running_amount / close
                    #running_amount = running_amount - bought_btc
                    running_amount = 0 # I put everything in
                    test_hold_security = True
                else:
                    pass
                    # print("Holding (Buy)")
            elif action == Controller.SELL:
                if test_hold_security:
                    # print("Selling")
                    total_commission = total_commission + 3.4
                    sell_price = bought_btc * close
                    #total_profit = total_profit + (sell_price - running_amount)
                    running_amount = sell_price - 3.4
                    test_hold_security = False
                else:
                    pass
                    # print("Holding (Sell)")

        # at the end of the test day sell if still holding
        if test_hold_security:
            # print("Selling")
            total_commission = total_commission + 3.4
            sell_price = bought_btc * row["close"]
            #total_profit = total_profit + (sell_price - 3000)
            running_amount = running_amount + sell_price - 3.4

        return total_profit, total_commission, running_amount

    def __silent_wealth_start_live(self, ib,
                                   contract,
                                   frame_size,
                                   unit_type,
                                   ticker_name,
                                   output_data,
                                   ema_short,
                                   ema_medium,
                                   ema_long,
                                   atr_period,
                                   rsi,
                                   rsi_bottom,
                                   rsi_top):
        ema = ExpMovingAverage(ib, contract, frame_size, unit_type, ticker_name, output_data)
        df = ema.calculate_exp_moving_average([ema_short, ema_medium, ema_long])

        close = df["close"].iloc[-1]
        ema_short_value = df[f"{ema_short}_day_EMA"].iloc[-1]
        ema_medium_value = df[f"{ema_medium}_day_EMA"].iloc[-1]
        ema_long_value = df[f"{ema_long}_day_EMA"].iloc[-1]

        # calculate the Average True Range
        if atr_period > 0:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - prev_close).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=atr_period, adjust=False).mean()
            df[f"ATR_{atr_period}"] = atr
            if df[f"ATR_{atr_period}"].iloc[-1] > df[f"ATR_{atr_period}"].iloc[-2]:
                atr_value = True
            else:
                atr_value = False
        else:
            # always set True so this executes
            atr_value = True

        if rsi > 0:
            rsi_obj = RSI(df)
            df = rsi_obj.calculate_rsi(rsi)
            rsi_value = df["rsi"].iloc[-1]
        else:
            rsi_value = 0

        if ema_short_value <= ema_medium_value:
            print(f"......{ema_short_value} <= {ema_medium_value} -- ema_short_value <= ema_medium_value --> SELL")
            return Controller.SELL
        elif close < ema_medium:
            print(f"......{close} < {ema_medium_value} -- close < ema_medium_value --> SELL")
            return Controller.SELL
        elif ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value:
            if rsi_value > 0:
                if rsi_bottom < rsi_value <= rsi_top:
                    print(
                        f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top & atr_value == True --> BUY")
                    return Controller.BUY
                else:
                    print(
                        f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top & atr_value == True --> HOLD")
                    return Controller.HOLD
            else:  # when no RSI involved
                print(
                    f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value  --> BUY")
                return Controller.BUY
        else:
            print(
                f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> HOLD")
            return Controller.HOLD

    def _silent_wealth_start(self, ib,
                             contract,
                             frame_size,
                             unit_type,
                             ticker_name,
                             ema_short,
                             ema_medium,
                             ema_long,
                             vwap,
                             rsi,
                             rsi_top,
                             rsi_bottom,
                             atr_period,
                             output_data,
                             test_mode,
                             test_data):

        if test_mode:
            # This is for test only.....
            return self.__silent_wealth_start_test(ib, contract, frame_size, ticker_name, unit_type, ema_short,
                                            ema_medium, ema_long, vwap, rsi, rsi_top, rsi_bottom, atr_period,
                                            output_data, test_data)
        # when trading with a live (paper or otherwise) account i.e., none-test
        else:
            self.__silent_wealth_start_live(ib, contract, frame_size, ticker_name, unit_type, output_data,
                                                   ema_short, ema_medium, ema_long, atr_period, rsi, rsi_bottom,
                                                   rsi_top)

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def run(self):
        pass
