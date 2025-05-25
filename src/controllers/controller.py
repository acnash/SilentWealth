from abc import ABC, abstractmethod
from datetime import datetime
import random
from decimal import Decimal, getcontext

import pandas as pd
from ib_insync import IB, MarketOrder, LimitOrder, StopLimitOrder

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
        if not self.holding_stock:
            return
        else:
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

    def _place_market_crypto_order(self, ib, contract, dollar_amount, ticker_name):
        if not self.holding_stock:
            share_ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(2)
            btc_price = share_ticker.last if share_ticker.last > 0 else share_ticker.bid
            btc_quantity = dollar_amount / btc_price
            btc_quantity = round(btc_quantity, 8)

            order = MarketOrder(Controller.BUY, 0)
            order.cashQty = dollar_amount
            order.tif = "IOC"
            #order = LimitOrder(
            #    action='BUY',  # 'BUY' to purchase BTC
            #    totalQuantity=btc_quantity,  # Specify the calculated BTC quantity
            #    lmtPrice=btc_price  # Set limit price to 0 (market-like)
            #)
            #order.tif = "IOC"

            trade = ib.placeOrder(contract, order)
            self.holding_stock = True


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
                        test_data):

        if not self.commission_pot:
            self.commission_pot = commission_pot

        current_time = datetime.now().time()

        start = datetime.strptime(start_time, "%H:%M").time() if start_time else None
        stop = datetime.strptime(stop_time, "%H:%M").time() if stop_time else None

        if test_mode:
            self._silent_wealth_start(None,
                                      None,
                                      frame_size,
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
        else:
            if (start and stop and start <= current_time <= stop) or (not start and not stop):
                action = self._silent_wealth_start(ib,
                                                   contract,
                                                   frame_size,
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

                if action == Controller.HOLD:
                    print(f"...holding {ticker_name}.")
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
                        print("...holding. Holding stock/crypto.")
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

    def _silent_wealth_start(self, ib,
                             contract,
                             frame_size,
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
            test_hold_security = False

            ema = ExpMovingAverage(None, None, None, None, None)
            df_test = pd.read_csv(test_data)
            df_ema_short = ema.calculate_exp_moving_average(ema_short, df_test)
            # calculate the Average True Range
            df_ema_short['previous_close'] = df_ema_short['close'].shift(1)
            df_ema_short['high_low'] = df_ema_short['high'] - df_ema_short['low']
            df_ema_short['high_pc'] = (df_ema_short['high'] - df_ema_short['previous_close']).abs()
            df_ema_short['low_pc'] = (df_ema_short['low'] - df_ema_short['previous_close']).abs()
            df_ema_short['tr'] = df_ema_short[['high_low', 'high_pc', 'low_pc']].max(axis=1)
            if atr_period > 0:
                df_ema_short['atr'] = df_ema_short['tr'].rolling(window=atr_period).mean()
                vol_threshold = df_ema_short['atr'].mean()  # or a quantile
                df_ema_short['volatile_enough'] = df_ema_short['atr'] > vol_threshold
            else:
                # always set True so this executes
                df_ema_short['volatile_enough'] = True
            atr_value = df_ema_short["volatile_enough"].iloc[-1]

            df_ema_medium = ema.calculate_exp_moving_average(ema_medium, df_ema_short)
            df_ema_short[f"{ema_medium}_day_EMA"] = df_ema_medium[f"{ema_medium}_day_EMA"]
            if rsi > 0:
                rsi_obj = RSI(df_ema_short)
                rsi_value = rsi_obj.calculate_rsi(rsi)
            else:
                rsi_value = 0

                #loop over the data frame for close information and logic
                #action logic to buy etc is in here and not returned to the calling function
            for _, row in df_ema_short.iterrows():
                close = row["close"]
                print(close)
                ema_short_value = row[f"{ema_short}_day_EMA"]
                ema_medium_value = row[f"{ema_medium}_day_EMA"]
                ema_long_value = row[f"{ema_long}_day_EMA"]

                #    if commission >= com_pot:
                #        print("Ran out of commission.")
                #        break

                if ema_short_value <= ema_medium_value:
                    print(
                        f"......{ema_short_value} <= {ema_medium_value} -- ema_short_value <= ema_medium_value --> SELL")
                    action = Controller.SELL

                if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value:
                    if rsi_value > 0:
                        if rsi_bottom < rsi_value <= rsi_top:
                            print(f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top --> BUY")
                            action = Controller.BUY
                        else:
                            print(f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top --> HOLD")
                            action = Controller.HOLD
                    else:
                        print(f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> BUY")
                        action = Controller.BUY
                else:
                    print(f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> HOLD")
                    action = Controller.HOLD

                if action == Controller.HOLD:
                    pass
                elif action == Controller.BUY:
                    if test_hold_security == False:
                        test_hold_security = True
                        #self._place_market_crypto_order()
                else:
                    if test_hold_security == True:
                        test_hold_security = False
                        #self._sell_market_crypto_order()

        else:
            ema = ExpMovingAverage(ib, contract, frame_size, ticker_name, output_data)
            df_ema_short = ema.calculate_exp_moving_average(ema_short)

            ema_short_value = df_ema_short[f"{ema_short}_day_EMA"].iloc[-1]

            # calculate the Average True Range
            df_ema_short['previous_close'] = df_ema_short['close'].shift(1)
            df_ema_short['high_low'] = df_ema_short['high'] - df_ema_short['low']
            df_ema_short['high_pc'] = (df_ema_short['high'] - df_ema_short['previous_close']).abs()
            df_ema_short['low_pc'] = (df_ema_short['low'] - df_ema_short['previous_close']).abs()
            df_ema_short['tr'] = df_ema_short[['high_low', 'high_pc', 'low_pc']].max(axis=1)

            if atr_period > 0:
                df_ema_short['atr'] = df_ema_short['tr'].rolling(window=atr_period).mean()
                vol_threshold = df_ema_short['atr'].mean()  # or a quantile
                df_ema_short['volatile_enough'] = df_ema_short['atr'] > vol_threshold
            else:
                # always set True so this executes
                df_ema_short['volatile_enough'] = True
            atr_value = df_ema_short["volatile_enough"].iloc[-1]

            df_ema_medium = ema.calculate_exp_moving_average(ema_medium)
            ema_medium_value = df_ema_medium[f"{ema_medium}_day_EMA"].iloc[-1]

            if ema_long > 0:
                df_ema_long = ema.calculate_exp_moving_average(ema_long)
                ema_long_value = df_ema_long[f"{ema_long}_day_EMA"].iloc[-1]
            else:
                ema_long_value = 0

            close = df_ema_short["close"].iloc[-1]
            #if vwap > 0:
            #    vwap_obj = VolumeWeightedAverage(df_ema_short)
            #    df_vwap = vwap_obj.calculate_wva(vwap)
            #    vwap_value = df_vwap[f"rolling_vwap"].iloc[-1]
            #    if close > vwap_value:
            #        print(f"......{close} > {vwap_value} -- close > vwap_value --> HOLD")
            #        return Controller.HOLD

            typical_price = (df_ema_short["high"] + df_ema_short["low"] + df_ema_short["close"]) / 3
            vwap_numerator = (typical_price * df_ema_short["volume"]).cumsum()
            vwap_denominator = df_ema_short["volume"].cumsum()
            df_ema_short["VWAP"] = vwap_numerator / vwap_denominator
            if vwap:
                vwap_value = df_ema_short["VWAP"].iloc[-1]
            else:
                vwap_value = 0

            if (close > vwap_value) and vwap:
                print(f"......{close} > {vwap_value} -- close > vwap_value --> HOLD")
                return Controller.HOLD

            if rsi > 0:
                rsi_obj = RSI(df_ema_short)
                rsi_value = rsi_obj.calculate_rsi(rsi)
            else:
                rsi_value = 0

            if ema_short_value <= ema_medium_value:
                print(f"......{ema_short_value} <= {ema_medium_value} -- ema_short_value <= ema_medium_value --> SELL")
                return Controller.SELL

            if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value:
                if rsi_value > 0:
                    if rsi_bottom < rsi_value <= rsi_top:
                        print(f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top --> BUY")
                        return Controller.BUY
                    else:
                        print(f"......{rsi_bottom} < {rsi_value} <= {rsi_top} -- rsi_bottom < rsi_value <= rsi_top --> HOLD")
                        return Controller.HOLD
                else:
                    print(f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> BUY")
                    return Controller.BUY
            else:
                print(
                    f"......{ema_short_value} > {ema_medium_value} and {ema_short_value} > {ema_long_value} and {atr_value} -- ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value --> HOLD")
                return Controller.HOLD


    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def run(self):
        pass
