from abc import ABC, abstractmethod
from datetime import datetime
import random

from ib_insync import IB, MarketOrder, LimitOrder, StopLimitOrder

from src.monitor_tools.RSI import RSI
from src.monitor_tools.exp_moving_average import ExpMovingAverage
from src.monitor_tools.volume_weighted_average import VolumeWeightedAverage


class Controller(ABC):

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"

    def __init__(self):
        self.holding_stock = None

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
                print(f"You have {btc_quantity} crypto.")
            else:
                print("No crypto position found.")
                return

            # Create a market order to sell all BTC
            if btc_position and btc_quantity > 0:
                sell_order = MarketOrder(
                    action=Controller.SELL,
                    totalQuantity=abs(btc_quantity)  # Ensure quantity is positive
                )
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

            order = MarketOrder(Controller.BUY, btc_quantity)

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
                        start_time,
                        stop_time,
                        close_time,
                        ema_short,
                        ema_medium,
                        ema_long,
                        vwap,
                        rsi,
                        rsi_top,
                        rsi_bottom,
                        atr_period,
                        output_data):

        current_time = datetime.now().time()

        start = datetime.strptime(start_time, "%H:%M").time() if start_time else None
        stop = datetime.strptime(stop_time, "%H:%M").time() if stop_time else None

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
                                               output_data)

            if action == Controller.HOLD:
                pass
            elif action == Controller.SELL:
                if ticker_name == "BTC" or ticker_name == "SOL" or ticker_name == "ETH":
                    self._sell_market_crypto_order(ib, contract, ticker_name)
                else:
                    self._sell_market_order(ib, ticker_name, quantity)
            elif action == Controller.BUY:
                if ticker_name == "BTC" or ticker_name == "SOL" or ticker_name == "ETH":
                    self._place_market_crypto_order(ib, contract, dollar_amount, ticker_name)
                else:
                    self._place_market_order(ib, contract, quantity, ticker_name)
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
                             output_data):

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
        if vwap > 0:
            vwap_obj = VolumeWeightedAverage(df_ema_short)
            df_vwap = vwap_obj.calculate_wva(vwap)
            vwap_value = df_vwap[f"rolling_vwap"].iloc[-1]
            if close > vwap_value:
                return Controller.HOLD

        if rsi > 0:
            rsi_obj = RSI(df_ema_short)
            rsi_value = rsi_obj.calculate_rsi(rsi)
        else:
            rsi_value = 0

        if ema_short_value <= ema_medium_value:
            return Controller.SELL

        if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value:
            if rsi_value > 0:
                if rsi_bottom < rsi_value <= rsi_top:
                    return Controller.BUY
                else:
                    return Controller.HOLD
            else:
                return Controller.BUY
        else:
            return Controller.HOLD

        # if vwap == 0 and ema_long > 0 and rsi > 0:
        #     if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and (rsi_bottom < rsi_value <= rsi_top) and atr_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # elif ema_long == 0 and vwap > 0 and rsi == 0 and atr_value:
        #     if ema_short_value > ema_medium_value and ema_short_value > vwap_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # elif ema_long > 0 and vwap == 0 and rsi == 0:
        #     if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and atr_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # elif ema_long > 0 and vwap > 0 and rsi == 0:
        #     if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and ema_short_value > vwap_value and atr_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # elif ema_long == 0 and vwap == 0 and rsi > 0:
        #     if ema_short_value > ema_medium_value and (rsi_bottom < rsi_value <= rsi_top)  and atr_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # elif ema_long == 0 and vwap == 0 and rsi == 0  and atr_value:  # only the short and medium EMA
        #     if ema_short_value > ema_medium_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # elif ema_long == 0 and vwap > 0 and rsi > 0:
        #     if ema_short_value > ema_medium_value and ema_short_value > vwap_value and (rsi_bottom < rsi_value <= rsi_top) and atr_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD
        #
        # else:
        #     if ema_short_value > ema_medium_value and ema_short_value > ema_long_value and ema_short_value > vwap_value and (
        #             rsi_value > rsi_bottom and rsi_value <= rsi_top) and atr_value:
        #         return Controller.BUY
        #     else:
        #         return Controller.HOLD

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def run(self):
        pass
