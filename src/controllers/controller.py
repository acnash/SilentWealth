from abc import ABC, abstractmethod
from datetime import datetime
from random import random

from ib_insync import IB, MarketOrder, LimitOrder, StopLimitOrder

from src.monitor_tools.RSI import RSI
from src.monitor_tools.exp_moving_average import ExpMovingAverage
from src.monitor_tools.volume_weighted_average import VolumeWeightedAverage


class Controller(ABC):
    PAPER_PORT = 7497
    LIVE_PORT = 7496

    LSE_START_TIME = "08:10"
    LSE_END_TIME = "16:00"
    LSE_CLOSE_TIME = "16:10"

    NYSE_START_TIME = "14:30"
    NYSE_END_TIME = "20:30"
    NYSE_CLOSE_TIME = "20:58"

    BTC_PAXOS_START_TIME = "08:10"
    BTC_PAXOS_END_TIME = "20:00"
    BTC_PAXOS_CLOSE_TIME = "20:30"

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"

    holding_stock = None
    global_buy_price = None
    global_ema20 = None
    global_first_adjustment = True
    global_previous_ema = None

    def __int__(self, silent_wealth_inputs):
        pass

    def _connect_to_ib(self, account):
        client_id = random.randint(1, 9999)
        ib = IB()
        if account == "paper":
            ib.connect('127.0.0.1', Controller.PAPER_PORT, clientId=client_id)
        elif account == "live":
            ib.connect('127.0.0.1', Controller.LIVE_PORT, clientId=client_id)

        return ib

    def _sell_market_order(self, ib_input, symbol, quantity_input):
        # sell here...
        global holding_stock

        if not holding_stock:
            return
        else:
            print(f"...Selling {quantity_input} of {symbol}")
            sell_order = MarketOrder(Controller.SELL, quantity_input)
            trade = ib_input.placeOrder(stock, sell_order)
            holding_stock = False

    def _sell_market_BTC_order(self, ib_input, btc_contract):
        global holding_stock
        if not holding_stock:
            return
        else:
            positions = ib_input.positions()
            # Find the BTC position
            btc_position = next((pos for pos in positions if pos.contract.symbol == 'BTC'), None)

            if btc_position:
                btc_quantity = btc_position.position  # Quantity of BTC you hold
                print(f"You have {btc_quantity} BTC.")
            else:
                print("No BTC position found.")
                return

            # btc_contract = Contract(
            #    secType='CRYPTO',
            #    conId=490404295,  # Ensure this is the correct conId for BTC
            #    symbol='BTC',
            #    exchange='PAXOS',
            #    currency='USD'
            # )

            # Create a market order to sell all BTC
            if btc_position and btc_quantity > 0:
                sell_order = MarketOrder(
                    action=Controller.SELL,
                    totalQuantity=abs(btc_quantity)  # Ensure quantity is positive
                )
                sell_order.tif = "IOC"

                # Place the order
                trade = ib.placeOrder(btc_contract, sell_order)
                print(f"Sell order placed: {trade}")
                holding_stock = False
            else:
                print("No BTC to sell or invalid position.")

    def _place_market_BTC_order(self, ib_input, stock_input, quantity_input, cash, take_profit_input):
        global holding_stock
        global global_buy_price
        global global_ema20
        global global_first_adjustment
        global global_previous_ema

        if not holding_stock:
            share_ticker = ib_input.reqMktData(stock_input, '', False, False)
            ib_input.sleep(2)
            btc_price = share_ticker.last if share_ticker.last > 0 else share_ticker.bid
            btc_quantity = cash / btc_price
            btc_quantity = round(btc_quantity, 8)

            parent_order = LimitOrder(
                action='BUY',  # 'BUY' to purchase BTC
                totalQuantity=btc_quantity,  # Specify the calculated BTC quantity
                lmtPrice=btc_price  # Set limit price to 0 (market-like)
            )
            parent_order.tif = 'IOC'
            parent_id = parent_order.orderId

            # Create the Take Profit Limit Order
            take_profit_order = LimitOrder(Controller.SELL, totalQuantity=parent_order.totalQuantity, lmtPrice=btc_price + 20000)
            take_profit_order.parentId = parent_id  # Link to parent order

            # Create the Stop Loss Order
            stop_loss_order = StopLimitOrder(action=Controller.SELL,
                                             totalQuantity=parent_order.totalQuantity,
                                             stopPrice=btc_price - 20000,
                                             lmtPrice=btc_price - 19999)
            stop_loss_order.parentId = parent_id  # Link to parent order

            trade = ib_input.placeOrder(stock_input, parent_order)
            print(f"...Placing limit order of {btc_quantity} of BTC at bid price of {btc_price} per share")

            while not trade.isDone():
                # ib_input.waitOnUpdate()
                ib_input.sleep(2)

            if trade.fills:
                fill_price = trade.fills[0].execution.price
                global_buy_price = fill_price
                print(f"Trade filled at {fill_price} price per share")

                # set a stop loss and a take profit
                tp_trade = ib_input.placeOrder(stock_input, take_profit_order)
                ib_input.sleep(2)
                # while not tp_trade.isDone():
                #    print("Waiting to place take profit order")
                sl_trade = ib_input.placeOrder(stock_input, stop_loss_order)
                ib_input.sleep(2)

                #    #stop_loss_offset = 1.000 - stop_loss_percent
                #    #stop_loss_price = round(fill_price * stop_loss_offset, 3)
                #    #print(f"Stop-loss price set at {stop_loss_price:.3f}")
                #    #stop_order = StopOrder('SELL', btc_quantity, stop_loss_price)
                holding_stock = True
            else:
                print("Trade did not fill.")
                holding_stock = False

        else:
            print("Holding and looking to adjust the stop loss.")
            # if global_ema20 > global_buy_price and global_first_adjustment:
            #    print(
            #        f"EMA20 {global_ema20} > original buy price. "
            #        f"Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
            #    stop_order = StopOrder('SELL', quantity_input, global_ema20)
            #    trade = ib_input.placeOrder(stock_input, stop_order)
            #    global_previous_ema = global_ema20
            #    global_first_adjustment = False
            # elif global_ema20 > global_previous_ema and not global_first_adjustment:
            #    print(
            #        f"EMA20 {global_ema20} > previous EMA20 {global_previous_ema}. "
            #        f"Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
            #    stop_order = StopOrder('SELL', quantity_input, global_ema20)
            #    trade = ib_input.placeOrder(stock_input, stop_order)
            #    global_previous_ema = global_ema20

    def _place_market_order(self, ib_input,
                           stock_input,
                           quantity_input,
                           symbol,
                           stop_loss_input,
                           take_profit_input,
                           limit_order_input):
        # buy here...
        global holding_stock
        global global_buy_price
        global global_ema20
        global global_first_adjustment
        global global_previous_ema

        if holding_stock:
            print("Holding stock...")
            if global_ema20 > global_buy_price and global_first_adjustment:
                print(f"EMA20 {global_ema20} > original buy price. Adjusting stop loss to {global_ema20} "
                      f"to track the curve and avoid losses.")
                # stop_order = StopOrder('SELL', quantity_input, global_ema20)
                # trade = ib_input.placeOrder(stock_input, stop_order)
                global_previous_ema = global_ema20
                global_first_adjustment = False
            elif not global_first_adjustment and global_ema20 > global_previous_ema:
                print(f"EMA20 {global_ema20} > previous EMA20 {global_previous_ema}. "
                      f"Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
                # stop_order = StopOrder('SELL', quantity_input, global_ema20)
                # trade = ib_input.placeOrder(stock_input, stop_order)
                global_previous_ema = global_ema20
        else:
            # place a bracket order if a stop loss and take profit exists
            # ticker = ib_input.reqMktData(stock_input)
            # while not ticker.bid or not ticker.ask:
            #    ib.sleep(1)

            # Set the buy price as the bid (cheapest available price)
            # bid = ticker.bid
            # ask = ticker.ask
            # last = ticker.last
            # lower_quartile_price = bid + 0.25 * (ask - bid)
            if stop_loss_input:
                # Create the market order (parent)
                parent_order = MarketOrder(Controller.BUY, quantity_input)
                # parent_order.ocaGroup = f"oca_group_{parent_order.orderId}"

                # Place the parent market order
                parent_trade = ib_input.placeOrder(stock_input, parent_order)

                # Monitor the parent order's status through the Trade object
                while not parent_trade.isDone():
                    ib_input.waitOnUpdate()

                for fill in parent_trade.fills:
                    filled_price = fill.execution.price  # Corrected to access the price from the Fill object
                    print(f"Parent order filled at price: {filled_price} for order: {parent_order.orderId}")
                    holding_stock = True
                    global_buy_price = filled_price

                # take_order_price = filled_price * (1 + take_profit_input)
                # Create the take profit order (limit order)
                # take_profit_order = LimitOrder('SELL', quantity, take_order_price, parentId=parent_order.orderId)
                # take_profit_order.ocaGroup = f"oca_group_{parent_order.orderId}"

                # stop_loss_price = filled_price * 0.95 #* (1 - stop_loss_input)
                # stop_loss_price = ticker_name * 0.95  # * (1 - stop_loss_input)
                # Create the stop loss order (stop order)

                ticker = ib_input.reqMktData(stock_input)
                while not ticker.bid or not ticker.ask:
                    ib_input.sleep(5)

                # Set the buy price as the bid (cheapest available price)
                bid = ticker.bid
                ask = ticker.ask
                last = ticker.last

                print(f"Setting a stop loss of 0.95 from {last} to {round(last - 20, 4)}")
                # stop_loss_order = StopOrder('SELL', quantity, stopPrice=round(stop_loss_price-4,4), parentId=parent_order.orderId)
                stop_loss_order = LimitOrder(Controller.SELL, quantity_input, 300)
                # stop_loss_order.ocaGroup = f"oca_group_{parent_order.orderId}"
                # if parent_trade.orderStatus.status == 'Filled':
                # Parent order is filled, now place the child orders (take profit and stop loss)
                # ib_input.placeOrder(stock_input, take_profit_order)
                # print(f"Take profit at {take_order_price}")
                stop_order_trade = ib_input.placeOrder(stock_input, stop_loss_order)
                ib_input.sleep()
                print(stop_loss_order.orderRef)
                print(
                    f"Stop loss at {round(last, 4)} for order: {stop_loss_order.orderId} with parent id: {stop_loss_order.parentId}")
                # print("Child orders placed.")
                # else:
                #    print("Parent order was not filled.")

            else:  # a market order or limit order without stop loss or take profit
                if limit_order_input:
                    ticker_limit = ib_input.reqMktData(stock_input)
                    bid = ticker_limit.bid
                    ask = ticker_limit.ask
                    lower_quartile_price = round(bid + 0.25 * (ask - bid), 3)
                    order = LimitOrder(Controller.BUY, quantity_input, lower_quartile_price)
                    trade = ib_input.placeOrder(stock_input, order)
                    print("Trying to fill order.")
                    ib_input.sleep(5)
                    print(f"{trade.orderStatus.status}")
                    if trade.orderStatus.status == 'Filled':
                        print(f'Limit order {order.orderId} filled at {trade.fills[0].execution.price} submitted')
                        holding_stock = True
                        global_buy_price = trade.fills[0].execution.price
                    else:
                        print(
                            f'Warning: Limit order failed to fill at {lower_quartile_price} across the range: {bid} - {ask}.'
                            ' Attempting a market order instead.')
                        buy_order = MarketOrder(Controller.BUY,
                                                quantity_input)  # 'BUY' indicates the action and 10 is the quantity of shares
                        print(f"Placing market order of {quantity_input} of {symbol} shares.")
                        trade = ib_input.placeOrder(stock_input, buy_order)

                        # Wait for the order to fill
                        while not trade.isDone():
                            ib_input.waitOnUpdate()

                        if trade.fills:
                            holding_stock = True
                            fill_price = trade.fills[0].execution.price
                            global_buy_price = fill_price
                            print(f"Trade filled at {fill_price} price per share.")
                        else:
                            print("ERROR: trade not filled. Please check the Trader Workstation.")
                            print(f"{trade}")
                            holding_stock = False
                            return
                        # holding_stock = False
                else:  # place a market order instead
                    buy_order = MarketOrder(Controller.BUY,
                                            quantity_input)  # 'BUY' indicates the action and 10 is the quantity of shares
                    print(f"Placing market order of {quantity_input} of {symbol} shares.")
                    trade = ib_input.placeOrder(stock_input, buy_order)

                    # Wait for the order to fill
                    while not trade.isDone():
                        ib_input.waitOnUpdate()

                    if trade.fills:
                        holding_stock = True
                        fill_price = trade.fills[0].execution.price
                        global_buy_price = fill_price
                        print(f"Trade filled at {fill_price} price per share.")
                    else:
                        print("ERROR: trade not filled. Please check the Trader Workstation.")
                        print(f"{trade}")
                        return

    def _scheduled_task(self, ib_input,
                       stock_input,
                       ticker_name_input,
                       quantity_input,
                       frame_size,
                       stop_loss_percent,
                       dollar_amount,
                       start_time,
                       stop_time,
                       close_time,
                       ema_short,
                       ema_medium,
                       ema_long,
                       vwap,
                       rsi_period,
                       take_profit_input,
                       limit_order_input,
                       anchor_distance_input):
        global holding_stock
        current_time = datetime.now().time()

        start_process = False
        if datetime.strptime(start_time, "%H:%M").time() <= current_time <= datetime.strptime(stop_time,
                                                                                              "%H:%M").time():
            start_process = True

        if start_process:
            action = self._silent_wealth_start(ib_input,
                                         stock_input,
                                         frame_size,
                                         ticker_name_input,
                                         ema_short,
                                         ema_medium,
                                         ema_long,
                                         vwap,
                                         rsi_period,
                                         anchor_distance_input)

            if action == Controller.HOLD:
                pass
            elif action == Controller.SELL:
                if ticker_name_input == "BTC":
                    self._sell_market_BTC_order(ib_input, stock_input)
                else:
                    self._sell_market_order(ib_input, ticker_name_input, quantity_input)
            elif action == Controller.BUY:
                if ticker_name_input == "BTC":
                    self._place_market_BTC_order(ib_input, stock_input, quantity_input, dollar_amount, take_profit_input)
                else:
                    self._place_market_order(ib_input, stock_input, quantity_input, ticker_name_input, stop_loss_percent,
                                       take_profit_input, limit_order_input)
        else:
            # outside of trading hours for regular stock
            close_down_trades = False
            if datetime.strptime(stop_time, "%H:%M").time() < current_time <= datetime.strptime(close_time,
                                                                                                "%H:%M").time():
                close_down_trades = True

            if close_down_trades:
                positions = ib_input.positions()
                position = next((p for p in positions if p.contract.symbol == ticker_name_input), None)
                if position:
                    quantity_to_sell = position.position
                    print(f"Market nearing a close. Selling outstanding {quantity_to_sell} in {ticker_name_input}")
                    # Create a market order to sell all shares
                    sell_order = MarketOrder(Controller.SELL, quantity_to_sell)
                    # Place the order to sell the shares
                    ib_input.placeOrder(stock, sell_order)
                    exit()

    def _silent_wealth_start(self, ib_input,
                            stock_input,
                            frame_size_input,
                            stock_name,
                            ema_short_input,
                            ema_medium_input,
                            ema_long_input,
                            vwap_input,
                            rsi_period_input,
                            places_from_last_input):
        global global_ema20
        ema = ExpMovingAverage(ib_input, stock_input, frame_size_input, stock_name)

        df_ema_short = ema.calculate_exp_moving_average(ema_short_input)
        ema_short_value = df_ema_short[f"{ema_short_input}_day_EMA"].iloc[-1]

        df_ema_medium = ema.calculate_exp_moving_average(ema_medium_input)
        ema_medium_value = df_ema_medium[f"{ema_medium_input}_day_EMA"].iloc[-1]

        if ema_long_input > 0:
            df_ema_long = ema.calculate_exp_moving_average(ema_long_input)
            ema_long_input = df_ema_long[f"{ema_long_input}_day_EMA"].iloc[-1]

        if vwap_input > 0:
            vwap_obj = VolumeWeightedAverage(df_ema_short)
            df_vwap = vwap_obj.calculate_wva(vwap)
            vwap_value = df_vwap[f"rolling_vwap"].iloc[-1]
        else:
            vwap_value = 0

        if rsi_period_input > 0:
            rsi_obj = RSI(df_ema_short)
            rsi_value = rsi_obj.calculate_rsi(rsi_period_input)

        global_ema20 = ema_medium_input
        date_of_action = df_ema_short["date"].iloc[-1]

        # Should I attempt to buy based on where the 9 cross 20 occurred?
        # Find the index of the last entry where SHORT["short"] >= MEDIUM["medium"]
        # We iterate backwards to find the most recent match
        places_from_last = 0

        for i in range(len(df_ema_short) - 1, -1, -1):
            temp_short = df_ema_short[f"{ema_short_input}_day_EMA"][i]
            temp_medium = df_ema_medium[f"{ema_medium_input}_day_EMA"][i]
            if temp_short > temp_medium:
                # Calculate the number of places from the last entry
                # places_from_last = places_from_last + len(df_ema_short) - 1 - i
                places_from_last = places_from_last + 1
                # print(f"The value in 'short' is {temp_short:.3f} > 'medium' {temp_medium:.3f} at index {i}.")
                # print(f"The number of places from the last entry is: {places_from_last}")
            else:
                if places_from_last > places_from_last_input:
                    print(f"Too far from the cross-over BUY signal: {places_from_last}")
                    print(f"The value in 'short' is {temp_short:.3f} <= 'medium' {temp_medium:.3f} at index {i}.")
                    return Controller.HOLD
                else:
                    print(f"Within {places_from_last} cross-over to buy.")
                    break
        # else:
        #    print("No match found where 'short' is >= 'medium'.")

        if vwap_input == 0 and ema_long_input > 0 and rsi_period_input > 0:
            if ema_short_value > ema_medium_value and ema_short_value > ema_long_input and (50 < rsi_value <= 70):
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  RSI: {rsi_value:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  RSI: {rsi_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  RSI: {rsi_value:.3f}")
                return Controller.HOLD


        elif ema_long_input == 0 and vwap_input > 0 and rsi_period_input == 0:
            if ema_short_value > ema_medium_value and ema_short_value > vwap_value:
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  vwap: {vwap_value:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  vwap: {vwap_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  vwap: {vwap_value:.3f}")
                return Controller.HOLD


        elif ema_long_input > 0 and vwap == 0 and rsi_period_input == 0:
            if ema_short_value > ema_medium_value and ema_short_value > ema_long_input:
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}")
                return Controller.HOLD

        elif ema_long_input > 0 and vwap > 0 and rsi_period_input == 0:
            if ema_short_value > ema_medium_value and ema_short_value > ema_long_input and ema_short_input > vwap_value:
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  vwap: {vwap_value:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  vwap: {vwap_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  vwap: {vwap_value:.3f}")
                return Controller.HOLD

        elif ema_long_input == 0 and vwap == 0 and rsi_period_input > 0:
            if ema_short_value > ema_medium_value and (50 < rsi_value <= 70):
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.HOLD


        elif ema_long_input == 0 and vwap == 0 and rsi_period_input == 0:  # only the short and medium EMA
            if ema_short_value > ema_medium_value:
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}")
                return Controller.BUY
            elif ema_short_input <= ema_medium_input:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}")
                return Controller.HOLD

        elif ema_long_input == 0 and vwap_input > 0 and rsi_period_input > 0:
            if ema_short_value > ema_medium_value and ema_short_value > vwap_value and (50 < rsi_value <= 70):
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  vwap: {vwap_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  vwap: {vwap_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  vwap: {vwap_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.HOLD


        else:  # This is for all conditions
            if ema_short_value > ema_medium_value and ema_short_value > ema_long_input and ema_short_value > vwap_value and (
                    rsi_value > 50 and rsi_value <= 70):
                print(
                    f"BUY signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  vwap: {vwap_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.BUY
            elif ema_short_value <= ema_medium_value:
                print(
                    f"SELL signal (if holding) - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  vwap: {vwap_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.SELL
            else:
                print(
                    f"HOLD signal - {date_of_action} -- ema_short: {ema_short_value:.3f}  ema_medium: {ema_medium_value:.3f}  ema_long: {ema_long_input:.3f}  vwap: {vwap_value:.3f}  RSI: {rsi_value:.3f}")
                return Controller.HOLD


    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def run(self):
        pass
