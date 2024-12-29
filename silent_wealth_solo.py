import schedule
import time
from datetime import datetime
from ib_insync import IB, Stock, MarketOrder, StopOrder, Crypto, LimitOrder, Contract
import traceback
import argparse
import random
import pandas as pd

from exp_moving_average import ExpMovingAverage
from volume_weighted_average import VolumeWeightedAverage

PAPER_PORT = 7497
LIVE_PORT = 7496

LSE_START_TIME = "08:10"
LSE_END_TIME = "16:00"
LSE_CLOSE_TIME = "16:10"

NYSE_START_TIME = "14:10"
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


def sell_market_order(ib_input, symbol, quantity_input, stock_input):
    # sell here...
    global holding_stock

    if not holding_stock:
        return
    else:
        print(f"...Selling {quantity_input} of {symbol}")
        sell_order = MarketOrder('SELL', quantity_input)
        trade = ib_input.placeOrder(stock, sell_order)
        holding_stock = False


def sell_market_BTC_order(ib_input, btc_contract):
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
                action='SELL',
                totalQuantity=abs(btc_quantity)  # Ensure quantity is positive
            )
            sell_order.tif = "IOC"

            # Place the order
            trade = ib.placeOrder(btc_contract, sell_order)
            print(f"Sell order placed: {trade}")
            holding_stock = False
        else:
            print("No BTC to sell or invalid position.")


def place_market_BTC_order(ib_input, stock_input, quantity_input, cash):
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
        #order = MarketOrder('BUY', btc_quantity)
        #order.cashQty = cash

        order = LimitOrder(
            action='BUY',  # 'BUY' to purchase BTC
            totalQuantity=btc_quantity,  # Specify the calculated BTC quantity
            lmtPrice=0  # Set limit price to 0 (market-like)
        )
        order.cashQty = cash
        order.tif = 'IOC'
        trade = ib_input.placeOrder(stock_input, order)
        print(f"...Placing limit order of {btc_quantity} of BTC at bid price of {btc_price} per share")

        while not trade.isDone():
            #ib_input.waitOnUpdate()
            ib_input.sleep(2)

        if trade.fills:
            fill_price = trade.fills[0].execution.price
            global_buy_price = fill_price
            print(f"Trade filled at {fill_price} price per share")
            stop_loss_offset = 1.000 - stop_loss_percent
            stop_loss_price = round(fill_price * stop_loss_offset, 3)
            print(f"Stop-loss price set at {stop_loss_price:.3f}")
            stop_order = StopOrder('SELL', btc_quantity, stop_loss_price)
            holding_stock = True
        else:
            print("Trade did not fill.")
            holding_stock = False

    else:
        print("Holding and looking to adjust the stop loss.")
        #if global_ema20 > global_buy_price and global_first_adjustment:
        #    print(
        #        f"EMA20 {global_ema20} > original buy price. "
        #        f"Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
        #    stop_order = StopOrder('SELL', quantity_input, global_ema20)
        #    trade = ib_input.placeOrder(stock_input, stop_order)
        #    global_previous_ema = global_ema20
        #    global_first_adjustment = False
        #elif global_ema20 > global_previous_ema and not global_first_adjustment:
        #    print(
        #        f"EMA20 {global_ema20} > previous EMA20 {global_previous_ema}. "
        #        f"Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
        #    stop_order = StopOrder('SELL', quantity_input, global_ema20)
        #    trade = ib_input.placeOrder(stock_input, stop_order)
        #    global_previous_ema = global_ema20


def place_market_order(ib_input, stock_input, quantity_input, symbol, stop_loss_percent):
    # buy here...
    global holding_stock
    global global_buy_price
    global global_ema20
    global global_first_adjustment
    global global_previous_ema

    if holding_stock:
        if global_ema20 > global_buy_price and global_first_adjustment:
            print(
                f"EMA20 {global_ema20} > original buy price. Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
            stop_order = StopOrder('SELL', quantity_input, global_ema20)
            trade = ib_input.placeOrder(stock_input, stop_order)
            global_previous_ema = global_ema20
            global_first_adjustment = False
        elif not global_first_adjustment and global_ema20 > global_previous_ema:
            print(
                f"EMA20 {global_ema20} > previous EMA20 {global_previous_ema}. Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
            stop_order = StopOrder('SELL', quantity_input, global_ema20)
            trade = ib_input.placeOrder(stock_input, stop_order)
            global_previous_ema = global_ema20
    else:
        # btc_ticker = ib_input.reqMktData(stock_input, '', False, False)
        # ib_input.sleep(2)
        # bid_price = btc_ticker.bid
        # print(f"...Placing limit order of {quantity_input} of {symbol} shares at {bid_price} per share")
        # buy_order = LimitOrder('BUY', quantity_input, bid_price)
        buy_order = MarketOrder('BUY', quantity_input)  # 'BUY' indicates the action and 10 is the quantity of shares
        print(f"...Placing market order of {quantity_input} of {symbol} shares.")
        trade = ib_input.placeOrder(stock_input, buy_order)

        # Wait for the order to fill
        while not trade.isDone():
            ib_input.waitOnUpdate()

        if trade.fills:
            holding_stock = True
            fill_price = trade.fills[0].execution.price
            global_buy_price = fill_price
            print(f"Trade filled at {fill_price} price per share")
            stop_loss_offset = 1.000 - stop_loss_percent
            stop_loss_price = round(fill_price * stop_loss_offset, 3)
            print(f"Stop-loss price set at {stop_loss_price:.3f}")
            stop_order = StopOrder('SELL', quantity_input, stop_loss_price)
            trade = ib_input.placeOrder(stock_input, stop_order)
            print(trade)
        else:
            print("ERROR: trade not filled. Please check the Trader Workstation.")


# def silent_wealth_start(ib_input, stock_input, frame_size, stock_name):
#    # For the first trade the holding list is empty
#    return decision_maker(ib_input, stock_input, frame_size, stock_name)


def silent_wealth_start(ib_input,
                        stock_input,
                        frame_size_input,
                        stock_name,
                        ema_short_input,
                        ema_medium_input,
                        ema_long_input,
                        vwap_input):
    global global_ema20
    ema = ExpMovingAverage(ib_input, stock_input, frame_size_input, stock_name)

    df_ema_short = ema.calculate_exp_moving_average(ema_short_input)
    ema_short_input = df_ema_short[f"{ema_short_input}_day_EMA"].iloc[-1]

    df_ema_medium = ema.calculate_exp_moving_average(ema_medium_input)
    ema_medium_input = df_ema_medium[f"{ema_medium_input}_day_EMA"].iloc[-1]

    if ema_long_input > 0:
        df_ema_long = ema.calculate_exp_moving_average(ema_long_input)
        ema_long_input = df_ema_long[f"{ema_long_input}_day_EMA"].iloc[-1]

    if vwap_input > 0:
        vwap_obj = VolumeWeightedAverage(df_ema_short)
        df_vwap = vwap_obj.calculate_wva(vwap)
        vwap_value = df_vwap[f"rolling_vwap"].iloc[-1]
    else:
        vwap_value = 0

    global_ema20 = ema_medium_input
    date_of_action = df_ema_short["date"].iloc[-1]

    if vwap_input == 0 and ema_long_input > 0:
        if ema_short_input > ema_medium_input and ema_short_input > ema_long_input:
            print(
                f"BUY signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  ema200: {ema_long_input:.3f}")
            return BUY
        elif ema_short_input <= ema_medium_input:
            print(
                f"SELL signal (if holding) - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  ema200: {ema_long_input:.3f}")
            return SELL
        else:
            print(
                f"HOLD signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  ema200: {ema_long_input:.3f}")
            return HOLD
    elif ema_long_input == 0 and vwap_input > 0:
        if ema_short_input > ema_medium_input and ema_short_input > vwap_value:
            print(
                f"BUY signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  vwap: {vwap_value:.3f}")
            return BUY
        elif ema_short_input <= ema_medium_input:
            print(
                f"SELL signal (if holding) - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  vwap: {vwap_value:.3f}")
            return SELL
        else:
            print(
                f"HOLD signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  vwap: {vwap_value:.3f}")
            return HOLD
    elif ema_long_input == 0 and vwap == 0:
        if ema_short_input > ema_medium_input:
            print(f"BUY signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}")
            return BUY
        elif ema_short_input <= ema_medium_input:
            print(f"SELL signal (if holding) - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}")
            return SELL
        else:
            print(f"HOLD signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}")
            return HOLD
    else:  # This is for all conditions
        if ema_short_input > ema_medium_input and ema_short_input > ema_long_input and ema_short_input > vwap_value:
            print(
                f"BUY signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  ema200: {ema_long_input:.3f}  vwap: {vwap_value:.3f}")
            return BUY
        elif ema_short_input <= ema_medium_input:
            print(
                f"SELL signal (if holding) - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  ema200: {ema_long_input:.3f}  vwap: {vwap_value:.3f}")
            return SELL
        else:
            print(
                f"HOLD signal - {date_of_action} -- ema9: {ema_short_input:.3f}  ema20: {ema_medium_input:.3f}  ema200: {ema_long_input:.3f}  vwap: {vwap_value:.3f}")
            return HOLD


# Define a wrapper function to execute only within the desired hours
def scheduled_task(ib_input,
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
                   vwap):
    global holding_stock
    current_time = datetime.now().time()

    start_process = False
    if current_time >= datetime.strptime(start_time, "%H:%M").time() and \
            current_time <= datetime.strptime(stop_time, "%H:%M").time():
        start_process = True

    if start_process:
        action = silent_wealth_start(ib_input,
                                     stock_input,
                                     frame_size,
                                     ticker_name_input,
                                     ema_short,
                                     ema_medium,
                                     ema_long,
                                     vwap)

        if action == HOLD:
            pass
        elif action == SELL:
            if ticker_name_input == "BTC":
                sell_market_BTC_order(ib_input, stock_input)
            else:
                sell_market_order(ib_input, ticker_name_input, quantity_input, stock_input)
        elif action == BUY:
            if ticker_name_input == "BTC":
                place_market_BTC_order(ib_input, stock_input, quantity_input, dollar_amount)
            else:
                place_market_order(ib_input, stock_input, quantity_input, ticker_name_input, stop_loss_percent)
    else:
        # outside of trading hours for regular stock
        close_down_trades = False
        if current_time > datetime.strptime(stop_time, "%H:%M").time() and \
                current_time <= datetime.strptime(close_time, "%H:%M").time():
            close_down_trades = True

        if close_down_trades:
            positions = ib_input.positions()
            position = next((p for p in positions if p.contract.symbol == ticker_name_input), None)
            if position:
                quantity_to_sell = position.position
                print(f"Market nearing a close. Selling outstanding {quantity_to_sell} in {ticker_name_input}")
                # Create a market order to sell all shares
                sell_order = MarketOrder('SELL', quantity_to_sell)
                # Place the order to sell the shares
                ib_input.placeOrder(stock, sell_order)
                exit()


# =======================================================================================================================
parser = argparse.ArgumentParser(description="Silent Wealth")
parser.add_argument("--ticker_name", type=str,
                    help="Ticker name e.g., BP. SOXL SOXS, and BTC for Bitcoin", required=True)
parser.add_argument("--exchange", type=str,
                    help="Exchange e.g., LSE LSEETF ARCA. Not required for Bitcoin.", required=False)
parser.add_argument("--quantity", type=int,
                    help="The number of shares to buy/sell", required=False)
parser.add_argument("--frame_size", type=int,
                    help="Minute candle size e.g., 1, 5, or 10", required=True)
parser.add_argument("--account", type=str,
                    help="Account type e.g., paper or live", required=True)
parser.add_argument("--stop_loss_percent", type=float,
                    help="The percent below the buy position to take as a stop-loss", required=False)
parser.add_argument("--dollar_amount", type=int,
                    help="Amount of bitcoin to buy in dollars", required=False)
parser.add_argument("--ema_short", type=int, default=9, required=False,
                    help="Units to compute the short-range exponential moving average. Default is 9.")
parser.add_argument("--ema_medium", type=int, default=20, required=False,
                    help="Units to compute the mid-range exponential moving average. Default is 20.")
parser.add_argument("--ema_long", type=int, default=200, required=False,
                    help="Units to compute the long-range exponential moving average. Default is 200.")
parser.add_argument("--vwap", type=int, default=9, required=False,
                    help="Units to compute the volume-weighted average price. Default is 9.")
args = parser.parse_args()

ticker_name = args.ticker_name
exchange = args.exchange
quantity = args.quantity
frame_size = args.frame_size
account = args.account.lower()
stop_loss_percent = args.stop_loss_percent
dollar_amount = args.dollar_amount
ema_short = args.ema_short
ema_medium = args.ema_medium
ema_long = args.ema_long
vwap = args.vwap

client_id = random.randint(1, 9999)
ib = IB()
if account == "paper":
    ib.connect('127.0.0.1', PAPER_PORT, clientId=client_id)
elif account == "live":
    ib.connect('127.0.0.1', LIVE_PORT, clientId=client_id)
else:
    print(f"Unrecognised account port: {account}. It must be 'live' or 'paper'")

print("---------Silent Wealth------------")
print(f"Ticker nane: {ticker_name}")
print(f"Frame size (minutes): {frame_size}")
print(f"Client ID: {client_id}")

if exchange:
    print(f"Exchange: {exchange}")
    print(f"Quantity: {quantity}")
    if ["LSE"] in exchange:
        start_time = LSE_START_TIME
        stop_time = LSE_END_TIME
        close_time = LSE_CLOSE_TIME
        currency = "GBP"
    elif ["NASDAQ", "ARCA"] in exchange:
        start_time = NYSE_START_TIME
        stop_time = NYSE_END_TIME
        close_time = NYSE_CLOSE_TIME
        currency = "USD"
    else:
        print("Failed to establish trading time.")
        ib.disconnect()
        exit()
elif ticker_name == "BTC":
    currency = "USD"
    start_time = BTC_PAXOS_START_TIME
    stop_time = BTC_PAXOS_END_TIME
    close_time = BTC_PAXOS_CLOSE_TIME
    print(f"BTC Dollar amount: {dollar_amount}")
else:
    print("Failed to establish trading time.")
    ib.disconnect()
    exit()

print(f"Currency: {currency}")
print(f"Account type: {account}\n")
print("BUY/SELL conditions")
print(f"Short-range moving average: {ema_short}")
print(f"Medium-range moving average: {ema_medium}")
print(f"Long-range moving average: {ema_long}")
print(f"Volume-weighted average price: {vwap}\n")

if ticker_name == "BTC":
    #stock = Crypto('BTC', 'PAXOS', 'USD')
    stock = Contract(
        secType='CRYPTO',
        symbol='BTC',
        exchange='PAXOS',
        currency='USD'
    )
else:
    stock = Stock(symbol=ticker_name, exchange=exchange, currency=currency)
ib.qualifyContracts(stock)
ib.reqMktData(stock)
ib.sleep(2)
ticker = ib.ticker(stock)

market_price = None
if ticker.last:
    market_price = ticker.last
elif ticker.close:
    market_price = ticker.close
else:
    print("Cannot determine market price. Possible connection issue.")
    ib.disconnect()
    exit()

schedule.every(frame_size).minutes.do(scheduled_task,
                                      ib,
                                      stock,
                                      ticker_name,
                                      quantity,
                                      frame_size,
                                      stop_loss_percent,
                                      dollar_amount,
                                      start_time,
                                      stop_time,
                                      close_time,
                                      ema_short,
                                      ema_medium,
                                      ema_long,
                                      vwap)

# Keep the scheduler running
print(f"Starting run... {ticker_name}")
first_time = True
try:
    while True:
        if first_time:
            first_time = False
            scheduled_task(ib,
                           stock,
                           ticker_name,
                           quantity,
                           frame_size,
                           stop_loss_percent,
                           dollar_amount,
                           start_time,
                           stop_time,
                           close_time,
                           ema_short,
                           ema_medium,
                           ema_long,
                           vwap)
        else:
            schedule.run_pending()
            time.sleep(1)  # Sleep to prevent CPU overuse
except Exception as e:
    print(f"Scheduler stopped due to unknown error. {e}")
    traceback.print_exc()
finally:
    ib.disconnect()
