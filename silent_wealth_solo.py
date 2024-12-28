import schedule
import time
from datetime import datetime
from ib_insync import IB, Stock, MarketOrder, StopOrder, Crypto, LimitOrder
import traceback
import argparse
import random
import pandas as pd

from exp_moving_average import ExpMovingAverage
from volume_weighted_average import VolumeWeightedAverage

PAPER_PORT = 7497
LIVE_PORT = 7496

LSE_START_TIME = "08:30"
LSE_END_TIME = "15:30"
LSE_CLOSE_TIME = "15:58"

NYSE_START_TIME = "14:10"
NYSE_END_TIME = "20:30"
NYSE_CLOSE_TIME = "20:58"

HOLD = "hold"
BUY = "buy"
SELL = "sell"

holding_stock = None
global_buy_price = None
global_ema20 = None
global_first_adjustment = True
global_previous_ema = None

#number_shares = 0



def sell_market_order(ib_input, symbol, quantity_input, stock_input):
    # sell here...
    global holding_stock

    if not holding_stock:
        return
    else:
        #print("...Selling")
        #positions = ib_input.positions()
        #position = next((p for p in positions if p.contract.symbol == symbol), None)

        #quantity_to_sell = position.position
        print(f"...Selling {quantity_input} of {symbol}")

        # Create a market order to sell all shares
        sell_order = MarketOrder('SELL', quantity_input)

        # Place the order to sell the shares
        trade = ib_input.placeOrder(stock, sell_order)
        holding_stock = False

        # # Wait for the order to fill
        # while not trade.isDone():
        #     ib_input.waitOnUpdate()
        #
        # commission = 0
        # # Retrieve execution details
        # for execDetail in trade.fills:
        #     print(f"Execution: {execDetail.execution}")
        #
        #     # Retrieve commission report
        #     commissionReport = execDetail.commissionReport
        #     #print(f"Commission Report: {commissionReport}")
        #     #print(f"Commission Amount: {commissionReport.commission}")
        #     commission = commission + commissionReport.commission
        #
        # print(f"Commission cost: {commissionReport.commission}")
        # print(f"Accrued commission costs: {commission}")
        #
        # # get the trade price (roughly)
        # ticker = ib_input.reqMktData(stock_input, '', False, False)
        # ib_input.sleep(2)
        # #ticker = ib_input.ticker(stock_input)
        # if ticker.last:
        #     latest_price = ticker.last
        # elif ticker.close:
        #     latest_price = ticker.close  # Fallback to last close price if no live price is available
        # else:
        #     latest_price = None
        #
        # # Calculate the value of the shares
        # if latest_price:
        #     total_value = latest_price * 10  # Multiply by the number of shares
        #     print(f"Latest market price: ${latest_price}")
        #     print(f"Value of 10 shares: ${total_value}")
        # else:
        #     print("Market price not available.")

def place_market_BTC_order(ib_input, stock_input, quantity_input, cash):
    pass

def sell_market_BTC_order(ib_input, stock_input):
    pass

def place_market_order(ib_input, stock_input, quantity_input, symbol, stop_loss_percent, cash):
    # buy here...
    global holding_stock
    global global_buy_price
    global global_ema20
    global global_first_adjustment
    global global_previous_ema

    if holding_stock:
        if not symbol == "BTC":
            #ticker = ib_input.reqMktData(stock_input, '', False, False)
            #current_price = ticker.last
            if global_ema20 > global_buy_price and global_first_adjustment:
                print(f"EMA20 {global_ema20} > original buy price. Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
                stop_order = StopOrder('SELL', quantity_input, global_ema20)
                trade = ib_input.placeOrder(stock_input, stop_order)
                global_previous_ema = global_ema20
                global_first_adjustment = False
            elif not global_first_adjustment and global_ema20 > global_previous_ema:
                print(f"EMA20 {global_ema20} > previous EMA20 {global_previous_ema}. Adjusting stop loss to {global_ema20} to track the curve and avoid losses.")
                stop_order = StopOrder('SELL', quantity_input, global_ema20)
                trade = ib_input.placeOrder(stock_input, stop_order)
                global_previous_ema = global_ema20
        else:
            # For bitcoin - to finish
            pass
        return
    else:
        if not symbol == "BTC":
            #btc_ticker = ib_input.reqMktData(stock_input, '', False, False)
            #ib_input.sleep(2)
            #bid_price = btc_ticker.bid
            #print(f"...Placing limit order of {quantity_input} of {symbol} shares at {bid_price} per share")
            #buy_order = LimitOrder('BUY', quantity_input, bid_price)
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
        else:
            # This is for Bitcoin
            #order = MarketOrder('BUY', 0)
            share_ticker = ib_input.reqMktData(stock_input, '', False, False)
            ib_input.sleep(2)
            bid_price = share_ticker.bid
            print(f"...Placing limit order of {quantity_input} of BTC at {bid_price} per share")
            order = LimitOrder('BUY', quantity_input, bid_price)
            order.cashQty = cash
            order.tif = 'IOC'
            trade = ib_input.placeOrder(stock_input, order)
            holding_stock = True

            while not trade.isDone():
                ib_input.waitOnUpdate()

            if trade.fills:
                fill_price = trade.fills[0].execution.price
                print(f"Trade filled at {fill_price} price per share")
                stop_loss_offset = 1.000 - stop_loss_percent
                stop_loss_price = round(fill_price * stop_loss_offset, 3)
                print(f"Stop-loss price set at {stop_loss_price:.3f}")
                stop_order = StopOrder('SELL', quantity_input, stop_loss_price)


def silent_wealth_start(ib_input, stock_input, frame_size, stock_name):
    # For the first trade the holding list is empty
    decision = decision_maker(ib_input, stock_input, frame_size, stock_name)
    return decision


def decision_maker(ib_input, stock_input, frame_size, stock_name):
    global global_ema20
    ema = ExpMovingAverage(ib_input, stock_input, frame_size, stock_name)
    df_9 = ema.calculate_exp_moving_average(9)
    df_20 = ema.calculate_exp_moving_average(20)
    df_200 = ema.calculate_exp_moving_average(200)
    vwa = VolumeWeightedAverage(df_9)
    df_vwap = vwa.calculate_wva()

    ma200 = df_200["200_day_MA"].iloc[-1]
    ma20 = df_20["20_day_MA"].iloc[-1]
    global_ema20 = ma20
    ma9 = df_9["9_day_MA"].iloc[-1]
    vwap = df_vwap["rolling_vwap"].iloc[-1]
    date_of_action = df_9["date"].iloc[-1]

    print(f"{date_of_action} -- ema9: {ma9:.3f}  ema20: {ma20:.3f}  ema200: {ma200:.3f}  vwap (9 days): {vwap:.3f}")
    # The conditions are good and I don't have the security - buy it
    if pd.isna(vwap):
        if ma9 > ma20 and ma9 > ma200:
            return BUY
        elif ma9 <= ma20:
            return SELL
        else:
            return HOLD
    else:
        if ma9 > ma20 and ma9 > ma200 and ma9 > vwap:
            return BUY
        elif ma9 <= ma20:
            return SELL
        else:
            return HOLD


# Define a wrapper function to execute only within the desired hours
def scheduled_task(ib_input, stock_input, ticker_name_input, quantity_input, frame_size, stop_loss_percent, dollar_amount, trade_time):
    global holding_stock
    current_time = datetime.now().time()

    # bitcoin is 24/7
    if ticker_name_input == "BTC":
        action = silent_wealth_start(ib_input, stock_input, frame_size, ticker_name_input)
        if action == HOLD:
            pass
        elif action == SELL:
            sell_market_order(ib_input, ticker_name_input, quantity_input, stock_input)
        elif action == BUY:
            place_market_order(ib_input, stock_input, quantity_input, ticker_name_input, stop_loss_percent,
                               dollar_amount)
    else:
        start_process = False
        if trade_time == "US":
            if current_time >= datetime.strptime(NYSE_START_TIME, "%H:%M").time() and \
                    current_time <= datetime.strptime(NYSE_END_TIME, "%H:%M").time():
                start_process = True
        else:
            if current_time >= datetime.strptime(LSE_START_TIME, "%H:%M").time() and \
                    current_time <= datetime.strptime(LSE_END_TIME, "%H:%M").time():
                start_process = True

        if start_process:
            action = silent_wealth_start(ib_input, stock_input, frame_size, ticker_name_input)

            if action == HOLD:
                pass
            elif action == SELL:
                sell_market_order(ib_input, ticker_name_input, quantity_input, stock_input)
            elif action == BUY:
                place_market_order(ib_input, stock_input, quantity_input, ticker_name_input, stop_loss_percent, dollar_amount)
        else:
            # outside of trading hours for regular stock
            close_down_trades = False
            if trade_time == "US":
                if current_time > datetime.strptime(NYSE_END_TIME, "%H:%M").time() and \
                        current_time <= datetime.strptime(NYSE_CLOSE_TIME, "%H:%M").time():
                    close_down_trades = True
            else:
                if current_time > datetime.strptime(LSE_END_TIME, "%H:%M").time() and \
                        current_time <= datetime.strptime(LSE_CLOSE_TIME, "%H:%M").time():
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


# --ticker_name BP. --exchange LSE --quantity 100
parser = argparse.ArgumentParser(description="Silent Wealth")

# Add arguments
parser.add_argument("--ticker_name", type=str, help="Ticker name e.g., BP. SOXL SOXS, and BTC for Bitcoin", required=True)
parser.add_argument("--exchange", type=str, help="Exchange e.g., LSE LSEETF ARCA. Not required for Bitcoin.", required=False)
parser.add_argument("--currency", type=str, help="Currency for trade e.g., USD, GBP.", required=True)
parser.add_argument("--quantity", type=int, help="The number of shares to buy/sell", required=False)
parser.add_argument("--frame_size", type=int, help="Minute candle size e.g., 1, 5, or 10", required=True)
parser.add_argument("--account", type=str, help="Account type e.g., paper or live", required=True)
parser.add_argument("--stop_loss_percent", type=float, help="The percent below the buy position to take as a stop-loss", required=True)
parser.add_argument("--dollar_amount", type=int, help="Amount of bitcoin to buy in dollars", required=False)
parser.add_argument('--time', choices=['US', 'UK'], required=True, help='Specify the region (US or UK).')
args = parser.parse_args()

ticker_name = args.ticker_name
exchange = args.exchange
currency = args.currency
quantity = args.quantity
frame_size = args.frame_size
account = args.account.lower()
stop_loss_percent = args.stop_loss_percent
dollar_amount = args.dollar_amount
trade_time = args.time.upper()

client_id = random.randint(1, 9999)
ib = IB()
if account == "paper":
    ib.connect('127.0.0.1', PAPER_PORT, clientId=client_id)
elif account == "live":
    ib.connect('127.0.0.1', LIVE_PORT, clientId=client_id)
else:
    print(f"Unrecognised account port: {account}. It must be 'live' or 'paper'")

if ticker_name == "BTC":
    stock = Crypto('BTC', 'PAXOS', 'USD')
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
else:
    stock = Stock(symbol=ticker_name, exchange=exchange, currency=currency)
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


print("---------Silent Wealth------------")
print(f"Ticker nane: {ticker_name}")
print(f"Market price: {market_price}")
print(f"Exchange: {exchange}")
print(f"Currency: {currency}")
print(f"Quantity: {quantity}")
print(f"Frame size (minutes): {frame_size}")
print(f"Client ID: {client_id}")
print(f"Account type: {account}")
print(f"BTC Dollar amount: {dollar_amount}")
if trade_time == "US":
    print(f"Start time: {NYSE_START_TIME} (UK time)")
    print(f"Stop time: {NYSE_END_TIME} (UK time)\n")
else:
    print(f"Start time: {LSE_START_TIME}")
    print(f"Stop time: {LSE_END_TIME}\n")

schedule.every(frame_size).minutes.do(scheduled_task,
                                      ib,
                                      stock,
                                      ticker_name,
                                      quantity,
                                      frame_size,
                                      stop_loss_percent,
                                      dollar_amount,
                                      trade_time)

# Keep the scheduler running
print(f"Starting run... {ticker_name}")
first_time = True
try:
    while True:
        if first_time:
            first_time = False
            scheduled_task(ib, stock, ticker_name, quantity, frame_size, stop_loss_percent, dollar_amount, trade_time)
        else:
            schedule.run_pending()
            time.sleep(1)  # Sleep to prevent CPU overuse
except Exception as e:
    print(f"Scheduler stopped due to unknown error. {e}")
    traceback.print_exc()
finally:
    ib.disconnect()
