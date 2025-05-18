from ib_insync import *

# Connect to IB Gateway or Trader Workstation
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the stock contract (e.g., NVDA on SMART exchange)
nvda_contract = Stock('LLOY', 'LSE', 'GBP')

# Define the order details
quantity = 4
entry_price = None  # Set to None for Market Order
stop_loss_price = 52  # Example Stop Loss Price

# Create the Primary Order (Market Order for simplicity)
primary_order = MarketOrder('BUY', quantity)

# Place the primary order
trade = ib.placeOrder(nvda_contract, primary_order)
ib.sleep(2)  # Wait for the order to execute

# Get the filled price of the primary order
if trade.fills:
    filled_price = trade.fills[0].execution.price
else:
    raise Exception("The primary order was not filled. Cannot proceed with stop loss or take profit orders.")

# Calculate the take profit price as 1% above the filled price
take_profit_price = filled_price * 1.01

# Create the Take Profit Order
take_profit_order = LimitOrder('SELL', quantity, take_profit_price)
take_profit_order.parentId = primary_order.orderId

# Create the Stop Loss Order
stop_loss_order = StopOrder('SELL', quantity, stop_loss_price)
stop_loss_order.parentId = primary_order.orderId

# Assign an OCA Group to link the orders
oca_group = "OCA_Group_10"
take_profit_order.ocaGroup = oca_group
stop_loss_order.ocaGroup = oca_group

# Place the Take Profit and Stop Loss orders
ib.placeOrder(nvda_contract, take_profit_order)
ib.placeOrder(nvda_contract, stop_loss_order)

print(f"Bracket Order placed for NVDA: Buy {quantity} shares at {filled_price}, Stop Loss at {stop_loss_price}, Take Profit at {take_profit_price}")

# Disconnect after placing orders
ib.disconnect()
