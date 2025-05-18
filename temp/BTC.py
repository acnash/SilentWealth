from ib_insync import IB, Crypto, LimitOrder

# Connect to IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the cryptocurrency instrument (e.g., BTC/USD as Crypto, not Forex)
btc_usd = Crypto(symbol='BTC', currency='USD', exchange='PAXOS')  # Adjust exchange if needed

# Qualify the contract
contract_details = ib.qualifyContracts(btc_usd)
if not contract_details:
    raise ValueError("The BTC/USD contract could not be qualified. Check your settings.")

# Fetch current market price
market_data = ib.reqMktData(btc_usd)
ib.sleep(2)  # Allow time to fetch market data
#current_price = float(market_data.last)

# Set limit price (e.g., slightly below current market price)
#limit_price = round(current_price * 0.99, 2)  # 1% below the market price for demonstration

# Calculate the quantity to buy ($1000 worth)
quantity = round(1000 / market_data.last, 8)  # Adjust precision as needed

# Create a Limit Order
order = LimitOrder('BUY', quantity, market_data.last)
order.tif = "IOC"

# Place the order
trade = ib.placeOrder(btc_usd, order)

# Wait for the trade to be executed
ib.sleep(2)
print(f"Trade status: {trade.orderStatus.status}")
print(f"Filled quantity: {trade.orderStatus.filled}")
print(f"Average fill price: {trade.orderStatus.avgFillPrice}")

# Disconnect
ib.disconnect()
