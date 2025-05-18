from ib_insync import IB, Stock
import pandas as pd

# Connect to Interactive Brokers TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

# Define the stock contract for GLEN on the LSE
stock = Stock(symbol='LLOY', exchange='LSE', currency='GBP')

# Request historical data (last 30 trading days)
historical_data = ib.reqHistoricalData(
    stock,
    endDateTime='',
    durationStr='30 D',
    barSizeSetting='1 day',
    whatToShow='MIDPOINT',  # Options: TRADES, MIDPOINT, etc.
    useRTH=True
)

# Check if historical data was retrieved
if historical_data:
    # Manually extract data into a list of dictionaries
    data_dict = {
        "date": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": []
    }

    for bar in historical_data:
        data_dict["date"].append(bar.date),
        data_dict["open"].append(bar.open),
        data_dict["high"].append(bar.high),
        data_dict["low"].append(bar.low),
        data_dict["close"].append(bar.close),
        data_dict["volume"].append(bar.volume),

    # Convert to pandas DataFrame
    df = pd.DataFrame(data_dict)

    # Ensure dates are in proper datetime format and sorted
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Calculate the 20-day moving average for the 'close' column
    df['20_day_MA'] = df['close'].rolling(window=20).mean()

    # Display the most recent rows, including the 20-day moving average
    print(df[['date', 'close', '20_day_MA']].tail(5))
else:
    print("No historical data available for GLEN.")

# Disconnect from IBKR
ib.disconnect()
