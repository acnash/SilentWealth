import pandas as pd


class ExpMovingAverage:

    def __init__(self, ib, stock, frame_size, stock_name):
        if frame_size == 1:
            unit = "min"
        elif frame_size > 1:
            unit = "mins"
        if stock_name == "BTC":
            self.historical_data = ib.reqHistoricalData(
                stock,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting=f'{str(frame_size)} {unit}',
                whatToShow='MIDPOINT',
                useRTH=True
            )

        else:
            self.historical_data = ib.reqHistoricalData(
                stock,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting=f'{str(frame_size)} {unit}',
                whatToShow='TRADES',  # Options: TRADES, MIDPOINT, etc.
                useRTH=True
            )

    def calculate_exp_moving_average(self, days):
        # Check if historical data was retrieved
        if self.historical_data:
            # Manually extract data into a list of dictionaries
            data_dict = {
                "date": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            }

            for bar in self.historical_data:
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

            df[f"{days}_day_EMA"] = df['close'].ewm(span=days, adjust=False, min_periods=days).mean()

            return df
