import pandas as pd
from pathlib import Path

class ExpMovingAverage:

    def __init__(self, ib, stock, frame_size, unit_type, stock_name, output_data):

        # for testing just build the object of the class
        if not ib:
            return

        self.output_data = output_data
        self.frame_size = frame_size
        self.unit_type = unit_type

        if stock_name == "BTC" or stock_name == "ETH" or stock_name == "SOL":
            self.historical_data = ib.reqHistoricalData(
                stock,
                endDateTime='',
                durationStr='365 D',
                barSizeSetting=f'{str(frame_size)} {unit_type}',
                whatToShow='MIDPOINT',
                useRTH=True
            )

        else:
            self.historical_data = ib.reqHistoricalData(
                stock,
                endDateTime='',
                durationStr='365 D',
                barSizeSetting=f'{str(frame_size)} {unit_type}',
                whatToShow='TRADES',  # Options: TRADES, MIDPOINT, etc.
                useRTH=True
            )

    def calculate_exp_moving_average(self, days_list, test_df=None):
        if not isinstance(test_df, pd.DataFrame):
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

                if self.output_data:
                    filepath = Path(self.output_data)
                    if filepath.suffix == '':
                        filepath = filepath.with_suffix(".dat")

                    # Step 2: Modify the file name to include '_exp_20' before the extension
                    new_filename = filepath.stem + f"_frame_size_{self.frame_size}_{self.unit_type}" + filepath.suffix
                    new_filepath = filepath.with_name(new_filename)
                    df.to_csv(new_filepath, index=False)

        for day in days_list:
            if isinstance(test_df, pd.DataFrame):
                test_df[f"{day}_day_EMA"] = test_df['close'].ewm(span=day, adjust=False, min_periods=day).mean()
            else:
                df[f"{day}_day_EMA"] = df['close'].ewm(span=day, adjust=False, min_periods=day).mean()

        if isinstance(test_df, pd.DataFrame):
            return test_df
        else:
            return df
