import pandas as pd

class VolumeWeightedAverage:

    def __init__(self, df):
        self.df = df

    def calculate_wva(self):
        #self.df['price_volume'] = self.df['close'] * self.df['volume']  # Price * Volume for each row
        #vwap = self.df['price_volume'].sum() / self.df['volume'].sum()
        #return vwap

        # Set the timestamp as the index
        self.df['date'] = pd.to_datetime(self.df['date'])
        # Set a rolling window of 3 minutes
        window = 9  # rolling window size in minutes

        # Calculate rolling VWAP
        self.df['weighted_price'] = self.df['close'] * self.df['volume']
        self.df['rolling_vwap'] = (
                self.df['weighted_price'].rolling(window).sum() /
                self.df['volume'].rolling(window).sum()
        )

        return self.df
