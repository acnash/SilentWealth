import pandas as pd

class VolumeWeightedAverage:

    def __init__(self, df):
        self.df = df

    def calculate_wva(self, vwap):
        # Set the timestamp as the index
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Calculate rolling VWAP
        self.df['weighted_price'] = self.df['close'] * self.df['volume']
        self.df['rolling_vwap'] = (
                self.df['weighted_price'].rolling(vwap).sum() /
                self.df['volume'].rolling(vwap).sum()
        )

        return self.df
