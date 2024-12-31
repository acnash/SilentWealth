import pandas as pd
import numpy as np


class RSI:

    def __init__(self, df):
        self.df = df

    def calculate_rsi(self, period):
        close_prices = self.df['close']
        # Calculate the differences between consecutive prices
        delta = close_prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate the rolling average of gains and losses
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Compute the Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Compute the RSI
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        return last_rsi
