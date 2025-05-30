import pandas as pd
import numpy as np


class RSI:

    def __init__(self, df):
        self.df = df

    def calculate_rsi(self, period, wilder_smoothing=True):
        close_prices = self.df['close']
        # Calculate the differences between consecutive prices
        delta = close_prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate the rolling average of gains and losses
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        if wilder_smoothing:
            # Initialize empty RSI Series
            rsi = pd.Series(index=close_prices.index, dtype=float)

            # First average gain/loss (simple mean over first `period` values)
            avg_gain = gain.iloc[1:period + 1].mean()
            avg_loss = loss.iloc[1:period + 1].mean()
            # First RSI value (at index `period`)
            if avg_loss == 0:
                rsi.iloc[period] = 100
            else:
                rs = avg_gain / avg_loss
                rsi.iloc[period] = 100 - (100 / (1 + rs))

            # Wilder’s smoothing for the rest
            for i in range(period + 1, len(close_prices)):
                current_gain = gain.iloc[i]
                current_loss = loss.iloc[i]

                avg_gain = (avg_gain * (period - 1) + current_gain) / period
                avg_loss = (avg_loss * (period - 1) + current_loss) / period

                if avg_loss == 0:
                    rsi.iloc[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi.iloc[i] = 100 - (100 / (1 + rs))
        else:
            # Compute the Relative Strength (RS)
            rs = avg_gain / avg_loss
            # Compute the RSI
            rsi = 100 - (100 / (1 + rs))

        self.df["rsi"] = rsi

        return self.df
