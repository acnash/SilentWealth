import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_forward_returns(data, days=[1, 3, 5]):
    forward_returns = {}
    for d in days:
        # Subtracting 1 gives percent return
        forward_returns[f'{d}d'] = data['Close'].shift(-d) / data['Close'] - 1
    return pd.DataFrame(forward_returns, index=data.index)

def download_with_retry(ticker, start, end, retries=5, delay=15):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                raise ValueError("No data downloaded.")
            return data
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                raise
    return None

ticker = "NVDA"
start_date = "2025-01-01"
end_date = "2025-07-28"

data = download_with_retry(ticker, start_date, end_date)

if data.empty:
    raise ValueError(f"No data downloaded for {ticker}. Check internet connection or ticker symbol.")

data['RSI'] = calculate_rsi(data)

# Drop NaN rows caused by RSI calculation
data.dropna(subset=['RSI'], inplace=True)

# Add forward returns
forward_returns = get_forward_returns(data)
data = pd.concat([data, forward_returns], axis=1)

# Filter where RSI > 70
rsi_threshold = 70
high_rsi = data[data['RSI'] > rsi_threshold]

# Plot histogram of forward returns
plt.figure(figsize=(12, 6))
for i, col in enumerate(['1d', '3d', '5d']):
    plt.subplot(1, 3, i+1)
    plt.hist(high_rsi[col] * 100, bins=30, edgecolor='black')
    plt.title(f'Forward Return after RSI > {rsi_threshold} ({col})')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.show()
