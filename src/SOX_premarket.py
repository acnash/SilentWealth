from ib_insync import *
import pandas as pd
import time

from src.controllers.historial_data_controller import HistorialDataController


def main():

    #print("Pre-market:", info.get("preMarketPrice"))
    #print("After-hours:", info.get("postMarketPrice"))
    #print("Change %:", info.get("postMarketChangePercent"))

    tickers = [
        "AVGO", "NVDA", "TXN", "AMD", "QCOM", "MU", "KLAC", "MPWR", "LRCX", "MCHP",
        "AMAT", "TSM", "ADI", "ASML", "NXPI", "INTC", "MRVL", "ON", "TER", "SWKS",
        "ENTG", "QRVO", "STM", "OLED", "LSCC", "UMC", "ONTO", "ARM"
    ]

    historical_data_controller = HistorialDataController(tickers)
    change_dict = historical_data_controller.get_fyahoo_postmarket_change()
    marketcap_dict = historical_data_controller.get_fyahoo_marketcap()

    normalized_changes = {}  # Initialize an empty dictionary

    for ticker, change in change_dict.items():
        if not change:
            continue
        # Check if ticker exists in marketcap_dict and marketcap is not zero
        if ticker in marketcap_dict and marketcap_dict[ticker] != 0:
            # Calculate normalized change by dividing change by marketcap
            normalized_change = change / marketcap_dict[ticker]
            # Add result to the dictionary
            normalized_changes[ticker] = normalized_change

    #normalized_changes = {
    #    ticker: change / marketcap_dict[ticker]
    #    for ticker, change in change_dict.items()
    #    if ticker in marketcap_dict and marketcap_dict[ticker] != 0
    #}

    # Total market cap of all tracked stocks
    total_market_cap = sum(marketcap_dict[ticker] for ticker in normalized_changes)

    # Weighted signal
    weighted_signal = sum(
        (change * marketcap_dict[ticker]) / total_market_cap
        for ticker, change in normalized_changes.items()
    )

    influence = {
        ticker: abs(change * marketcap_dict[ticker])
        for ticker, change in normalized_changes.items()
    }

    most_influential = sorted(influence.items(), key=lambda x: x[1], reverse=True)
    print("Change")
    print(change_dict)
    print("-------------------------------\n")
    print("Normalised change")
    print(normalized_changes)
    print("-------------------------------\n")
    print("Most influential stock")
    print(most_influential)
    print("-------------------------------\n")
    print("Weighted signal: Positive --> Bullish || Negative --> Bearish")
    print(weighted_signal)

    #print(change_dict)

if __name__ == "__main__":
    main()
