from ib_insync import *
import pandas as pd
import time
import yfinance as yf

class HistorialDataController():

    def __init__(self, tickers, exchanges=None):
        self.tickers = tickers
        self.exchanges = exchanges

    def get_fyahoo_postmarket_change(self):
        results = {}
        #data = yf.download(self.tickers, period="1d")
        #for ticker in self.tickers:
        #    ticker_fyahoo = yf.Ticker(ticker)
        #    info = ticker_fyahoo.info

        #    time.sleep(2)
        #    results[ticker] = info.get("postMarketChangePercent")
        batch = yf.Tickers(' '.join(self.tickers))

        for ticker in self.tickers:
            info = batch.tickers[ticker].info
            results[ticker] = info.get('postMarketChangePercent', None)

        return results

    def get_fyahoo_marketcap(self):
        results = {}
        for ticker in self.tickers:
            ticker_fyahoo = yf.Ticker(ticker)
            info = ticker_fyahoo.info
            time.sleep(1)
            results[ticker] = info.get("marketCap")
        return results

    def get_n_day_momentum(self, ib, n_days):
        results = []
        for ticker in self.tickers:
            try:
                contract = Stock(ticker, "NASDAQ", "USD")
                momentum = self.__get_historical_data(ib, contract, n_days)
                results.append({'Ticker': ticker, '5-Day Percent Momentum': momentum})
                time.sleep(1.5)
            except Exception as e:
                print(f"Something went terribly wrong.")
                exit()
        momentum_df = pd.DataFrame(results)
        momentum_df_sorted = momentum_df.sort_values(by='5-Day Percent Momentum', ascending=False)
        momentum_df_sorted = momentum_df_sorted.reset_index(drop=True)

        return momentum_df_sorted

    def get_last_extended_prices(self, ib):
        results = []
        for ticker, exchange in zip(self.tickers, self.exchanges):
            try:
                contract = Stock(ticker, exchange, "USD")
                print(f"{ticker}:{exchange}")
                df = self.__get_extended_historical_data(ib, contract)
                latest_bar = df['close'].iloc[-1]
                results.append(latest_bar)
            except Exception as e:
                print(f"Something went terribly wrong.")
                exit()

        return results

    def __get_extended_historical_data(self, ib, contract):
        # only works if I'm paying for the prescription
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',  # last 2 days to be sure
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,  # include extended hours
            formatDate=1
        )

        return util.df(bars)

    def __get_historical_data(self, ib, contract, duration):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{duration} D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        # Convert to DataFrame
        df = util.df(bars)

        # Drop today’s partial bar if still forming
        df = df[df['close'] > 0]

        if len(df) >= 5:
            start_price = df['close'].iloc[-5]
            end_price = df['close'].iloc[-1]
            percent_momentum = ((end_price - start_price) / start_price) * 100
            return percent_momentum
        else:
            return None
