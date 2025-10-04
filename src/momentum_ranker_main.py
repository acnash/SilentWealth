import random
from ib_insync import IB
from src.controllers.historial_data_controller import HistorialDataController


def main():
    IBG_PAPER_PORT = 4002
    #IBG_LIVE_PORT = 4001

    client_id = random.randint(1, 9999)
    ib = IB()
    ib.connect('127.0.0.1', IBG_PAPER_PORT, clientId=client_id)

    tickers = ["NVDA",
               "MSFT",
               "AAPL",
               "AMZN",
               "GOOGL",
               "GOOG",
               "META",
               "TSLA",
               "AVGO",
               "NFLX",
               "COST",
               "ASML",
               "TMUS",
               "CSCO",
               "ISRG",
               "AZN",
               "LIN",
               "PEP",
               "AMD",
               "QCOM"]

    historial_data_controller = HistorialDataController(tickers)
    momentum_df = historial_data_controller.get_n_day_momentum(ib, 5)
    print(momentum_df)

if __name__ == "__main__":
    main()
