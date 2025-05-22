import schedule

from ib_insync import Stock

from src.controllers.controller import Controller


class StockController(Controller):

    NYSE_START_TIME = "14:30"
    NYSE_END_TIME = "20:30"
    NYSE_CLOSE_TIME = "20:58"

    def __init__(self, silent_wealth_inputs):
        super().__init__()
        self.silent_wealth_inputs = silent_wealth_inputs
        self.account = self.silent_wealth_inputs.account
        self.platform = self.silent_wealth_inputs.platform
        self.ticker_name = self.silent_wealth_inputs.ticker_name
        self.exchange = self.silent_wealth_inputs.exchange
        self.frame_size = self.silent_wealth_inputs.frame_size
        self.quantity = self.silent_wealth_inputs.quantity
        self.purchase_type = self.silent_wealth_inputs.purchase_type
        self.stop_loss = self.silent_wealth_inputs.stop_loss
        self.take_profit = self.silent_wealth_inputs.take_profit
        self.ema_short = self.silent_wealth_inputs.ema_short
        self.ema_medium = self.silent_wealth_inputs.ema_medium
        self.ema_long = self.silent_wealth_inputs.ema_long
        self.vwap = self.silent_wealth_inputs.vwap
        self.rsi_period = self.silent_wealth_inputs.rsi_period
        self.anchor_distance = self.silent_wealth_inputs.anchor_distance
        self.output_data = self.silent_wealth_inputs.output_data

        if self.silent_wealth_inputs.exchange == "LSE":
            self.currency = "GBP"
        else:
            self.currency = "USD"

    def validate(self):
        return True

    def run(self):
        ib = super()._connect_to_ib(self.silent_wealth_inputs.account)

        stock = Stock(symbol=self.silent_wealth_inputs.ticker_name, exchange=self.silent_wealth_inputs.exchange, currency=self.currency)

        ib.qualifyContracts(stock)
        ib.reqMktData(stock)
        ib.sleep(2)
        ticker = ib.ticker(stock)

        ib.disconnect()
        exit()
