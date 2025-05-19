from ib_insync import Stock

from src.controllers.controller import Controller


class StockController(Controller):

    def __int__(self):
        super().__init__()
        self.silent_wealth_inputs = None
        self.currency = None

    def validate(self, silent_wealth_inputs):
        self.silent_wealth_inputs = silent_wealth_inputs
        if self.silent_wealth_inputs.exchange == "LSE":
            self.currency = "GBP"
        else:
            self.currency = "USD"

    def run(self):
        ib = super()._connect_to_ib(self.silent_wealth_inputs.account)

        stock = Stock(symbol=self.silent_wealth_inputs.ticker_name, exchange=self.silent_wealth_inputs.exchange, currency=self.currency)

        ib.disconnect()
        exit()
