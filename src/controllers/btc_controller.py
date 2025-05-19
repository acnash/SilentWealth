import schedule
import time
import traceback

from ib_insync import Contract

from src.controllers.controller import Controller


class BTCController(Controller):
    # BTC_PAXOS_START_TIME = "08:10"
    # BTC_PAXOS_END_TIME = "20:00"
    # BTC_PAXOS_CLOSE_TIME = "20:30"

    def __init__(self, silent_wealth_inputs):
        super().__init__()
        self.silent_wealth_inputs = silent_wealth_inputs
        self.account = self.silent_wealth_inputs.account
        self.ticker_name = self.silent_wealth_inputs.ticker_name
        self.exchange = self.silent_wealth_inputs.exchange
        self.frame_size = self.silent_wealth_inputs.frame_size
        self.dollar_amount = self.silent_wealth_inputs.dollar_amount
        self.purchase_type = self.silent_wealth_inputs.purchase_type
        self.stop_loss = self.silent_wealth_inputs.stop_loss
        self.take_profit = self.silent_wealth_inputs.take_profit
        self.ema_short = self.silent_wealth_inputs.ema_short
        self.ema_medium = self.silent_wealth_inputs.ema_medium
        self.ema_long = self.silent_wealth_inputs.ema_long
        self.vwap = self.silent_wealth_inputs.vwap
        self.rsi = self.silent_wealth_inputs.rsi
        self.anchor_distance = self.silent_wealth_inputs.anchor_distance

        # self.start_time = BTCController.BTC_PAXOS_START_TIME
        # self.stop_time = BTCController.BTC_PAXOS_END_TIME
        # self.close_time = BTCController.BTC_PAXOS_CLOSE_TIME

    def validate(self):
        return True

    def run(self):
        ib = super()._connect_to_ib(self.silent_wealth_inputs.account)

        contract = Contract(secType='CRYPTO', symbol=self.ticker_name, exchange=self.exchange, currency='USD')

        ib.qualifyContracts(contract)
        ib.reqMktData(contract)
        ib.sleep(2)
        ticker = ib.ticker(contract)

        if not ticker.last or not ticker.close:
            print("Error: Cannot determine market price. Possible connection issue.")
            ib.disconnect()
            exit()

        schedule.every(self.frame_size).minutes.do(self._scheduled_task,
                                                   ib,
                                                   contract,
                                                   self.ticker_name,
                                                   None,
                                                   self.frame_size,
                                                   self.dollar_amount,
                                                   None,
                                                   None,
                                                   None,
                                                   self.ema_short,
                                                   self.ema_medium,
                                                   self.ema_long,
                                                   self.vwap,
                                                   self.rsi)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)  # Sleep to prevent CPU overuse
        except Exception as e:
            print(f"Scheduler stopped due to unknown error. {e}")
            traceback.print_exc()
        finally:
            ib.disconnect()
