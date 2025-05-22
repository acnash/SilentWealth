from datetime import datetime

import schedule
import time
import traceback

from ib_insync import Contract

from src.controllers.controller import Controller


class BTCController(Controller):

    BTC_PAXOS_START_TIME = "00:10" # start trading if connected (running)
    BTC_PAXOS_END_TIME = "23:45"  # force a sale if we're holding, close connection, then exit

    def __init__(self, silent_wealth_inputs):
        super().__init__()
        self.silent_wealth_inputs = silent_wealth_inputs
        self.account = self.silent_wealth_inputs.account
        self.platform = self.silent_wealth_inputs.platform
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
        self.output_data = self.silent_wealth_inputs.output_data

    def validate(self):
        return True

    def run(self):
        ib = super()._connect_to_ib(self.silent_wealth_inputs.port)

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
                                                   self.rsi,
                                                   self.output_data)

        try:
            start_time = datetime.strptime(BTCController.BTC_PAXOS_START_TIME, "%H:%M").time()
            end_time = datetime.strptime(BTCController.BTC_PAXOS_END_TIME, "%H:%M").time()
            while True:
                current_time = datetime.now().time()
                if start_time <= current_time <= end_time:
                    schedule.run_pending()
                    time.sleep(1)  # Sleep to prevent CPU overuse
                else:
                    print(f"Closing. Outside of trading hours and the {self.platform} platform needs to reset.")
                    ib.disconnect()
                    exit()
        except Exception as e:
            print(f"Error: Scheduler stopped due to unknown error. {e}")
            traceback.print_exc()
        finally:
            ib.disconnect()
