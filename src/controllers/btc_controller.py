from datetime import datetime

import schedule
import time
import traceback

from ib_insync import Contract

from src.controllers.controller import Controller
from src.testing.bootstrap import Bootstrap


class BTCController(Controller):

    BTC_PAXOS_START_TIME = "00:01" # start trading if connected (running)
    BTC_PAXOS_END_TIME = "23:45"  # force a sale if we're holding, close connection, then exit

    def __init__(self, silent_wealth_inputs):
        super().__init__()
        self.silent_wealth_inputs = silent_wealth_inputs
        self.account = self.silent_wealth_inputs.account
        self.platform = self.silent_wealth_inputs.platform
        self.ticker_name = self.silent_wealth_inputs.ticker_name
        self.exchange = self.silent_wealth_inputs.exchange
        self.frame_size = self.silent_wealth_inputs.frame_size
        self.unit_type = self.silent_wealth_inputs.unit_type
        self.dollar_amount = self.silent_wealth_inputs.dollar_amount
        self.commission_pot_child = self.silent_wealth_inputs.commission_pot
        self.purchase_type = self.silent_wealth_inputs.purchase_type
        self.stop_loss = self.silent_wealth_inputs.stop_loss
        self.take_profit = self.silent_wealth_inputs.take_profit
        self.ema_short = self.silent_wealth_inputs.ema_short
        self.ema_medium = self.silent_wealth_inputs.ema_medium
        self.ema_long = self.silent_wealth_inputs.ema_long
        self.vwap = self.silent_wealth_inputs.vwap
        self.rsi = self.silent_wealth_inputs.rsi
        self.rsi_top = self.silent_wealth_inputs.rsi_top
        self.rsi_bottom = self.silent_wealth_inputs.rsi_bottom
        self.atr = self.silent_wealth_inputs.atr
        self.output_data = self.silent_wealth_inputs.output_data
        self.test_mode = self.silent_wealth_inputs.test_mode
        self.test_data = self.silent_wealth_inputs.test_data

        self.bootstrap_data = self.silent_wealth_inputs.bootstrap_data
        if self.bootstrap_data:
            self.bootstrap_sample_size = self.silent_wealth_inputs.bootstrap_sample_size
            self.bootstrap_number_samples = self.silent_wealth_inputs.bootstrap_number_samples



    def validate(self):
        return True

    def run(self):
        if self.test_mode:
            ib = None
            contract = None
        else:
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
                                                   self.unit_type,
                                                   self.dollar_amount,
                                                   self.commission_pot_child,
                                                   BTCController.BTC_PAXOS_START_TIME,
                                                   BTCController.BTC_PAXOS_END_TIME,
                                                   self.ema_short,
                                                   self.ema_medium,
                                                   self.ema_long,
                                                   self.vwap,
                                                   self.rsi,
                                                   self.rsi_top,
                                                   self.rsi_bottom,
                                                   self.atr,
                                                   self.output_data,
                                                   self.test_mode,
                                                   self.test_data,
                                                   None)

        if self.test_mode:
            # for testing only
            #set bootstrap data if it's available
            if self.bootstrap_data:
                bootstrap = Bootstrap(self.bootstrap_data, self.bootstrap_sample_size, self.bootstrap_number_samples)
                self.silent_wealth_inputs.prep_bootstrap()
                for ema_short in self.silent_wealth_inputs.ema_short:
                    for ema_medium in self.silent_wealth_inputs.ema_medium:
                        for ema_long in self.silent_wealth_inputs.ema_long:
                            for rsi in self.silent_wealth_inputs.rsi:
                                for rsi_top in self.silent_wealth_inputs.rsi_top:
                                    for rsi_bottom in self.silent_wealth_inputs.rsi_bottom:
                                        for atr in self.silent_wealth_inputs.atr:
                                            print("************************************************************")
                                            print(f"ema_short: {ema_short}")
                                            print(f"ema_medium: {ema_medium}")
                                            print(f"ema_long: {ema_long}")
                                            print(f"rsi: {rsi}")
                                            print(f"rsi_top: {rsi_top}")
                                            print(f"rsi_bottom: {rsi_bottom}")
                                            print(f"atr: {atr}")
                                            self._scheduled_task(
                                                None,
                                                None,
                                                self.ticker_name,
                                                None,
                                                self.frame_size,
                                                self.unit_type,
                                                self.dollar_amount,
                                                self.commission_pot_child,
                                                BTCController.BTC_PAXOS_START_TIME,
                                                BTCController.BTC_PAXOS_END_TIME,
                                                ema_short,
                                                ema_medium,
                                                ema_long,
                                                self.vwap,
                                                rsi,
                                                rsi_top,
                                                rsi_bottom,
                                                atr,
                                                self.output_data,
                                                self.test_mode,
                                                self.test_data,
                                                bootstrap)

            else:
                self._scheduled_task(
                    None,
                    None,
                    self.ticker_name,
                    None,
                    self.frame_size,
                    self.unit_type,
                    self.dollar_amount,
                    self.commission_pot_child,
                    BTCController.BTC_PAXOS_START_TIME,
                    BTCController.BTC_PAXOS_END_TIME,
                    self.ema_short,
                    self.ema_medium,
                    self.ema_long,
                    self.vwap,
                    self.rsi,
                    self.rsi_top,
                    self.rsi_bottom,
                    self.atr,
                    self.output_data,
                    self.test_mode,
                    self.test_data,
                    None)
        else:
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
                if not self.test_mode:
                    ib.disconnect()
