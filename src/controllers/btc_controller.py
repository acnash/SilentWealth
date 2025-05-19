import schedule

from ib_insync import Contract

from src.controllers.controller import Controller


class BTCController(Controller):

    def __int__(self, silent_wealth_inputs):
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
        self.rsi_period = self.silent_wealth_inputs.rsi_period
        self.anchor_distance = self.silent_wealth_inputs.anchor_distance

    def validate(self):
        return True

    def run(self):
        ib = super()._connect_to_ib(self.silent_wealth_inputs.account)

        stock = Contract(secType='CRYPTO', symbol='BTC', exchange='PAXOS', currency='USD')

        ib.qualifyContracts(stock)
        ib.reqMktData(stock)
        ib.sleep(2)
        ticker = ib.ticker(stock)

        schedule.every(frame_size).minutes.do(self._scheduled_task,
                                              ib,
                                              stock,
                                              ticker_name,
                                              quantity,
                                              frame_size,
                                              stop_loss_percent,
                                              dollar_amount,
                                              start_time,
                                              stop_time,
                                              close_time,
                                              ema_short,
                                              ema_medium,
                                              ema_long,
                                              vwap,
                                              rsi_period,
                                              take_profit,
                                              limit_order,
                                              anchor_distance)



        ib.disconnect()
        exit()
