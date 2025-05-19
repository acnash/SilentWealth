from ib_insync import Contract

from src.controllers.controller import Controller

class BTCController(Controller):

    def __int__(self, silent_wealth_inputs):
        super().__init__()
        self.silent_wealth_inputs = None

    def validate(self, silent_wealth_inputs):
        self.silent_wealth_inputs = silent_wealth_inputs

    def run(self):
        ib = super()._connect_to_ib(self.silent_wealth_inputs.account)

        stock = Contract(secType='CRYPTO', symbol='BTC', exchange='PAXOS', currency='USD')

        ib.disconnect()
        exit()
