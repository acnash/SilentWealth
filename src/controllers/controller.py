from abc import ABC, abstractmethod
from random import random

from ib_insync import IB


class Controller(ABC):
    PAPER_PORT = 7497
    LIVE_PORT = 7496

    def __int__(self):
        pass

    def _connect_to_ib(self, account):
        client_id = random.randint(1, 9999)
        ib = IB()
        if account == "paper":
            ib.connect('127.0.0.1', Controller.PAPER_PORT, clientId=client_id)
        elif account == "live":
            ib.connect('127.0.0.1', Controller.LIVE_PORT, clientId=client_id)
        return ib

    @abstractmethod
    def validate(self, silent_wealth_inputs):
        pass

    @abstractmethod
    def run(self):
        pass
