from typing import Dict


class SilentWealthInputs:

    def __init__(self, yaml_inputs: Dict):
        try:
            account_data = yaml_inputs["account_data"]
        except KeyError:
            print("Error: account_data not provided in YAML input file. Exiting.")
            exit()
        self.account = account_data.get("account", "paper")

        try:
            stock = yaml_inputs["stock"]
        except KeyError:
            print("Error: stock not provided in YAML input file. Exiting.")
            exit()
        try:
            self.ticker_name = stock["ticker_name"]
            self.exchange = stock["exchange"]
            self.frame_size = stock["frame_size"]
            if self.ticker_name == "BTC":
                self.dollar_amount = stock["dollar_amount"]
            else:
                self.quantity = stock["quantity"]
        except KeyError:
            print("Error: incorrect entry in stock")
            exit()

        try:
            buy_sell_conditions = yaml_inputs["buy_sell_conditions"]
        except KeyError:
            print("Error: buy_sell_conditions not provided in YAML input file. Exiting.")
            exit()
        try:
            self.purchase_type = buy_sell_conditions.get("purchase_type", "market_order")
            self.stop_loss = buy_sell_conditions.get("stop_loss", 0)
            self.take_profit = buy_sell_conditions.get("take_profit", 0)
        except KeyError:
            print("Error: incorrect entry in buy_sell_conditions. Exiting.")
            exit()

        try:
            monitor_conditions = yaml_inputs["monitor_conditions"]
        except KeyError:
            print("Error: monitor_conditions not provided in YAML input file. Exiting.")
            exit()
        try:
            self.ema_short = monitor_conditions.get("ema_short", 9)
            self.ema_medium = monitor_conditions.get("ema_medium", 20)
            self.ema_long = monitor_conditions.get("ema_long", 200)
            self.vwap = monitor_conditions.get("vwap", 9)
            self.rsi_period = monitor_conditions.get("rsi_period", 14)
            self.anchor_distance = monitor_conditions.get("anchor_distance", 0)
        except KeyError:
            print("Error: incorrect entry in monitor_conditions. Exiting.")
