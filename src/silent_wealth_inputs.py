from typing import Dict


class SilentWealthInputs:

    def __init__(self, yaml_inputs: Dict):
        try:
            account_data = yaml_inputs["account_data"]
        except KeyError:
            print("Error: account_data not provided in YAML input file. Exiting.")
            exit()
        self.account = account_data.get("account", "paper")  # always returns paper if no value given

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
            self.stop_loss = buy_sell_conditions.get("")
        except KeyError:
            print("Error: incorrect entry in buy_sell_conditions")
            exit()

        try:
            monitor_conditions = yaml_inputs["monitor_conditions"]
        except KeyError:
            print("Error: monitor_conditions not provided in YAML input file. Exiting.")
            exit()

buy_sell_conditions:
purchase_type: limit_order
stop_loss: 0
take_profit: 0
limit_order: False

monitor_conditions:
ema_short: 9
ema_medium: 20
ema_long: 200
vwap: 9
rsi_period: 14
anchor_distance: 0

# ticker_name = args.ticker_name
# exchange = args.exchange
# quantity = args.quantity
# frame_size = args.frame_size
# account = args.account.lower()
# stop_loss_percent = args.stop_loss
# dollar_amount = args.dollar_amount
# ema_short = args.ema_short
# ema_medium = args.ema_medium
# ema_long = args.ema_long
# vwap = args.vwap
# rsi_period = args.rsi_period
# take_profit = args.take_profit
# limit_order = args.limit_order
# anchor_distance = args.anchor_distance