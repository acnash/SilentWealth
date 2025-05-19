from typing import Dict


class SilentWealthInputs:

    def __init__(self, yaml_inputs: Dict):
        print("Setting input parameters")
        print("-------------------------------------------------------------------")
        try:
            account_data = yaml_inputs["account_data"]
            print(f"Account data...")
        except KeyError:
            print("Error: account_data not provided in YAML input file. Exiting.")
            exit()
        self.account = account_data.get("account", "paper")
        print(f"...using a {self.account} account.\n")

        try:
            stock = yaml_inputs["stock"]
            print(f"Stock data...")
        except KeyError:
            print("Error: stock not provided in YAML input file. Exiting.")
            exit()
        try:
            self.ticker_name = stock["ticker_name"]
            print(f"...trading {self.ticker_name}.")
            self.exchange = stock["exchange"]
            print(f"...on the {self.exchange} exchange.")
            self.frame_size = stock["frame_size"]
            print(f"...resolution size {self.frame_size}.")
            if self.ticker_name == "BTC" or self.ticker_name == "SOL" or self.ticker_name == "ETH":
                self.dollar_amount = stock["dollar_amount"]
                print(f"...trading with {self.dollar_amount} for bitcoin.")
            else:
                self.quantity = stock["quantity"]
                print(f"...trading {self.quantity} amount of stock.\n")
        except KeyError:
            print("Error: incorrect entry in stock. Exiting.")
            exit()

        try:
            buy_sell_conditions = yaml_inputs["buy_sell_conditions"]
            print("Buy/Sell conditions...")
        except KeyError:
            print("Error: buy_sell_conditions not provided in YAML input file. Exiting.")
            exit()
        try:
            self.purchase_type = buy_sell_conditions.get("purchase_type", "market_order")
            print(f"...looking to place {self.purchase_type}")
            self.stop_loss = buy_sell_conditions.get("stop_loss", 0)
            print(f"...setting stop loss to {self.stop_loss}.")
            self.take_profit = buy_sell_conditions.get("take_profit", 0)
            print(f"...setting take profit to {self.take_profit}.\n")
        except KeyError:
            print("Error: incorrect entry in buy_sell_conditions. Exiting.")
            exit()

        try:
            monitor_conditions = yaml_inputs["monitor_conditions"]
            print("Monitor conditions...")
        except KeyError:
            print("Error: monitor_conditions not provided in YAML input file. Exiting.")
            exit()
        try:
            self.ema_short = monitor_conditions.get("ema_short", 9)
            print(f"...setting EMA short {self.ema_short}.")
            self.ema_medium = monitor_conditions.get("ema_medium", 20)
            print(f"...setting EMA medium {self.ema_medium}.")
            self.ema_long = monitor_conditions.get("ema_long", 200)
            print(f"...setting EMA long {self.ema_long}.")
            self.vwap = monitor_conditions.get("vwap", 9)
            print(f"...setting vwap {self.vwap}.")
            self.rsi = monitor_conditions.get("rsi", 14)
            print(f"...setting RSI {self.rsi}.")
            self.anchor_distance = monitor_conditions.get("anchor_distance", 0)
            print(f"...setting anchor distance {self.anchor_distance}.\n")
        except KeyError:
            print("Error: incorrect entry in monitor_conditions. Exiting.")
