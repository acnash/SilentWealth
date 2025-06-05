from typing import Dict


class SilentWealthInputs:

    TW_PAPER_PORT = 7497
    TW_LIVE_PORT = 7496

    IBG_PAPER_PORT = 4002
    IBG_LIVE_PORT = 4001

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
        self.account = self.account.lower()
        print(f"...using a {self.account} account.")

        self.platform = account_data.get("platform", "ibg")
        self.platform = self.platform.lower()
        print(f"...using platform {self.platform}.")

        if self.platform == "ibg" and self.account == "paper":
            self.port = SilentWealthInputs.IBG_PAPER_PORT
        elif self.platform == "ibg" and self.account == "live":
            self.port = SilentWealthInputs.IBG_LIVE_PORT
        elif self.platform == "tw" and self.account == "paper":
            self.port = SilentWealthInputs.TW_PAPER_PORT
        elif self.platform == "tw" and self.account == "live":
            self.port = SilentWealthInputs.TW_LIVE_PORT
        else:
            print(f"Error: platform {self.platform} and account {self.account} not recognised. Cannot assign port. Exiting.")
            exit()
        print(f"...using port {self.port}.\n")

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
            self.unit_type = stock["unit_type"]
            print(f"...unit type {self.unit_type}")
            self.commission_pot = stock.get("commission_pot", 200)
            print(f"...commission pot size ${self.commission_pot}")
            if self.ticker_name == "BTC" or self.ticker_name == "SOL" or self.ticker_name == "ETH":
                self.dollar_amount = stock["dollar_amount"]
                print(f"...trading with {self.dollar_amount} for bitcoin.\n")
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
            self.vwap = monitor_conditions.get("vwap", 0)
            print(f"...setting vwap {self.vwap}.")
            self.rsi = monitor_conditions.get("rsi", 0)
            print(f"...setting RSI {self.rsi}.")
            self.rsi_top = monitor_conditions.get("rsi_top", 0)
            print(f"...setting RSI top condition {self.rsi_top}.")
            self.rsi_bottom = monitor_conditions.get("rsi_bottom", 0)
            print(f"...setting RSI bottom condition {self.rsi_bottom}.")
            self.atr = monitor_conditions.get("atr", 0)
            print(f"...setting ATR period {self.atr}\n")
        except KeyError:
            print("Error: incorrect entry in monitor_conditions. Exiting.")
            exit()

        self.debug = yaml_inputs.get("debug")
        if self.debug:
            print("Loading debug YAML options.")
            self.output_data = self.debug["output_data"]
            print(f"...output_data file location {self.output_data}")
            self.test_mode = self.debug.get("test_mode", False)
            print(f"...test_mode {self.test_mode}")
            self.test_data = self.debug.get("test_data", None)
            print(f"...loading test data: {self.test_data}\n")
            self.bootstrap_data = self.debug.get("bootstrap_data", None)
            print(f"...loading bootstrap data: {self.bootstrap_data}")
            if self.bootstrap_data:
                try:
                    self.bootstrap_sample_size = self.debug["bootstrap_sample_size"]
                    print(f"...size of bootstrap sample: {self.bootstrap_sample_size}")
                    self.bootstrap_number_samples = self.debug["bootstrap_number_samples"]
                    print(f"...bootstrap number of samples: {self.bootstrap_number_samples}")
                except KeyError:
                    print("Error: incorrect entries for bootstrap variables. Exiting")
                    exit()

    def prep_bootstrap(self):
        self.ema_short = [9, 12, 14]
        self.ema_medium = [16, 18, 21]
        self.ema_long = [100, 150]

        self.rsi = [9, 12, 14]
        #self.rsi_top = [70]
        #self.rsi_bottom = [50]

        #self.vwap = [9, 12, 14]
        self.atr = [9, 12, 14]


