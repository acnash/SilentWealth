import argparse

import yaml

from src.controllers.btc_controller import BTCController
from src.controllers.stock_controller import StockController
from src.models.silent_wealth_inputs import SilentWealthInputs


def main():
    parser = argparse.ArgumentParser(description="Silent Wealth")

    parser.add_argument("--input", type=str, required=True,
                        help="YAML input file.")
    args = parser.parse_args()
    input_yaml_file = args.input
    with open(input_yaml_file, "r") as file:
        yaml_inputs = yaml.safe_load(file)

    silent_wealth_inputs = SilentWealthInputs(yaml_inputs)

    if silent_wealth_inputs.ticker_name == "BTC":
        controller = BTCController(silent_wealth_inputs)
    else:
        controller = StockController(silent_wealth_inputs)

    validated = controller.validate()

    if validated:
        controller.run()

if __name__ == "__main__":
    main()
