#!/usr/bin/env python3
"""Build inventories for raw acoustic data.

See `./config/inventory.toml` for configuration options.
"""

import argparse
import logging
from pathlib import Path
import tomllib

import numpy as np
from tritonoa.data.inventory import Inventory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import ClockParameters

from vineyard.config import get_path


def main(config_file: Path) -> None:
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    for key in config:
        logging.info(f"Building inventory for {key}...")

        data_cfg = config[key]["data"]
        clock_cfg = config[key].get("clock", None)
        hyd_cfg = config[key].get("hydrophone", None)

        if clock_cfg is None:
            clock_params = ClockParameters()
        else:
            time_check_0 = np.datetime64(clock_cfg["time_check_0"])
            time_check_1 = np.datetime64(clock_cfg["time_check_1"])
            clock_params = ClockParameters(
                time_check_0=time_check_0,
                time_check_1=time_check_1,
                offset_0=clock_cfg["offset_0"],
                offset_1=clock_cfg["offset_1"],
            )

        if hyd_cfg is None:
            signal_params = SignalParams()
        else:
            signal_params = SignalParams(
                gain=hyd_cfg["fixed_gain"],
                sensitivity=hyd_cfg["sensitivity"],
            )

        inv = Inventory()
        inv.build(
            dataset_path=data_cfg["directory"],
            glob_pattern=data_cfg["glob_pattern"],
            clock_params=clock_params,
            conditioner=signal_params,
            file_format=data_cfg.get("file_format", None),
        )
        savepath = Path(data_cfg["destination"])
        inv.save(savepath)

        logging.info(f"Inventory saved to {savepath.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save inventories")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file",
        default=get_path("inventory_config"),
    )
    args = parser.parse_args()
    main(args.config)
