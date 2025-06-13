#!/usr/bin/env python
"""Plot SHRU data.

Suggested times:
VLA 1:
- Inventory: 2023-12-01T21:26:00 to 2023-12-02T01:13:00
- Fin whale: 2023-12-01T21:26:00 to 2023-12-01T23:19:00

VLA 2:
- Inventory: 2023-12-01T21:44:00 to 2023-12-02T01:31:00
- Fin whale: 2023-12-01T21:26:00 to 2023-12-01T23:19:00
"""

from argparse import ArgumentParser
import logging

import dotenv
import matplotlib.pyplot as plt
import numpy as np
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION, convert_datetime64_to_string
from tritonoa.data.reader import read_inventory

from vineyard.config import get_path
from vineyard.plotting import plot_shru_pectrograms, savefig_kwargs

dotenv.load_dotenv()

STFT_PARAMS = {
    "nperseg": 64,
    "noverlap": 60,
    "nfft": 2**8,
}


def condition_data(ds: DataStream, decimation_factor: int = 40) -> DataStream:
    ds_filt = ds.copy()
    ds_filt.decimate(decimation_factor)
    # ds_filt.filter(
    #     filt_type="highpass",
    #     freq=1.0,
    # )
    return ds_filt


def main(
    sensor: str,
    start: str,
    end: str,
    decimation: int,
    multi: bool,
    interval: float,
    savefig: bool,
) -> None:

    inv = get_path(f"{sensor}_inventory")
    start = np.datetime64(start, TIME_PRECISION)
    end = np.datetime64(end, TIME_PRECISION)

    if multi:
        interval = np.timedelta64(interval, "s")
    else:
        interval = end - start

    for start_time in np.arange(start, end, interval):
        end_time = start_time + interval
        logging.info(f"Plotting {sensor.upper()} data from {start_time} to {end_time}")
        ds = condition_data(
            read_inventory(inv, start_time, end_time), decimation_factor=decimation
        )

        start_str = convert_datetime64_to_string(start_time)
        end_str = convert_datetime64_to_string(end_time)
        start_str_read = convert_datetime64_to_string(start_time, readable=True)
        end_str_read = convert_datetime64_to_string(end_time, readable=True)

        fig = plot_shru_pectrograms(ds, **STFT_PARAMS)
        fig.suptitle(
            f"{sensor.upper()} Data from {start_str_read}Z to {end_str_read}Z",
            fontsize=12,
        )
        if savefig:
            fname = f"{sensor}_{start_str}-{end_str}.png"
            fig.savefig(get_path(f"{sensor}_data_view") / fname, **savefig_kwargs)
            plt.close(fig)
            continue
        if multi:
            plt.draw()

    if not savefig:
        plt.show()


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Plot SHRU data from a given inventory.")
    parser.add_argument(
        "--sensor",
        type=str,
        default="vla2",
        choices=["vla1", "vla2"],
        help="Sensor to plot data for. Default is 'vla1'.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-12-01T21:44:00",
        help="Start time of the data to extract.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-02T01:31:00",
        help="End time of the data to extract.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=40,
        help="Decimation factor for the data.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        default=True,
        help="Generate multiple figures with a specified interval.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval in seconds for generating multiple figures.",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        default=True,
        help="Save the figure instead of showing it.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(
        args.sensor,
        args.start,
        args.end,
        args.decimation,
        args.multi,
        args.interval,
        args.savefig,
    )
