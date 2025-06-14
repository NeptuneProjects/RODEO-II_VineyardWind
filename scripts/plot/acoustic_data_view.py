#!/usr/bin/env python
"""Plot acoustic data (time series & spectrograms) from 3DVHA, VLA1, and VLA2.

Times of interest:
3DVHA:
VLA 1: 2023-12-01T21:26:00 to 2023-12-02T01:13:00
VLA 2: 2023-12-01T21:44:00 to 2023-12-02T01:31:00
Fin whale: 2023-12-01T21:26:00 to 2023-12-01T23:19:00
"""

from argparse import ArgumentParser
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Callable

import dotenv
import matplotlib.pyplot as plt
import numpy as np
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION, convert_datetime64_to_string
from tritonoa.data.reader import read_inventory

from vineyard.config import get_path
from vineyard.plotting import (
    plot_3dvha_spectrograms,
    plot_shru_spectrograms,
    savefig_kwargs,
)

dotenv.load_dotenv()

SENSORS = ["3dvha", "vla1", "vla2"]
nperseg = 256
STFT_PARAMS = {
    "nperseg": nperseg,
    "hop": int(0.05 * nperseg),
    "nfft": 2**12,
    "fmin": 0.0,
    "fmax": 250.0,
}


def condition_data(
    ds: DataStream,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> DataStream:
    if target_fs is None and filt_type is None and filt_freq is None:
        return ds

    ds_filt = ds.copy()
    if target_fs is not None:
        dec_factor = int(np.round(ds_filt.stats.sampling_rate / target_fs))
        ds_filt.decimate(dec_factor)
    if filt_type is not None and filt_freq is not None:
        ds_filt.filter(
            filt_type=filt_type,
            freq=filt_freq,
        )
    return ds_filt


def dataloader_3dvha(
    inventory: Path,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
):
    def dataloader(start_time: np.datetime64, end_time: np.datetime64) -> DataStream:
        ds_raw = condition_data(
            read_inventory(inventory, start_time, end_time),
            target_fs=target_fs,
            filt_type=filt_type,
            filt_freq=filt_freq,
        )
        ds = ds_raw.copy()
        ds.data[0:3] = -ds_raw.data[0:3]
        return ds

    return dataloader


def dataloader_all_sensors():
    return


def dataloader_shru(
    inventory: Path,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> Callable:
    def dataloader(start_time: np.datetime64, end_time: np.datetime64) -> DataStream:
        return condition_data(
            read_inventory(inventory, start_time, end_time),
            target_fs=target_fs,
            filt_type=filt_type,
            filt_freq=filt_freq,
        )

    return dataloader


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Plot SHRU data from a given inventory.")
    parser.add_argument(
        "--sensor",
        type=str,
        default="3dvha",
        choices=SENSORS + ["all"],
        help="Sensor to plot data for. Default is 'vla1'.",
    )
    parser.add_argument(
        "--start",
        type=str,
        # default="2023-12-01T21:44:00",
        default="2023-12-01T22:25:00",
        help="Start time of the data to extract.",
    )
    parser.add_argument(
        "--end",
        type=str,
        # default="2023-12-02T01:31:00",
        default="2023-12-01T22:26:00",
        help="End time of the data to extract.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        default=False,
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
        default=False,
        help="Save the figure instead of showing it.",
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=1000.0,
        help="Decimation factor for the data.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="highpass",
        choices=["lowpass", "highpass", "bandpass", None],
        help="Filter type to apply to the data.",
    )
    parser.add_argument(
        "--filt_freq",
        type=float,
        nargs="+",
        default=10.0,
        help="Frequency or frequencies for the filter.",
    )
    return parser


def setup_plotting(
    sensor: str,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> tuple[Callable, Callable]:
    match sensor:
        case "all":
            invs = [get_path(f"{s}_inventory") for s in SENSORS]
            [
                logging.info(f"Using inventory for {sensor.upper()}: {inv.resolve()}")
                for sensor, inv in zip(SENSORS, invs)
            ]
        case "3dvha":
            inv = get_path(f"{sensor}_inventory")
            plotter = plot_3dvha_spectrograms
            dataloader = dataloader_3dvha(
                inv, target_fs=target_fs, filt_type=filt_type, filt_freq=filt_freq
            )
            logging.info(f"Using inventory for {sensor.upper()}: {inv.resolve()}")
        case "vla1" | "vla2":
            inv = get_path(f"{sensor}_inventory")
            plotter = plot_shru_spectrograms
            dataloader = dataloader_shru(
                inv, target_fs=target_fs, filt_type=filt_type, filt_freq=filt_freq
            )
            logging.info(f"Using inventory for {sensor.upper()}: {inv.resolve()}")
    return plotter, dataloader


def main(
    sensor: str,
    start: str,
    end: str,
    multi: bool,
    interval: float,
    savefig: bool,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> None:

    plotter, dataloader = setup_plotting(
        sensor,
        target_fs=target_fs,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )

    start = np.datetime64(start, TIME_PRECISION)
    end = np.datetime64(end, TIME_PRECISION)

    if multi:
        interval = np.timedelta64(interval, "s")
    else:
        interval = end - start

    for start_time in np.arange(start, end, interval):
        end_time = start_time + interval
        logging.info(f"Plotting {sensor.upper()} data from {start_time} to {end_time}")
        ds = dataloader(start_time, end_time)
        fig = plotter(ds, **STFT_PARAMS)

        start_str = convert_datetime64_to_string(start_time)
        end_str = convert_datetime64_to_string(end_time)
        start_str_read = convert_datetime64_to_string(start_time, readable=True)
        end_str_read = convert_datetime64_to_string(end_time, readable=True)

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


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(
        args.sensor,
        args.start,
        args.end,
        args.multi,
        args.interval,
        args.savefig,
        target_fs=args.target_fs,
        filt_type=args.filter,
        filt_freq=args.filt_freq,
    )
