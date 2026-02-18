#!/usr/bin/env python
"""Plot acoustic data (time series & spectrograms) from 3DVHA, VLA1, and VLA2.

Times of interest:
  ALL: 2023-12-01T21:44:00 to 2023-12-01T23:30:00
3DVHA: 2023-12-01T21:00:00 to 2023-12-01T23:30:00
VLA 1: 2023-12-01T21:26:00 to 2023-12-02T01:13:00
VLA 2: 2023-12-01T21:44:00 to 2023-12-02T01:31:00

Fin whale: 2023-12-01T21:00:00 to 2023-12-01T23:19:00
"""

from argparse import ArgumentParser
from collections.abc import Sequence
from functools import partial
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
    plot_3dvha_data,
    plot_all_acoustic_data,
    plot_shru_data,
    SAVEFIG_KWARGS,
)

dotenv.load_dotenv()

nperseg = 256
STFT_PARAMS = {
    "nperseg": nperseg,
    "hop": int(0.05 * nperseg),
    "nfft": 2**12,
}
SENSORS = {
    "all": {
        "channels_to_plot": {
            "3dvha": list(range(4, 8)),
            "vla1": [3],
            "vla2": [1],
        },
    },
    "3dvha": {
        "metadata": {
            "channel_names": [
                "3DVHA Front Hydrophone",
                "3DVHA Right Hydrophone",
                "3DVHA Left Hydrophone",
                "3DVHA Back Hydrophone",
                "3DVHA Particle Motion X",
                "3DVHA Particle Motion Y",
                "3DVHA Particle Motion Z",
                "3DVHA Omni Hydrophone",
            ]
        }
    },
    "vla1": {
        "metadata": {
            "channel_names": [
                "VLA1 Channel 1",
                "VLA1 Channel 2",
                "VLA1 Channel 3",
                "VLA1 Channel 4",
            ]
        }
    },
    "vla2": {
        "metadata": {
            "channel_names": [
                "VLA2 Channel 1",
                "VLA2 Channel 2",
                "VLA2 Channel 3",
                "VLA2 Channel 4",
            ]
        }
    },
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
    start_time: np.datetime64,
    end_time: np.datetime64,
    channels: int | list[int] | None = None,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    metadata: dict | None = None,
) -> DataStream:
    ds_raw = condition_data(
        read_inventory(
            inventory, start_time, end_time, channels=channels, metadata=metadata
        ),
        target_fs=target_fs,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )
    ds = ds_raw.copy()
    ds.data[0:3] = -ds_raw.data[0:3]
    return ds


def dataloader_all_sensors(
    inventories: list[Path],
    start_time: np.datetime64,
    end_time: np.datetime64,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> list[DataStream]:
    def load_and_format_data(
        sensor: str, dataloader: Callable, inventory: Path
    ) -> DataStream:
        channels = SENSORS["all"]["channels_to_plot"][sensor]
        metadata = SENSORS[sensor]["metadata"].copy()
        metadata["channel_names"] = remove_channel_metadata(
            channels, metadata["channel_names"]
        )
        return dataloader(
            inventory,
            start_time,
            end_time,
            channels=channels,
            target_fs=target_fs,
            filt_type=filt_type,
            filt_freq=filt_freq,
            metadata=metadata,
        )

    def remove_channel_metadata(indexes: list[int], data: list) -> None:
        return [i for j, i in enumerate(data) if j in indexes]

    sensors = [s for s in SENSORS.keys() if s != "all"]
    dataloaders = [dataloader_3dvha, dataloader_shru, dataloader_shru]

    data = []
    for sensor, dataloader, inventory in zip(sensors, dataloaders, inventories):
        logging.info(f"Loading data for {sensor.upper()} from {inventory.resolve()}")
        data.append(load_and_format_data(sensor, dataloader, inventory))

    return data


def dataloader_shru(
    inventory: Path,
    start_time: np.datetime64,
    end_time: np.datetime64,
    channels: int | list[int] | None = None,
    target_fs: float | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    metadata: dict | None = None,
) -> DataStream:
    return condition_data(
        read_inventory(
            inventory, start_time, end_time, channels=channels, metadata=metadata
        ),
        target_fs=target_fs,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )


def format_times(
    start_time: np.datetime64, end_time: np.datetime64
) -> tuple[str, str, str, str]:
    start_str = convert_datetime64_to_string(start_time)
    end_str = convert_datetime64_to_string(end_time)
    start_str_read = convert_datetime64_to_string(start_time, readable=True)
    end_str_read = convert_datetime64_to_string(end_time, readable=True)
    return start_str, end_str, start_str_read, end_str_read


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Plot SHRU data from a given inventory.")
    parser.add_argument(
        "--sensor",
        type=str,
        default="all",
        choices=list(SENSORS.keys()) + ["all"],
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
        "--target-fs",
        type=float,
        default=100.0,
        help="Decimation factor for the data.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="bandpass",
        choices=["lowpass", "highpass", "bandpass", None],
        help="Filter type to apply to the data.",
    )
    parser.add_argument(
        "--filt-freq",
        type=float,
        nargs="+",
        default=[15.0, 35.0],
        help="Frequency or frequencies for the filter.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=15.0,
        help="Minimum frequency for the spectrogram.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=35.0,
        help="Maximum frequency for the spectrogram.",
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
            invs = [get_path(f"{s}_inventory") for s in SENSORS.keys() if s != "all"]
            plotter = plot_all_acoustic_data
            dataloader = partial(
                dataloader_all_sensors,
                invs,
                target_fs=target_fs,
                filt_type=filt_type,
                filt_freq=filt_freq,
            )
            [
                logging.info(f"Using inventory for {sensor.upper()}: {inv.resolve()}")
                for sensor, inv in zip(SENSORS.keys(), invs)
            ]
        case "3dvha":
            inv = get_path(f"{sensor}_inventory")
            plotter = plot_3dvha_data
            dataloader = partial(
                dataloader_3dvha,
                inv,
                target_fs=target_fs,
                filt_type=filt_type,
                filt_freq=filt_freq,
                metadata=SENSORS[sensor]["metadata"],
            )
            logging.info(f"Using inventory for {sensor.upper()}: {inv.resolve()}")
        case "vla1" | "vla2":
            inv = get_path(f"{sensor}_inventory")
            plotter = plot_shru_data
            dataloader = partial(
                dataloader_shru,
                inv,
                target_fs=target_fs,
                filt_type=filt_type,
                filt_freq=filt_freq,
                metadata=SENSORS[sensor]["metadata"],
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
    fmin: float = 10.0,
    fmax: float = 500.0,
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
        start_str, end_str, start_str_read, end_str_read = format_times(
            start_time, end_time
        )

        title = f"{sensor.upper()} data from {start_str_read}Z to {end_str_read}Z"

        logging.info(f"Plotting {title}")
        data = dataloader(start_time, end_time)
        fig = plotter(data, title=title, **(STFT_PARAMS | {"fmin": fmin, "fmax": fmax}))

        if savefig:
            savepath = (get_path(f"{sensor}_data_view") / f"{fmin}-{fmax}Hz")
            savepath.mkdir(parents=True, exist_ok=True)
            file = savepath / f"{sensor}_{start_str}-{end_str}.png"

            fig.savefig(file, **SAVEFIG_KWARGS)
            plt.close(fig)
            logging.info(f"Saved figure to {file.resolve()}")
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
        filt_freq=args.filt_freq[0] if len(args.filt_freq) == 1 else args.filt_freq,
        fmin=args.fmin,
        fmax=args.fmax,
    )
