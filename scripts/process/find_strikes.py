#!/usr/bin/env python3
"""Find pile-driving strikes in the dataset using peak-finding."""

from argparse import ArgumentParser
from collections.abc import Sequence
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame, concat
from rodeo import utils
from scipy.signal import find_peaks
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION

from vineyard.config import get_path

SENSORS = [
    {"name": "3dvha", "channel": 7, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla1", "channel": 3, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla2", "channel": 0, "distance_sec": 1.0, "threshold": 0.02},
]


def characteristic_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    xsq = x**2
    return xsq / np.max(xsq)


def find_strikes(
    sensor: dict,
    time_start: np.datetime64,
    time_end: np.datetime64,
    taper_pc: float = 1e-4,
    dec_factor: int = 20,
    filt_type: str = "bandpass",
    freq: float | Sequence[float] = [100.0, 300.0],
) -> DataFrame:
    name, channel, distance_sec, threshold = tuple(sensor.values())
    logging.info(f"Processing sensor: {name}, channel: {channel}")

    inventory = get_path(f"{name}_inventory")
    ds = (
        read_inventory(
            inventory, time_start=time_start, time_end=time_end, channels=channel
        )
        .taper(taper_pc)
        .decimate(dec_factor)
        .filter(filt_type, freq)
    )

    cf = characteristic_function(ds.data[0])

    peaks = find_peaks(
        cf, height=threshold, distance=int(distance_sec * ds.stats.sampling_rate)
    )[0]
    logging.info(f"Found {len(peaks)} peaks for sensor {name}.")
    return DataFrame(
        {
            "sensor": name,
            "channel": channel,
            "strike_index": np.arange(len(peaks)),
            "time": ds.time_vector[peaks],
        }
    )


def main(start: np.datetime64, end: np.datetime64, output: Path) -> None:
    time_start = start.astype(f"datetime64[{TIME_PRECISION}]")
    time_end = end.astype(f"datetime64[{TIME_PRECISION}]")

    dfs = []
    for sensor in SENSORS:
        dfs.append(find_strikes(sensor, time_start, time_end))

    concat(dfs).write_csv(output)
    logging.info(f"Strikes extracted and saved to {output}.")


if __name__ == "__main__":
    logging.basicConfig(**utils.logging_kwargs)
    parser = ArgumentParser(description="Extract all strikes from the dataset.")
    parser.add_argument(
        "--start",
        type=np.datetime64,
        default="2023-12-01T21:51:15.00",
        help="Start time for extracting strikes.",
    )
    parser.add_argument(
        "--end",
        type=np.datetime64,
        default="2023-12-01T22:26:00.00",
        help="End time for extracting strikes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("strike_index"),
        help="Output file to save the extracted strikes.",
    )
    args = parser.parse_args()
    main(args.start, args.end, args.output)
