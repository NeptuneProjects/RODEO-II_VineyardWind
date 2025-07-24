#!/usr/bin/env python3
"""Find pile-driving strikes in the dataset using peak-finding."""

from argparse import ArgumentParser
from collections.abc import Sequence
import logging
from pathlib import Path

import numpy as np
from polars import DataFrame, concat
from tritonoa.data.time import TIME_PRECISION

from rodeo import utils
from vineyard.config import get_path
from vineyard.readers import read_acoustic_data
from vineyard.signal import find_strikes

SENSORS = [
    {"name": "3dvha", "channel": 7, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla1", "channel": 3, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla2", "channel": 0, "distance_sec": 1.0, "threshold": 0.02},
]


def build_strikes_df(
    sensor: dict,
    time_start: np.datetime64,
    time_end: np.datetime64,
    taper_pc: float = 1e-4,
    dec_factor: int | None = None,
    filt_type: str = "bandpass",
    freq: float | Sequence[float] = [100.0, 300.0],
) -> DataFrame:
    name, channel, distance_sec, threshold = tuple(sensor.values())
    logging.info(f"Processing sensor: {name}, channel: {channel}")

    ds = read_acoustic_data(
        get_path(f"{name}_inventory"),
        time_start,
        time_end,
        channels=channel,
        taper_pc=taper_pc,
        dec_factor=dec_factor,
        filt_type=filt_type,
        filt_freq=freq,
    )
    peaks = find_strikes(ds.data[0], ds.stats.sampling_rate, threshold, distance_sec)

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
        raw_df = build_strikes_df(sensor, time_start, time_end)
        # refined_df = refine_strikes_df(raw_df)
        dfs.append(raw_df)

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
