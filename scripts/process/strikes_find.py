#!/usr/bin/env python3
"""Find pile-driving strikes in the dataset using peak-finding."""

from argparse import ArgumentParser
from collections.abc import Sequence
import logging
from pathlib import Path

import numpy as np
from polars import DataFrame, concat

from rodeo import utils
from vineyard.config import get_path, time_ranges
from vineyard.readers import read_acoustic_data
from vineyard.signal import find_strikes

sensors = [
    {"name": "3dvha", "channel": 7, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla1", "channel": 3, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla2", "channel": 0, "distance_sec": 1.0, "threshold": 0.02},
]


def build_strikes_df(
    sensor: dict,
    time_start: np.datetime64,
    time_end: np.datetime64,
    strike_index_offset: int = 0,
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
            "strike_index": np.arange(len(peaks)) + strike_index_offset,
            "time": ds.time_vector[peaks],
            "sample": peaks,
        }
    )


def main(output: Path) -> None:
    all_dfs = []

    for sensor in sensors:
        sensor_dfs = []
        strike_index_offset = 0

        for i, (time_start, time_end) in enumerate(time_ranges):
            logging.info(
                f"Processing sensor {sensor['name']}, time range "
                f"{i+1}/{len(time_ranges)}: {time_start} to {time_end}"
            )

            df = build_strikes_df(sensor, time_start, time_end, strike_index_offset)
            sensor_dfs.append(df)
            strike_index_offset += len(df)

        all_dfs.extend(sensor_dfs)

    concat(all_dfs).write_csv(output)
    logging.info(f"Strikes extracted and saved to {output}.")


if __name__ == "__main__":
    logging.basicConfig(**utils.logging_kwargs)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("strike_index"),
        help="Output file to save the extracted strikes.",
    )
    args = parser.parse_args()
    main(args.output)
