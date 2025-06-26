#!/usr/bin/env python3
"""Read index of pile driving strikes and save pulses to an HDF5 file."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
import logging

import h5py
import polars as pl
from polars import DataFrame
from tqdm import tqdm
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

from rodeo.utils import logging_kwargs
from vineyard.config import get_path
from vineyard.readers import read_acoustic_data


def load_index(index: Path, buffer_start: float, buffer_end: float) -> DataFrame:
    start = pl.duration(
        microseconds=buffer_start * TIME_CONVERSION_FACTOR, time_unit=TIME_PRECISION
    )
    end = pl.duration(
        microseconds=buffer_end * TIME_CONVERSION_FACTOR, time_unit=TIME_PRECISION
    )
    return (
        pl.read_csv(index)
        .with_columns(
            pl.col("time").str.to_datetime(time_unit=TIME_PRECISION).alias("peak_time")
        )
        .drop("time")
        .with_columns((pl.col("peak_time") - start).alias("start_time"))
        .with_columns((pl.col("peak_time") + end).alias("end_time"))
    )


def main(
    index: Path,
    output: Path,
    buffer_start: float,
    buffer_end: float,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> None:
    df = load_index(index, buffer_start, buffer_end)

    with h5py.File(output, "w") as file:
        for row in tqdm(df.iter_rows(), desc="Processing strikes", total=df.shape[0]):
            sensor, channel, strike_index, _, time_start, time_end = row
            ds = read_acoustic_data(
                get_path(f"{sensor}_inventory"),
                time_start,
                time_end,
                channel,
                dec_factor=dec_factor,
                filt_type=filt_type,
                filt_freq=filt_freq,
                taper_pc=taper_pc,
            )
            g = file.create_group(f"{sensor}/{strike_index:04d}")
            ds.create_hdf5_dataset(g)


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    parser = ArgumentParser(description="Save strike data to an HDF file.")
    parser.add_argument(
        "--index",
        type=Path,
        default=get_path("strike_index"),
        help="Path to the index file containing the strike times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("strike_data"),
        help="Path to the output HDF5 file.",
    )
    parser.add_argument(
        "--buffer_start",
        type=float,
        default=0.75,
        help="Buffer start time in seconds before the strike time.",
    )
    parser.add_argument(
        "--buffer_end",
        type=float,
        default=0.75,
        help="Buffer end time in seconds after the strike time.",
    )
    parser.add_argument(
        "--taper_pc",
        type=float,
        default=0.05,
        help="Percentage of taper to apply to the data.",
    )
    parser.add_argument(
        "--dec_factor",
        type=int,
        default=20,
        help="Decimation factor for the data.",
    )
    parser.add_argument(
        "--filt_type",
        type=str,
        default=None,
        help="Type of filter to apply to the data (e.g., 'lowpass', 'highpass').",
    )
    parser.add_argument(
        "--filt_freq",
        type=float,
        nargs="+",
        default=None,
        help="Frequency or frequencies for the filter.",
    )
    args = parser.parse_args()
    main(
        args.index,
        args.output,
        args.buffer_start,
        args.buffer_end,
        args.taper_pc,
        args.dec_factor,
        args.filt_type,
        args.filt_freq,
    )
