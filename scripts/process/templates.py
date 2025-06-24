#!/usr/bin/env python3

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION

from vineyard.config import get_path


def extract_template(
    sensor: str,
    channel: int,
    start: str,
    end: str,
    target_fs: float | None = None,
    filt_type: str | None = None,
    freq: float | Sequence[float] | None = None,
) -> DataStream:
    time_start = np.datetime64(start, TIME_PRECISION)
    time_end = np.datetime64(end, TIME_PRECISION)

    inventory = get_path(f"{sensor}_inventory")
    ds = read_inventory(
        inventory,
        time_start=time_start,
        time_end=time_end,
        channels=channel,
    )
    if target_fs is not None:
        ds.resample(target_fs)
    if filt_type is not None and freq is not None:
        ds.filter(filt_type=filt_type, freq=freq)
    return ds


def main(input: Path, output: Path) -> None:
    df = pl.read_csv(input)

    with h5py.File(output, "w") as f:
        for row in df.iter_rows():
            sensor, channel, start, end, description = row
            ds = extract_template(sensor, channel, start, end)

            dataset_name = f"{sensor}_{description.replace(' ', '_').lower()}"
            g = f.create_group(dataset_name)
            ds.create_hdf5_dataset(g)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract template data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=get_path("template_config"),
        help="Input file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("template_data"),
        help="Output file for templates.",
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=None,
        help="Target sampling frequency for the templates.",
    )
    parser.add_argument(
        "--filt_type",
        type=str,
        default=None,
        help="Filter type to apply to the templates.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        nargs="*",
        default=None,
        help="Frequency or frequencies for filtering the templates.",
    )
    args = parser.parse_args()
    main(args.input, args.output, args.target_fs, args.filt_type, args.freq)
