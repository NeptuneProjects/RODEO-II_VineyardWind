#!/usr/bin/env python3
"""Read index of pile driving strikes and save pulses to an HDF5 database."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
import logging

import h5py
from tqdm import tqdm

from rodeo.utils import logging_kwargs
from vineyard.config import get_path
from vineyard.readers import read_acoustic_data, read_strike_index


def main(
    index: Path,
    output: Path,
    buffer_start: float,
    buffer_end: float,
    detrend: bool = True,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
) -> None:
    df = read_strike_index(index, buffer_start, buffer_end)

    with h5py.File(output, "w") as file:
        for row in tqdm(df.iter_rows(), desc="Processing strikes", total=df.shape[0]):
            sensor, channel, strike_index, _, _, time_start, time_end = row
            ds = read_acoustic_data(
                get_path(f"{sensor}_inventory"),
                time_start,
                time_end,
                channel,
                detrend=detrend,
                dec_factor=dec_factor,
                filt_type=filt_type,
                filt_freq=filt_freq,
                taper_pc=taper_pc,
            )
            g = file.create_group(f"{sensor}/{strike_index:04d}")
            ds.create_hdf5_dataset(g)


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    parser = ArgumentParser(description=__doc__)
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
        "--detrend",
        action="store_true",
        default=False,
        help="Whether to detrend the data before processing.",
    )
    parser.add_argument(
        "--taper_pc",
        type=float,
        default=None,
        help="Percentage of taper to apply to the data.",
    )
    parser.add_argument(
        "--dec_factor",
        type=int,
        default=None,
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
        args.detrend,
        args.taper_pc,
        args.dec_factor,
        args.filt_type,
        args.filt_freq,
    )
