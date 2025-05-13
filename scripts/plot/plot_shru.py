#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
from tritonoa.data.stream import DataStream

import vwdas.paths as paths
from vwdas.plotting import plot_spectrogram, savefig_kwargs
import vwdas.readers as readers

dotenv.load_dotenv()


def condition_data(ds: DataStream) -> DataStream:
    ds_filt = ds.copy()
    ds_filt.decimate(20)
    ds_filt.filter(
        filt_type="highpass",
        freq=1.0,
    )
    return ds_filt


def main(args: argparse.Namespace) -> None:
    ds = readers.read_shru_data(args.inv, args.start, args.end)
    ds = condition_data(ds)

    fig = plot_spectrogram(ds, channel=3, xlabel=f"Time (s) after {args.start}")
    fig.savefig(paths.reports.figures / "finwhale_spec.png", **savefig_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=str,
        default="2023-12-01T22:00:40",
        help="Start time of the data to extract.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-01T22:01:10",
        help="End time of the data to extract.",
    )
    parser.add_argument(
        "--inv",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("VLA2_INV")),
    )
    args = parser.parse_args()
    main(args)
