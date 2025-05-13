#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
from tritonoa.data.stream import DataStream

import vwdas.paths as paths
from vwdas.plotting import savefig_kwargs
import vwdas.readers as readers

dotenv.load_dotenv()


def condition_data(ds: DataStream, target_sampling_rate: float) -> DataStream:
    ds_filt = ds.copy()
    decimation_factor = int(ds.stats.sampling_rate // target_sampling_rate)
    ds_filt.decimate(decimation_factor)
    ds_filt.filter(
        filt_type="highpass",
        freq=10.0,
    )
    return ds_filt


def main(args: argparse.Namespace) -> None:
    ds = readers.read_shru_data(args.inv, args.start, args.end)
    ds = condition_data(ds, args.target_sampling_rate)
    
    ds.write(args.out)

    fig, ax = plt.figure()
    ax.plot(ds.time_vector, ds.data[args.channel])
    ax.set_title("Pile Driving Template")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude ($\\mu$Pa)")
    fig.savefig(paths.reports.figures / "pile_driving_template.png", **savefig_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pile driving pattern from SHRU data.",
    )
    parser.add_argument(
        "--inv",
        type=Path,
        default=Path(os.getenv("VLA1_INV")),
        help="Path to the SHRU inventory file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/acoustic/pile_driving_pattern"),
        help="Path to the output file.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-12-01T22:00:51.5",
        help="Start time of the data to extract.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-01T22:00:52.75",
        help="End time of the data to extract.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel number to extract.",
    )
    parser.add_argument(
        "--target-sampling-rate",
        type=float,
        default=250.0,
        help="Target sampling rate for the output data.",
    )
    args = parser.parse_args()

    main(args)
