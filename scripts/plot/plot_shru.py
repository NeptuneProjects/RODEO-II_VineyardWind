#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
from tritonoa.data.stream import DataStream

import vwdas.paths as paths
from vwdas.plotting import plot_spectrogram, savefig_kwargs
import vwdas.readers as readers

dotenv.load_dotenv()


def condition_data(ds: DataStream) -> DataStream:
    ds_filt = ds.copy()
    ds_filt.decimate(40)
    ds_filt.filter(
        filt_type="highpass",
        freq=1.0,
    )
    return ds_filt


def main(args: argparse.Namespace) -> None:
    ds = readers.read_shru_data(args.inv, args.start, args.end)
    ds = condition_data(ds)

    for channel in tqdm(range(0, 4)):
        fig = plot_spectrogram(ds, channel=channel, xlabel=f"Time (s) after {args.start}")
        title = args.inv.name[:12]
        fig.suptitle(f"{title},  channel {channel}")
        plt.draw()
        # fig.savefig(paths.reports.figures / f"finwhale_spec_{title}_ch{channel}.png", **savefig_kwargs)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=str,
        default="2023-12-01T21:06:00",
        help="Start time of the data to extract.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-01T21:09:00",
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
