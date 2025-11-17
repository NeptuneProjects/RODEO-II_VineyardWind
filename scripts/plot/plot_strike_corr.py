#!/usr/bin/env python3
"""Plot correlation coefficients between pairs of pile-driving strikes."""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from vineyard.config import get_path
from vineyard.plotting import SAVEFIG_KWARGS, plot_corr


def main(database: Path, output: Path, window: float, show: bool) -> None:
    corrs, time_diffs = [], []
    with h5py.File(database, "r") as f:
        sensors = list(f.keys())
        for group in f.values():
            corrs.append(group["corr"][:])
            time_diffs.append(group["time_diff"][:])

    fig = plot_corr(sensors, corrs, time_diffs, window=window)
    fig.savefig(output, **SAVEFIG_KWARGS)
    if show:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot strike correlation data.")
    parser.add_argument(
        "--database",
        type=Path,
        default=get_path("strike_corr").parent / "strike_corr_19-25.hdf5",
        help="Path to the strike correlation data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("figures") / "strike_correlation" / "strike_corr_19-25.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=300.0,
        help="Window size for the correlation plot in seconds.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the plot interactively.",
    )
    args = parser.parse_args()
    main(args.database, args.output, args.window, args.show)
