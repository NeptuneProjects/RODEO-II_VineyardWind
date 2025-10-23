#!/usr/bin/env python3

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator

from vineyard.config import get_path
from vineyard.plotting import SAVEFIG_KWARGS


def plot_bearings(df: pl.DataFrame):
    sensors = ["3dvha", "vla1", "vla2"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for sensor in sensors:
        bearing_col = f"{sensor}_brg"

        ax.plot(
            df["timestamp"],
            df[bearing_col],
            "o",
            label=f"{sensor.upper()} Bearing",
        )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    ax.set_ylim(160, 180)

    ax.set_xlabel("Time")
    ax.set_ylabel("Bearing (degrees)")
    ax.legend()
    plt.tight_layout()

    return fig


def main():
    df = pl.read_csv(get_path("tdoa_data") / "tdoa_with_locations.csv").cast(
        {"timestamp": pl.Datetime}
    )
    fig = plot_bearings(df)
    fig.savefig(get_path("figures") / "tdoa" / "whale_bearings.png", **SAVEFIG_KWARGS)


if __name__ == "__main__":
    main()
