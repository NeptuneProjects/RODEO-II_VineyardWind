#!/usr/bin/env python

import matplotlib.pyplot as plt
from tritonoa.data.reader import read_hdf5
from tritonoa.data.stream import DataStream

from vineyard.config import get_path
from vineyard.plotting import SAVEFIG_KWARGS


def plot_denoised_data(ds: DataStream, sensor: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ds.time_vector, ds.data[0], label="Original Signal")
    ax.plot(ds.time_vector, ds.data[1], label="Filtered Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Denoising Results for {sensor.upper()}")
    ax.legend(loc="upper left")
    return fig


def main():
    sensors = ["3dvha", "vla1", "vla2"]
    savepath = get_path("figures") / "denoised"
    savepath.mkdir(parents=True, exist_ok=True)

    for sensor in sensors:
        ds = read_hdf5(get_path("denoised_data") / f"{sensor}.h5")

        fig = plot_denoised_data(ds, sensor)
        fig.savefig(savepath / f"denoised_{sensor}.png", **SAVEFIG_KWARGS)
        plt.close(fig)


if __name__ == "__main__":
    main()
