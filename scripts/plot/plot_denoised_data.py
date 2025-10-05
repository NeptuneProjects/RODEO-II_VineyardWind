#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from tritonoa.data.reader import read_hdf5
from tritonoa.data.stream import DataStream

from vineyard.config import get_path
from vineyard.plotting import SAVEFIG_KWARGS


def plot_denoised_data(ds: DataStream, sensor: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ds.time_vector, ds.data[0], label="Original Signal")
    ax.plot(ds.time_vector, ds.data[1], label="Filtered Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Denoising Results for {sensor.upper()}")
    ax.legend(loc="upper left")
    return fig


def plot_pulse_compressed_data(
    ds: DataStream, ds_pc: DataStream, sensor: str
) -> plt.Figure:

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"hspace": 0.25})
    ax = axes[0]
    ax.plot(ds.time_vector, ds.data[1], label="Denoised Signal", zorder=20)
    ax.plot(ds.time_vector, ds.data[0], label="Original Signal", alpha=0.5, zorder=10)
    ax.legend(loc="upper left")
    ax.set_title(f"Denoised Signal - {sensor.upper()}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Amplitude")

    ax = axes[1]
    ax.plot(ds_pc.time_vector, ds_pc.data.squeeze())
    ax.set_title(f"Denoised, Pulse Compressed Signal - {sensor.upper()}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Amplitude")

    return fig


def plot_all_channels(
    streams: dict[str, DataStream], channel: int = 0, title=""
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(streams), 1, figsize=(10, 8), gridspec_kw={"hspace": 0.25}
    )

    for ax, (sensor, ds) in zip(axes, streams.items()):
        ax.plot(ds.time_vector, ds.data[channel])
        ax.set_title(f"{title}{sensor.upper()}")
        ax.grid()

    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")

    return fig


def main():
    sensors = ["3dvha", "vla1", "vla2"]
    savepath = get_path("figures") / "denoised"
    savepath.mkdir(parents=True, exist_ok=True)

    streams = {}
    streams_pc = {}
    for sensor in sensors:
        ds = read_hdf5(get_path("denoised_data") / f"{sensor}.h5")
        ds_pc = read_hdf5(get_path("pulse_comp_data") / f"{sensor}_pc.h5")

        # fig = plot_denoised_data(ds, sensor)
        # fig.savefig(savepath / f"{sensor}_denoised.png", **SAVEFIG_KWARGS)
        # plt.close(fig)

        # fig = plot_pulse_compressed_data(ds, ds_pc, sensor)
        # fig.savefig(savepath / f"{sensor}_denoised_pc.png", **SAVEFIG_KWARGS)
        # plt.close(fig)

        streams[sensor] = ds
        streams_pc[sensor] = ds_pc

    fig = plot_all_channels(streams, channel=1, title="Denoised Signal - ")
    fig.savefig(savepath / f"all_denoised_channels.png", **SAVEFIG_KWARGS)
    plt.close(fig)

    fig = plot_all_channels(streams_pc, title="Denoised, Pulse Compressed Signal - ")
    fig.savefig(savepath / f"all_denoised_pc_channels.png", **SAVEFIG_KWARGS)
    plt.close(fig)


if __name__ == "__main__":
    main()
