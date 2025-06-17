#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from tritonoa.data.reader import read_inventory
from tritonoa.data.signal import pulse_compression
from tritonoa.data.time import TIME_PRECISION

from vineyard.config import get_path


def main():
    channels = list(range(4, 8))
    time_start = np.datetime64("2023-12-01 22:25:00", TIME_PRECISION)
    time_end = np.datetime64("2023-12-01 22:26:00", TIME_PRECISION)
    ds = (
        read_inventory(
            get_path("3dvha_inventory"),
            time_start=time_start,
            time_end=time_end,
            channels=channels,
        )
        .decimate(20)
        .filter(filt_type="bandpass", freq=[15.0, 35.0])
    )

    excerpt_start = np.datetime64("2023-12-01 22:25:46.08", TIME_PRECISION)
    excerpt_end = np.datetime64("2023-12-01 22:25:47", TIME_PRECISION)
    buffer_start = excerpt_start - np.timedelta64(30, "s")
    buffer_end = excerpt_end + np.timedelta64(30, "s")
    excerpt = (
        read_inventory(
            get_path("3dvha_inventory"),
            time_start=buffer_start,
            time_end=buffer_end,
            channels=7,
        )
        .decimate(20)
        .filter(filt_type="bandpass", freq=[15.0, 35.0])
        .trim(excerpt_start, excerpt_end)
    )

    plt.figure()
    plt.plot(excerpt.time_vector, excerpt.data[0], label="Excerpt Signal")
    plt.plot(
        excerpt.time_vector,
        np.imag(hilbert(excerpt.data[0])),
        label="Analytical Signal",
    )
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Template Signal")
    plt.draw()

    dspc = ds.copy()
    dspc = dspc.pulse_compression(np.imag(hilbert(excerpt.data)))
    # dspc = dspc.pulse_compression(excerpt.data[0])
    dspc.data[-1] = pulse_compression(excerpt.data[0], ds.data[-1])

    channel_names = [
        "3DVHA Particle Motion X",
        "3DVHA Particle Motion Y",
        "3DVHA Particle Motion Z",
        "3DVHA Omni Hydrophone",
    ]

    fig, axs = plt.subplots(
        nrows=dspc.num_channels, ncols=2, figsize=(10, 16), sharex=True
    )
    for i in range(dspc.num_channels):
        ax = axs[i, 0]
        ax.plot(ds.time_vector, ds.data[i], label=f"Channel {i+1}")
        ax.set_xlim(ds.time_vector[0], ds.time_vector[-1])
        if i != 3:
            ax.set_ylim(-0.2, 0.2)
        if i == 0:
            ax.set_title(f"Original\n{channel_names[i]}", rotation=0)
        else:
            ax.set_title(channel_names[i], rotation=0)
        ax.grid()

    for i in range(dspc.num_channels):
        ax = axs[i, 1]
        ax.plot(dspc.time_vector, ds.data[i], label=f"Channel {i+1}")
        # ax.plot(dspc.time_vector, np.real(dspc.data[i]), label=f"Channel {i+1}")
        ax.plot(dspc.time_vector, dspc.data[i], label=f"Channel {i+1}")
        ax.set_xlim(dspc.time_vector[0], dspc.time_vector[-1])
        ax.set_title(channel_names[i], rotation=0)
        if i != 3:
            ax.set_ylim(-0.2, 0.2)
        if i == 0:
            ax.set_title(f"Pulse-compressed\n{channel_names[i]}", rotation=0)
        else:
            ax.set_title(channel_names[i], rotation=0)
        ax.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
