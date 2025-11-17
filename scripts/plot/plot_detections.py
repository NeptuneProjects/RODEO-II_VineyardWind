#!

import matplotlib.pyplot as plt
import numpy as np
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION
from scipy.signal import find_peaks

from vineyard.config import get_path
import vineyard.plotting as plotting


def plot_detections(ds: DataStream, threshold: float) -> plt.Figure:
    distance = 1.0 * ds.stats.sampling_rate
    cf = ds.data[0] ** 2
    cf /= np.max(cf)
    peaks = find_peaks(cf, height=threshold, distance=distance)[0]

    fig, axes = plt.subplots(nrows=2, figsize=(8, 2.5), sharex=True)
    ax = axes[0]
    ax.plot(ds.time_vector, ds.data[0], c="k")
    ax.set_ylabel("Bandpassed Input\nSignal ($\mathrm{\mu}$Pa)")

    ax = axes[1]
    ax.plot(ds.time_vector, cf, c="k", label="Char. Func.")
    ax.plot(ds.time_vector[peaks], cf[peaks], "ro", label="Peaks")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_ylabel("Characteristic\nFunction")
    ax.legend()

    ax.set_xlim(ds.time_vector[0], ds.time_vector[-1])

    return fig


def main():
    time_start = np.datetime64("2023-12-01T22:24:00.00", TIME_PRECISION)
    time_end = np.datetime64("2023-12-01T22:26:00.00", TIME_PRECISION)
    sensor = "vla1"
    channel = 3
    threshold = 0.05

    ds = (
        read_inventory(
            get_path(f"{sensor}_inventory"),
            channels=channel,
            time_start=time_start,
            time_end=time_end,
        )
        .taper(max_percentage=1e-4)
        .decimate(20)
        .filter("bandpass", [100.0, 300.0])
    )

    fig = plot_detections(ds, threshold)
    savepath = get_path("figures") / "strike_detection.png"
    fig.savefig(savepath, **plotting.SAVEFIG_KWARGS)


if __name__ == "__main__":
    main()
