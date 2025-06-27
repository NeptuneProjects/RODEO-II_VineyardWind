#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy import stats
from tritonoa.data.time import TIME_CONVERSION_FACTOR

from rodeo.plotting import regularize_grid, stack_data
from vineyard.config import get_path


def plot_corr(sensor: str, corr: NDArray, time_diff: NDArray):

    window = 1800.0
    M = corr.shape[0]
    stacked_corr = np.full((M, M), np.nan)
    stacked_dt = np.full((M, M), np.nan)

    for i in range(M):
        stacked_corr[i, : M - i] = corr[i, i:]
        stacked_dt[i, : M - i] = time_diff[i, i:] / TIME_CONVERSION_FACTOR

    num_points = 1000
    tvec = np.linspace(0.0, np.nanmax(stacked_dt), num_points)

    resampled_data = np.full((M, num_points), np.nan)
    for i in range(M):
        signal = stacked_corr[i, :]
        dt = stacked_dt[i, :]

        valid_idx = ~np.isnan(signal) & ~np.isnan(dt)
        valid_time = dt[valid_idx]
        valid_signal = signal[valid_idx]

        sort_idx = np.argsort(valid_time)
        valid_time = valid_time[sort_idx]
        valid_signal = valid_signal[sort_idx]

        if len(valid_time) < 2:
            continue

        f = interp1d(
            valid_time,
            valid_signal,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        resampled_data[i, :] = f(tvec)

    mean_corr = np.nanmean(resampled_data, axis=0)
    std_corr = np.nanstd(resampled_data, axis=0)
    ucb = mean_corr + 2 * std_corr
    lcb = mean_corr - 2 * std_corr
    # PDF for plotting:


    n_time_points = resampled_data.shape[1]
    # pdf_matrix = np.zeros((num_points, n_time_points))
    
    # Calculate PDF for each time point
    # pdfs = []
    confidence = 0.95
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100
    lower_bounds = []
    upper_bounds = []
    for i in range(n_time_points):
        data_values = resampled_data[:, i]
        lower_bound = np.nanpercentile(data_values, lower_percentile)
        upper_bound = np.nanpercentile(data_values, upper_percentile)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        # kde = stats.gaussian_kde(resampled_data[:, i])
        # x = np.linspace(0.0, 1.0, resampled_data.shape[0])
        # pdfs.append(kde(x))

    plt.figure()
    plt.imshow(stacked_corr, aspect="auto", cmap="inferno", origin="lower")
    plt.colorbar(label="Correlation")
    plt.title(f"Strike Correlation for {sensor}")
    plt.draw()

    plt.figure()
    for i in range(M):
        plt.plot(stacked_dt[i], stacked_corr[i], "k", alpha=0.01)
    plt.plot(tvec, mean_corr, "r", label="Mean Correlation")
    plt.plot(tvec, upper_bounds, "b", label="Upper 2 Std Dev")
    plt.plot(tvec, lower_bounds, "b", label="Lower 2 Std Dev")
    plt.xlim(0, window)
    plt.ylim(0, 1.0)
    plt.xlabel("Time Difference (s)")
    plt.ylabel("Correlation")
    plt.title(f"Strike Correlation for {sensor}")
    plt.draw()

    # plt.figure()
    # plt.plot(tvec, mean_corr, "k", label="Mean Correlation")
    # plt.fill_between(tvec, lcb, ucb, color="gray", alpha=0.5, label="2 Std Dev")
    # plt.xlim(0, window)
    # plt.ylim(0, 1.0)
    # plt.xlabel("Time Elapsed (s)")
    # plt.ylabel("Correlation Coefficient")
    # plt.title(f"Strike Correlation for {sensor}")
    # plt.legend()
    # plt.draw()

    return


def main(database: Path):
    with h5py.File(database, "r") as f:
        for sensor, group in f.items():
            corr = group["corr"][:]
            time_diff = group["time_diff"][:]

            plot_corr(sensor, corr, time_diff)
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot strike correlation data.")
    parser.add_argument(
        "--database",
        type=Path,
        default=get_path("strike_corr").parent / "strike_corr_15-35.hdf5",
        help="Path to the strike correlation data file.",
    )
    args = parser.parse_args()
    main(args.database)
