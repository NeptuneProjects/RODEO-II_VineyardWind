"""Plot correlation coefficients between pairs of pile-driving strikes."""

import string
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from tritonoa.data.time import TIME_CONVERSION_FACTOR

SECONDS_BETWEEN_STRIKES = 1.7

sensor_map = {
    "3dvha": "Site A",
    "vla1": "Site B",
    "vla2": "Site C",
}


def _compute_pdf_vs_time(
    data: NDArray, confidence: float = 0.95
) -> tuple[NDArray, NDArray, NDArray]:
    mean = np.nanmean(data, axis=0)
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100

    lower_bounds = []
    upper_bounds = []
    epdf = []
    for i in range(data.shape[1]):
        data_values = data[:, i]
        lower_bounds.append(np.nanpercentile(data_values, lower_percentile))
        upper_bounds.append(np.nanpercentile(data_values, upper_percentile))
        hist, bin_edges = np.histogram(
            data_values, bins=50, density=False, range=(0.0, 1.0)
        )
        if i == 0:
            epdf_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        epdf.append(hist / np.sum(hist))

    epdf = np.array(epdf).T
    return mean, lower_bounds, upper_bounds, epdf, epdf_bins


def _format_data(
    corr: NDArray, time_diff: NDArray, window: float, n_resampled: int = 3000
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    M = corr.shape[0]
    stacked_corr = np.full((M, M), np.nan)
    stacked_dt = np.full((M, M), np.nan)
    for i in range(M):
        stacked_corr[i, : M - i] = corr[i, i:]
        stacked_dt[i, : M - i] = time_diff[i, i:] / TIME_CONVERSION_FACTOR

    tvec, resampled_data = _resample_to_grid(
        stacked_dt, stacked_corr, window, num_points=n_resampled
    )
    _, _, _, epdf, _ = _compute_pdf_vs_time(resampled_data, confidence=0.95)
    Tgrid, Mgrid = np.meshgrid(tvec, np.arange(M), indexing="ij")
    return (
        resampled_data,
        epdf,
        Tgrid,
        Mgrid,
    )


def _load_corr_data(corr_file: Path) -> tuple[list[str], list[NDArray], list[NDArray]]:
    sensors, corrs, time_diffs = [], [], []
    with h5py.File(corr_file, "r") as f:
        sensors = list(f.keys())
        for group in f.values():
            corrs.append(group["corr"][:])
            time_diffs.append(group["time_diff"][:])
    return sensors, corrs, time_diffs


def _plot_corr(
    sensors: list[str],
    corrs: list[NDArray],
    time_diffs: list[NDArray],
    window: float = 300.0,
    strike_window_size: int | None = None,
) -> Figure:
    fig, axes = plt.subplots(
        figsize=(6.5, 3.75),
        nrows=2,
        ncols=3,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    for j, (sensor, corr, time_diff) in enumerate(zip(sensors, corrs, time_diffs)):
        (
            resampled_data,
            epdf,
            Tgrid,
            Mgrid,
        ) = _format_data(corr, time_diff, window)

        epdf_vmin = 0.01
        epdf_vmax = 0.1
        epdf[epdf < epdf_vmin] = np.nan

        tvec = Tgrid[:, 0]

        for i in range(2):
            ax = axes[i, j]
            if i == 0:
                im = ax.pcolormesh(
                    Tgrid,
                    Mgrid,
                    resampled_data.T,
                    shading="nearest",
                    cmap="cmo.thermal",
                    vmin=0.2,
                    vmax=1.0,
                )
                ax.set_xticklabels([])
                ax.set_xlabel(None)
                ax.set_title(sensor_map.get(sensor, sensor.upper()))
                if j == 0:
                    ax.set_ylabel("Strike index ($i$)")
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel(None)
                if j == 2:
                    cax = fig.add_axes([0.91, 0.52, 0.02, 0.35])
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label("Correlation coefficient ($\\rho_{ij}$)")
            if i == 1:
                im = ax.imshow(
                    epdf,
                    aspect="auto",
                    cmap="cmo.dense",
                    origin="lower",
                    extent=(tvec[0], tvec[-1], 0.0, 1.0),
                    vmin=epdf_vmin,
                    vmax=epdf_vmax,
                )
                if j == 0:
                    ax.set_xlabel("Reduced time $\\tau = t_j - t_i$ (s)")
                    ax.set_ylabel("Correlation coefficient ($\\rho_{ij}$)")
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel(None)
                if j == 2:
                    cax = fig.add_axes([0.91, 0.12, 0.02, 0.35])
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label("Empirical PDF")

                ax.set_ylim(0.2, 1.0)

            ax.set_xlim(0, window)

    for label, ax in zip(string.ascii_lowercase[0:6], axes.flat):
        ax.text(
            0.95,
            0.05,
            f"{label})",
            transform=ax.transAxes,
            fontsize=plt.rcParams["font.size"],
            va="bottom",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.0},
        )

    if strike_window_size is not None:
        template_window = SECONDS_BETWEEN_STRIKES * (strike_window_size // 2)
        for ax in axes.flat:
            ax.axvline(template_window, color="tab:red", linestyle="--", linewidth=1.5)

        axes[1, 0].text(
            template_window + 3,
            0.08,
            f"Template window = {template_window:.0f} s",
            color="tab:red",
            va="center",
            ha="left",
            transform=axes[1, 0].get_xaxis_transform(),
            fontsize=plt.rcParams["font.size"] * 0.9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.0},
        )

    return fig


def plot_correlations(
    corr_file: Path, window: float, strike_window_size: int | None = None
) -> plt.Figure:
    """Plot correlation coefficients between pairs of pile-driving strikes.

    Args:
        corr_file: Path to the HDF5 file containing correlation data.
        window: Time window in seconds for plotting correlations.
        strike_window_size: Optional size of the strike window in number of strikes.

    Returns:
        A matplotlib Figure object containing the correlation plot.
    """
    sensors, corrs, time_diffs = _load_corr_data(corr_file)
    return _plot_corr(sensors, corrs, time_diffs, window, strike_window_size)


def _resample_to_grid(
    dt_values: NDArray, corr_values: NDArray, window: float, num_points: int
) -> tuple[NDArray, NDArray]:
    M = corr_values.shape[0]
    tvec = np.linspace(0.0, np.nanmax(dt_values), num_points)

    resampled_data = np.full((M, num_points), np.nan)
    for i in range(M):
        signal = corr_values[i, :]
        dt = dt_values[i, :]

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

    points_to_keep = np.where(tvec <= window)[0]
    return tvec[points_to_keep], resampled_data[:, points_to_keep]
