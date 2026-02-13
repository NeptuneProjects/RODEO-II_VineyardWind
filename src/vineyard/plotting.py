from collections.abc import Sequence
from pathlib import Path
import string

import cmasher as cmr
import cmocean as cmo
import matplotlib
from matplotlib import font_manager
import matplotlib.colors as colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.basemap import Basemap
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.interpolate import interp1d
import scipy.signal as signal
import scipy.stats
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_CONVERSION_FACTOR

FIG_STYLE = Path("config/scirep_fig_style.mplstyle")
# FONT_PATH = Path("/System/Library/Fonts/HelveticaNeue.ttc")
plt.style.use(FIG_STYLE)

# print(matplotlib.rcParams)

# font_dirs = [Path("/System/Library/Fonts")]
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# font_list = font_manager.createFontList(font_files)
# font_manager.fontManager.ttflist.extend(font_list)

# matplotlib.rcParams["font.family"] = "Helvetica Neue"

# font_manager.fontManager.addfont(FONT_PATH)
# prop = font_manager.FontProperties(fname=FONT_PATH)
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = prop.get_name()


SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
    "facecolor": "white",
}


def draw_polygon(
    m: Basemap,
    longitudes: list[float],
    latitudes: list[float],
    fill: bool = False,
    alpha: float = 1.0,
    edgecolor: str = "red",
    linewidth: float = 1.0,
    zorder: int = 10,
):
    x, y = m(longitudes, latitudes)
    polygon = Polygon(
        xy=list(zip(x, y)),
        closed=True,
        fill=fill,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    return polygon


def find_closest_contour_index(levels: NDArray[np.float64], value: float) -> int:
    """
    Find the index of the contour level that is closest to a given value.

    Args
    levels: The contour levels.
    value: The value to find the closest contour level for.

    Returns
        The index of the closest contour level.
    """
    return np.argmin(np.abs(levels - value))


def plot_bathy(
    data: NDArray[np.float64],
    lonvec: NDArray[np.float64],
    latvec: NDArray[np.float64],
    m: Basemap,
    ax: Axes | None = None,
    shallowest_contour_depth: float = 0.0,
    levelsf=np.arange(-100, 10, 5),
    levelsc=np.arange(-100, 1, 5),
) -> tuple[plt.contourf, Axes]:

    data[data > 0] = 0.1

    if ax is None:
        ax = plt.gca()

    # Create a modified colormap truncated for shallow water and gray for
    # positive values
    n_bins = 256
    colors_array = cmr.get_sub_cmap("cmo.deep_r", 0.5, 1.0)(np.linspace(0, 1, n_bins))
    colors_list = np.vstack((colors_array, np.array([0.8, 0.8, 0.8, 0.8])))
    custom_cmap = colors.ListedColormap(colors_list)

    vmin = data.min()
    vmax = max(data.max(), 0.1)  # Ensure positive range exists

    # Create boundaries with n_bins below zero, 1 above zero
    boundaries = np.linspace(vmin, 0, n_bins)
    boundaries = np.append(boundaries, vmax)

    # Create the BoundaryNorm
    norm = colors.BoundaryNorm(boundaries, custom_cmap.N)

    lonlon, latlat = np.meshgrid(lonvec, latvec)
    im = m.contourf(
        lonlon,
        latlat,
        np.flipud(data),
        cmap=custom_cmap,
        norm=norm,
        levels=levelsf,
        latlon=True,
        ax=ax,
    )
    idx = np.argmin(np.abs(levelsc - shallowest_contour_depth))
    CS_water = m.contour(
        lonlon,
        latlat,
        np.flipud(data),
        colors="k",
        levels=levelsc[0 : idx + 1],
        linewidths=0.5,
        latlon=True,
        ax=ax,
    )
    m.contour(
        lonlon,
        latlat,
        np.flipud(data),
        colors="k",
        levels=levelsc[idx:],
        linewidths=0.5,
        latlon=True,
        ax=ax,
    )
    ax.clabel(
        CS_water,
        inline=True,
        fmt=lambda x: f"{abs(x):.0f}",
        fontsize=plt.rcParams["font.size"] - 2,
    )
    return im, ax


def plot_corr(
    sensors: list[str],
    corrs: list[NDArray],
    time_diffs: list[NDArray],
    window: float = 300.0,
) -> Figure:
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
    ) -> None:
        M = corr.shape[0]
        stacked_corr = np.full((M, M), np.nan)
        stacked_dt = np.full((M, M), np.nan)
        for i in range(M):
            stacked_corr[i, : M - i] = corr[i, i:]
            stacked_dt[i, : M - i] = time_diff[i, i:] / TIME_CONVERSION_FACTOR

        tvec, resampled_data = _resample_to_grid(
            stacked_dt, stacked_corr, window, num_points=n_resampled
        )
        mean_corr, lower_bounds, upper_bounds, epdf, epdf_bins = _compute_pdf_vs_time(
            resampled_data, confidence=0.95
        )
        Tgrid, Mgrid = np.meshgrid(tvec, np.arange(M), indexing="ij")
        return (
            resampled_data,
            mean_corr,
            lower_bounds,
            upper_bounds,
            epdf,
            epdf_bins,
            Tgrid,
            Mgrid,
        )

    def _resample_to_grid(dt_values, corr_values, window: float, num_points: int):
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

    # TODO: Once manuscript is ready, update y-axis label to math notation.
    fig, axes = plt.subplots(
        figsize=(6.5, 3.75),
        nrows=2,
        ncols=3,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    for j, (sensor, corr, time_diff) in enumerate(zip(sensors, corrs, time_diffs)):
        (
            resampled_data,
            mean_corr,
            lower_bounds,
            upper_bounds,
            epdf,
            epdf_bins,
            Tgrid,
            Mgrid,
        ) = _format_data(corr, time_diff, window)

        epdf_vmin = 0.01
        epdf_vmax = 0.1
        epdf[epdf < epdf_vmin] = np.nan

        tvec = Tgrid[:, 0]
        M = resampled_data.shape[0]

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
                ax.set_title(sensor.upper())
                if j == 0:
                    ax.set_ylabel("Strike index")
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel(None)
                if j == 2:
                    cax = fig.add_axes([0.91, 0.52, 0.02, 0.35])
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label("Correlation coefficient")
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
                    ax.set_xlabel("$\\tau = t_j - t_i$ (s)")
                    ax.set_ylabel("Correlation coefficient")
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
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.0),
        )

    return fig


def plot_3dvha_data(
    ds: DataStream,
    nperseg: int = 64,
    hop: int = 4,
    nfft: int | None = 2**12,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float = -60.0,
    vmax: float = 0.0,
    figsize: tuple[float] = (16, 9),
    xlabel_t: str = "Time",
    xlabel_f: str = "Time (s)",
    ylabel_t: str = "Amplitude",
    ylabel_f: str = "Frequency (Hz)",
    title: str = None,
) -> Figure:
    subplot_hspace = 0.25
    title_kwargs = {"ha": "left", "x": 0.0, "y": 0.95}

    fs = ds.stats.sampling_rate
    channels = np.arange(ds.num_channels)
    window = signal.windows.hann(nperseg)
    STFT = signal.ShortTimeFFT(window, hop, fs, mfft=nfft, scale_to="psd")

    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=-0.15)

    tfig = subfigs[0]
    taxs = tfig.subplots(ds.num_channels, 1, gridspec_kw={"hspace": subplot_hspace})
    ffig = subfigs[1]
    faxs = ffig.subplots(ds.num_channels, 1, gridspec_kw={"hspace": subplot_hspace})

    for i, channel in enumerate(channels):
        tax = taxs[i]
        tax.plot(ds.time_vector, ds.data[i] / np.max(np.abs(ds.data[i])))
        amp_lim = 1.1
        tax.set_xlim(ds.time_vector[0], ds.time_vector[-1])
        tax.set_ylim(-amp_lim, amp_lim)

        if ds.stats.metadata.get("channel_names", None) is not None:
            tax.set_title(ds.stats.metadata["channel_names"][channel], **title_kwargs)
        else:
            tax.set_title(f"Channel {channel}", **title_kwargs)

        Sxx = STFT.spectrogram(ds.data[i])
        f = STFT.f
        t = STFT.t(ds.num_samples)

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        # Normalize:
        Sxx /= Sxx.max()

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

        fax = faxs[i]
        im = plot_spectrogram(f, t, Sxx, ax=fax, vmin=vmin, vmax=vmax)

        if channel == channels[-1]:
            tax.set_xlabel(xlabel_t)
            tax.set_ylabel(f"{ylabel_t} ($\\mathrm{{{ds.stats.units}}}$)")
            fax.set_xlabel(xlabel_f)
            fax.set_ylabel(ylabel_f)
        else:
            tax.set_xticklabels([])
            tax.set_xlabel("")
            fax.set_xticklabels([])
            fax.set_xlabel("")

        cax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD ($\\mathrm{{{ds.stats.units}}}^2 / \\mathrm{{Hz}}$)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.92)
    return fig


def plot_3dvha_spectrograms(
    ds: DataStream,
    nperseg: int = 64,
    hop: int = 4,
    nfft: int | None = 2**12,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float = -60.0,
    vmax: float = 0.0,
    figsize: tuple[float] = (8, 10),
    xlabel: str = "Time (s)",
    ylabel: str = "Frequency (Hz)",
    title: str = None,
) -> Figure:
    title_kwargs = {"ha": "left", "x": 0, "y": 0.95}
    fs = ds.stats.sampling_rate
    channels = np.arange(ds.num_channels)
    window = signal.windows.hann(nperseg)
    STFT = signal.ShortTimeFFT(window, hop, fs, mfft=nfft, scale_to="psd")

    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=figsize, gridspec_kw={"hspace": 0.3}
    )
    for i, channel in enumerate(channels):
        Sxx = STFT.spectrogram(ds.data[i])
        f = STFT.f
        t = STFT.t(ds.num_samples)

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

        # Normalize:
        Sxx /= Sxx.max()

        ax = axs[i]
        im = plot_spectrogram(f, t, Sxx, ax=ax, vmin=vmin, vmax=vmax)

        if ds.stats.metadata.get("channel_names", None) is not None:
            ax.set_title(ds.stats.metadata["channel_names"][channel], **title_kwargs)
        else:
            ax.set_title(f"Channel {channel}", **title_kwargs)
        if channel == channels[-1]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD ($\\mathrm{{{ds.stats.units}}}^2 / \\mathrm{{Hz}}$)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.92)
    return fig


def plot_all_acoustic_data(
    data: list[DataStream],
    nperseg: int = 64,
    hop: int = 4,
    nfft: int | None = 2**12,
    fmin: float | None = None,
    fmax: float | None = None,
    figsize: tuple[float] = (16, 9),
    title: str = None,
) -> None:
    def extract_data(data: list[DataStream]):
        ch_names = []
        data_vectors = []
        time_vectors = []
        sampling_rates = []
        Sxx_matrices = []
        Sxx_freqs = []
        Sxx_times = []
        units = []

        for ds in data:
            ch_names.extend(ds.stats.metadata["channel_names"])
            for i in range(ds.num_channels):
                data_vectors.append(ds.data[i])
                time_vectors.append(ds.time_vector)
                sampling_rates.append(ds.stats.sampling_rate)

                fs = ds.stats.sampling_rate
                window = signal.windows.hann(nperseg)
                STFT = signal.ShortTimeFFT(window, hop, fs, mfft=nfft, scale_to="psd")

                Sxx_matrices.append(STFT.spectrogram(ds.data[i]))
                Sxx_freqs.append(STFT.f)
                Sxx_times.append(STFT.t(ds.num_samples))

                units.append(ds.stats.units)

        return (
            ch_names,
            data_vectors,
            time_vectors,
            sampling_rates,
            Sxx_matrices,
            Sxx_freqs,
            Sxx_times,
            units,
        )

    (
        ch_names,
        data_vectors,
        time_vectors,
        sampling_rates,
        Sxx_matrices,
        Sxx_freqs,
        Sxx_times,
        units,
    ) = extract_data(data)

    num_rows = len(ch_names)

    subplot_hspace = 0.25
    title_kwargs = {"ha": "left", "x": 0.05, "y": 0.95}

    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=-0.15)

    tfig = subfigs[0]
    taxs = tfig.subplots(num_rows, 1, gridspec_kw={"hspace": subplot_hspace})
    ffig = subfigs[1]
    faxs = ffig.subplots(num_rows, 1, gridspec_kw={"hspace": subplot_hspace})

    for i in range(num_rows):
        time = time_vectors[i]
        tax = taxs[i]
        tax.plot(time, data_vectors[i])
        tax.set_xlim(time[0], time[-1])
        tax.set_title(ch_names[i], **title_kwargs)

        f = Sxx_freqs[i]
        t = Sxx_times[i]
        Sxx = Sxx_matrices[i]
        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        # Normalize:
        Sxx /= Sxx.max()

        fax = faxs[i]
        im = plot_spectrogram(f, t, Sxx, ax=fax, vmin=-60.0, vmax=0.0)

        if i == num_rows - 1:
            tax.set_xlabel("Time")
            tax.set_ylabel(f"Amplitude ($\\mathrm{{{units[i]}}}$)")
            fax.set_xlabel("Time (s)")
            fax.set_ylabel("Frequency (Hz)")
        else:
            tax.set_xticklabels([])
            tax.set_xlabel("")
            fax.set_xticklabels([])
            fax.set_xlabel("")

        cax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD (Normalized dB)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.92)
    return fig


def plot_template(
    traces: NDArray,
    template: NDArray | None = None,
    reference_ind: int | None = None,
    title: str = None,
    ylim: Sequence[float] | None = None,
    figsize: tuple[float] = (4, 10),
) -> Figure:
    fig, axes = plt.subplots(
        nrows=4,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 0.25, 0.25, 0.25], "hspace": 0.3},
    )
    ax = axes[0]

    # Determine vertical spacing for stacking
    n_traces = traces.shape[0]
    trace_max = np.nanmax(np.abs(traces))
    # Handle NaN or zero trace_max
    if np.isnan(trace_max) or trace_max == 0:
        offset_spacing = 1.0
    else:
        offset_spacing = 2.5 * trace_max

    # Plot traces stacked from top, with reference trace in blue
    offset_idx = 0
    ytick_positions = []
    ytick_labels = []

    for i in range(n_traces):
        if i == reference_ind:
            ax.plot(
                traces[i] - offset_idx * offset_spacing,
                "b",
                label="Reference",
                linewidth=0.8,
            )
        else:
            ax.plot(
                traces[i] - offset_idx * offset_spacing,
                "k",
                linewidth=0.8,
                alpha=0.7,
            )
        ytick_positions.append(-offset_idx * offset_spacing)
        ytick_labels.append(f"{i}")
        offset_idx += 1

    # Plot template at the bottom if provided
    ax.plot(template - offset_idx * offset_spacing, "r", label="Template", linewidth=1)
    ytick_positions.append(-offset_idx * offset_spacing)
    ytick_labels.append("Template")

    ax.set_xlim(0, traces.shape[1])
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.legend(loc="upper left")
    ax.set_xlabel("Sample Index")

    ax = axes[1]
    for i in range(n_traces):
        if i == reference_ind:
            ax.plot(traces[i], "b", label="Reference", linewidth=0.8)
        else:
            ax.plot(traces[i], "k", linewidth=0.8, alpha=0.7)
    ax.plot(template, "r", label="Template", zorder=20)
    ax.set_xlim(0, traces.shape[1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")

    ax = axes[2]
    ax.plot(traces[reference_ind], "b", label="Reference", linewidth=0.8)
    ax.plot(template, "r", label="Template", zorder=20)
    ax.set_xlim(0, traces.shape[1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")

    ax = axes[3]
    diff = template - traces[reference_ind]
    ax.plot(diff, "tab:green", label="Residual")
    ax.set_xlim(0, traces.shape[1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")

    fig.suptitle(title, y=0.92)
    return fig


def plot_template_detail(
    reference: NDArray,
    template: NDArray,
    xcorr: NDArray,
    lags: NDArray,
    title: str | None = None,
) -> Figure:

    peak_lag = lags[np.argmax(xcorr)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    ax = axes[0]
    ax.plot(reference, "k", label="Reference")
    ax.plot(template, "b", label="Aligned Template")
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()

    ax = axes[1]
    ax.plot(lags, xcorr, label="Cross-correlation")
    ax.axvline(peak_lag, color="r", linestyle="--", label="Peak Lag")
    ax.grid()
    ax.set_title("Cross-correlation of Aligned Template and Reference")
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Cross-correlation")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_shru_data(
    ds: DataStream,
    nperseg: int = 64,
    hop: int = 4,
    nfft: int | None = 2**12,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float = 70.0,
    vmax: float = 130.0,
    figsize: tuple[float] = (16, 9),
    xlabel_t: str = "Time",
    xlabel_f: str = "Time (s)",
    ylabel_t: str = "Amplitude",
    ylabel_f: str = "Frequency (Hz)",
    title: str = None,
) -> Figure:
    subplot_hspace = 0.1

    fs = ds.stats.sampling_rate
    channels = np.arange(ds.num_channels)
    window = signal.windows.hann(nperseg)
    STFT = signal.ShortTimeFFT(window, hop, fs, mfft=nfft, scale_to="psd")

    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=-0.15)

    tfig = subfigs[0]
    taxs = tfig.subplots(ds.num_channels, 1, gridspec_kw={"hspace": subplot_hspace})
    ffig = subfigs[1]
    faxs = ffig.subplots(ds.num_channels, 1, gridspec_kw={"hspace": subplot_hspace})

    for i, channel in enumerate(channels):
        tax = taxs[i]
        tax.plot(ds.time_vector, ds.data[i])
        amp_lim = 1.1 * np.max(np.abs(ds.data))
        tax.set_xlim(ds.time_vector[0], ds.time_vector[-1])
        tax.set_ylim(-amp_lim, amp_lim)
        tax.set_title(
            f"Channel {channel}", ha="center", va="center", x=-0.075, y=0.5, rotation=90
        )

        Sxx = STFT.spectrogram(ds.data[i])
        f = STFT.f
        t = STFT.t(ds.num_samples)

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

        fax = faxs[i]
        im = plot_spectrogram(f, t, Sxx, ax=fax, vmin=vmin, vmax=vmax)

        if channel == channels[-1]:
            tax.set_xlabel(xlabel_t)
            tax.set_ylabel(f"{ylabel_t} ($\\mathrm{{{ds.stats.units}}}$)")
            fax.set_xlabel(xlabel_f)
            fax.set_ylabel(ylabel_f)
        else:
            tax.set_xticklabels([])
            tax.set_xlabel("")
            fax.set_xticklabels([])
            fax.set_xlabel("")

        cax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD ($\\mathrm{{{ds.stats.units}}}^2 / \\mathrm{{Hz}}$)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.92)
    return fig


def plot_shru_spectrograms(
    ds: DataStream,
    nperseg: int = 64,
    hop: int = 4,
    nfft: int | None = 2**12,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float = 70.0,
    vmax: float = 130.0,
    figsize: tuple[float] = (8, 6),
    xlabel: str = "Time (s)",
    ylabel: str = "Frequency (Hz)",
    title: str = None,
) -> Figure:
    fs = ds.stats.sampling_rate
    channels = np.arange(ds.num_channels)
    window = signal.windows.hann(nperseg)
    STFT = signal.ShortTimeFFT(window, hop, fs, mfft=nfft, scale_to="psd")

    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=figsize, gridspec_kw={"hspace": 0.3}
    )
    if ds.num_channels == 1:
        axs = [axs]

    for i, channel in enumerate(channels):
        Sxx = STFT.spectrogram(ds.data[i])
        f = STFT.f
        t = STFT.t(ds.num_samples)

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

        ax = axs[i]
        im = plot_spectrogram(f, t, Sxx, ax=ax, vmin=vmin, vmax=vmax)

        ax.set_title(f"Channel {channel}", fontsize=10, ha="left", x=0)
        if channel == channels[-1]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD ($\\mathrm{{{ds.stats.units}}}^2 / \\mathrm{{Hz}}$)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.95)
    return fig


def plot_spectrogram(f, t, Sxx, ax=None, vmin=None, vmax=None) -> Axes:
    if ax is None:
        ax = plt.gca()
    extent = (t[0], t[-1], f[0], f[-1])
    return ax.imshow(
        10 * np.log10(Sxx),
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )


# def plot_spectrogram(
#     ds: DataStream,
#     channel=0,
#     nfft: int = 2**15,
#     nperseg: int = 128,
#     noverlap: int = 64,
#     xlabel: str = "Time (s)",
#     ylabel: str = "Frequency (Hz)",
#     title: str = None,
#     vmin: float = 70,
#     vmax: float = 130,
#     ax=None,
# ) -> Axes:
#     fs = ds.stats.sampling_rate

#     f, t, Sxx = signal.spectrogram(
#         ds.data[channel],
#         fs=fs,
#         noverlap=noverlap,
#         nperseg=nperseg,
#         nfft=nfft,
#     )

#     fig, ax = plt.subplots(figsize=(10, 5))
#     im = ax.pcolormesh(
#         t,
#         f,
#         10 * np.log10(Sxx),
#         shading="auto",
#         cmap="magma",
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("Power Spectral Density (dB re 1 $\\mu$Pa/Hz$^2$)")

#     return fig


def plot_study_area(
    bathy_data: NDArray[np.float64],
    lonvec: NDArray[np.float64],
    latvec: NDArray[np.float64],
    equipment_df: pd.DataFrame | None = None,
    turbines_df: pd.DataFrame | None = None,
    active_turbine: dict | None = None,
    bounds: list[list[float]] | None = None,
    ax: Axes | None = None,
    scale_bar: int = 1,
    shallowest_contour_depth: float = 0.0,
    levelsf=np.arange(-100, 10, 5),
    levelsc=np.arange(-100, 1, 5),
    meridians: float = 0.2,
    parallels: float = 0.2,
    meridian_labels: list[int] = [0, 0, 1, 0],
    parallel_labels: list[int] = [1, 0, 0, 0],
    show_legend: bool = True,
) -> tuple[Axes, Basemap]:
    if bounds is None:
        llcrnrlat = np.min(latvec)
        urcrnrlat = np.max(latvec)
        llcrnrlon = np.min(lonvec)
        urcrnrlon = np.max(lonvec)
        bounds = np.array([[llcrnrlon, urcrnrlon], [llcrnrlat, urcrnrlat]])
    else:
        llcrnrlon = bounds[0][0]
        urcrnrlon = bounds[0][1]
        llcrnrlat = bounds[1][0]
        urcrnrlat = bounds[1][1]

    if ax is None:
        ax = plt.gca()

    m = Basemap(
        projection="tmerc",
        llcrnrlat=llcrnrlat,
        urcrnrlat=urcrnrlat,
        llcrnrlon=llcrnrlon,
        urcrnrlon=urcrnrlon,
        resolution="f",
        lon_0=np.mean(lonvec),
        lat_0=np.mean(latvec),
    )
    m.drawmeridians(
        np.arange(llcrnrlon, urcrnrlon, meridians), labels=meridian_labels, ax=ax
    )
    m.drawparallels(
        np.arange(llcrnrlat, urcrnrlat, parallels), labels=parallel_labels, ax=ax
    )
    xlim = m(np.array(bounds[0]), np.ones_like(bounds[0]) * np.mean(bounds[1]))[0]
    ylim = m(np.ones_like(bounds[0]) * np.mean(bounds[0]), np.array(bounds[1]))[1]

    _, ax = plot_bathy(
        bathy_data,
        lonvec=lonvec,
        latvec=latvec,
        m=m,
        ax=ax,
        shallowest_contour_depth=shallowest_contour_depth,
        levelsf=levelsf,
        levelsc=levelsc,
    )

    if turbines_df is not None:
        ax.scatter(
            *m(turbines_df["lon"], turbines_df["lat"]),
            marker="h",
            c="yellow",
            edgecolors="k",
            # linewidth=1,
            # s=150,
            zorder=20,
            label="Turbines",
        )

    if active_turbine is not None:
        ax.scatter(
            *m(active_turbine["longitude"], active_turbine["latitude"]),
            marker="h",
            c="tab:orange",
            edgecolors="k",
            # linewidth=1,
            # s=150,
            zorder=30,
            label=active_turbine["label"],
        )

    if equipment_df is not None:
        ax.scatter(
            *m(equipment_df["longitude"], equipment_df["latitude"]),
            marker="v",
            c="tab:green",
            edgecolors="k",
            # linewidth=1,
            zorder=20,
            label="Hydrophone Array",
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if show_legend:
        leg = ax.legend(
            facecolor="white",
            edgecolor="black",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
        )
        leg.get_frame().set_alpha(None)

    scalebar = AnchoredSizeBar(
        ax.transData,
        scale_bar * 1e3,
        f"{scale_bar:d} km",
        "lower right",
        pad=0.1,
        color="k",
        frameon=True,
        size_vertical=20 * scale_bar,
        zorder=50,
    )
    ax.add_artist(scalebar)

    return ax, m
