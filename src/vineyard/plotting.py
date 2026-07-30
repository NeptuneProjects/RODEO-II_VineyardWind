import string
from pathlib import Path

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import signal
from scipy.interpolate import interp1d
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_CONVERSION_FACTOR

FIG_STYLE = Path("config/scirep_fig_style.mplstyle")
plt.style.use(FIG_STYLE)

SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
    "facecolor": "white",
}


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
            _,
            _,
            _,
            epdf,
            _,
            Tgrid,
            Mgrid,
        ) = _format_data(corr, time_diff, window)

        epdf_vmin = 0.01
        epdf_vmax = 0.1
        epdf[epdf < epdf_vmin] = np.nan

        tvec = Tgrid[:, 0]
        # M = resampled_data.shape[0]

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
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.0},
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
    title: str | None = None,
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
    title: str | None = None,
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
    title: str | None = None,
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
        _,
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
        cbar.set_label("PSD (Normalized dB)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.92)
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
    title: str | None = None,
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
    title: str | None = None,
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
