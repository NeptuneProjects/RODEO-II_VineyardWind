from collections.abc import Sequence
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT, find_peaks, get_window, hilbert
from tritonoa.data.reader import read_hdf5, read_hdf5_group
from tritonoa.data.stream import DataStream

from vineyard.figures import add_panel_label


def load_plotting_data(
    data_dir: Path,
    sensor: str,
    template_data: Path,
    time_start: np.datetime64,
    time_end: np.datetime64,
    filt_type: str | None = None,
    filt_freq: Sequence[float] | float | None = None,
) -> tuple[DataStream, DataStream]:
    with h5py.File(template_data, "r") as f:
        whale_ds = read_hdf5_group(f[sensor]["type2"])

    ds = (
        read_hdf5(data_dir / f"{sensor}_pc.h5")
        .trim(time_start, time_end)
        .filter(filt_type, filt_freq)
        if filt_type and filt_freq
        else read_hdf5(data_dir / f"{sensor}_pc.h5").trim(time_start, time_end)
    )
    return ds, whale_ds


def plot_denoising(
    data_dir: Path,
    sensor: str,
    template_data: Path,
    time_start: np.datetime64,
    time_end: np.datetime64,
    freq_time_start: np.datetime64,
    freq_time_end: np.datetime64,
    filt_type: str | None = None,
    filt_freq: Sequence[float] | float | None = None,
    window: str = "hann",
    nperseg: int = 16384,
    hop: int = 8192,
    flim: list[float] = [15, 50],
) -> plt.Figure:
    ds, whale_ds = load_plotting_data(
        data_dir,
        sensor,
        template_data,
        time_start,
        time_end,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )

    # Create main figure with two subfigures side-by-side
    fig = plt.figure(figsize=(10, 6))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=-0.12)

    # Plot into each subfigure
    plot_denoising_time(
        ds,
        whale_ds,
        subfig=subfigs[0],
        freq_time_start=freq_time_start,
        freq_time_end=freq_time_end,
    )
    plot_denoising_freq(
        ds.trim(freq_time_start, freq_time_end),
        window=window,
        nperseg=nperseg,
        hop=hop,
        flim=flim,
        subfig=subfigs[1],
    )

    return fig


def plot_denoising_freq(
    ds: DataStream,
    window: str = "hann",
    nperseg: int = 16384,
    hop: int = 8192,
    flim: list[float] = [15, 50],
    subfig: plt.Figure | None = None,
) -> plt.Figure:
    fs = ds.stats.sampling_rate
    STFT = ShortTimeFFT(
        fs=fs, hop=hop, mfft=nperseg, win=get_window(window, nperseg), scale_to="psd"
    )
    freq = STFT.f
    df = freq[1] - freq[0]

    Zxx_orig = STFT.spectrogram(ds.data[0])
    Zxx_orig[Zxx_orig == 0] = 1e-12
    Zxx_orig_db = 10 * np.log10(Zxx_orig)

    Zxx_filt = STFT.spectrogram(ds.data[1])
    Zxx_filt[Zxx_filt == 0] = 1e-12

    Zxx_filt_db = 10 * np.log10(Zxx_filt)
    Zxx_diff_db = Zxx_filt_db - Zxx_orig_db

    if flim is not None:
        fmin, fmax = flim
        Zxx_orig_db = Zxx_orig_db[(freq >= fmin - df) & (freq <= fmax + df), :]
        Zxx_filt_db = Zxx_filt_db[(freq >= fmin - df) & (freq <= fmax + df), :]
        Zxx_diff_db = Zxx_diff_db[(freq >= fmin - df) & (freq <= fmax + df), :]
        freq = freq[(freq >= fmin - df) & (freq <= fmax + df)]

    time = STFT.t(ds.num_samples)
    time_string = ds.stats.time_init + time.astype("timedelta64[s]")
    time_mpl = mdates.date2num(time_string.astype("datetime64[ms]").astype(object))

    plotting_data = [
        {
            "data": Zxx_orig_db,
            "title": "Original signal",
            "vmin": 60,
            "vmax": 120,
            "cmap": "inferno",
            "cblabel": "PSD (dB re 1 μPa²/Hz)",
        },
        {
            "data": Zxx_filt_db,
            "title": "Denoised signal",
            "vmin": 60,
            "vmax": 120,
            "cmap": "inferno",
            "cblabel": "PSD (dB re 1 μPa²/Hz)",
        },
        {
            "data": Zxx_diff_db,
            "title": "Difference (Denoised - Original)",
            "vmin": -30,
            "vmax": 30,
            "cmap": "bwr",
            "cblabel": "PSD Difference (dB re 1 μPa²/Hz)",
        },
    ]

    # Create subplots within subfigure if provided, otherwise create new figure
    if subfig is not None:
        axes = subfig.subplots(nrows=3)
        fig = subfig.figure
    else:
        fig, axes = plt.subplots(nrows=3, figsize=(8, 8))

    last_row = len(plotting_data) - 1
    extent = [time_mpl[0], time_mpl[-1], freq[0], freq[-1]]

    for i, (ax, plot_data) in enumerate(zip(axes, plotting_data)):
        im = ax.imshow(
            plot_data["data"],
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=plot_data["vmin"],
            vmax=plot_data["vmax"],
            cmap=plot_data["cmap"],
            interpolation="none",
        )
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(
            locator, offset_formats=["%Y-%b-%d"] * 6
        )
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        ax.set_ylim(flim)
        ax.set_title(plot_data["title"])

        if i != last_row:
            ax.set_xticklabels([])

        if i == last_row:
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Frequency (Hz)")

        add_panel_label(ax, label=chr(101 + i))

        # Add colorbar to the right of each subplot
        if subfig is not None:
            # For subfigures, use subfig.colorbar with axes positioning
            cax = subfig.add_axes(
                [0.92, ax.get_position().y0, 0.02, ax.get_position().height]
            )
            cbar = subfig.colorbar(im, cax=cax)
        else:
            # For standalone figures
            cax = fig.add_axes(
                [0.92, ax.get_position().y0, 0.02, ax.get_position().height]
            )
            cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(plot_data["cblabel"], rotation=270, labelpad=10)

    return fig


def plot_denoising_time(
    signal_ds: DataStream,
    template_ds: DataStream,
    subfig: plt.Figure | None = None,
    freq_time_start: np.datetime64 | None = None,
    freq_time_end: np.datetime64 | None = None,
) -> plt.Figure:
    fs = signal_ds.stats.sampling_rate
    time = signal_ds.time_vector
    template = template_ds.data[0]
    template_energy = np.sum(np.abs(template) ** 2)

    plotting_data = [
        {
            "data": signal_ds.data[0],
            "ylabel": "Amplitude (μPa)",
            "ylim": (-5e7, 5e7),
            "title": "Original signal",
            "color": "tab:blue",
        },
        {
            "data": signal_ds.data[1],
            "ylabel": "Amplitude (μPa)",
            "ylim": (-5e7, 5e7),
            "title": "Denoised signal",
            "color": "tab:green",
        },
        {
            "data": np.abs(hilbert(signal_ds.data[3] / template_energy)),
            "ylabel": "Normalized correlation",
            "ylim": (0, 2.1),
            "title": "Matched filter output with original signal",
            "color": "tab:blue",
            "threshold": 1.2,
            "distance": 7.0,
        },
        {
            "data": np.abs(hilbert(signal_ds.data[5] / template_energy)),
            "ylabel": "Normalized correlation",
            "ylim": (0, 2.1),
            "title": "Matched filter output with denoised signal",
            "color": "tab:green",
            "threshold": 0.27,
            "distance": 7.0,
        },
    ]

    # Create subplots within subfigure if provided, otherwise create new figure
    if subfig is not None:
        axes = subfig.subplots(nrows=4, gridspec_kw={"hspace": 0.25})
        fig = subfig.figure
    else:
        fig, axes = plt.subplots(nrows=4, figsize=(8, 8))

    last_row = len(plotting_data) - 1

    for i, (ax, plot_data) in enumerate(zip(axes, plotting_data)):
        ax.plot(time, plot_data["data"], color=plot_data["color"])
        if freq_time_start and freq_time_end:
            ax.axvspan(
                freq_time_start.astype("datetime64[ms]").astype(object),
                freq_time_end.astype("datetime64[ms]").astype(object),
                facecolor="gray",
                edgecolor="none",
                alpha=0.25,
            )
        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel(plot_data.get("ylabel", None))
        ax.set_ylim(plot_data.get("ylim", None))
        ax.set_title(plot_data.get("title", None), y=0.97)
        ax.grid()

        # Use ConciseDateFormatter for automatic date/time display
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(
            locator, offset_formats=["%Y-%b-%d"] * 6
        )
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        if i != last_row:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (UTC)")

        if i == last_row - 1:
            peaks, _ = find_peaks(
                plot_data["data"],
                height=plot_data["threshold"],
                distance=plot_data["distance"] * fs,
            )
            ax.axhline(
                plot_data["threshold"],
                color="tab:red",
                linestyle="--",
                label="Detection threshold",
            )
            ax.plot(
                time[peaks],
                plot_data["data"][peaks],
                "x",
                color="tab:red",
                label="Detections",
            )
            ax.legend(
                loc="lower left",
                framealpha=1.0,
                borderpad=0.2,
            )
        if i == last_row:
            peaks, _ = find_peaks(
                plot_data["data"],
                height=plot_data["threshold"],
                distance=plot_data["distance"] * fs,
            )
            ax.axhline(
                plot_data["threshold"],
                color="tab:red",
                linestyle="--",
                label="Detection threshold",
            )
            ax.plot(
                time[peaks],
                plot_data["data"][peaks],
                "x",
                color="tab:red",
                label="Peaks",
            )

        add_panel_label(ax, label=chr(97 + i))

    return fig
