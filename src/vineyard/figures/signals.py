from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

import vineyard.readers as readers


def next_pow2(n: int) -> int:
    """Calculate the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


def _plot_sensor_column(
    ax_time: Axes,
    ax_spec: Axes,
    times: np.ndarray,
    data: np.ndarray,
    fs: float,
    col_title: str,
    is_first_col: bool,
    nperseg: int,
    hop: int,
    nfft: int,
    flim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    vmin: float | None = 70,
    vmax: float | None = 130,
):
    """Plot time series and spectrogram for a single sensor column.

    Returns the image object from the spectrogram plot.
    """
    # Compute spectrogram
    window = signal.windows.hann(nperseg)
    STFT = signal.ShortTimeFFT(window, hop, fs, mfft=nfft, scale_to="psd")
    Sxx = STFT.spectrogram(data)

    f = STFT.f
    df = f[1] - f[0]
    t = STFT.t(len(data))

    if flim is not None:
        fmin, fmax = flim
        Sxx = Sxx[(f >= fmin - df) & (f <= fmax + df), :]
        f = f[(f >= fmin - df) & (f <= fmax + df)]

    # Plot time series
    ax_time.plot(times, data, color="k")
    ax_time.set_title(col_title)
    if is_first_col:
        ax_time.set_ylabel("Amplitude (μPa)")
    ax_time.set_xlim(times[0], times[-1])
    ax_time.set_ylim(ylim)
    ax_time.grid()
    ax_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_time.set_ylim(ylim)  # Re-apply ylim to ensure it sticks

    # Plot spectrogram
    im = plot_spectrogram(f, t, Sxx, ax=ax_spec, vmin=vmin, vmax=vmax)
    if is_first_col:
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlim(times[0], times[-1])
    ax_spec.set_ylim(flim)

    return im


def plot_signal_templates(
    inventory_dir: Path,
    whale_sensors: list[dict[str, any]],
    strike_sensors: list[dict[str, any]],
    col_titles: list[str],
    filt_type: str | None = None,
    filt_freq: list[float] | float | None = None,
    nperseg: int = 4096,
    hop: int = 2048,
    nfft: int | None = None,
    flim: tuple[float, float] | None = None,
    whale_ylim: tuple[float, float] | None = None,
    calibration_dir: Path | None = None,
    strike_ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> Figure:

    fig = plt.figure(figsize=figsize)

    # Create main grid: 2 rows, 3 columns
    # Top row for whale signals, bottom row for other signals
    main_gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.15)

    # Create nested subgrids for each column in both rows
    # Each column will have time series above spectrogram
    whale_axes = []
    strike_axes = []
    for i in range(3):
        # Top row (whale signals)
        sub_gs_whale = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=main_gs[0, i], hspace=0.0, height_ratios=[0.3, 0.7]
        )
        ax_time_whale = fig.add_subplot(sub_gs_whale[0])
        ax_spec_whale = fig.add_subplot(sub_gs_whale[1])
        whale_axes.append((ax_time_whale, ax_spec_whale))

        # Bottom row (other signals)
        sub_gs_other = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=main_gs[1, i], hspace=0.0, height_ratios=[0.3, 0.7]
        )
        ax_time_other = fig.add_subplot(sub_gs_other[0])
        ax_spec_other = fig.add_subplot(sub_gs_other[1])
        strike_axes.append((ax_time_other, ax_spec_other))

    if nfft is None:
        nfft = next_pow2(nperseg)

    # Collect spectrogram axes for colorbar
    spec_image = None

    # Plot whale sensors (top row)
    for i, sensor in enumerate(whale_sensors):
        ds = readers.read_and_process(
            inventory_dir / f"inventory_{sensor['name']}.csv",
            start=np.datetime64(sensor["peak_time"], TIME_PRECISION)
            - np.timedelta64(
                int(TIME_CONVERSION_FACTOR * sensor["start_buffer"]), TIME_PRECISION
            ),
            end=np.datetime64(sensor["peak_time"], TIME_PRECISION)
            + np.timedelta64(
                int(TIME_CONVERSION_FACTOR * sensor["end_buffer"]), TIME_PRECISION
            ),
            channels=sensor["channel"],
            filt_type=filt_type,
            filt_freq=filt_freq,
        )

        fs = ds.stats.sampling_rate
        times = np.arange(ds.data.shape[1]) / fs
        data = ds.data[0]

        if calibration_dir is not None:
            match sensor["name"]:
                case "3dvha":
                    cal_file = Path(calibration_dir) / f"{sensor['name']}_cal.csv"
                    data = readers.calibrate_3dvha(cal_file, data, fs)
                case "vla1" | "vla2":
                    cal_file = Path(calibration_dir) / f"{sensor['name']}_cal.toml"
                    data = readers.calibrate_vla(cal_file, data, fs)

        ax_time, ax_spec = whale_axes[i]
        im = _plot_sensor_column(
            ax_time=ax_time,
            ax_spec=ax_spec,
            times=times,
            data=data,
            fs=fs,
            col_title=f"Fin whale call at {col_titles[i]}",
            is_first_col=(i == 0),
            nperseg=nperseg,
            hop=hop,
            nfft=nfft,
            flim=flim,
            ylim=whale_ylim,
            vmin=70,
            vmax=130,
        )
        if spec_image is None:
            spec_image = im

    # Plot other sensors (bottom row) if provided
    if strike_sensors is not None:
        for i, sensor in enumerate(strike_sensors):
            ds = readers.read_and_process(
                inventory_dir / f"inventory_{sensor['name']}.csv",
                start=np.datetime64(sensor["peak_time"], TIME_PRECISION)
                - np.timedelta64(
                    int(TIME_CONVERSION_FACTOR * sensor["start_buffer"]), TIME_PRECISION
                ),
                end=np.datetime64(sensor["peak_time"], TIME_PRECISION)
                + np.timedelta64(
                    int(TIME_CONVERSION_FACTOR * sensor["end_buffer"]), TIME_PRECISION
                ),
                channels=sensor["channel"],
                filt_type=filt_type,
                filt_freq=filt_freq,
            )

            fs = ds.stats.sampling_rate
            times = np.arange(ds.data.shape[1]) / fs
            data = ds.data[0]

            if calibration_dir is not None:
                match sensor["name"]:
                    case "3dvha":
                        cal_file = Path(calibration_dir) / f"{sensor['name']}_cal.csv"
                        data = readers.calibrate_3dvha(cal_file, data, fs)
                    case "vla1" | "vla2":
                        cal_file = Path(calibration_dir) / f"{sensor['name']}_cal.toml"
                        data = readers.calibrate_vla(cal_file, data, fs)

            ax_time, ax_spec = strike_axes[i]
            im = _plot_sensor_column(
                ax_time=ax_time,
                ax_spec=ax_spec,
                times=times,
                data=data,
                fs=fs,
                col_title=f"Pile driving strikes at {col_titles[i]}",
                is_first_col=(i == 0),
                nperseg=nperseg,
                hop=hop,
                nfft=nfft,
                flim=flim,
                ylim=strike_ylim,
                vmin=70,
                vmax=130,
            )
            if spec_image is None:
                spec_image = im

    # Add colorbar manually positioned on the right side
    if spec_image is not None:
        # Create a colorbar axis manually: [left, bottom, width, height] in figure coordinates
        cbar_ax = fig.add_axes(
            [0.92, 0.15, 0.02, 0.7]
        )  # Adjust position and size as needed
        cbar = fig.colorbar(spec_image, cax=cbar_ax)
        cbar.set_label(
            "Power Spectral Density (dB re 1 μPa²/Hz)", rotation=270, labelpad=15
        )

    return fig


def plot_spectrogram(f, t, Sxx, ax=None, vmin=None, vmax=None) -> Axes:
    if ax is None:
        ax = plt.gca()
    extent = (t[0], t[-1], f[0], f[-1])
    Sxx_dB = 10 * np.log10(Sxx)
    return ax.imshow(
        Sxx_dB,
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
