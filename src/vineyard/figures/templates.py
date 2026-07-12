"""Signal template construction figure creation."""

import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

import vineyard.readers as readers
from vineyard.figures.common import add_panel_label, format_tick_scientific
from vineyard.process_utils import (
    enforce_same_size,
    extract_trace,
    get_anchor_trace,
    sample_delay,
)


def plot_strike_template(
    strike_index_path: Path,
    strike_corr_path: Path,
    inventory_path: Path,
    calibration_dir: Path,
    sensor_name: str,
    channel: int,
    strike_indices: list[int],
    start_time: np.datetime64,
    end_time: np.datetime64,
    buffer_start: float,
    buffer_end: float,
    window_size: int,
    ylim: tuple[float, float] | None = None,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | list[float] | None = None,
) -> Figure:
    """Plot templates for multiple strike indices in a multi-panel figure.

    Args:
        strike_index_path: Path to the strike index CSV file.
        strike_corr_path: Path to the correlation matrix HDF5 file.
        inventory_path: Path to the inventory CSV file for the sensor.
        calibration_dir: Path to the calibration directory.
        sensor_name: Name of the sensor.
        channel: Channel number.
        strike_indices: List of strike indices to plot.
        start_time: Start time of the data range.
        end_time: End time of the data range.
        buffer_start: Buffer before the strike peak (seconds).
        buffer_end: Buffer after the strike peak (seconds).
        window_size: Size of the rolling window.
        ylim: Optional y-axis limits.
        taper_pc: Taper percentage for data processing.
        dec_factor: Decimation factor for data processing.
        filt_type: Filter type for data processing.
        filt_freq: Filter frequency for data processing.

    Returns:
        Matplotlib Figure.
    """
    # Read the acoustic data
    logging.info(f"Reading acoustic data for {sensor_name}...")
    ds, strike_index = readers.read_strike_data(
        inventory_path,
        strike_index_path,
        sensor_name,
        channel,
        start_time,
        end_time,
        buffer_start,
        buffer_end,
        taper_pc=taper_pc,
        dec_factor=dec_factor,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )

    # Apply calibration
    ds.data = readers.calibrate(
        calibration_dir, ds.data, ds.stats.sampling_rate, sensor_name
    )
    ds.stats.units = "uPa"

    # Read correlation matrix
    logging.info(f"Reading correlation matrix for {sensor_name}...")
    corr_matrix, _, _ = readers.read_xcorr_data(strike_corr_path, sensor_name)

    # Get strike information
    num_strikes = len(strike_index)
    fs = ds.stats.sampling_rate
    half_window = window_size // 2

    # Create figure with multiple columns
    n_panels = len(strike_indices)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=n_panels,
        figsize=(3 * n_panels, 4.5),
        gridspec_kw={
            "height_ratios": [1, 0.25, 0.25, 0.25],
            "hspace": 0.0,
            "wspace": 0.05,
        },
    )

    # Ensure axes is 2D even if only one strike
    if n_panels == 1:
        axes = axes[:, np.newaxis]

    # Process each strike
    for panel_idx, strike_idx in enumerate(strike_indices):
        # Validate strike index
        if strike_idx < 0 or strike_idx >= num_strikes:
            raise ValueError(
                f"Strike index {strike_idx} out of range [0, {num_strikes - 1}]"
            )

        # Determine window boundaries
        if strike_idx < half_window:
            window_start = 0
            window_end = min(window_size, num_strikes)
        elif strike_idx >= num_strikes - half_window:
            window_start = max(0, num_strikes - window_size)
            window_end = num_strikes
        else:
            window_start = strike_idx - half_window
            window_end = strike_idx + half_window

        logging.info(
            f"Processing strike {strike_idx} with window [{window_start}, {window_end})"
        )

        # Get indices of strikes in the window
        template_inds = list(range(window_start, window_end))

        # Get the anchor trace
        corr_matrix_window = corr_matrix[
            window_start:window_end, window_start:window_end
        ]
        anchor_index = get_anchor_trace(corr_matrix_window)
        anchor_trace = extract_trace(
            ds,
            np.datetime64(strike_index.item(template_inds[anchor_index], "start_time")),
            np.datetime64(strike_index.item(template_inds[anchor_index], "end_time")),
        )

        # Extract and align all traces in the window
        traces = []
        reference_ind = None

        for idx, j in enumerate(template_inds):
            tr_start = np.datetime64(strike_index.item(j, "start_time"))
            tr_end = np.datetime64(strike_index.item(j, "end_time"))

            if idx == anchor_index:
                aligned_tr = anchor_trace
            else:
                tr = extract_trace(ds, tr_start, tr_end)
                shift_samples = sample_delay(anchor_trace, tr)
                shift_seconds = shift_samples / fs

                aligned_tr = extract_trace(
                    ds,
                    tr_start
                    - np.timedelta64(
                        int(shift_seconds * TIME_CONVERSION_FACTOR), TIME_PRECISION
                    ),
                    tr_end
                    - np.timedelta64(
                        int(shift_seconds * TIME_CONVERSION_FACTOR), TIME_PRECISION
                    ),
                )

            if strike_idx == j:
                reference_ind = idx

            traces.append(aligned_tr)

        # Compute template from aligned traces
        traces = np.array(enforce_same_size(traces))
        template = np.median(traces, axis=0)
        time = np.arange(0, template.shape[0]) / fs

        # Create the plot in the current column
        title = (
            f"Strike $i = {strike_idx}$, "
            f"{strike_index.item(strike_idx, 'start_time').strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        plot_template(
            time,
            traces,
            template,
            reference_ind,
            title=title,
            ylim=ylim,
            offset_spacing_factor=1.0,
            axes=axes[:, panel_idx],
            show_ylabel=(panel_idx == 0),  # Only show y-label on first panel
            show_legend=(panel_idx == 0),  # Only show legend on first panel
        )

        add_panel_label(axes[0, panel_idx], label=chr(97 + panel_idx))

    return fig


def plot_template(
    time: NDArray,
    traces: NDArray,
    template: NDArray | None = None,
    reference_ind: int | None = None,
    title: str = None,
    ylim: Sequence[float] | None = None,
    figsize: tuple[float] = (4, 6),
    offset_spacing_factor: float = 2.5,
    axes: NDArray | None = None,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> Figure:
    """Plot template construction showing aligned traces and the resulting template.

    Args:
        time: Time array.
        traces: Array of aligned traces.
        template: Template trace (median of aligned traces).
        reference_ind: Index of the reference trace.
        title: Plot title.
        ylim: Y-axis limits.
        figsize: Figure size (only used if axes is None).
        offset_spacing_factor: Spacing factor for stacked traces.
        axes: Optional array of axes to plot into. If None, creates a new figure.
        show_ylabel: Whether to show y-axis labels.
        show_legend: Whether to show legends.

    Returns:
        Figure object.
    """
    if axes is None:
        fig, axes = plt.subplots(
            nrows=4,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1, 0.25, 0.25, 0.25], "hspace": 0.0},
        )
        return_fig = True
    else:
        fig = axes[0].get_figure()
        return_fig = False

    xlim = (time[0], time[-1])
    ax = axes[0]

    # Determine vertical spacing for stacking
    n_traces = traces.shape[0]
    trace_max = np.nanmax(np.abs(traces))
    # Handle NaN or zero trace_max
    if np.isnan(trace_max) or trace_max == 0:
        offset_spacing = 1.0
    else:
        offset_spacing = offset_spacing_factor * trace_max

    # Plot traces stacked from top, with reference trace in blue
    offset_idx = 0
    ytick_positions = []
    ytick_labels = []

    for i in range(n_traces):
        if i == reference_ind:
            ax.plot(
                time,
                traces[i] - offset_idx * offset_spacing,
                "tab:blue",
                label="Reference",
                linewidth=1.2,
            )
            ytick_labels.append("Reference $p_i$")
        else:
            if i < reference_ind:
                label = f"$p_{{i - {np.abs(i - reference_ind)}}}$"
            elif i > reference_ind:
                label = f"$p_{{i + {np.abs(i - reference_ind)}}}$"
            ax.plot(
                time,
                traces[i] - offset_idx * offset_spacing,
                "k",
                linewidth=0.7,
                alpha=0.7,
            )
            ytick_labels.append(label)
        ytick_positions.append(-offset_idx * offset_spacing)
        offset_idx += 1

    # Plot template at the bottom if provided
    ax.plot(
        time,
        template - offset_idx * offset_spacing,
        "tab:red",
        label="Template",
        linewidth=1,
    )
    ytick_positions.append(-offset_idx * offset_spacing)
    ytick_labels.append("Template $x_i$")

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_yticks(ytick_positions)
    ax.set_ylim(-(offset_idx + 1) * offset_spacing, offset_spacing)
    if show_ylabel:
        ax.set_yticklabels(ytick_labels)
    else:
        ax.set_yticklabels([])

    ax = axes[1]
    # Plot reference first
    if reference_ind is not None:
        ax.plot(
            time,
            traces[reference_ind],
            "tab:blue",
            label="$p_i$",
            linewidth=1.0,
            zorder=2,
        )
    others_labeled = False
    for i in range(n_traces):
        if i != reference_ind:
            label = "$p_{i + j, j\\neq 0}$" if not others_labeled else None
            ax.plot(
                time, traces[i], "k", label=label, linewidth=0.8, alpha=0.7, zorder=1
            )
            others_labeled = True
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_yticklabels([])
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.01, 1.06),
            framealpha=1.0,
            borderpad=0.1,
            labelspacing=0.0,
            ncol=1,
        )

    ax = axes[2]
    ax.plot(time, traces[reference_ind], "tab:blue", label="$p_i$", zorder=1)
    ax.plot(time, template, "tab:red", label="$x_i$", zorder=2)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_yticklabels([])
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.01, 1.06),
            framealpha=1.0,
            borderpad=0.1,
            labelspacing=0.0,
            ncol=1,
        )

    ax = axes[3]
    diff = template - traces[reference_ind]
    ax.plot(time, diff, "tab:green", label="$p_i - x_i$")
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.01, 1.06),
            framealpha=1.0,
            borderpad=0.2,
            labelspacing=0.0,
        )
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Amplitude (μPa)")
        ax.yaxis.set_major_formatter(
            FuncFormatter(partial(format_tick_scientific, mathtext=False))
        )
        ax.yaxis.get_offset_text().set_visible(False)
    else:
        ax.set_yticklabels([])

    if title is not None:
        if return_fig:
            fig.suptitle(title, y=0.91)
        else:
            # For subplots, use column title
            axes[0].set_title(title, fontsize=8)

    if return_fig:
        return fig
    else:
        return None
