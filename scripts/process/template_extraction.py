#!/usr/bin/env python3
from argparse import ArgumentParser
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

from rodeo.utils import logging_kwargs
from vineyard.config import get_path
from vineyard.plotting import plot_template, plot_template_detail
from vineyard.readers import read_acoustic_data, read_strike_index

SENSORS = [
    {
        "name": "3dvha",
        "channel": 7,
        "distance_sec": 1.0,
        "threshold": 0.05,
        "ylim": [-0.015, 0.015],
    },
    {
        "name": "vla1",
        "channel": 3,
        "distance_sec": 1.0,
        "threshold": 0.05,
        "ylim": [-1.0e7, 1.0e7],
    },
    {
        "name": "vla2",
        "channel": 0,
        "distance_sec": 1.0,
        "threshold": 0.02,
        "ylim": [-5.0e6, 5.0e6],
    },
]
SMOKE_TEST = False


def enforce_same_size(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Ensure all arrays in the list have the same size by padding with NaNs."""
    max_length = max(arr.shape[0] for arr in arrays)
    return [
        np.pad(arr, (0, max_length - arr.shape[0]), constant_values=np.nan)
        for arr in arrays
    ]


def load_data(
    sensor: str,
    channel: int,
    time_start: np.datetime64,
    time_end: np.datetime64,
    start_buffer: float,
    end_buffer: float,
):
    ds = read_acoustic_data(
        get_path(f"{sensor}_inventory"),
        time_start,
        time_end,
        channels=channel,
        dec_factor=None,
        filt_type="bandpass",
        filt_freq=[19.0, 25.0],
    )

    strike_index = (
        read_strike_index(get_path("strike_index"), start_buffer, end_buffer)
        .filter(pl.col("sensor") == sensor)
        .drop(["sensor", "channel"])
    )

    with h5py.File(get_path("strike_corr"), "r") as f:
        group = f[sensor]
        corrs = group["corr"][:]

    return ds, strike_index, corrs


def get_template_inds(
    num_signals: int, corrs: np.ndarray, threshold: float = 0.9, window_size: int = 20
) -> list[list[int]]:
    """Get indices of templates for each signal based on correlation matrix.

    Args:
        num_signals (int): Number of signals.
        corrs (np.ndarray): Correlation matrix.
        threshold (float): Correlation threshold to consider as template.
        window_size (int): Number of signals to consider on each side.

    Returns:
        list[list[int]]: List of lists containing template indices for each signal.
    """
    inds = []
    for i in range(num_signals):
        start_idx = max(0, i - window_size)
        end_idx = min(num_signals - 1, i + window_size)
        template_inds = []
        for j in range(start_idx, end_idx + 1):
            if corrs[i, j] > threshold:
                template_inds.append(j)
        inds.append(template_inds)
    return inds


def get_window_inds(
    sampling_rate: float, peak_index: int, start_buffer: float, end_buffer: float
) -> tuple[int, int]:
    start_index = peak_index - int(start_buffer * sampling_rate)
    end_index = peak_index + int(end_buffer * sampling_rate)
    return start_index, end_index


def process_datastream(
    ds,
    strike_index,
    corrs,
    name: str,
    start_buffer: float,
    end_buffer: float,
    corr_cutoff: float = 0.9,
    window_size: int = 20,
    save_plots: bool = False,
) -> None:
    all_template_inds = get_template_inds(
        strike_index.shape[0], corrs, threshold=corr_cutoff, window_size=window_size
    )

    num_strikes = len(strike_index)
    # Calculate maximum possible template length
    max_template_length = int((start_buffer + end_buffer) * ds.stats.sampling_rate) + 1

    # Initialize HDF file and datasets - keep file open during processing
    hdf_path = get_path("template_data")
    hdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(hdf_path, "a") as f:
        if name in f:
            logging.warning(
                f"Group {name} already exists in template_data. Overwriting."
            )
            del f[name]
        g = f.create_group(name)
        g.attrs["sampling_rate"] = ds.stats.sampling_rate
        g.create_dataset("start_sample", shape=(num_strikes,), dtype=int)
        g.create_dataset("end_sample", shape=(num_strikes,), dtype=int)
        g.create_dataset(
            "data",
            shape=(num_strikes, max_template_length),
            dtype=float,
            fillvalue=np.nan,
        )

        template = None
        previous_template_inds = []
        traces = None

        for i in tqdm(
            range(num_strikes),
            desc=f"Processing {name}",
            total=num_strikes,
            unit="strike",
        ):
            ref_start = np.datetime64(strike_index.item(i, "start_time"))
            ref_end = np.datetime64(strike_index.item(i, "end_time"))
            reference_trace = (
                ds.copy().trim(starttime=ref_start, endtime=ref_end).data[0]
            )

            start_index, end_index = get_window_inds(
                ds.stats.sampling_rate,
                strike_index.item(i, "sample"),
                start_buffer,
                end_buffer,
            )

            template_inds = all_template_inds[i]
            template_inds.remove(i)

            # Case 1: No templates found, and no previous template
            # Action: Use the reference trace as the template
            if len(template_inds) < 1 and template is None:
                logging.warning(
                    f"Strike {i} has no templates. Using reference trace as template."
                )
                template_inds = [i]
                traces = np.atleast_2d(reference_trace)
                template = reference_trace
            # Case 2: No templates found, but previous template exists
            # Action: Use the previous template
            elif len(template_inds) < 5:
                logging.warning(
                    f"Strike {i} has insufficient candidates. Using previous template."
                )
                template_inds = previous_template_inds.copy()
                tmp_traces, tmp_ref = enforce_same_size([traces, reference_trace])
                template = np.mean(
                    np.hstack([tmp_traces, tmp_ref[:, np.newaxis]]), axis=1
                )
            # Case 3: Templates found
            # Action: Generate a new template from the reference trace and the templates
            else:
                traces = [reference_trace]
                for j in template_inds:
                    orig_template_start = np.datetime64(
                        strike_index.item(j, "start_time")
                    )
                    orig_template_end = np.datetime64(strike_index.item(j, "end_time"))
                    tr = (
                        ds.copy()
                        .trim(starttime=orig_template_start, endtime=orig_template_end)
                        .data[0]
                    )
                    fs = ds.stats.sampling_rate

                    xcorr = correlate(reference_trace, tr, mode="same")
                    lags = correlation_lags(len(reference_trace), len(tr), mode="same")
                    peak_lag = lags[np.argmax(xcorr)]

                    dt = np.timedelta64(
                        int(peak_lag / fs * TIME_CONVERSION_FACTOR), TIME_PRECISION
                    )

                    corrected_template_start = (
                        np.datetime64(strike_index.item(j, "start_time")) - dt
                    )
                    corrected_template_end = (
                        np.datetime64(strike_index.item(j, "end_time")) - dt
                    )

                    template_tr = (
                        ds.copy()
                        .trim(
                            starttime=corrected_template_start,
                            endtime=corrected_template_end,
                        )
                        .data[0]
                    )

                    traces.append(template_tr)

                    if SMOKE_TEST:
                        fig = plot_template_detail(
                            reference_trace,
                            template_tr,
                            xcorr,
                            lags,
                            title=f"Strike {i} - Template {j}",
                        )
                        plt.show()

                traces = np.array(enforce_same_size(traces)).T
                template = np.mean(traces, axis=1)
                previous_template_inds = template_inds.copy()

            template_length = len(template)
            if template_length < max_template_length:
                padded_template = np.full(max_template_length, np.nan)
                padded_template[:template_length] = template
            else:
                padded_template = template[:max_template_length]

            g["data"][i, :] = padded_template
            g["start_sample"][i] = start_index
            g["end_sample"][i] = end_index

            if save_plots:
                print_corrs = ", ".join(
                    [
                        f"{x:.2f}"
                        for x in np.atleast_1d(corrs[i, template_inds]).tolist()
                    ]
                )
                title = (
                    f"{name.upper()} - Strike {i} - {strike_index.item(i, 'start_time')}\n"
                    f"Template Indices: {template_inds}\nMax Corr: [{print_corrs}]"
                )
                ylim = [sensor["ylim"] for sensor in SENSORS if sensor["name"] == name][
                    0
                ]
                savepath = get_path("figures") / "strike_templates" / name
                savepath.mkdir(parents=True, exist_ok=True)
                fig = plot_template(traces, template, title=title, ylim=ylim)
                fig.savefig(
                    savepath / f"{name}_strike_{i:04d}_template.png",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

    logging.info(f"Templates for {name.upper()} saved to {hdf_path}")


def main(
    time_start: np.datetime64,
    time_end: np.datetime64,
    start_buffer: float,
    end_buffer: float,
    corr_cutoff: float,
    window_size: float,
    save_plots: bool,
) -> None:
    for sensor in SENSORS:
        name = sensor["name"]
        channel = sensor["channel"]
        logging.info(f"Processing sensor: {name} channel {channel}.")

        ds, strike_index, corrs = load_data(
            name, channel, time_start, time_end, start_buffer, end_buffer
        )
        process_datastream(
            ds,
            strike_index,
            corrs,
            name,
            start_buffer,
            end_buffer,
            corr_cutoff=corr_cutoff,
            window_size=window_size,
            save_plots=save_plots,
        )


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    parser = ArgumentParser(description="Process Vineyard Wind acoustic data.")
    parser.add_argument(
        "--start",
        type=str,
        default="2023-12-01T21:06:00.00",
        help="Start time in ISO format (default: 2023-12-01T21:51:15.00)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-01T22:26:00.00",
        help="End time in ISO format (default: 2023-12-01T22:26:00.00)",
    )
    parser.add_argument(
        "--start-buffer", type=float, default=0.75, help="Buffer before peak (s)."
    )
    parser.add_argument(
        "--end-buffer", type=float, default=0.85, help="Buffer after peak (s)."
    )
    parser.add_argument(
        "--corr-cutoff",
        type=float,
        default=0.8,
        help="Correlation cutoff for template selection.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Window size for template selection.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save a plot of each template.",
    )
    args = parser.parse_args()
    time_start = np.datetime64(args.start, TIME_PRECISION)
    time_end = np.datetime64(args.end, TIME_PRECISION)
    main(
        time_start,
        time_end,
        args.start_buffer,
        args.end_buffer,
        args.corr_cutoff,
        args.window_size,
        args.save_plots,
    )
