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
    num_signals: int, corrs: np.ndarray, threshold: float = 0.9, window_size: int = 5
) -> list[list[int]]:
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
    save_plots: bool = False,
) -> None:
    corr_cutoff = 0.9
    all_template_inds = get_template_inds(
        strike_index.shape[0], corrs, threshold=corr_cutoff, window_size=5
    )

    template_data = []
    start_inds = []
    end_inds = []
    template = None
    previous_template_inds = []

    for i in tqdm(
        range(len(strike_index)),
        desc=f"Processing {name}",
        total=len(strike_index),
        unit="strike",
    ):
        ref_start = np.datetime64(strike_index.item(i, "start_time"))
        ref_end = np.datetime64(strike_index.item(i, "end_time"))
        reference_trace = ds.copy().trim(starttime=ref_start, endtime=ref_end).data[0]

        start_index, end_index = get_window_inds(
            ds.stats.sampling_rate,
            strike_index.item(i, "sample"),
            start_buffer,
            end_buffer,
        )
        start_inds.append(start_index)
        end_inds.append(end_index)

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
        elif len(template_inds) < 1:
            logging.warning(f"Strike {i} has no templates. Using previous template.")
            template_inds = previous_template_inds.copy()
            traces = np.array(enforce_same_size([reference_trace, template])).T
            template = np.mean(traces, axis=1)
        # Case 3: Templates found
        # Action: Generate a new template from the reference trace and the templates
        else:
            traces = [reference_trace]
            for j in template_inds:
                orig_template_start = np.datetime64(strike_index.item(j, "start_time"))
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

        template_data.append(template)

        if save_plots:
            print_corrs = ", ".join(
                [f"{x:.2f}" for x in np.atleast_1d(corrs[i, template_inds]).tolist()]
            )
            title = (
                f"{name.upper()} - Strike {i} - {strike_index.item(i, 'start_time')}\n"
                f"Template Indices: {template_inds}\nMax Corr: [{print_corrs}]"
            )
            ylim = [sensor["ylim"] for sensor in SENSORS if sensor["name"] == name][0]
            savepath = get_path("figures") / "strike_templates" / name
            savepath.mkdir(parents=True, exist_ok=True)
            fig = plot_template(traces, template, title=title, ylim=ylim)
            fig.savefig(
                savepath / f"{name}_strike_{i:04d}_template.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    template_db = np.array(enforce_same_size(template_data))
    with h5py.File(get_path("template_data"), "a") as f:
        if name in f:
            logging.warning(
                f"Group {name} already exists in template_data. Overwriting."
            )
            del f[name]
        g = f.create_group(name)
        g.attrs["sampling_rate"] = ds.stats.sampling_rate
        g.create_dataset("start_sample", data=start_inds)
        g.create_dataset("end_sample", data=end_inds)
        g.create_dataset("data", data=template_db)
        logging.info(
            f"Templates for {name.upper()} saved to {get_path("template_data")}"
        )


def main(
    time_start: np.datetime64,
    time_end: np.datetime64,
    start_buffer: float,
    end_buffer: float,
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
            ds, strike_index, corrs, name, start_buffer, end_buffer, save_plots
        )


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    parser = ArgumentParser(description="Process Vineyard Wind acoustic data.")
    parser.add_argument(
        "--start",
        type=str,
        default="2023-12-01T21:51:15.00",
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
        "--save-plots",
        action="store_true",
        default=False,
        help="Save a plot of each template.",
    )
    args = parser.parse_args()
    time_start = np.datetime64(args.start, TIME_PRECISION)
    time_end = np.datetime64(args.end, TIME_PRECISION)
    main(time_start, time_end, args.start_buffer, args.end_buffer, args.save_plots)
