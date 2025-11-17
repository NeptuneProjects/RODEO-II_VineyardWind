#!/usr/bin/env python

from argparse import ArgumentParser
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import polars as pl
from tqdm import tqdm
from tritonoa.data.signal import taper
from tritonoa.data.time import TIME_PRECISION

from rodeo.utils import logging_kwargs
from vineyard.config import get_path
from vineyard.readers import read_acoustic_data, read_strike_index

SENSORS = [
    {
        "name": "3dvha",
        "channel": 7,
    },
    {
        "name": "vla1",
        "channel": 3,
    },
    {
        "name": "vla2",
        "channel": 0,
    },
]


def adaptive_filter(
    signal: ArrayLike, template: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    tap = taper(len(signal), max_percentage=0.02)
    y = tap * template
    signal_fft = np.fft.fft(signal)
    y_fft = np.fft.fft(y)
    e_fft = signal_fft - y_fft
    e = np.fft.ifft(e_fft).real
    return e, y


def denoise_data(
    signal: ArrayLike, strike_index, templates, start_samples, end_samples
) -> tuple[np.ndarray, np.ndarray]:
    filtered_signal = signal.copy()
    y_full = np.zeros_like(signal)

    strike_inds = strike_index["strike_index"].to_list()

    for strike_ind, start_ind, end_ind in tqdm(
        zip(strike_inds, start_samples, end_samples),
        desc="Denoising Signals",
        total=len(strike_inds),
    ):
        template = templates[strike_ind]
        segment = signal[start_ind:end_ind]

        min_length = min(len(template), len(segment))
        template = template[:min_length]
        segment = segment[:min_length]

        e, y = adaptive_filter(segment, template)

        y_full[start_ind:end_ind] = y
        filtered_signal[start_ind:end_ind] = e

    return filtered_signal, y_full


def load_data(sensor: str, channel: int, start: np.datetime64, end: np.datetime64):
    # Entire data stream
    ds = read_acoustic_data(
        get_path(f"{sensor}_inventory"),
        start,
        end,
        channels=channel,
        dec_factor=None,
        filt_type="bandpass",
        filt_freq=[19.0, 25.0],
    )

    strike_index = (
        read_strike_index(get_path("strike_index"), 0.75, 0.85)
        .filter(pl.col("sensor") == sensor)
        .drop(["sensor", "channel"])
    )

    with h5py.File(get_path("template_data"), "r") as f:
        g = f.get(sensor)
        template_fs = g.attrs["sampling_rate"]
        templates = g["data"][:]
        start_samples = g["start_sample"][:]
        end_samples = g["end_sample"][:]

    if template_fs != ds.stats.sampling_rate:
        raise ValueError(
            f"Template sampling rate {template_fs} does not match "
            f"data sampling rate {ds.stats.sampling_rate}"
        )

    return ds, strike_index, templates, start_samples, end_samples


def main(start: np.datetime64, end: np.datetime64) -> None:
    for sensor in SENSORS:
        ds, strike_index, templates, start_samples, end_samples = load_data(
            sensor["name"], sensor["channel"], start, end
        )
        x_filtered, y = denoise_data(
            ds.data[0],
            strike_index,
            templates,
            start_samples,
            end_samples,
        )
        new_data = np.vstack((ds.data[0], x_filtered, y))

        ds.data = new_data
        ds.stats.channels = [0, 1, 2]
        ds.stats.metadata = {
            "sensor": sensor["name"],
            "channel": sensor["channel"],
            "channel_names": {
                0: "Original Signal",
                1: "Filtered Signal",
                2: "Rejected Signal",
            },
        }
        ds.write_hdf5(get_path("denoised_data") / f"{sensor['name']}.h5")


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    parser = ArgumentParser()
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
    args = parser.parse_args()
    start = np.datetime64(args.start, TIME_PRECISION)
    end = np.datetime64(args.end, TIME_PRECISION)

    main(start, end)
