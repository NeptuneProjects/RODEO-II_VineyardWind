#!/usr/bin/env python3

from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve, correlate, correlation_lags
from tritonoa.data.stream import DataStream, DataStreamStats

from rodeo import utils
from vineyard.config import get_path
from vineyard.readers import read_acoustic_data
from vineyard.signal import find_strikes, roll_and_pad

SENSORS = [
    {"name": "3dvha", "channel": 7, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla1", "channel": 3, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla2", "channel": 0, "distance_sec": 1.0, "threshold": 0.02},
]
SP_KWARGS = {
    "taper_pc": 1e-4,
    "dec_factor": 40,
    "filt_type": "bandpass",
    "filt_freq": [10.0, 300.0],
}


def main(
    sensor: str,
    start: np.datetime64,
    end: np.datetime64,
    template_start: np.datetime64,
    template_end: np.datetime64,
    output: Path,
) -> None:
    # 1. Load & condition acoustic data
    sensor_info = next((s for s in SENSORS if s["name"] == sensor), None)
    channel = sensor_info["channel"]
    ds = read_acoustic_data(
        get_path(f"{sensor}_inventory"), start, end, channels=channel, **SP_KWARGS
    )
    logging.info(
        f"Loaded acoustic data for sensor {sensor.upper()} on channel {channel}."
    )
    logging.info(
        f"Data shape: {ds.data.shape}, Sampling rate: {ds.stats.sampling_rate} Hz."
    )

    # 2. Find strike peaks
    peaks = find_strikes(
        ds.data[0],
        ds.stats.sampling_rate,
        threshold=sensor_info["threshold"],
        distance_sec=sensor_info["distance_sec"],
    )
    logging.info(f"Found {len(peaks)} peaks for sensor {sensor.upper()}.")

    # 3. Load template of first strike
    ds_template = read_acoustic_data(
        get_path(f"{sensor}_inventory"),
        template_start,
        template_end,
        channels=channel,
        **(SP_KWARGS | {"taper_pc": 0.05, "filt_freq": [15.0, 30.0]}),
    ).taper(max_percentage=0.05)
    template = ds_template.data[0]

    logging.info(f"Loaded template data for sensor {sensor.upper()}.")
    logging.info(
        f"Template data shape: {ds_template.data.shape}, "
        f"Sampling rate: {ds_template.stats.sampling_rate} Hz."
    )

    # 4. Convolve template with peaks to generate synthetic strikes
    synthetic_strikes = np.zeros_like(ds.data[0])
    synthetic_strikes[peaks] = 1.0
    synth_data = convolve(
        synthetic_strikes,
        template,
        mode="same",
        method="auto",
    )
    logging.info(
        f"Convolved template with data to generate synthetic strikes for sensor {sensor.upper()}."
    )
    logging.info(
        f"Synthetic strikes shape: {synthetic_strikes.shape}, Data shape: {ds.data[0].shape}."
    )

    # 5. Store synthetic strikes in a DataStream
    synth_ds = DataStream(
        stats=DataStreamStats(
            time_init=ds.stats.time_init,
            time_end=ds.stats.time_end,
            sampling_rate=ds.stats.sampling_rate,
        ),
        data=synth_data.reshape(1, -1),
    )

    # 6. Correct timing of synthetic strikes by finding the lag between the
    # first strikes of the original and synthetic data
    ds_lf = read_acoustic_data(
        get_path(f"{sensor}_inventory"),
        start,
        end,
        channels=channel,
        **(SP_KWARGS | {"filt_freq": [15.0, 30.0]}),
    )
    # plt.figure()
    # plt.plot(ds_lf.time_vector, ds_lf.data[0], label="Original Data")
    # plt.show()
    lf_peaks = find_strikes(
        ds_lf.data[0],
        ds_lf.stats.sampling_rate,
        threshold=sensor_info["threshold"],
        distance_sec=sensor_info["distance_sec"],
    )

    synth_peaks = find_strikes(
        synth_ds.data[0],
        synth_ds.stats.sampling_rate,
        threshold=sensor_info["threshold"],
        distance_sec=sensor_info["distance_sec"],
    )

    first_strike_start = ds_lf.time_vector[lf_peaks[0]] - np.timedelta64(1, "s")
    first_strike_end = synth_ds.time_vector[synth_peaks[0]] + np.timedelta64(1, "s")


    ds_strk = ds_lf.copy().trim(starttime=first_strike_start, endtime=first_strike_end)
    ds_strk_synth = synth_ds.copy().trim(starttime=first_strike_start, endtime=first_strike_end)
    data_strk = ds_strk.data[0]
    data_strk_synth = ds_strk_synth.data[0]
    # t_strk = ds_strk.time_vector

    xcorr = correlate(data_strk, data_strk_synth, mode="full")
    lags = correlation_lags(len(data_strk), len(data_strk_synth), mode="full")
    max_lag = lags[np.argmax(xcorr)]
    logging.info(f"Max lag found: {max_lag} samples.")

    synth_ds.data[0] = roll_and_pad(synth_ds.data[0], max_lag)

    savename = output / f"{sensor}_synthetic_strikes.hdf5"
    synth_ds.write_hdf5(savename)
    logging.info(f"Saving synthetic strikes to {savename}")


if __name__ == "__main__":
    logging.basicConfig(**utils.logging_kwargs)
    parser = ArgumentParser(description="Extract all strikes from the dataset.")
    parser.add_argument(
        "--sensor",
        type=str,
        default="vla1",
        help="Sensor name from which to extract strikes. Options: '3dvha', 'vla1', 'vla2'.",
    )
    parser.add_argument(
        "--start",
        type=np.datetime64,
        default="2023-12-01T21:51:15.00",
        help="Start time for extracting strikes.",
    )
    parser.add_argument(
        "--end",
        type=np.datetime64,
        default="2023-12-01T22:26:00.00",
        help="End time for extracting strikes.",
    )
    parser.add_argument(
        "--template_start",
        type=np.datetime64,
        default="2023-12-01T21:51:23.5",
        help="Start time for extracting strikes.",
    )
    parser.add_argument(
        "--template_end",
        type=np.datetime64,
        default="2023-12-01T21:51:25.65",
        help="End time for extracting strikes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("acoustic_data"),
        help="Output file to save the extracted strikes.",
    )
    args = parser.parse_args()
    main(
        args.sensor,
        args.start,
        args.end,
        args.template_start,
        args.template_end,
        args.output,
    )
