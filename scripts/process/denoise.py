#!/usr/bin/env python3
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tritonoa.data.stream import DataStream
from tritonoa.data.reader import read_hdf5, read_inventory
from tritonoa.data.time import TIME_PRECISION

from rodeo.utils import logging_kwargs
from vineyard.config import get_path
from vineyard.readers import read_acoustic_data, read_strike_index

time_start = np.datetime64("2023-12-01T21:51:15.00", TIME_PRECISION)
time_end = np.datetime64("2023-12-01T22:26:00.00", TIME_PRECISION)
SENSORS = [
    {"name": "3dvha", "channel": 7, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla1", "channel": 3, "distance_sec": 1.0, "threshold": 0.05},
    {"name": "vla2", "channel": 0, "distance_sec": 1.0, "threshold": 0.02},
]


def load_data(sensor: str, channel: int):
    ds = read_acoustic_data(
        get_path(f"{sensor}_inventory"),
        time_start,
        time_end,
        channels=channel,
        dec_factor=20,
        filt_type="bandpass",
        filt_freq=[19.0, 25.0],
    )
    # ds = None
    strike_index = (
        read_strike_index(get_path("strike_index"), 0.75, 0.75)
        .filter(pl.col("sensor") == sensor)
        .drop(["sensor", "channel"])
    )


    print(strike_index.head())
    with h5py.File(get_path("strike_corr"), "r") as f:
        group = f[sensor]
        corrs = group["corr"][:]
        time_diffs = group["time_diff"][:]

    return ds, strike_index, corrs, time_diffs


def process_datastream(ds, strike_index, corrs, time_diffs):
    inds = []
    for i, strike_time, start_time, end_time in strike_index.iter_rows():
        corr_cutoff = 0.9
        corr_cutoff_inds = np.where(corrs[i] >= corr_cutoff)[0]
        inds.append(corr_cutoff_inds)

        traces = []
        for j in corr_cutoff_inds.tolist():
            traces.append(
                ds.copy()
                .trim(
                    starttime=np.datetime64(strike_index.item(j, "start_time")),
                    endtime=np.datetime64(strike_index.item(j, "end_time")),
                )
                .data[0]
            )
        
        traces = np.array(traces).T

        plt.figure(figsize=(12, 6))
        plt.plot(traces, "k")
        plt.plot(traces[:, corr_cutoff_inds == i], "b", linewidth=2, label="Reference")
        plt.plot(np.mean(traces, axis=1), "r", linewidth=2, label="Mean")
        plt.show()

        # breakpoint()
    return


def main():
    for sensor in SENSORS:
        name = sensor["name"]
        channel = sensor["channel"]
        logging.info(f"Processing sensor: {name} channel {channel}.")

        ds, strike_index, corrs, time_diffs = load_data(name, channel)
        process_datastream(ds, strike_index, corrs, time_diffs)

        break


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    main()
