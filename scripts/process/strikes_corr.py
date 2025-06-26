#!/usr/bin/env python3
"""Compute correlation coefficients between all signal pairs for each station."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path

import dotenv
import h5py
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from tritonoa.data.reader import read_hdf5_group

from rodeo.utils import (
    compute_array_size,
    initialize_julia,
    logging_kwargs,
)
from vineyard.config import get_path
from vineyard.signal import process_datastream

dotenv.load_dotenv()


@dataclass
class Record:
    sensor: str
    time_diff: np.ndarray
    corr: np.ndarray

    def save_h5(self, path: Path) -> None:
        """Save the record to an HDF5 file.

        The record is saved to the output file in the following format:
        ```
        /sensor
            /time_diff
            /corr
        ```
        `time_diff` and `corr` are 2D arrays with the shape
        `(num_detections, num_detections)`.

        Args:
            path: Path to the output HDF5 file.
        """
        with h5py.File(path, "a") as file:
            # Create a new group for the station if it doesn't exist:
            if self.sensor not in file:
                file.create_group(self.sensor)

            # Save the data:
            grp = file[f"{self.sensor}"]
            grp.attrs["sensor"] = self.sensor
            grp.create_dataset("time_diff", data=self.time_diff)
            grp.create_dataset("corr", data=self.corr)


def process_sensor(
    sensor: str, sensor_group: h5py.Group, sp_kwargs: dict = {}
) -> Record:
    data, t0 = retrieve_data(sensor_group, **sp_kwargs)

    num_detections = data.shape[0]
    time_diff = np.full((num_detections, num_detections), np.nan)
    max_corr = np.full((num_detections, num_detections), np.nan)

    logging.info(f"Data shape: {num_detections} detections.")
    logging.info(
        f"Size of arrays: {compute_array_size([max_corr, time_diff]) / (1024 ** 3):.2f} GB."
    )

    jl = initialize_julia("CrossCorr")
    threads = jl.seval("Threads.nthreads()")
    logging.info(f"Number of Julia threads: {threads}.")

    jl_data = jl.seval("x -> Matrix{Float64}(x)")(data)
    jl_time = jl.seval("x -> Vector{Float64}(x)")(t0)

    time_diff = np.array(jl.CrossCorr.dt_matrix(jl_time))
    max_corr = np.array(jl.CrossCorr.corr_matrix(jl_data))

    logging.info(f"Computed time_diff and max_corr for sensor {sensor.upper()}.")
    logging.info(f"Shape of time_diff: {time_diff.shape}, max_corr: {max_corr.shape}.")

    return Record(
        sensor=sensor,
        time_diff=time_diff,
        corr=max_corr,
    )


def retrieve_data(sensor_group: h5py.Group, **kwargs) -> tuple[NDArray, NDArray]:
    num_detections = len(sensor_group)

    data = None
    t0 = np.full((num_detections,), np.nan)
    for i, strike_group in tqdm(
        enumerate(sensor_group.values()), desc="Loading data", total=num_detections
    ):
        ds = process_datastream(read_hdf5_group(strike_group), **kwargs)
        # Initialize within loop since num_samples depends on target_fs:
        if data is None:
            data = np.full((num_detections, ds.num_samples), np.nan)

        if ds.num_samples < data.shape[1]:
            ds.data = np.pad(ds.data, ((0, 0), (0, data.shape[1] - ds.num_samples)))
        elif ds.num_samples > data.shape[1]:
            ds.data = ds.data[:, : data.shape[1]]

        data[i] = ds.data.squeeze()
        t0[i] = ds.stats.time_init

    return data, t0


def main(
    strike_db: Path,
    output: Path,
    detrend: bool,
    taper_pc: float,
    dec_factor: int,
    filt_type: str,
    filt_freq: float | Sequence[float],
) -> None:
    sp_kwargs = {
        "detrend": detrend,
        "taper_pc": taper_pc,
        "dec_factor": dec_factor,
        "filt_type": filt_type,
        "filt_freq": filt_freq,
    }
    with h5py.File(strike_db, "r") as file:
        for sensor, sensor_group in file.items():
            logging.info(f"Processing sensor {sensor.upper()}.")
            record = process_sensor(sensor, sensor_group, sp_kwargs)
            record.save_h5(output)
            logging.info(f"Processed sensor {sensor.upper()}.")


if __name__ == "__main__":
    logging.basicConfig(**logging_kwargs)
    parser = argparse.ArgumentParser(
        description="Compute the coherence between two signals."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=get_path("strike_data"),
        help="Path to the input HDF5 file containing the signals.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Number of workers to use for parallel processing.",
        default=12,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output HDF5 file to save the results.",
        default=get_path("strike_corr"),
    )
    parser.add_argument(
        "--detrend",
        action="store_true",
        default=False,
        help="Apply detrending to the signals before processing.",
    )
    parser.add_argument(
        "--taper_pc",
        type=float,
        default=None,
        help="Percentage of the signal to taper at the beginning and end.",
    )
    parser.add_argument(
        "--dec_factor",
        type=int,
        default=None,
        help="Decimation factor to apply to the signals.",
    )
    parser.add_argument(
        "--filt_type",
        type=str,
        default="bandpass",
        help="Type of filter to apply to the signals.",
    )
    parser.add_argument(
        "--filt_freq",
        type=float,
        nargs="+",
        default=[15.0, 35.0],
        help="Frequency range for the filter. Provide two values for bandpass filter.",
    )
    args = parser.parse_args()
    os.environ["JULIA_NUM_THREADS"] = str(args.max_workers)
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    main(
        args.db,
        args.output,
        args.detrend,
        args.taper_pc,
        args.dec_factor,
        args.filt_type,
        args.filt_freq,
    )
