#!/usr/bin/env python3
"""Find pile-driving strikes in the dataset using peak-finding."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from polars import DataFrame, concat
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
from tritonoa.data.reader import read_hdf5_group
from tritonoa.data.time import TIME_PRECISION

from rodeo.utils import compute_array_size, initialize_julia
from vineyard.readers import read_acoustic_data, read_strike_index
from vineyard.signal_proc import find_strikes, process_datastream


class StrikeCorrConfig(BaseModel):
    max_workers: int = 10
    detrend: bool = True
    taper_pc: float = 0.05
    dec_factor: int = 20
    filt_type: str | None = "bandpass"
    filt_freq: float | Sequence[float] | None = [19.0, 25.0]


class StrikeFindConfig(BaseModel):
    sensors: list[dict] | None = None
    taper_pc: float = 1e-4
    dec_factor: int | None = None
    filt_type: str = "bandpass"
    filt_freq: float | Sequence[float] = [100.0, 300.0]


class StrikeSaveConfig(BaseModel):
    buffer_start: float = 0.75
    buffer_end: float = 0.75
    detrend: bool = False
    taper_pc: float | None = None
    dec_factor: int | None = None
    filt_type: str | None = None
    filt_freq: float | Sequence[float] | None = None


class StrikeConfig(BaseModel):
    strike_index: Path = "data/acoustic/strike_index.csv"
    strike_data: Path = "data/acoustic/strike_data.h5"
    strike_corr: Path = "data/acoustic/strike_corr"
    strike_find_config: StrikeFindConfig = Field(alias="find")
    strike_save_config: StrikeSaveConfig = Field(alias="save")
    strike_corr_config: StrikeCorrConfig = Field(alias="correlation")

    @model_validator(mode="after")
    def validate_paths(cls, config: "StrikeConfig") -> "StrikeConfig":
        """Append filtering info to strike_corr path if filters are configured."""
        if (
            config.strike_corr_config.filt_type is None
            or config.strike_corr_config.filt_freq is None
        ):
            if config.strike_corr.suffix != ".h5":
                config.strike_corr = config.strike_corr.with_suffix(".h5")
            return config

        if isinstance(config.strike_corr_config.filt_freq, (list, tuple)):
            freq_str = f"{config.strike_corr_config.filt_freq[0]}-{config.strike_corr_config.filt_freq[1]}"
        else:
            freq_str = str(config.strike_corr_config.filt_freq)

        new_name = f"{config.strike_corr.stem}_{config.strike_corr_config.filt_type}_{freq_str}.h5"
        config.strike_corr = config.strike_corr.parent / new_name

        return config


class ProcessConfig(BaseModel):
    """Configuration for the strike-finding process."""

    inventory_path: Path | None = None
    time_ranges: list[list[str]] | None = None
    strike_config: StrikeConfig = Field(alias="strike")

    def get_time_ranges_as_datetime(self) -> list[tuple[np.datetime64, np.datetime64]]:
        """Convert time_ranges from strings to numpy datetime64 tuples."""
        return [
            (
                np.datetime64(start, TIME_PRECISION),
                np.datetime64(end, TIME_PRECISION),
            )
            for start, end in self.time_ranges
        ]


@dataclass
class Record:
    """Data class to hold the results of the cross-correlation for a single sensor."""

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


def build_strikes_df_per_sensor(
    inventory_path: Path,
    sensor: dict,
    time_start: np.datetime64,
    time_end: np.datetime64,
    strike_index_offset: int = 0,
    taper_pc: float = 1e-4,
    dec_factor: int | None = None,
    filt_type: str = "bandpass",
    filt_freq: float | Sequence[float] = [100.0, 300.0],
) -> DataFrame:
    """Build a DataFrame of detected strikes for a single sensor and time range.

    The DataFrame has the following columns:
    - sensor: Name of the sensor.
    - channel: Channel used for detection.
    - strike_index: Unique index for each detected strike (across all time ranges).
    - time: Time of the detected strike.
    - sample: Sample index of the detected strike in the original data.

    Args:
        inventory_path: Path to the inventory CSV file for the sensor.
        sensor: Dictionary containing sensor configuration (name, channel, etc.).
        time_start: Start time of the time range to process.
        time_end: End time of the time range to process.
        strike_index_offset: Offset to add to the strike_index to ensure uniqueness across time ranges.
        taper_pc: Percentage of the data to taper on each side before processing.
        dec_factor: Decimation factor to apply to the data before processing.
        filt_type: Type of filter to apply to the data before processing (e.g., "bandpass").
        filt_freq: Frequency or frequencies to use for filtering the data before processing.

    Returns:
        A DataFrame containing the detected strikes for the sensor and time range.
    """
    name, channel, distance_s, threshold = tuple(sensor.values())
    logging.info(f"Processing sensor: {name}, channel: {channel}")

    ds = read_acoustic_data(
        inventory_path,
        time_start,
        time_end,
        channels=channel,
        taper_pc=taper_pc,
        dec_factor=dec_factor,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )
    peaks = find_strikes(ds.data[0], ds.stats.sampling_rate, threshold, distance_s)

    logging.info(f"Found {len(peaks)} peaks for sensor {name}.")
    return DataFrame(
        {
            "sensor": name,
            "channel": channel,
            "strike_index": np.arange(len(peaks)) + strike_index_offset,
            "time": ds.time_vector[peaks],
            "sample": peaks,
        }
    )


def build_strikes_df(
    config: StrikeConfig,
    time_ranges: list[tuple[np.datetime64, np.datetime64]],
    inventory_path: Path,
) -> None:
    """Build a DataFrame of detected strikes for all sensors and time ranges,
    and save it to a CSV file.

    The DataFrame has the following columns:
    - sensor: Name of the sensor.
    - channel: Channel used for detection.
    - strike_index: Unique index for each detected strike (across all sensors and time ranges).
    - time: Time of the detected strike.
    - sample: Sample index of the detected strike in the original data.

    Args:
        config: StrikeConfig instance containing the configuration for strike finding.
        time_ranges: List of tuples containing the start and end times for each time range to process.
        inventory_path: Path to the inventory CSV files for the sensors.
    """
    all_dfs = []

    for sensor in config.strike_find_config.sensors:
        sensor_dfs = []
        strike_index_offset = 0

        for i, (time_start, time_end) in enumerate(time_ranges):
            logging.info(
                f"Processing sensor {sensor['name']}, time range "
                f"{i+1}/{len(time_ranges)}: {time_start} to {time_end}"
            )

            df = build_strikes_df_per_sensor(
                inventory_path / f"inventory_{sensor['name']}.csv",
                sensor,
                time_start,
                time_end,
                strike_index_offset,
                **config.strike_find_config.model_dump(exclude={"sensors"}),
            )
            sensor_dfs.append(df)
            strike_index_offset += len(df)

        all_dfs.extend(sensor_dfs)

    concat(all_dfs).write_csv(config.strike_index)
    logging.info(f"Strikes extracted and saved to {config.strike_index}.")


def xcorr_strike_pairs(
    config: StrikeCorrConfig, strike_data_path: Path, output_path: Path
) -> None:
    """Compute the cross-correlation of strike pairs for each sensor and save
    the results.

    The results for each sensor are saved to the output HDF5 file in the
    following format:
    ```
    /sensor
        /time_diff
        /corr
    ```
    `time_diff` and `corr` are 2D arrays with the shape `(num_detections, num_detections)`.

    Args:
        config: StrikeCorrConfig instance containing the configuration for cross-correlation.
        strike_data_path: Path to the input HDF5 file containing the strike data.
        output_path: Path to the output HDF5 file where the results will be saved.
    """
    os.environ["JULIA_NUM_THREADS"] = str(config.max_workers)
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    with h5py.File(strike_data_path, "r") as file:
        for sensor, sensor_group in file.items():
            logging.info(f"Processing sensor {sensor.upper()}.")
            record = xcorr_sensor(
                sensor, sensor_group, config.model_dump(exclude={"max_workers"})
            )
            record.save_h5(output_path)
            logging.info(f"Processed sensor {sensor.upper()}.")
    return


def process_data(config: ProcessConfig) -> None:
    """Run data processing steps based on the provided configuration.

    Args:
        config: ProcessConfig instance containing the configuration for
            data processing.
    """
    # build_strikes_df(
    #     config.strike_config,
    #     config.get_time_ranges_as_datetime(),
    #     config.inventory_path,
    # )
    # save_strikes(config.strike_config, config.inventory_path)
    xcorr_strike_pairs(
        config.strike_config.strike_corr_config,
        config.strike_config.strike_data,
        config.strike_config.strike_corr,
    )


def read_strikes(sensor_group: h5py.Group, **kwargs) -> tuple[NDArray, NDArray]:
    """Read the strike data for a single sensor from the HDF5 group and
    process it.

    Args:
        sensor_group: HDF5 group containing the strike data for a single sensor.
        **kwargs: Additional keyword arguments to pass to the process_datastream
            function.

    Returns:
        A tuple containing:
        - data: 2D array of shape (num_detections, num_samples) containing the
            processed strike data for the sensor.
        - t0: 1D array of shape (num_detections,) containing the initial time
            of each strike detection.
    """
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


def save_strikes(config: StrikeConfig, inventory_path: Path) -> None:
    """Save the strike data to an HDF5 file.

    Args:
        config: StrikeConfig instance containing the configuration for saving strikes.
        inventory_path: Path to the inventory directory containing sensor CSV files.
    """
    save_config = config.strike_save_config

    df = read_strike_index(
        config.strike_index,
        save_config.buffer_start,
        save_config.buffer_end,
    )

    with h5py.File(config.strike_data, "w") as file:
        for row in tqdm(
            df.iter_rows(), desc="Extracting & saving strikes", total=df.shape[0]
        ):
            sensor, channel, strike_index, _, _, time_start, time_end = row
            ds = read_acoustic_data(
                inventory_path / f"inventory_{sensor}.csv",
                time_start,
                time_end,
                channel,
                detrend=save_config.detrend,
                dec_factor=save_config.dec_factor,
                filt_type=save_config.filt_type,
                filt_freq=save_config.filt_freq,
                taper_pc=save_config.taper_pc,
            )
            g = file.create_group(f"{sensor}/{strike_index:04d}")
            ds.create_hdf5_dataset(g)


def xcorr_sensor(sensor: str, sensor_group: h5py.Group, sp_kwargs: dict = {}) -> Record:
    """Compute the cross-correlation of strike pairs for a single sensor.

    Args:
        sensor: Name of the sensor.
        sensor_group: HDF5 group containing the strike data for the sensor.
        sp_kwargs: Additional keyword arguments to pass to the process_datastream
            function when reading the strike data.

    Returns:
        A Record instance containing the results of the cross-correlation for the sensor.
    """
    data, t0 = read_strikes(sensor_group, **sp_kwargs)

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
