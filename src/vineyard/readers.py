"""Functions to read data from various file formats used in the project."""

import logging
import tomllib
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from tqdm import tqdm
from tritonoa.data.reader import read_and_process, read_hdf5_group
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION


def calibrate_3dvha(cal_file: Path, signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply frequency-dependent sensitivity calibration to convert voltage to micropascals.

    Parameters
    ----------
    cal_file : Path
        Path to calibration file with frequency (Hz) and sensitivity (dB re 1V/uPa) columns
    signal : np.ndarray
        Time-series signal in volts
    fs : float
        Sampling rate in Hz

    Returns
    -------
    np.ndarray
        Calibrated signal in micropascals
    """
    # Load calibration data
    data = np.loadtxt(cal_file, skiprows=1, delimiter=",")
    freq_cal = data[:, 0]
    sensitivity_dB = data[:, 1] - 2.5

    # Convert sensitivity from dB re 1V/uPa to linear scale (V/uPa)
    sensitivity_linear = 10 ** (sensitivity_dB / 20)

    # Compute FFT of the signal
    signal_fft = np.fft.rfft(signal)

    # Get frequency bins for the FFT
    freq_fft = np.fft.rfftfreq(len(signal), d=1 / fs)

    # Interpolate calibration sensitivity to match FFT frequency bins
    sensitivity_interp = np.interp(freq_fft, freq_cal, sensitivity_linear)

    # Apply calibration: divide voltage by sensitivity to get pressure in uPa
    # P (uPa) = V (volts) / S (V/uPa)
    signal_fft_calibrated = signal_fft / sensitivity_interp

    # Convert back to time domain
    calibrated_signal = np.fft.irfft(signal_fft_calibrated, n=len(signal))

    return calibrated_signal


def calibrate_vla(cal_file: Path, signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply frequency-independent sensitivity calibration to convert voltage to micropascals.

    Parameters
    ----------
    cal_file : Path
        Path to TOML calibration file with fixed_gain and sensitivity fields (both in dB)
    signal : np.ndarray
        Time-series signal in volts
    fs : float
        Sampling rate in Hz (unused but kept for API consistency)

    Returns
    -------
    np.ndarray
        Calibrated signal in micropascals
    """
    with open(cal_file, "rb") as f:
        cal_data = tomllib.load(f)

    fixed_gain_dB = cal_data.get("fixed_gain", 0)
    sensitivity_dB = cal_data.get("sensitivity", 0)

    # Total system sensitivity in dB re 1V/uPa
    total_sensitivity_dB = sensitivity_dB + fixed_gain_dB

    # Convert to linear scale (V/uPa)
    total_sensitivity_linear = 10 ** (total_sensitivity_dB / 20)

    # Apply calibration: P (uPa) = V (volts) / S (V/uPa)
    calibrated_signal = signal / total_sensitivity_linear / 1e6

    return calibrated_signal


def process_datastream(
    ds: DataStream,
    detrend: bool = True,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    detrend_kwargs: dict = {},
) -> DataStream:
    if detrend:
        ds.detrend(**detrend_kwargs)
    if taper_pc is not None:
        ds.taper(max_percentage=taper_pc)
    if dec_factor is not None:
        ds.decimate(dec_factor)
    if filt_type is not None and filt_freq is not None:
        ds.filter(filt_type, filt_freq)
    return ds


def read_bathymetry(
    file: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Read bathymetry data from an HDF5 file.

    Args:
        file: Path to the HDF5 file.
    Returns:
        Tuple of numpy arrays containing bathymetry data, longitude vector, and latitude vector.
    """
    logging.info(f"Reading file: {file}")
    with h5py.File(file, "r") as f:
        data = f.get("data")[:]
        lonvec = f.get("lonvec")[:]
        latvec = f.get("latvec")[:]
    return data, lonvec, latvec


def read_denoise_data(
    inventory_path: Path,
    strike_index_path: Path,
    template_path: Path,
    sensor: str,
    channel: int,
    start: np.datetime64,
    end: np.datetime64,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    buffer_start: float = 0.75,
    buffer_end: float = 0.85,
) -> tuple[DataStream, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    ds, strike_index = read_strike_data(
        inventory_path,
        strike_index_path,
        sensor,
        channel,
        start,
        end,
        buffer_start,
        buffer_end,
        taper_pc,
        dec_factor,
        filt_type,
        filt_freq,
    )

    with h5py.File(template_path, "r") as f:
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


def read_distances(lut_file: Path) -> tuple[float, float, float]:
    distance_lut = pl.read_csv(lut_file)
    d_3dvha_vla1 = (
        distance_lut.filter(pl.col("from_equipment") == "3DVHA")
        .filter(pl.col("to_equipment") == "VLA1")["distance_meters"]
        .item()
    )
    d_3dvha_vla2 = (
        distance_lut.filter(pl.col("from_equipment") == "3DVHA")
        .filter(pl.col("to_equipment") == "VLA2")["distance_meters"]
        .item()
    )
    d_vla1_vla2 = (
        distance_lut.filter(pl.col("from_equipment") == "VLA1")
        .filter(pl.col("to_equipment") == "VLA2")["distance_meters"]
        .item()
    )
    return d_3dvha_vla1, d_3dvha_vla2, d_vla1_vla2


def read_sensor_positions(
    sensor_data_file: Path,
) -> tuple[list[float], list[float], float, float]:
    """Load sensor positions from equipment config and compute ENU coordinates."""
    df = pl.read_csv(sensor_data_file)

    lat0 = df["ref_lat"].head(1).item()
    lon0 = df["ref_lon"].head(1).item()

    # Return sensor positions (excluding last row which is reference) and reference lat/lon
    sensor_eastings = df["easting"].unique().to_list()
    sensor_northings = df["northing"].unique().to_list()

    return sensor_eastings, sensor_northings, lat0, lon0


def read_strike_data(
    inventory_path: Path,
    strike_index_path: Path,
    sensor: str,
    channel: int,
    time_start: np.datetime64,
    time_end: np.datetime64,
    buffer_start: float,
    buffer_end: float,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: str | None = None,
) -> tuple[DataStream, pl.DataFrame]:
    ds = read_and_process(
        inventory_path,
        time_start,
        time_end,
        channel,
        taper_pc=taper_pc,
        dec_factor=dec_factor,
        filt_type=filt_type,
        filt_freq=filt_freq,
    )
    strike_index = (
        read_strike_index(strike_index_path, buffer_start, buffer_end)
        .filter(pl.col("sensor") == sensor)
        .drop(["sensor", "channel"])
    )

    return ds, strike_index


def read_strike_index(index: Path, buffer_start: float, buffer_end: float) -> DataFrame:
    """Read the strike index from a CSV file and apply time buffers.

    Args:
        index (Path): Path to the CSV file containing the strike index.
        buffer_start (float): Buffer time in seconds to subtract from the peak time.
        buffer_end (float): Buffer time in seconds to add to the peak time.
    Returns:
        DataFrame: Polars DataFrame with adjusted start and end times.
    """
    start = pl.duration(
        microseconds=buffer_start * TIME_CONVERSION_FACTOR, time_unit=TIME_PRECISION
    )
    end = pl.duration(
        microseconds=buffer_end * TIME_CONVERSION_FACTOR, time_unit=TIME_PRECISION
    )
    return (
        pl.read_csv(index)
        .with_columns(
            pl.col("time").str.to_datetime(time_unit=TIME_PRECISION).alias("peak_time")
        )
        .drop("time")
        .with_columns((pl.col("peak_time") - start).alias("start_time"))
        .with_columns((pl.col("peak_time") + end).alias("end_time"))
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


def read_turbine_locations(file: Path) -> pd.DataFrame:
    """Read the turbine locations from a CSV file.

    Args:
        file (Path): Path to the CSV file.
    Returns:
        Dataframe with turbine locations in latitudes/longitudes.
    """
    logging.info(f"Reading file: {file}")
    df = pd.read_csv(file)
    return df.drop(index=[63, 64])[["Latitude dd", "longitude dd"]].rename(
        columns={
            "Latitude dd": "latitude",
            "longitude dd": "longitude",
        }
    )


def read_whale_call_times(whale_call_data: Path):
    df = pl.read_csv(whale_call_data)

    times = {}

    for sensor in df["sensor"].unique():
        times[sensor] = (
            df.filter(pl.col("sensor") == sensor)["timestamp"]
            .str.to_datetime()
            .to_numpy()
        )

    return times


def read_whale_template(template_path: Path, sensor: str, call_type: str) -> DataStream:
    """Read a whale call template from an HDF5 file.

    Args:
        template_path: Path to the HDF5 file containing the whale templates.
        sensor: Name of the sensor.
        call_type: Type of whale call (e.g., "type1", "type2").
    Returns:
        DataStream containing the whale template data.
    """
    group_name = f"{sensor}/{call_type}"
    logging.info(f"Reading whale template {group_name} from: {template_path}")
    with h5py.File(template_path, "r") as f:
        template_group = f[group_name]
        return read_hdf5_group(template_group)


def read_xcorr_data(data_path: Path, sensor: str) -> tuple[NDArray, NDArray, NDArray]:
    """Read the cross-correlation data from an HDF5 file for a specific sensor.

    Args:
        data_path: Path to the HDF5 file containing the cross-correlation data.
        sensor: Name of the sensor to read data for.
    Returns:
        Tuple of numpy arrays containing the cross-correlation data, shift matrix, and time differences.
    """
    logging.info(f"Reading xcorr data for sensor {sensor} from: {data_path}")
    with h5py.File(data_path, "r") as f:
        sensor_group = f[sensor]
        corr_matrix = sensor_group["corr"][:]
        shift_matrix = sensor_group["shifts"][:]
        time_diff = sensor_group["time_diff"][:]
    return corr_matrix, shift_matrix, time_diff
