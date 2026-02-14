#!/usr/bin/env python3
"""Find pile-driving strikes in the dataset using peak-finding."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars import DataFrame, concat
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
from tritonoa.data.reader import read_and_process, read_hdf5, read_hdf5_group
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

import vineyard.readers as readers
from rodeo.utils import compute_array_size, initialize_julia
from vineyard.plotting import plot_template
from vineyard.readers import read_whale_template
from vineyard.signal_proc import denoise_data, find_strikes, sample_delay


class DenoiseConfig(BaseModel):
    denoised_data: Path = "data/acoustic/denoised"
    sensors: list[dict] | None = None
    detrend: bool | None = None
    taper_pc: float | None = None
    dec_factor: int | None = None
    filt_type: str = "bandpass"
    filt_freq: float | Sequence[float] = [19.0, 25.0]
    buffer_start: float = 0.75
    buffer_end: float = 0.85
    template_taper_pc: float = 0.05


class StrikeCorrConfig(BaseModel):
    max_workers: int = 10
    detrend: bool = True
    taper_pc: float = 0.05
    dec_factor: int | None = None
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


class TDOAConfig(BaseModel): ...


class TemplateConfig(BaseModel):
    """Configuration for the template-building process."""

    sensors: list[dict] | None = None
    buffer_start: float = 0.75
    buffer_end: float = 0.85
    taper_pc: float | None = None
    dec_factor: int | None = None
    filt_type: str | None = None
    filt_freq: float | Sequence[float] | None = None
    corr_cutoff: float = 0.8
    window_size: int = 20
    plot_dir: Path | None = None
    template_data: Path = "data/acoustic/templates/strike_templates.h5"


class WhaleTemplateConfig(BaseModel):
    """Configuration for whale call templates.

    Structure: calls[sensor][type] -> {start: datetime, end: datetime}
    Example: calls["3dvha"]["type1"]["start"]
    """

    template_data: Path
    filt_type: str | None = None
    filt_freq: float | Sequence[float] | None = None
    taper_pc: float | None = None
    calls: dict[str, dict[str, int | dict[str, str]]]

    @model_validator(mode="after")
    def convert_to_np_datetime(self) -> "WhaleTemplateConfig":
        """Convert start and end times from strings to numpy datetime64."""
        for sensor_calls in self.calls.values():
            for key, value in sensor_calls.items():
                # Skip non-dict fields like 'channel'
                if isinstance(value, dict) and "start" in value and "end" in value:
                    value["start"] = np.datetime64(value["start"], TIME_PRECISION)
                    value["end"] = np.datetime64(value["end"], TIME_PRECISION)
        return self


class ProcessConfig(BaseModel):
    """Configuration for the strike-finding process."""

    inventory_path: Path | None = None
    start_time: str | None = None
    end_time: str | None = None
    time_ranges: list[list[str]] | None = None
    strike_config: StrikeConfig = Field(alias="strike")
    template_config: TemplateConfig = Field(alias="template")
    whale_template_config: WhaleTemplateConfig | None = Field(alias="whale_template")
    denoise_config: DenoiseConfig | None = Field(alias="denoise")

    @model_validator(mode="after")
    def convert_to_np_datetime(self) -> "ProcessConfig":
        """Convert time_ranges from lists of strings to lists of numpy datetime64 tuples."""
        if self.start_time is not None:
            self.start_time = np.datetime64(self.start_time, TIME_PRECISION)
        if self.end_time is not None:
            self.end_time = np.datetime64(self.end_time, TIME_PRECISION)
        if self.time_ranges is not None:
            self.time_ranges = [
                (
                    np.datetime64(start, TIME_PRECISION),
                    np.datetime64(end, TIME_PRECISION),
                )
                for start, end in self.time_ranges
            ]
        return self


@dataclass
class Record:
    """Data class to hold the results of the cross-correlation for a single sensor."""

    sensor: str
    time_diff: np.ndarray
    corr: np.ndarray
    shifts: np.ndarray

    def save_h5(self, path: Path) -> None:
        """Save the record to an HDF5 file.

        The record is saved to the output file in the following format:
        ```
        /sensor
            /time_diff
            /corr
            /shifts
        ```
        `time_diff`, `corr`, and `shifts` are 2D arrays with the shape
        `(num_detections, num_detections)`.

        Args:
            path: Path to the output HDF5 file.
        """
        with h5py.File(path, "a") as file:
            # Create a new group for the station if it doesn't exist:
            if self.sensor in file:
                logging.warning(
                    f"Group {self.sensor} already exists in {path}. Overwriting."
                )
                del file[self.sensor]

            file.create_group(self.sensor)

            # Save the data:
            grp = file[f"{self.sensor}"]
            grp.attrs["sensor"] = self.sensor
            grp.create_dataset("time_diff", data=self.time_diff)
            grp.create_dataset("corr", data=self.corr)
            grp.create_dataset("shifts", data=self.shifts)


def _build_strikes_df_per_sensor(
    inventory_path: Path,
    sensor: dict,
    reference_time: np.datetime64,
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

    ds = read_and_process(
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
    samples_since_reference = (
        (ds.time_vector[peaks] - reference_time)
        / np.timedelta64(1, "s")
        * ds.stats.sampling_rate
    ).astype(int)

    logging.info(f"Found {len(peaks)} peaks for sensor {name}.")
    return DataFrame(
        {
            "sensor": name,
            "channel": channel,
            "strike_index": np.arange(len(peaks)) + strike_index_offset,
            "time": ds.time_vector[peaks],
            "sample": peaks,
            "global_sample": samples_since_reference,
        }
    )


def build_strikes_df(
    config: StrikeConfig,
    reference_time: np.datetime64,
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
        reference_time: Reference time to use for calculating time differences.
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

            df = _build_strikes_df_per_sensor(
                inventory_path / f"inventory_{sensor['name']}.csv",
                sensor,
                reference_time,
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


def build_templates(
    config: TemplateConfig,
    start_time: np.datetime64,
    end_time: np.datetime64,
    inventory_path: Path,
    strike_index_path: Path,
    strike_corr_path: Path,
) -> None:
    for sensor in config.sensors:
        name = sensor["name"]
        channel = sensor["channel"]
        ylim = sensor["ylim"]
        logging.info(f"Processing sensor: {name} channel {channel}.")

        ds, strike_index = readers.read_strike_data(
            inventory_path / f"inventory_{name}.csv",
            strike_index_path,
            name,
            channel,
            start_time,
            end_time,
            config.buffer_start,
            config.buffer_end,
            taper_pc=config.taper_pc,
            dec_factor=config.dec_factor,
            filt_type=config.filt_type,
            filt_freq=config.filt_freq,
        )
        corr_matrix, shift_matrix, _ = readers.read_xcorr_data(strike_corr_path, name)
        _build_sensor_templates_rolling(
            ds,
            strike_index,
            corr_matrix,
            shift_matrix,
            config.template_data,
            name,
            config.buffer_start,
            config.buffer_end,
            window_size=config.window_size,
            max_shift=config.max_shift,
            plot_dir=config.plot_dir,
            ylim=ylim,
        )


def _build_sensor_templates_rolling(
    ds,
    strike_index: pl.DataFrame,
    corr_matrix: np.ndarray,
    template_data_path: Path,
    name: str,
    buffer_start: float,
    buffer_end: float,
    window_size: int = 20,
    plot_dir: Path | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Build templates using a rolling median window with iterative refinement.

    This iterative approach combines the robustness of consensus alignment with the
    precision of template-based alignment, ensuring all traces align to the same
    waveform features. For each strike i, the template is computed from strikes
    [i - window_size//2, i + window_size//2].

    Args:
        ds: DataStream containing the acoustic data.
        strike_index: DataFrame containing strike indices and times.
        corr_matrix: Pre-computed correlation matrix (n_strikes x n_strikes).
        template_data_path: Path to save the template data.
        name: Name of the sensor.
        buffer_start: Buffer before the strike peak (in seconds).
        buffer_end: Buffer after the strike peak (in seconds).
        window_size: Number of strikes to include in the rolling window.
        plot_dir: Optional directory to save template plots.
        ylim: Optional y-axis limits for plots.
    """
    num_strikes = len(strike_index)
    max_template_length = int((buffer_start + buffer_end) * ds.stats.sampling_rate)
    half_window = window_size // 2
    fs = ds.stats.sampling_rate

    # Initialize HDF file and datasets
    template_data_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(template_data_path, "a") as f:
        if name in f:
            logging.warning(
                f"Group {name} already exists in template_data. Overwriting."
            )
            del f[name]
        g = f.create_group(name)
        g.attrs["sampling_rate"] = fs
        g.create_dataset("start_sample", shape=(num_strikes,), dtype=int)
        g.create_dataset("end_sample", shape=(num_strikes,), dtype=int)
        g.create_dataset(
            "data",
            shape=(num_strikes, max_template_length),
            dtype=float,
            fillvalue=0.0,
        )

        for i in tqdm(
            range(num_strikes),
            desc=f"Processing {name}",
            total=num_strikes,
            unit="strike",
        ):
            # Determine window boundaries with constant window size
            # For the first half_window strikes, use [0, window_size)
            # For the last half_window strikes, use [num_strikes - window_size, num_strikes)
            # For middle strikes, use [i - half_window, i + half_window)
            if i < half_window:
                window_start = 0
                window_end = min(window_size, num_strikes)
            elif i >= num_strikes - half_window:
                window_start = max(0, num_strikes - window_size)
                window_end = num_strikes
            else:
                window_start = i - half_window
                window_end = i + half_window

            # Get indices of strikes in the window
            template_inds = list(range(window_start, window_end))

            # STEP 1: Get the anchor trace to which all other traces in window
            # will be initially aligned.
            corr_matrix_window = corr_matrix[
                window_start:window_end, window_start:window_end
            ]
            anchor_index = _get_anchor_trace(corr_matrix_window)
            anchor_trace = _extract_trace(
                ds,
                np.datetime64(
                    strike_index.item(template_inds[anchor_index], "start_time")
                ),
                np.datetime64(
                    strike_index.item(template_inds[anchor_index], "end_time")
                ),
            )

            traces = []
            for idx, j in enumerate(template_inds):
                tr_start = np.datetime64(strike_index.item(j, "start_time"))
                tr_end = np.datetime64(strike_index.item(j, "end_time"))

                if idx == anchor_index:
                    aligned_tr = anchor_trace
                else:
                    tr = _extract_trace(ds, tr_start, tr_end)
                    shift_samples = sample_delay(anchor_trace, tr)
                    shift_seconds = shift_samples / fs

                    aligned_tr = _extract_trace(
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

                if i == j:
                    reference_ind = idx

                traces.append(aligned_tr)

            # STEP 4: Compute final template from template-aligned traces
            traces = np.array(_enforce_same_size(traces))
            template = np.median(traces, axis=0)

            # Get window indices for this strike
            start_index, end_index = _get_window_inds(
                ds.stats.sampling_rate,
                strike_index.item(i, "global_sample"),
                buffer_start,
                buffer_end,
            )

            # Extract the ORIGINAL trace from the SAME window we'll use for placement
            original_trace = ds.data[0, start_index:end_index]

            # Ensure both signals are the same length for accurate cross-correlation
            min_length = min(len(original_trace), len(template))
            original_trace_trimmed = original_trace[:min_length]
            template_trimmed = template[:min_length]

            ref_shift = sample_delay(original_trace_trimmed, template_trimmed)

            # Pad or trim template to match max_template_length
            template_length = len(template)
            if template_length < max_template_length:
                padded_template = np.full(max_template_length, 0.0)
                padded_template[:template_length] = template
            else:
                padded_template = template[:max_template_length]

            g["data"][i, :] = padded_template
            g["start_sample"][i] = start_index + ref_shift
            g["end_sample"][i] = end_index + ref_shift

            if plot_dir:
                title = (
                    f"{name.upper()} - Strike {i} - {strike_index.item(i, 'start_time')}\n"
                    f"Rolling Window: [{window_start}, {window_end})"
                )
                savepath = plot_dir / name
                savepath.mkdir(parents=True, exist_ok=True)
                fig = plot_template(
                    traces, template, reference_ind, title=title, ylim=ylim
                )
                fig.savefig(
                    savepath / f"{name}_strike_{i:04d}_template.png",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

    logging.info(f"Rolling templates for {name.upper()} saved to {template_data_path}")


def denoise_strikes(
    config: DenoiseConfig,
    start_time: np.datetime64,
    end_time: np.datetime64,
    inventory_path: Path,
    strike_index_path: Path,
    template_path: Path,
) -> None:
    """Denoise strike data for each sensor using the provided templates and
    strike indices.

    Args:
        config: DenoiseConfig instance containing the configuration for denoising.
        start_time: Start time of the time range to process.
        end_time: End time of the time range to process.
        inventory_path: Path to the inventory CSV files for the sensors.
        strike_index_path: Path to the CSV file containing strike indices.
        template_path: Path to the HDF5 file containing templates for each sensor.
    """
    for sensor in config.sensors:
        ds, strike_index, templates, start_samples, end_samples = (
            readers.read_denoise_data(
                inventory_path / f"inventory_{sensor['name']}.csv",
                strike_index_path,
                template_path,
                sensor["name"],
                sensor["channel"],
                start_time,
                end_time,
                taper_pc=config.taper_pc,
                dec_factor=config.dec_factor,
                filt_type=config.filt_type,
                filt_freq=config.filt_freq,
                buffer_start=config.buffer_start,
                buffer_end=config.buffer_end,
            )
        )
        x_filtered, y = denoise_data(
            ds.data[0],
            strike_index,
            templates,
            start_samples,
            end_samples,
            taper_pc=config.template_taper_pc,
        )

        ds.data = np.vstack((ds.data[0], x_filtered, y))
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
        save_dir = config.denoised_data
        save_dir.mkdir(parents=True, exist_ok=True)
        ds.write_hdf5(save_dir / f"{sensor['name']}.h5")


def _enforce_same_size(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Ensure all arrays in the list have the same size by padding with zeros."""
    max_length = max(arr.shape[0] for arr in arrays)
    return [
        np.pad(arr, (0, max_length - arr.shape[0]), constant_values=0.0)
        for arr in arrays
    ]


def _extract_trace(
    ds: DataStream, start_time: np.datetime64, end_time: np.datetime64
) -> np.ndarray:
    """Extract trace from datastream using direct array indexing.

    Args:
        ds: DataStream containing the acoustic data.
        start_time: Start time of the trace to extract.
        end_time: End time of the trace to extract.
    Returns:
        Extracted trace data.
    """
    # Convert times to sample indices
    fs = ds.stats.sampling_rate
    start_sample = int((start_time - ds.stats.time_init) / np.timedelta64(1, "s") * fs)
    end_sample = int((end_time - ds.stats.time_init) / np.timedelta64(1, "s") * fs)

    # Clip to valid range
    start_sample = max(0, start_sample)
    end_sample = min(ds.num_samples, end_sample)

    return ds.data[0, start_sample:end_sample]


def extract_whale_templates(config: WhaleTemplateConfig, inventory_path: Path) -> None:
    with h5py.File(config.template_data, "w") as f:
        for sensor_name, sensor_data in config.calls.items():
            channel = sensor_data.pop("channel")
            for call_type, call_times in sensor_data.items():
                time_start = call_times["start"]
                time_end = call_times["end"]
                template = read_and_process(
                    inventory_path / f"inventory_{sensor_name}.csv",
                    time_start,
                    time_end,
                    channels=channel,
                    taper_pc=config.taper_pc,
                    filt_type=config.filt_type,
                    filt_freq=config.filt_freq,
                )

                g = f.create_group(f"{sensor_name}_{call_type}")
                template.create_hdf5_dataset(g)
                logging.info(
                    f"Saved template for whale {call_type} on {sensor_name} to {config.template_data}"
                )


def _get_anchor_trace(corr_matrix_window: np.ndarray) -> int:
    """Get the "anchor trace" for a window of strikes, defined as the
    trace with the highest median correlation to all others.

    Args:
        corr_matrix_window (np.ndarray): Correlation matrix for the strikes
            in the current window.
    Returns:
        int: Index of the anchor trace within the window.
    """
    corr_matrix_masked = np.where(corr_matrix_window < 1.0, corr_matrix_window, np.nan)
    median_corrs = np.nanmedian(corr_matrix_masked, axis=0)
    anchor_index = np.nanargmax(median_corrs)
    return anchor_index


def _get_template_inds(
    num_signals: int, corrs: np.ndarray, threshold: float = 0.9, window_size: int = 35
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
    # Create a mask for the window constraint (vectorized)
    row_idx, col_idx = np.meshgrid(
        np.arange(num_signals), np.arange(num_signals), indexing="ij"
    )
    window_mask = np.abs(row_idx - col_idx) <= window_size

    # Apply correlation threshold and window mask
    valid_mask = (corrs > threshold) & window_mask

    # Extract indices for each row
    return [np.where(valid_mask[i])[0].tolist() for i in range(num_signals)]


def _get_window_inds(
    sampling_rate: float, peak_index: int, buffer_start: float, buffer_end: float
) -> tuple[int, int]:
    start_index = peak_index - int(buffer_start * sampling_rate)
    end_index = peak_index + int(buffer_end * sampling_rate)
    return start_index, end_index


def process_data(config: ProcessConfig) -> None:
    """Run data processing steps based on the provided configuration.

    Args:
        config: ProcessConfig instance containing the configuration for
            data processing.
    """
    build_strikes_df(
        config.strike_config,
        config.start_time,
        config.time_ranges,
        config.inventory_path,
    )
    save_strikes(config.strike_config, config.inventory_path)
    xcorr_strike_pairs(
        config.strike_config.strike_corr_config,
        config.strike_config.strike_data,
        config.strike_config.strike_corr,
    )
    build_templates(
        config.template_config,
        config.start_time,
        config.end_time,
        config.inventory_path,
        config.strike_config.strike_index,
        config.strike_config.strike_corr,
    )
    denoise_strikes(
        config.denoise_config,
        config.start_time,
        config.end_time,
        config.inventory_path,
        config.strike_config.strike_index,
        config.template_config.template_data,
    )
    extract_whale_templates(config.whale_template_config, config.inventory_path)
    pulse_compress(config.denoise_config, config.whale_template_config)


def pulse_compress(denoise_config: DenoiseConfig, config: WhaleTemplateConfig) -> None:
    """Apply pulse compression to the denoised strike data using the whale call templates.

    Args:
        denoise_config: DenoiseConfig instance containing the configuration
            for denoising and pulse compression.
        config: WhaleTemplateConfig instance containing the configuration
            for whale call templates.
    """
    for sensor in denoise_config.sensors:
        logging.info(f"Processing sensor: {sensor['name']} for pulse compression.")
        sensor_name = sensor["name"]

        ds = read_hdf5(denoise_config.denoised_data / f"{sensor_name}.h5")
        ds_orig = ds.copy()

        ds.data = ds.data[1]
        ds_orig.data = ds_orig.data[0]

        template_type1 = read_whale_template(
            config.template_data, sensor_name, "type1"
        ).data.squeeze()
        template_type2 = read_whale_template(
            config.template_data, sensor_name, "type2"
        ).data.squeeze()

        ds_pc_orig1 = ds_orig.copy().pulse_compression(template_type1).data
        ds_pc_orig2 = ds_orig.copy().pulse_compression(template_type2).data
        ds_pc_dn1 = ds.copy().pulse_compression(template_type1).data
        ds_pc_dn2 = ds.copy().pulse_compression(template_type2).data

        new_ds = ds.copy()
        new_ds.data = np.vstack(
            (ds_orig.data, ds.data, ds_pc_orig1, ds_pc_orig2, ds_pc_dn1, ds_pc_dn2)
        )
        new_ds.stats.channels = [0, 1, 2, 3, 4, 5]
        new_ds.stats.metadata = {
            "sensor": sensor_name,
            "channel": sensor["channel"],
            "channel_names": {
                0: "Original Signal",
                1: "Denoised Signal",
                2: "Pulse Compression Original Type 1",
                3: "Pulse Compression Original Type 2",
                4: "Pulse Compression Denoised Type 1",
                5: "Pulse Compression Denoised Type 2",
            },
        }

        new_ds.write_hdf5(denoise_config.denoised_data / f"{sensor_name}_pc.h5")
        logging.info(
            f"Pulse compression completed for sensor: {sensor_name}. "
            f"Saved to {denoise_config.denoised_data / f'{sensor_name}_pc.h5'}"
        )


def save_strikes(config: StrikeConfig, inventory_path: Path) -> None:
    """Save the strike data to an HDF5 file.

    Args:
        config: StrikeConfig instance containing the configuration for saving strikes.
        inventory_path: Path to the inventory directory containing sensor CSV files.
    """
    save_config = config.strike_save_config

    df = readers.read_strike_index(
        config.strike_index,
        save_config.buffer_start,
        save_config.buffer_end,
    )

    with h5py.File(config.strike_data, "w") as file:
        for row in tqdm(
            df.iter_rows(), desc="Extracting & saving strikes", total=df.shape[0]
        ):
            sensor, channel, strike_index, _, _, _, time_start, time_end = row
            ds = read_and_process(
                inventory_path / f"inventory_{sensor}.csv",
                time_start,
                time_end,
                channel,
                detrend=save_config.detrend,
                taper_pc=save_config.taper_pc,
                dec_factor=save_config.dec_factor,
                filt_type=save_config.filt_type,
                filt_freq=save_config.filt_freq,
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
    data, t0 = readers.read_strikes(sensor_group, **sp_kwargs)

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
    max_corr, shifts = jl.CrossCorr.corr_matrix(jl_data)
    max_corr = np.array(max_corr)
    shifts = np.array(shifts)

    logging.info(
        f"Computed time_diff, max_corr, and shifts for sensor {sensor.upper()}."
    )
    logging.info(
        f"Shape of time_diff: {time_diff.shape}, max_corr: {max_corr.shape}, "
        f"shifts: {shifts.shape}."
    )

    return Record(
        sensor=sensor,
        time_diff=time_diff,
        corr=max_corr,
        shifts=shifts,
    )


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
