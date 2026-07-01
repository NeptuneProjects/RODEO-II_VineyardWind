import gc
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from polars import DataFrame, concat
from pydantic import BaseModel, Field, model_validator
from rodeo.utils import compute_array_size, initialize_julia
from scipy.signal import find_peaks, hilbert
from tqdm import tqdm
from tritonoa.data.reader import read_and_process, read_hdf5
from tritonoa.data.signal import taper
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

import vineyard.readers as readers
from vineyard.figures.templates import plot_template
from vineyard.process_utils import (
    enforce_same_size,
    extract_trace,
    get_anchor_trace,
    sample_delay,
)

logger = logging.getLogger(__name__)


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


class WhaleDetectionConfig(BaseModel):
    sensors: list[dict] | None = None
    channel: int = 5
    filt_type: str | None = None
    filt_freq: float | Sequence[float] | None = None
    output_file: Path = "data/acoustic/whale_detections.csv"


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
    calibration_dir: Path | None = None
    start_time: str | None = None
    end_time: str | None = None
    time_ranges: list[list[str]] | None = None
    strike_config: StrikeConfig = Field(alias="strike")
    template_config: TemplateConfig = Field(alias="template")
    whale_template_config: WhaleTemplateConfig | None = Field(alias="whale_template")
    denoise_config: DenoiseConfig | None = Field(alias="denoise")
    whale_detection_config: WhaleDetectionConfig | None = Field(alias="whale_detection")

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
                logger.warning(
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
            logger.warning(
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
            anchor_index = get_anchor_trace(corr_matrix_window)
            anchor_trace = extract_trace(
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

                if i == j:
                    reference_ind = idx

                traces.append(aligned_tr)

            # STEP 4: Compute final template from template-aligned traces
            traces = np.array(enforce_same_size(traces))
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
                time = np.arange(len(template)) / fs
                title = (
                    f"{name.upper()} - Strike {i} - {strike_index.item(i, 'start_time')}\n"
                    f"Rolling Window: [{window_start}, {window_end})"
                )
                savepath = plot_dir / name
                savepath.mkdir(parents=True, exist_ok=True)
                fig = plot_template(
                    time, traces, template, reference_ind, title=title, ylim=ylim
                )
                fig.savefig(
                    savepath / f"{name}_strike_{i:04d}_template.png",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

    logger.info(f"Rolling templates for {name.upper()} saved to {template_data_path}")


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
    logger.info(f"Processing sensor: {name}, channel: {channel}")

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

    peaks = _find_strikes(ds.data[0], ds.stats.sampling_rate, threshold, distance_s)
    samples_since_reference = (
        (ds.time_vector[peaks] - reference_time)
        / np.timedelta64(1, "s")
        * ds.stats.sampling_rate
    ).astype(int)

    logger.info(f"Found {len(peaks)} peaks for sensor {name}.")
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
            logger.info(
                f"Processing sensor {sensor['name']}, time range "
                f"{i + 1}/{len(time_ranges)}: {time_start} to {time_end}"
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
    logger.info(f"Strikes extracted and saved to {config.strike_index}.")


def build_templates(
    config: TemplateConfig,
    start_time: np.datetime64,
    end_time: np.datetime64,
    inventory_path: Path,
    calibration_dir: Path,
    strike_index_path: Path,
    strike_corr_path: Path,
) -> None:
    for sensor in config.sensors:
        name = sensor["name"]
        channel = sensor["channel"]
        ylim = sensor["ylim"]
        logger.info(f"Processing sensor: {name} channel {channel}.")

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
        ds.data = readers.calibrate(
            calibration_dir, ds.data, ds.stats.sampling_rate, name
        )
        ds.stats.units = "uPa"

        corr_matrix, _, _ = readers.read_xcorr_data(strike_corr_path, name)
        _build_sensor_templates_rolling(
            ds,
            strike_index,
            corr_matrix,
            config.template_data,
            name,
            config.buffer_start,
            config.buffer_end,
            window_size=config.window_size,
            plot_dir=config.plot_dir,
            ylim=ylim,
        )

    logger.info(f"Templates built and saved to {config.template_data}.")


def _compute_noise_reduction(
    denoised_file: Path, strike_index_file: Path
) -> pl.DataFrame:
    ds = read_hdf5(denoised_file)
    fs = ds.stats.sampling_rate
    start_buffer = int(0.9 * fs)
    end_buffer = int(0.7 * fs)

    df = pl.read_csv(strike_index_file, try_parse_dates=True).filter(
        pl.col("sensor") == denoised_file.stem
    )

    rms_df = pl.DataFrame(
        {
            "global_sample": df["global_sample"],
            "global_start": df["global_sample"] - start_buffer,
            "global_end": df["global_sample"] + end_buffer,
            "orig_rms_db": np.zeros(len(df)),
            "filt_rms_db": np.zeros(len(df)),
            "rms_diff_db": np.zeros(len(df)),
        }
    )

    for i, row in enumerate(df.iter_rows(named=True)):
        strike_index = row["global_sample"]
        orig_data = ds.data[0, strike_index - start_buffer : strike_index + end_buffer]
        filt_data = ds.data[1, strike_index - start_buffer : strike_index + end_buffer]

        # Calculate RMS in dB re 1 uPa
        orig_data_rms_db = 20 * np.log10(np.sqrt(np.mean(orig_data**2)))
        filt_data_rms_db = 20 * np.log10(np.sqrt(np.mean(filt_data**2)))

        rms_df[i, "orig_rms_db"] = orig_data_rms_db
        rms_df[i, "filt_rms_db"] = filt_data_rms_db
        rms_df[i, "rms_diff_db"] = orig_data_rms_db - filt_data_rms_db

    return rms_df


def _construct_template_signal(
    signal: ArrayLike,
    strike_inds: list[int],
    templates: list[NDArray[np.float64]],
    start_samples: list[int],
    end_samples: list[int],
    taper_pc: float | None = None,
) -> np.ndarray:
    """Construct a template signal by placing templates at the given strike
    indices.

    This function handles overlapping templates by trimming the template
    from the beginning, ensuring that the strike indices remain consistent
    with the original signal.

    Args:
        signal: The original signal to which the templates will be added.
        strike_inds: List of indices corresponding to the strikes in the signal.
        templates: List of template signals corresponding to each strike index.
        start_samples: List of start sample indices for each template.
        end_samples: List of end sample indices for each template.
        taper_pc: Optional percentage for tapering the templates to reduce
            edge effects.

    Returns:
        A signal constructed by adding the templates at the specified strike
            indices, with handling for overlapping templates.
    """
    template_signal = np.zeros_like(signal)
    previous_end = 0

    for strike_ind, start_ind, end_ind in tqdm(
        zip(strike_inds, start_samples, end_samples),
        desc="Constructing template signal",
        total=len(strike_inds),
    ):
        template = templates[strike_ind]

        # Handle overlap by trimming template from the beginning, not shifting position
        # However, if the previous placement would consume the entire current window
        # (likely due to noise), trust the current index and place it normally
        template_offset = 0
        if start_ind < previous_end < end_ind:
            # Normal overlap: trim from beginning
            template_offset = previous_end - start_ind
            start_ind = previous_end

        # Apply offset to skip overlapping part of template
        template = template[template_offset:]

        min_length = min(len(template), end_ind - start_ind)
        updated_template = template[:min_length]
        window = (
            taper(len(updated_template), max_percentage=taper_pc)
            if taper_pc is not None
            else np.ones(len(updated_template))
        )

        template_signal[start_ind : start_ind + min_length] += updated_template * window
        previous_end = start_ind + min_length

    return template_signal


def _denoise_data(
    signal: ArrayLike,
    strike_index: pl.DataFrame,
    templates: list[NDArray[np.float64]],
    start_samples: list[int],
    end_samples: list[int],
    taper_pc: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    strike_inds = strike_index["strike_index"].to_list()
    template_signal = _construct_template_signal(
        signal, strike_inds, templates, start_samples, end_samples, taper_pc=taper_pc
    )
    error = signal - template_signal
    return error, template_signal


def denoise_strikes(
    config: DenoiseConfig,
    start_time: np.datetime64,
    end_time: np.datetime64,
    inventory_path: Path,
    calibration_dir: Path,
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
    mean_reduction, median_reduction, std_reduction = [], [], []
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
        ds.data = readers.calibrate(
            calibration_dir, ds.data, ds.stats.sampling_rate, sensor["name"]
        )
        ds.stats.units = "uPa"

        x_filtered, y = _denoise_data(
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
        output_file = config.denoised_data / f"{sensor['name']}.h5"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ds.write_hdf5(output_file)

        df = _compute_noise_reduction(output_file, strike_index_path)
        df.write_csv(config.denoised_data / f"{sensor['name']}_noise_reduction.csv")
        mean_reduction.append(df["rms_diff_db"].mean())
        median_reduction.append(df["rms_diff_db"].median())
        std_reduction.append(df["rms_diff_db"].std())
        logger.info(f"Denoised data for {sensor['name']} saved to {output_file}.")

    pl.DataFrame(
        {
            "sensor": [s["name"] for s in config.sensors],
            "mean_reduction_db": mean_reduction,
            "median_reduction_db": median_reduction,
            "std_reduction_db": std_reduction,
        }
    ).write_csv(config.denoised_data / "noise_reduction_summary.csv")
    logger.info(
        f"Noise reduction summary saved to {config.denoised_data / 'noise_reduction_summary.csv'}."
    )


def _detect_on_channel(
    config: WhaleDetectionConfig, data_path: Path, channel: int, output_file: Path
) -> None:
    dfs = []
    for sensor in config.sensors:
        ds = readers.process_datastream(
            read_hdf5(data_path / f"{sensor['name']}_pc.h5"),
            filt_type=config.filt_type,
            filt_freq=config.filt_freq,
        )
        fs = ds.stats.sampling_rate
        cf = np.abs(hilbert(ds.data[channel]))
        peaks = find_peaks(
            cf / np.max(cf),
            height=sensor["threshold"],
            distance=int(sensor["distance_s"] * fs),
        )[0]
        times = _refine_peak_times(cf, peaks, ds.time_vector, fs)
        del cf, ds
        gc.collect()
        logger.info(
            f"Detected {len(peaks)} whale calls on sensor {sensor['name']} "
            f"(channel {channel})."
        )
        dfs.append(
            pl.DataFrame(
                {
                    "sensor": sensor["name"],
                    "timestamp": times,
                    "sample": peaks,
                    "channel": channel,
                    "threshold": sensor["threshold"],
                    "distance_s": sensor["distance_s"],
                }
            )
        )
    df = pl.concat(dfs).with_columns(
        pl.col("timestamp").dt.epoch(time_unit="us").alias("unix_time_us")
    )
    df.write_csv(output_file)
    logger.info(f"Whale call detections saved to {output_file}.")


def detect_whale_calls(config: WhaleDetectionConfig, data_path: Path) -> None:
    """Detect whale calls in the dataset using the Hilbert transform and peak finding.

    Args:
        config: WhaleDetectionConfig instance containing the configuration
            for whale call detection.
        data_path: Path to the directory containing the denoised data files.
    """
    _detect_on_channel(config, data_path, config.channel, config.output_file)


_NOISE_CORRECTIONS: dict[str, float] = {
    "rayleigh": np.log(2),  # median(e²) = E[e²]·ln(2) for Rayleigh envelope
    "none": 1.0,
}


def estimate_detection_snr(
    pc_path: Path,
    channel: int,
    detection_time: np.datetime64,
    template_duration_s: float,
    f_low_hz: float,
    f_high_hz: float,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    window_s: float = 5.0,
    noise_correction: float | str = "rayleigh",
) -> dict[str, float]:
    """Estimate SNR and CRLB timing uncertainty for a single whale call detection.

    Loads the pulse-compressed data in a ±window_s window around the detection,
    computes the amplitude envelope via the Hilbert transform, and partitions it
    into signal and noise regions.  Noise power is estimated using the median of
    the squared envelope, which is robust to sparse pile driving transients that
    survive denoising.  Signal power is the mean squared envelope in the signal
    window minus the noise floor — this naturally weights a triangular (or any
    non-flat) envelope correctly, unlike using the peak alone.

    The signal window spans ±template_duration_s around the detection peak,
    covering the full matched-filter output width (~2T at the base).

    Timing uncertainty follows the CRLB for a linear FM chirp in AWGN:
    σ_t = 1 / (2π·β·√SNR), where β² = (f₁²+f₁f₂+f₂²)/3 is the mean-square
    bandwidth of the swept band [f_low_hz, f_high_hz].

    Args:
        pc_path: Path to the pulse-compressed HDF5 file ({sensor}_pc.h5).
        channel: Integer channel index (same value passed to detect_whale_calls).
        detection_time: Refined peak timestamp from the detection DataFrame.
        template_duration_s: Duration of the whale call template in seconds.
            Defines the signal exclusion window for noise estimation.
        f_low_hz: Lower frequency of the linear FM chirp template (Hz).
        f_high_hz: Upper frequency of the linear FM chirp template (Hz).
        filt_type: Optional filter type forwarded to process_datastream (should
            match the filter applied during detection).
        filt_freq: Optional filter frequencies forwarded to process_datastream.
        window_s: Half-width of the data window loaded around the detection (s).
            Must satisfy window_s > template_duration_s; 5 s is recommended.
        noise_correction: Divisor applied to median(e²) to recover an unbiased
            estimate of E[e²].  "rayleigh" uses ln(2), appropriate when the
            bandpass noise envelope is Rayleigh-distributed (Gaussian noise).
            "none" uses the raw median (distribution-free but biased low).
            A float value is used directly as the divisor.

    Returns:
        Dict with keys:
            snr_p       — linear peak signal-to-noise power ratio
            sigma_t_s   — 1-sigma single-channel timing uncertainty (s)
            sigma_tdoa_s— 1-sigma TDOA uncertainty between two channels (s)
            p_noise     — estimated noise power (envelope-squared units)
            p_signal    — estimated signal power (envelope-squared units)
    """
    if isinstance(noise_correction, str):
        if noise_correction not in _NOISE_CORRECTIONS:
            raise ValueError(
                f"noise_correction must be one of {list(_NOISE_CORRECTIONS)!r} or a float"
            )
        correction = _NOISE_CORRECTIONS[noise_correction]
    else:
        correction = float(noise_correction)

    # Load a single sample to obtain file metadata without reading all data.
    meta = read_hdf5(pc_path, start=0, stop=1)
    fs = meta.stats.sampling_rate
    t_file_start = meta.stats.time_init

    # Convert detection timestamp to sample index within the file.
    dt_s = (detection_time - t_file_start) / np.timedelta64(1, "s")
    peak_sample = int(round(float(dt_s) * fs))

    # Load ±window_s window, clamped to file bounds.
    half_win = int(window_s * fs)
    start_idx = max(0, peak_sample - half_win)
    stop_idx = peak_sample + half_win

    ds = readers.process_datastream(
        read_hdf5(pc_path, start=start_idx, stop=stop_idx),
        filt_type=filt_type,
        filt_freq=filt_freq,
    )

    env_sq = np.abs(hilbert(ds.data[channel])) ** 2

    # Peak location within the loaded window.
    peak_local = peak_sample - start_idx
    n = len(env_sq)

    # Signal window covers ±T around the peak (full MF output width ≈ 2T).
    sig_half = int(template_duration_s * fs)
    sig_start = max(0, peak_local - sig_half)
    sig_stop = min(n, peak_local + sig_half)

    noise_mask = np.ones(n, dtype=bool)
    noise_mask[sig_start:sig_stop] = False

    # Noise power: median / correction converts median(e²) → E[e²].
    p_noise = np.median(env_sq[noise_mask]) / correction

    # Signal power: mean envelope power in signal window minus noise floor.
    # Correctly weights non-flat (e.g. triangular) envelopes; mean(A²)/3 for
    # a triangle vs A²_peak, giving the true mean rather than inflated peak.
    p_signal = float(max(0.0, np.mean(env_sq[sig_start:sig_stop]) - p_noise))

    snr_p = p_signal / p_noise if p_noise > 0.0 else 0.0

    # β² = (f₁² + f₁f₂ + f₂²)/3 — mean-square bandwidth of linear FM chirp
    beta_sq = (f_low_hz**2 + f_low_hz * f_high_hz + f_high_hz**2) / 3.0
    if snr_p > 0.0:
        sigma_t = 1.0 / (2.0 * np.pi * np.sqrt(beta_sq * snr_p))
        sigma_tdoa = np.sqrt(2.0) * sigma_t
    else:
        sigma_t = sigma_tdoa = float("inf")

    return {
        "snr_p": snr_p,
        "sigma_t_s": sigma_t,
        "sigma_tdoa_s": sigma_tdoa,
        "p_noise": float(p_noise),
        "p_signal": p_signal,
    }


_SENSOR_PAIRS = [("3dvha", "vla1"), ("3dvha", "vla2"), ("vla1", "vla2")]
_SENSORS = ["3dvha", "vla1", "vla2"]

# Maps estimate_detection_snr key → output column suffix (without sensor name).
# sigma_t_s is abbreviated to sigma_t; snr_p gets _db suffix.
_RESULT_FIELD_MAP = {
    "snr_p": "snr_p_db",
    "sigma_t_s": "sigma_t",
    "sigma_tdoa_s": "sigma_tdoa_s",
    "p_noise": "p_noise",
    "p_signal": "p_signal",
}
# Fields whose linear values are converted to dB (10·log10) before storage.
_DB_FIELDS = {"snr_p"}


def _sigma_tdoa_for_row(
    i: int,
    row: dict,
    pc_data_path: Path,
    channel: int,
    template_duration_s: float,
    f_low_hz: float,
    f_high_hz: float,
    filt_type: str | None,
    filt_freq: float | Sequence[float] | None,
    window_s: float,
    noise_correction: float | str,
) -> tuple[int, dict[str, dict[str, float]], float]:
    """Compute σ_TDOA for a single detection row.

    Returns (index, per_sensor_full_results, combined_var_tdoa).
    per_sensor_full_results maps sensor name → full estimate_detection_snr dict.
    """
    ref_time_np = np.datetime64(row["timestamp"])
    sensor_results: dict[str, dict[str, float]] = {}

    for sensor in _SENSORS:
        pc_path = pc_data_path / f"{sensor}_pc.h5"
        if not pc_path.exists():
            continue

        t_sensor = ref_time_np + np.timedelta64(int(row[sensor] * 1e6), "us")
        try:
            result = estimate_detection_snr(
                pc_path=pc_path,
                channel=channel,
                detection_time=t_sensor,
                template_duration_s=template_duration_s,
                f_low_hz=f_low_hz,
                f_high_hz=f_high_hz,
                filt_type=filt_type,
                filt_freq=filt_freq,
                window_s=window_s,
                noise_correction=noise_correction,
            )
            if np.isfinite(result["sigma_t_s"]):
                sensor_results[sensor] = result
        except Exception:
            logger.warning(
                "SNR estimation failed for sensor %s at detection %d.",
                sensor,
                i,
                exc_info=True,
            )

    var_t = {s: r["sigma_t_s"] ** 2 for s, r in sensor_results.items()}
    pair_variances = [
        var_t[s1] + var_t[s2] for s1, s2 in _SENSOR_PAIRS if s1 in var_t and s2 in var_t
    ]
    combined_var = float(np.mean(pair_variances)) if pair_variances else float("nan")
    return i, sensor_results, combined_var


def compute_sigma_tdoa_per_detection(
    df: pl.DataFrame,
    pc_data_path: Path,
    channel: int,
    template_duration_s: float,
    f_low_hz: float,
    f_high_hz: float,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    window_s: float = 5.0,
    noise_correction: float | str = "rayleigh",
    max_workers: int = 10,
) -> pl.DataFrame:
    """Compute per-detection TDOA timing uncertainty from pulse-compressed data.

    For each detection, estimates the single-channel timing uncertainty σ_t at
    every sensor by calling estimate_detection_snr, then combines them into a
    single representative σ_TDOA per detection.  The combination is the mean of
    the three pairwise values: σ_TDOA_ij = √(σ_t_i² + σ_t_j²).

    Per-sensor timestamps are recovered from the TDOA DataFrame: the reference
    site timestamp plus the stored TDOA offset gives each sensor's absolute
    detection time.  Sensors whose PC file is missing or whose SNR estimate
    fails are skipped; if all three sensors fail for a detection, the
    corresponding entry is NaN.

    Detections are processed concurrently with a ThreadPoolExecutor; the inner
    sensor loop within each detection remains serial so that concurrent reads
    from the same HDF5 file are minimised.

    Args:
        df: TDOA DataFrame as returned by estimate_tdoa / correlations_to_dataframe,
            with columns: timestamp (pl.Datetime), 3dvha, vla1, vla2 (seconds).
        pc_data_path: Directory containing {sensor}_pc.h5 files.
        channel: Integer channel index (same as used during detection).
        template_duration_s: Template duration in seconds.
        f_low_hz: Lower frequency of the linear FM chirp template (Hz).
        f_high_hz: Upper frequency of the linear FM chirp template (Hz).
        filt_type: Filter type forwarded to process_datastream.
        filt_freq: Filter frequencies forwarded to process_datastream.
        window_s: Half-width of the data window around each detection (s).
        noise_correction: Noise correction for estimate_detection_snr.
        max_workers: Number of parallel threads.

    Returns:
        DataFrame with one column per (field, sensor) combination — named
        ``{col}_{sensor}`` where col follows _RESULT_FIELD_MAP — plus a
        ``var_tdoa`` column containing the mean pairwise TDOA variance
        (σ_t,i² + σ_t,j²) averaged over all sensor pairs.  Entries are NaN
        where estimation failed.
    """
    n = len(df)
    # col_name -> per-detection array
    arrays: dict[str, np.ndarray] = {
        f"{col}_{sensor}": np.full(n, np.nan)
        for col in _RESULT_FIELD_MAP.values()
        for sensor in _SENSORS
    }
    var_tdoa_out = np.full(n, np.nan)
    rows = list(df.iter_rows(named=True))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _sigma_tdoa_for_row,
                i,
                row,
                pc_data_path,
                channel,
                template_duration_s,
                f_low_hz,
                f_high_hz,
                filt_type,
                filt_freq,
                window_s,
                noise_correction,
            ): i
            for i, row in enumerate(rows)
        }
        for future in tqdm(
            as_completed(futures), total=len(rows), desc="Estimating σ_TDOA"
        ):
            i, sensor_results, combined_var = future.result()
            var_tdoa_out[i] = combined_var
            for sensor, result in sensor_results.items():
                for src_key, col in _RESULT_FIELD_MAP.items():
                    v = result[src_key]
                    if src_key in _DB_FIELDS:
                        v = 10.0 * np.log10(v) if v > 0.0 else float("-inf")
                    arrays[f"{col}_{sensor}"][i] = v

    return pl.DataFrame({**arrays, "var_tdoa": var_tdoa_out})


def extract_whale_templates(
    config: WhaleTemplateConfig, inventory_path: Path, calibration_dir: Path
) -> None:
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
                template.data = readers.calibrate(
                    calibration_dir,
                    template.data,
                    template.stats.sampling_rate,
                    sensor_name,
                )
                template.stats.units = "uPa"
                g = f.create_group(f"{sensor_name}/{call_type}")
                template.create_hdf5_dataset(g)
                logger.info(
                    f"Saved template for whale {call_type} on {sensor_name} to {config.template_data}"
                )


def _find_strikes(
    data: NDArray[np.float64],
    sampling_rate: float,
    threshold: float,
    distance_sec: float,
) -> NDArray[np.int32]:
    def _characteristic_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
        xsq = x**2
        return xsq / np.max(xsq)

    cf = _characteristic_function(data)
    peaks = find_peaks(
        cf, height=threshold, distance=int(distance_sec * sampling_rate)
    )[0]
    return peaks


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
    logger.info("==== BEGIN STRIKE DETECTION ====")
    build_strikes_df(
        config.strike_config,
        config.start_time,
        config.time_ranges,
        config.inventory_path,
    )
    logger.info("==== STRIKE DETECTION COMPLETE ====")

    logger.info("==== BEGIN STRIKE SAVING ====")
    save_strikes(config.strike_config, config.inventory_path, config.calibration_dir)
    logger.info("==== STRIKE SAVING COMPLETE ====")

    logger.info("==== BEGIN STRIKE CROSS-CORRELATION ====")
    xcorr_strike_pairs(
        config.strike_config.strike_corr_config,
        config.strike_config.strike_data,
        config.strike_config.strike_corr,
    )
    logger.info("==== STRIKE CROSS-CORRELATION COMPLETE ====")

    logger.info("==== BEGIN TEMPLATE BUILDING ====")
    build_templates(
        config.template_config,
        config.start_time,
        config.end_time,
        config.inventory_path,
        config.calibration_dir,
        config.strike_config.strike_index,
        config.strike_config.strike_corr,
    )
    logger.info("==== TEMPLATE BUILDING COMPLETE ====")

    logger.info("==== BEGIN STRIKE DENOISING ====")
    denoise_strikes(
        config.denoise_config,
        config.start_time,
        config.end_time,
        config.inventory_path,
        config.calibration_dir,
        config.strike_config.strike_index,
        config.template_config.template_data,
    )
    logger.info("==== STRIKE DENOISING COMPLETE ====")

    logger.info("==== BEGIN WHALE TEMPLATE EXTRACTION ====")
    extract_whale_templates(
        config.whale_template_config, config.inventory_path, config.calibration_dir
    )
    logger.info("==== WHALE TEMPLATE EXTRACTION COMPLETE ====")

    logger.info("==== BEGIN PULSE COMPRESSION ====")
    pulse_compress(config.denoise_config, config.whale_template_config)
    logger.info("==== PULSE COMPRESSION COMPLETE ====")

    logger.info("==== BEGIN WHALE CALL DETECTION ====")
    detect_whale_calls(
        config.whale_detection_config, config.denoise_config.denoised_data
    )
    logger.info("==== WHALE CALL DETECTION COMPLETE ====")


def pulse_compress(denoise_config: DenoiseConfig, config: WhaleTemplateConfig) -> None:
    """Apply pulse compression to the denoised strike data using the whale call templates.

    Args:
        denoise_config: DenoiseConfig instance containing the configuration
            for denoising and pulse compression.
        config: WhaleTemplateConfig instance containing the configuration
            for whale call templates.
    """
    for sensor in denoise_config.sensors:
        logger.info(f"Processing sensor: {sensor['name']} for pulse compression.")
        sensor_name = sensor["name"]

        ds = read_hdf5(denoise_config.denoised_data / f"{sensor_name}.h5")
        ds_orig = ds.copy()

        ds.data = ds.data[1]
        ds_orig.data = ds_orig.data[0]

        template_type1 = readers.read_whale_template(
            config.template_data, sensor_name, "type1"
        ).data.squeeze()
        template_type2 = readers.read_whale_template(
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
        logger.info(
            f"Pulse compression completed for sensor: {sensor_name}. "
            f"Saved to {denoise_config.denoised_data / f'{sensor_name}_pc.h5'}"
        )


def _refine_peak_times(
    cf: NDArray[np.float64],
    peaks: NDArray[np.intp],
    time_vector: NDArray,
    fs: float,
) -> NDArray:
    """Refine integer-sample peak locations to sub-sample precision via parabolic interpolation.

    Fits a parabola through each peak and its two immediate neighbors to
    estimate the true maximum to fractional-sample accuracy, then converts
    the offset to a timedelta and adds it to the sample time.  Peaks at the
    array boundary (index 0 or N-1) are returned at their original sample time.

    Args:
        cf: Characteristic function from which peaks were detected.
        peaks: Integer sample indices returned by find_peaks.
        time_vector: Time vector aligned with cf (numpy datetime64 array).
        fs: Sampling rate in Hz.

    Returns:
        Array of refined peak times (same dtype as time_vector).
    """
    n = len(cf)
    deltas = np.zeros(len(peaks))
    interior = (peaks > 0) & (peaks < n - 1)

    if interior.any():
        p = peaks[interior]
        a, b, c = cf[p - 1], cf[p], cf[p + 1]
        denom = a - 2.0 * b + c
        with np.errstate(invalid="ignore", divide="ignore"):
            deltas[interior] = np.where(denom != 0.0, 0.5 * (a - c) / denom, 0.0)

    offsets_us = np.round(deltas * 1e6 / fs).astype(np.int64)
    return time_vector[peaks] + offsets_us.astype("timedelta64[us]")


def save_strikes(
    config: StrikeConfig, inventory_path: Path, calibration_dir: Path
) -> None:
    """Save the strike data to an HDF5 file.

    Args:
        config: StrikeConfig instance containing the configuration for saving strikes.
        inventory_path: Path to the inventory directory containing sensor CSV files.
        calibration_dir: Path to the calibration directory containing calibration files.
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
            ds.data = readers.calibrate(
                calibration_dir, ds.data, ds.stats.sampling_rate, sensor
            )
            ds.stats.units = "uPa"
            g = file.create_group(f"{sensor}/{strike_index:04d}")
            ds.create_hdf5_dataset(g)

    logger.info(f"Strikes saved to {config.strike_data}.")


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

    logger.info(f"Data shape: {num_detections} detections.")
    logger.info(
        f"Size of arrays: {compute_array_size([max_corr, time_diff]) / (1024**3):.2f} GB."
    )

    jl = initialize_julia("CrossCorr")
    threads = jl.seval("Threads.nthreads()")
    logger.info(f"Number of Julia threads: {threads}.")

    jl_data = jl.seval("x -> Matrix{Float64}(x)")(data)
    jl_time = jl.seval("x -> Vector{Float64}(x)")(t0)

    time_diff = np.array(jl.CrossCorr.dt_matrix(jl_time))
    max_corr, shifts = jl.CrossCorr.corr_matrix(jl_data)
    max_corr = np.array(max_corr)
    shifts = np.array(shifts)

    logger.info(
        f"Computed time_diff, max_corr, and shifts for sensor {sensor.upper()}."
    )
    logger.info(
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
            logger.info(f"Processing sensor {sensor.upper()}.")
            record = xcorr_sensor(
                sensor, sensor_group, config.model_dump(exclude={"max_workers"})
            )
            record.save_h5(output_path)
            logger.info(f"Processed sensor {sensor.upper()}.")

    logger.info(f"Cross-correlation of strike pairs saved to {output_path}.")
