"""Detection performance evaluation."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel
from scipy.signal import find_peaks, hilbert
from tqdm import tqdm
from tritonoa.data.reader import read_hdf5

import vineyard.readers as readers
from vineyard.process import estimate_detection_snr


class EvaluationConfig(BaseModel):
    annotations_file: Path
    detections_file: Path
    report_dir: Path | None = None
    time_window_s: float = 2.0
    max_travel_s: float = 4.0
    min_sensors: int = 2
    # SNR at annotated call times (denoised channel only)
    pc_data_dir: Path | None = None
    compute_snr: bool = True
    denoised_pc_channel: int = 5
    template_duration_s: float = 1.0
    f_low_hz: float = 19.0
    f_high_hz: float = 24.0
    snr_max_workers: int = 10
    pr_sweep: "PRSweepConfig | None" = None


class PRSweepConfig(BaseModel):
    """Configuration for detection threshold sweep to generate PR curve data."""

    output: Path = Path("reports/evaluation/pr_curve_data.csv")
    n_thresholds: int = 40
    # Synced from process.whale_detection in workflow.Config.sync_attributes
    sensors: list[dict] | None = None
    denoised_channel: int = 5
    filt_type: str = "bandpass"
    filt_freq: list[float] | float = [15.0, 50.0]
    # Synced from evaluation config
    pc_data_dir: Path | None = None
    match_window_s: float = 2.0


def load_annotations(path: Path) -> pl.DataFrame:
    """Load the manual annotation CSV and parse the 'time' column as datetime."""
    return pl.read_csv(path).with_columns(
        pl.col("time").str.to_datetime(time_unit="us")
    )


def load_detections(path: Path) -> pl.DataFrame:
    """Load the automatic detection CSV and parse the 'timestamp' column as datetime."""
    return pl.read_csv(path).with_columns(
        pl.col("timestamp").str.to_datetime(time_unit="us")
    )


def label_condition(
    df: pl.DataFrame,
    time_col: str,
    piling_ranges: list[tuple[np.datetime64, np.datetime64]],
) -> pl.DataFrame:
    """Add a 'condition' column indicating the noise environment.

    Args:
        df: DataFrame containing a datetime column.
        time_col: Name of the datetime column to test.
        piling_ranges: Active pile-driving intervals as (start, end) pairs of
            np.datetime64, e.g. from ProcessConfig.time_ranges.

    Returns:
        df with an additional string column 'condition': 'piling' or 'quiet'.
    """
    exprs = [
        (pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) <= pl.lit(end))
        for start, end in piling_ranges
    ]
    is_piling = reduce(lambda a, b: a | b, exprs)
    return df.with_columns(
        pl.when(is_piling)
        .then(pl.lit("piling"))
        .otherwise(pl.lit("quiet"))
        .alias("condition")
    )


def _match_sensor(
    ann_times_us: np.ndarray,
    det_times_us: np.ndarray,
    window_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbor matching within window_us microseconds.

    Both input arrays must be sorted in ascending order.

    Returns:
        (ann_matched, det_matched): boolean arrays indicating which rows were matched.
    """
    ann_matched = np.zeros(len(ann_times_us), dtype=bool)
    det_matched = np.zeros(len(det_times_us), dtype=bool)

    for i, t in enumerate(ann_times_us):
        lo = np.searchsorted(det_times_us, t - window_us)
        hi = np.searchsorted(det_times_us, t + window_us, side="right")
        candidates = np.where(~det_matched[lo:hi])[0] + lo

        if len(candidates) == 0:
            continue

        j = candidates[np.argmin(np.abs(det_times_us[candidates] - t))]
        ann_matched[i] = True
        det_matched[j] = True

    return ann_matched, det_matched


def match_detections(
    annotations: pl.DataFrame,
    detections: pl.DataFrame,
    time_window_s: float,
    annotation_time_col: str = "time",
    detection_time_col: str = "timestamp",
    sensor_col: str = "sensor",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Match detections to annotations within a time window, per sensor.

    Args:
        annotations: Annotations DataFrame; must include sensor_col and
            annotation_time_col. Assumed to be sorted by time within each sensor.
        detections: Detections DataFrame; must include sensor_col and
            detection_time_col. Assumed to be sorted by time within each sensor.
        time_window_s: Half-width of the matching window in seconds.
        annotation_time_col: Name of the datetime column in annotations.
        detection_time_col: Name of the datetime column in detections.
        sensor_col: Name of the column identifying the sensor.

    Returns:
        (annotations, detections) each with a boolean 'matched' column appended.
        TP: annotation matched. FP: detection not matched. FN: annotation not matched.
    """
    window_us = int(time_window_s * 1e6)
    sensors = annotations[sensor_col].unique().sort().to_list()

    ann_matched_all = np.zeros(len(annotations), dtype=bool)
    det_matched_all = np.zeros(len(detections), dtype=bool)

    ann_times_all = annotations[annotation_time_col].cast(pl.Int64).to_numpy()
    det_times_all = detections[detection_time_col].cast(pl.Int64).to_numpy()
    ann_sensors = annotations[sensor_col].to_numpy()
    det_sensors = detections[sensor_col].to_numpy()

    for sensor in sensors:
        ann_idx = np.where(ann_sensors == sensor)[0]
        det_idx = np.where(det_sensors == sensor)[0]

        ann_order = np.argsort(ann_times_all[ann_idx])
        det_order = np.argsort(det_times_all[det_idx])

        ann_m, det_m = _match_sensor(
            ann_times_all[ann_idx][ann_order],
            det_times_all[det_idx][det_order],
            window_us,
        )

        ann_matched_all[ann_idx[ann_order]] = ann_m
        det_matched_all[det_idx[det_order]] = det_m

    return (
        annotations.with_columns(pl.Series("matched", ann_matched_all)),
        detections.with_columns(pl.Series("matched", det_matched_all)),
    )


def compute_metrics(
    annotations: pl.DataFrame,
    detections: pl.DataFrame,
    group_cols: list[str] = ["sensor", "condition"],
) -> pl.DataFrame:
    """Compute precision, recall, and F1 from matched annotations and detections.

    Args:
        annotations: Output from match_detections; must have 'matched' column.
        detections: Output from match_detections; must have 'matched' column.
        group_cols: Columns to group by when computing metrics.

    Returns:
        DataFrame with TP, FP, FN, precision, recall, and F1 per group.
        Precision and recall are NaN where the denominator is zero.
    """
    ann_agg = annotations.group_by(group_cols).agg(
        pl.col("matched").sum().alias("tp"),
        (pl.len() - pl.col("matched").sum()).alias("fn"),
    )
    det_agg = detections.group_by(group_cols).agg(
        (pl.len() - pl.col("matched").sum()).alias("fp"),
    )
    return (
        ann_agg.join(det_agg, on=group_cols, how="left")
        .fill_null(0)
        .with_columns(
            (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).alias("precision"),
            (pl.col("tp") / (pl.col("tp") + pl.col("fn"))).alias("recall"),
        )
        .with_columns(
            (
                2
                * pl.col("precision")
                * pl.col("recall")
                / (pl.col("precision") + pl.col("recall"))
            ).alias("f1")
        )
        .sort(group_cols)
    )


def aggregate_events(
    annotations: pl.DataFrame,
    max_travel_s: float = 4.0,
    min_sensors: int = 2,
    annotation_time_col: str = "time",
) -> pl.DataFrame:
    """Group per-sensor annotations into call events and compute detection rate.

    Consecutive annotations (across all sensors, sorted by time) separated by
    less than max_travel_s are assumed to be arrivals of the same call at
    different sensors.

    Args:
        annotations: Output from match_detections; must have 'matched' and
            'condition' columns.
        max_travel_s: Maximum cross-site propagation time used to cluster
            per-sensor arrivals into a single call event.
        min_sensors: Minimum number of TP matches required for an event to be
            counted as detected.
        annotation_time_col: Name of the datetime column.

    Returns:
        DataFrame with one row per event: event_id, event_time, condition,
        n_sensors_annotated, n_sensors_detected, is_detected. Sorted by
        event_time.
    """
    ann_sorted = annotations.sort(annotation_time_col)
    times_us = ann_sorted[annotation_time_col].cast(pl.Int64).to_numpy()
    gaps_us = np.diff(times_us, prepend=times_us[0])
    event_ids = np.cumsum(gaps_us > int(max_travel_s * 1e6)).astype(np.int32)

    return (
        ann_sorted.with_columns(pl.Series("event_id", event_ids))
        .group_by("event_id")
        .agg(
            pl.col(annotation_time_col).min().alias("event_time"),
            pl.first("condition"),
            pl.len().alias("n_sensors_annotated"),
            pl.col("matched").sum().alias("n_sensors_detected"),
        )
        .with_columns(
            (pl.col("n_sensors_detected") >= min_sensors).alias("is_detected")
        )
        .sort("event_time")
    )


def _snr_for_row(
    row: dict,
    pc_data_dir: Path,
    channel: int,
    template_duration_s: float,
    f_low_hz: float,
    f_high_hz: float,
    annotation_time_col: str,
) -> dict | None:
    sensor = row["sensor"]
    pc_path = pc_data_dir / f"{sensor}_pc.h5"
    if not pc_path.exists():
        logging.warning("PC file not found: %s — skipping", pc_path)
        return None

    t = np.datetime64(row[annotation_time_col])
    try:
        result = estimate_detection_snr(
            pc_path=pc_path,
            channel=channel,
            detection_time=t,
            template_duration_s=template_duration_s,
            f_low_hz=f_low_hz,
            f_high_hz=f_high_hz,
        )
        snr_db = 10 * np.log10(result["snr_p"]) if result["snr_p"] > 0 else float("nan")
        sigma_t_ms = (
            result["sigma_t_s"] * 1e3
            if np.isfinite(result["sigma_t_s"])
            else float("nan")
        )
        var_t_ms2 = sigma_t_ms**2 if np.isfinite(sigma_t_ms) else float("nan")
    except Exception:
        logging.warning(
            "SNR estimation failed for %s at %s (channel %d)",
            sensor,
            t,
            channel,
            exc_info=True,
        )
        snr_db = float("nan")
        sigma_t_ms = float("nan")
        var_t_ms2 = float("nan")

    return {
        "sensor": sensor,
        "time": row[annotation_time_col],
        "condition": row["condition"],
        "snr_db": snr_db,
        "sigma_t_ms": sigma_t_ms,
        "var_t_ms2": var_t_ms2,
    }


def estimate_snr_at_annotations(
    annotations: pl.DataFrame,
    pc_data_dir: Path,
    channel: int,
    template_duration_s: float,
    f_low_hz: float,
    f_high_hz: float,
    max_workers: int = 4,
    annotation_time_col: str = "time",
) -> pl.DataFrame:
    """Estimate SNR on the denoised PC channel at each annotated whale call.

    Returns:
        DataFrame with one row per annotation: sensor, time, condition,
        snr_db, sigma_t_ms, var_t_ms2, unix_time_us.
    """
    rows = list(annotations.iter_rows(named=True))
    records: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _snr_for_row,
                row,
                pc_data_dir,
                channel,
                template_duration_s,
                f_low_hz,
                f_high_hz,
                annotation_time_col,
            ): row
            for row in rows
        }
        for future in tqdm(
            as_completed(futures), total=len(rows), desc="Estimating SNR"
        ):
            result = future.result()
            if result is not None:
                records.append(result)

    return pl.DataFrame(records).with_columns(
        pl.col("time").dt.epoch(time_unit="us").alias("unix_time_us")
    )


def _event_summary(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.group_by("condition")
        .agg(
            pl.len().alias("n_events"),
            pl.col("is_detected").sum().alias("n_detected"),
            pl.col("is_detected").mean().alias("detection_rate"),
        )
        .sort("condition")
    )


def run_evaluation(
    config: EvaluationConfig,
    piling_ranges: list[tuple[np.datetime64, np.datetime64]],
) -> None:
    """Run the full detection evaluation pipeline and print results to stdout."""
    logging.info("Loading annotations from %s", config.annotations_file)
    ann = load_annotations(config.annotations_file)
    ann = label_condition(ann, "time", piling_ranges)

    logging.info("Matching detections with %.1f s window", config.time_window_s)
    det = load_detections(config.detections_file)
    det = label_condition(det, "timestamp", piling_ranges)
    ann_m, det_m = match_detections(ann, det, config.time_window_s)

    per_sensor = compute_metrics(ann_m, det_m, group_cols=["sensor", "condition"])
    pooled = compute_metrics(ann_m, det_m, group_cols=["condition"])
    events = _event_summary(
        aggregate_events(ann_m, config.max_travel_s, config.min_sensors)
    )

    print("\n=== Per-sensor metrics by condition ===")
    print(per_sensor)
    print("\n=== Metrics by condition (pooled across sensors) ===")
    print(pooled)
    print("\n=== Event-level detection rate by condition ===")
    print(events)

    snr_df: pl.DataFrame | None = None
    if config.pc_data_dir is not None and config.compute_snr:
        logging.info(
            "Estimating SNR at annotated call times from %s", config.pc_data_dir
        )
        snr_df = estimate_snr_at_annotations(
            ann,
            config.pc_data_dir,
            config.denoised_pc_channel,
            config.template_duration_s,
            config.f_low_hz,
            config.f_high_hz,
            config.snr_max_workers,
        )
        snr_summary = (
            snr_df.group_by("sensor", "condition")
            .agg(
                pl.col("snr_db").mean().alias("mean_snr_db"),
                pl.col("snr_db").std().alias("std_snr_db"),
                pl.col("snr_db").median().alias("median_snr_db"),
                pl.col("var_t_ms2").mean().alias("mean_var_t_ms2"),
                pl.col("var_t_ms2").std().alias("std_var_t_ms2"),
                pl.col("var_t_ms2").median().alias("median_var_t_ms2"),
                pl.col("var_t_ms2").quantile(0.25).alias("q1_var_t_ms2"),
                pl.col("var_t_ms2").quantile(0.75).alias("q3_var_t_ms2"),
                pl.len().alias("n"),
            )
            .sort("sensor", "condition")
        )
        print("\n=== SNR at annotated call times (denoised channel) ===")
        print(snr_summary)

    if config.report_dir is not None:
        config.report_dir.mkdir(parents=True, exist_ok=True)
        per_sensor.write_csv(config.report_dir / "per_sensor_metrics.csv")
        pooled.write_csv(config.report_dir / "pooled_metrics.csv")
        events.write_csv(config.report_dir / "event_metrics.csv")
        if snr_df is not None:
            snr_df.write_csv(config.report_dir / "snr_comparison.csv")
            snr_summary.write_csv(config.report_dir / "snr_summary.csv")
        logging.info("Evaluation report saved to %s", config.report_dir)


def sweep_sensor(
    sensor: str,
    channel: int,
    ann_times_us: np.ndarray,
    piling_ranges: list[tuple],
    pc_data_dir: Path,
    distance_s: float,
    filt_type: str,
    filt_freq: list[float] | float,
    thresholds: np.ndarray,
    match_window_us: int,
) -> list[dict]:
    """Sweep detection thresholds for one sensor×channel.

    Loads the pulse-compressed HDF5 file once, then evaluates every threshold
    value in a single pass.  Detections are restricted to piling periods before
    matching so that results are directly comparable to the piling-only
    annotations.

    Args:
        sensor: Sensor name (e.g. 'vla1').
        channel: HDF5 channel index to sweep.
        ann_times_us: Sorted annotation timestamps in microseconds.
        piling_ranges: Active piling intervals as (start, end) np.datetime64 pairs.
        pc_data_dir: Directory containing {sensor}_pc.h5 files.
        distance_s: Minimum peak separation in seconds (same as detect_whale_calls).
        filt_type: Filter type string passed to readers.process_datastream.
        filt_freq: Filter frequency/frequencies.
        thresholds: Array of threshold values to sweep.
        match_window_us: Half-width of the matching window in microseconds.

    Returns:
        List of dicts with keys: sensor, threshold, tp, fp, fn.
    """

    pc_path = pc_data_dir / f"{sensor}_pc.h5"
    if not pc_path.exists():
        logging.warning("PC file not found: %s — skipping", pc_path)
        return []

    ds = read_hdf5(pc_path)
    ds = readers.process_datastream(ds, filt_type=filt_type, filt_freq=filt_freq)

    fs = ds.stats.sampling_rate
    cf = np.abs(hilbert(ds.data[channel]))
    cf /= np.maximum(np.max(cf), 1e-10)

    piling_us = [
        (
            np.datetime64(s, "us").astype(np.int64),
            np.datetime64(e, "us").astype(np.int64),
        )
        for s, e in piling_ranges
    ]

    distance_samples = int(distance_s * fs)
    records = []
    for t in thresholds:
        peaks, _ = find_peaks(cf, height=t, distance=distance_samples)
        all_det_us = ds.time_vector[peaks].astype(np.int64)

        in_piling = np.zeros(len(all_det_us), dtype=bool)
        for s_us, e_us in piling_us:
            in_piling |= (all_det_us >= s_us) & (all_det_us <= e_us)
        det_times_us = np.sort(all_det_us[in_piling])

        ann_matched, det_matched = _match_sensor(
            ann_times_us, det_times_us, match_window_us
        )
        records.append(
            {
                "sensor": sensor,
                "threshold": float(t),
                "tp": int(ann_matched.sum()),
                "fn": int(len(ann_matched) - ann_matched.sum()),
                "fp": int(len(det_matched) - det_matched.sum()),
            }
        )

    return records


def run_pr_sweep(
    config: PRSweepConfig,
    annotations_file: Path,
    piling_ranges: list[tuple],
) -> None:
    """Sweep detection thresholds across all configured sensors (denoised channel).

    Loads annotations, filters to piling condition, then iterates over every
    sensor, calling sweep_sensor for each.  Results are written as a CSV to
    config.output with columns: sensor, threshold, tp, fp, fn.

    Args:
        config: PRSweepConfig specifying sensors, filter params, and output.
        annotations_file: Path to the cleaned annotations CSV.
        piling_ranges: Active piling intervals from process config.
    """
    if config.sensors is None:
        raise ValueError(
            "PRSweepConfig.sensors is not set. "
            "Ensure [process.whale_detection] sensors are configured."
        )
    if config.pc_data_dir is None:
        raise ValueError(
            "PRSweepConfig.pc_data_dir is not set. "
            "Ensure [evaluation] pc_data_dir is configured."
        )

    ann = load_annotations(annotations_file)
    ann = label_condition(ann, "time", piling_ranges)
    ann_piling = ann.filter(pl.col("condition") == "piling")

    thresholds = np.linspace(0.02, 0.98, config.n_thresholds)
    match_window_us = int(config.match_window_s * 1e6)

    all_records: list[dict] = []
    with tqdm(total=len(config.sensors), desc="Sweeping PR thresholds") as pbar:
        for sensor_dict in config.sensors:
            sensor = sensor_dict["name"]
            distance_s = sensor_dict.get("distance_s", 7.0)
            ann_sensor = ann_piling.filter(pl.col("sensor") == sensor)
            ann_times_us = (
                ann_sensor.sort("time")["time"].dt.epoch(time_unit="us").to_numpy()
            )
            pbar.set_postfix(sensor=sensor)
            records = sweep_sensor(
                sensor=sensor,
                channel=config.denoised_channel,
                ann_times_us=ann_times_us,
                piling_ranges=piling_ranges,
                pc_data_dir=config.pc_data_dir,
                distance_s=distance_s,
                filt_type=config.filt_type,
                filt_freq=config.filt_freq,
                thresholds=thresholds,
                match_window_us=match_window_us,
            )
            all_records.extend(records)
            pbar.update(1)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(all_records).write_csv(config.output)
    logging.info("PR sweep data saved to %s", config.output)
