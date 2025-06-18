"""Module for common signal processing functions used in the project."""

import numpy as np
import numpy.typing as npt
from tritonoa.data.stream import DataStream
from tritonoa.signal.util import resample_ratio


def convert_to_strain_rate(
    data: npt.NDArray[np.float64],
    scale_factor: float,
    calibration_factor: float,
    sampling_rate_hz: float,
    gauge_length_m: float,
) -> np.ndarray:
    conversion_factor = (
        scale_factor * calibration_factor * sampling_rate_hz / gauge_length_m
    )
    return data * conversion_factor


def resample_datastreams(data: list[DataStream], target_fs: float) -> list[DataStream]:
    if isinstance(data, DataStream):
        data = [data]

    resampled_data = []
    for ds in data:
        fs = ds.stats.sampling_rate
        p, q = resample_ratio(fs, target_fs)
        resampled_data.append(ds.filter("lowpass", target_fs / 2).resample_poly(p, q))

    return resampled_data


def subtract_median(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Subtract the median from each row of the data array."""
    return data - np.median(data, axis=1, keepdims=True)
