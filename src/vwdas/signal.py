# -*- coding: utf-8 -*-
"""Module for common signal processing functions used in the project."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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


def subtract_median(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Subtract the median from each row of the data array."""
    return data - np.median(data, axis=1, keepdims=True)