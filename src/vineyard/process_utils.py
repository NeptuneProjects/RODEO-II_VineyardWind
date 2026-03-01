import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags
from tritonoa.data.stream import DataStream


def enforce_same_size(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Ensure all arrays in the list have the same size by padding with zeros."""
    max_length = max(arr.shape[0] for arr in arrays)
    return [
        np.pad(arr, (0, max_length - arr.shape[0]), constant_values=0.0)
        for arr in arrays
    ]


def extract_trace(
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


def get_anchor_trace(corr_matrix_window: np.ndarray) -> int:
    """Get the "anchor trace" for a window of strikes, defined as the
    trace with the highest median correlation to all others.

    Args:
        corr_matrix_window: Correlation matrix for the strikes in the current window.

    Returns:
        Index of the anchor trace within the window.
    """
    corr_matrix_masked = np.where(corr_matrix_window < 1.0, corr_matrix_window, np.nan)
    median_corrs = np.nanmedian(corr_matrix_masked, axis=0)
    anchor_index = np.nanargmax(median_corrs)
    return anchor_index


def sample_delay(sig1: NDArray, sig2: NDArray) -> int:
    """Calculate the sample delay between two signals using cross-correlation.

    Args:
        sig1: First signal.
        sig2: Second signal.

    Returns:
        Time delay in seconds between the two signals.
    """
    xcorr = correlate(sig1, sig2, mode="same")
    lags = correlation_lags(len(sig1), len(sig2), mode="same")
    peak_lag = lags[np.argmax(xcorr)]
    return peak_lag
