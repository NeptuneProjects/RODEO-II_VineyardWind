"""Module for common signal processing functions used in the project."""

from collections.abc import Sequence
from curses import window
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy.signal import correlate, correlation_lags, find_peaks
from tqdm import tqdm
from tritonoa.data.signal import taper
from tritonoa.data.stream import DataStream
from tritonoa.signal.util import resample_ratio


def complex_cepstrum(
    x: NDArray[np.float64], n: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    r"""Compute the complex cepstrum of a real sequence.

    The complex cepstrum is given by:
    $$
    c[n] = F^{-1}\left[\log_{10}\left(F{x[n]}\right)\right]
    $$

    where $x[n]$ is the input signal and $F$ and $F^{-1}$
    are respectively the forward and backward Fourier transform.


    Args:
      x: Real sequence to compute complex cepstrum of.
      n: Length of the Fourier transform.

    Returns:
      The complex cepstrum of the real data sequence `x` computed using the Fourier transform.
      The amount of samples of circular delay added to `x`.


    See Also:
      - [`real_cepstrum`][acoustic_toolbox.cepstrum.real_cepstrum]: Compute the real cepstrum.
      - [`inverse_complex_cepstrum`][acoustic_toolbox.cepstrum.inverse_complex_cepstrum]: Compute the inverse complex cepstrum of a real sequence.

    Examples:
      In the following example we use the cepstrum to determine the fundamental
      frequency of a set of harmonics. There is a distinct peak at the quefrency
      corresponding to the fundamental frequency. To be more precise, the peak
      corresponds to the spacing between the harmonics.

      >>> import numpy as np
      >>> import matplotlib.pyplot as plt
      >>> from acoustic_toolbox.cepstrum import complex_cepstrum

      >>> duration = 5.0
      >>> fs = 8000.0
      >>> samples = int(fs*duration)
      >>> t = np.arange(samples) / fs

      >>> fundamental = 100.0
      >>> harmonics = np.arange(1, 30) * fundamental
      >>> signal = np.sin(2.0*np.pi*harmonics[:,None]*t).sum(axis=0)
      >>> ceps, _ = complex_cepstrum(signal)

      >>> fig = plt.figure()
      >>> ax0 = fig.add_subplot(211)
      >>> ax0.plot(t, signal)
      >>> ax0.set_xlabel('time in seconds')
      >>> ax0.set_xlim(0.0, 0.05)
      >>> ax1 = fig.add_subplot(212)
      >>> ax1.plot(t, ceps)
      >>> ax1.set_xlabel('quefrency in seconds')
      >>> ax1.set_xlim(0.005, 0.015)
      >>> ax1.set_ylim(-5., +10.)

    References:
      1. Wikipedia, "Cepstrum".
            [http://en.wikipedia.org/wiki/Cepstrum](http://en.wikipedia.org/wiki/Cepstrum)
      2. M.P. Norton and D.G. Karczub, D.G.,
            "Fundamentals of Noise and Vibration Analysis for Engineers", 2003.
      3. B. P. Bogert, M. J. R. Healy, and J. W. Tukey:
            "The Quefrency Analysis of Time Series for Echoes: Cepstrum, Pseudo
            Autocovariance, Cross-Cepstrum and Saphe Cracking".
            Proceedings of the Symposium on Time Series Analysis
            Chapter 15, 209-243. New York: Wiley, 1963.
    """

    def _unwrap(
        phase: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Unwrap phase values.

        Args:
          phase: Phase values to unwrap.

        Returns:
          Unwrapped phase values
          Number of delay samples
        """
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay

    spectrum = np.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay


def construct_template_signal(
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
        import matplotlib.pyplot as plt

        template = templates[strike_ind]

        # Handle overlap by trimming template from the beginning, not shifting position
        template_offset = 0
        if start_ind < previous_end:
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


def convert_to_strain_rate(
    data: NDArray[np.float64],
    scale_factor: float,
    calibration_factor: float,
    sampling_rate_hz: float,
    gauge_length_m: float,
) -> np.ndarray:
    conversion_factor = (
        scale_factor * calibration_factor * sampling_rate_hz / gauge_length_m
    )
    return data * conversion_factor


def denoise_data(
    signal: ArrayLike,
    strike_index: pl.DataFrame,
    templates: list[NDArray[np.float64]],
    start_samples: list[int],
    end_samples: list[int],
    taper_pc: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    strike_inds = strike_index["strike_index"].to_list()
    template_signal = construct_template_signal(
        signal, strike_inds, templates, start_samples, end_samples, taper_pc=taper_pc
    )
    error = signal - template_signal
    return error, template_signal


def inverse_complex_cepstrum(
    ceps: NDArray[np.float64], ndelay: NDArray[np.int_]
) -> NDArray[np.float64]:
    r"""Compute the inverse complex cepstrum of a real sequence.

    The inverse complex cepstrum is given by:
    $$
    x[n] = F^{-1}\left[\exp(F(c[n]))\right]
    $$

    where $c[n]$ is the input signal and $F$ and $F^{-1}$ are respectively the forward and backward Fourier transform.

    Args:
      ceps: Real sequence to compute inverse complex cepstrum of.
      ndelay: The amount of samples of circular delay added to `x`.

    Returns:
      The inverse complex cepstrum of the real sequence `ceps`.

    See Also:
      - [`complex_cepstrum`][acoustic_toolbox.cepstrum.complex_cepstrum]: Compute the complex cepstrum of a real sequence.
      - [`real_cepstrum`][acoustic_toolbox.cepstrum.real_cepstrum]: Compute the real cepstrum of a real sequence.

    Examples:
      Taking the complex cepstrum and then the inverse complex cepstrum results
      in the original sequence.

      >>> import numpy as np
      >>> from acoustic_toolbox.cepstrum import inverse_complex_cepstrum
      >>> x = np.arange(10)
      >>> ceps, ndelay = complex_cepstrum(x)
      >>> y = inverse_complex_cepstrum(ceps, ndelay)
      >>> print(x)
      >>> print(y)

    References:
      1. Wikipedia, "Cepstrum".
          [http://en.wikipedia.org/wiki/Cepstrum](http://en.wikipedia.org/wiki/Cepstrum)
    """

    def _wrap(
        phase: NDArray[np.float64], ndelay: NDArray[np.int_]
    ) -> NDArray[np.float64]:
        """Wrap phase values.

        Args:
          phase: Phase values to wrap.
          ndelay: Number of delay samples.

        Returns:
          Wrapped phase values.
        """
        ndelay = np.array(ndelay)
        samples = phase.shape[-1]
        center = (samples + 1) // 2
        wrapped = phase + np.pi * ndelay[..., None] * np.arange(samples) / center
        return wrapped

    log_spectrum = np.fft.fft(ceps)
    spectrum = np.exp(log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay))
    x = np.fft.ifft(spectrum).real
    return x


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


def real_cepstrum(x: NDArray[np.float64], n: int | None = None) -> NDArray[np.float64]:
    r"""Compute the real cepstrum of a real sequence.

    The real cepstrum is given by:
    $$
    c[n] = F^{-1}\left[\log_{10}\left|F{x[n]}\right|\right]
    $$

    where $x[n]$ is the input signal and $F$ and $F^{-1}$ are respectively
    the forward and backward Fourier transform.

    Note that contrary to the complex cepstrum the magnitude is taken of the spectrum.

    Args:
      x: Real sequence to compute real cepstrum of.
      n: Length of the Fourier transform.

    Returns:
      The real cepstrum.


    See Also:
      - [`complex_cepstrum`][acoustic_toolbox.cepstrum.complex_cepstrum]: Compute the complex cepstrum of a real sequence.
      - [`inverse_complex_cepstrum`][acoustic_toolbox.cepstrum.inverse_complex_cepstrum]: Compute the inverse complex cepstrum of a real sequence.

    Examples:
      >>> from acoustic_toolbox.cepstrum import real_cepstrum

    References:
      1. Wikipedia, "Cepstrum".
          [http://en.wikipedia.org/wiki/Cepstrum](http://en.wikipedia.org/wiki/Cepstrum)
    """
    spectrum = np.fft.fft(x, n=n)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
    return ceps


def resample_datastreams(data: list[DataStream], target_fs: float) -> list[DataStream]:
    if isinstance(data, DataStream):
        data = [data]

    resampled_data = []
    for ds in data:
        fs = ds.stats.sampling_rate
        p, q = resample_ratio(fs, target_fs)
        resampled_data.append(ds.filter("lowpass", target_fs / 2).resample_poly(p, q))

    return resampled_data


def subtract_median(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Subtract the median from each row of the data array."""
    return data - np.median(data, axis=1, keepdims=True)


def find_strikes(
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


def roll_and_pad(data: NDArray, shift: int, fill: Any = 0.0, axis: int = -1) -> NDArray:
    shifted_data = np.roll(data, shift, axis=axis)
    if shift > 0:
        shifted_data[:shift] = fill
    elif shift < 0:
        shifted_data[shift:] = fill
    return shifted_data


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
