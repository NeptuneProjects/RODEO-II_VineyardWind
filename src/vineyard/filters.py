from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import pywt

from vineyard.signal import complex_cepstrum, inverse_complex_cepstrum, real_cepstrum


def plot_signal_cwt(
    signal_data,
    sampling_rate=1.0,
    wavelet="morl",
    scales=None,
    title="Continuous Wavelet Transform",
    cmap="viridis",
    figsize=(10, 8),
):
    """
    Plot a signal in Continuous Wavelet Transform (CWT) space.

    Parameters:
    -----------
    signal_data : array_like
        The input signal to analyze
    sampling_rate : float, optional
        Sampling rate of the signal in Hz (default: 1.0)
    wavelet : str, optional
        Wavelet to use (default: 'morl' for Morlet wavelet)
        Options include: 'morl', 'cmor', 'gaus', 'mexh', etc.
    scales : array_like, optional
        Scales for the CWT (default: None, will generate appropriate scales)
    title : str, optional
        Title for the plot (default: 'Continuous Wavelet Transform')
    cmap : str, optional
        Colormap for the CWT plot (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 8))

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plots
    """
    # Generate appropriate scales if not provided
    # if scales is None:
    #     # Create logarithmically spaced scales
    #     width = min(len(signal_data), 1024)  # Use smaller of signal length or 1024
    #     scales = np.arange(1, width // 2)

    # Compute the CWT
    coef, frequencies = pywt.cwt(signal_data, scales, wavelet, 1.0 / sampling_rate)

    # Create figure with subplots
    fig, axs = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 3]}
    )

    # Plot the original signal
    time = np.arange(len(signal_data)) / sampling_rate
    axs[0].plot(time, signal_data)
    axs[0].set_title("Original Signal")
    axs[0].set_xlabel("Time [s]" if sampling_rate != 1.0 else "Samples")
    axs[0].set_ylabel("Amplitude")

    # Plot the CWT
    # Convert scales to frequencies for better interpretation
    # if wavelet == "morl":
    #     # For Morlet, the relationship is approximately scale = central_freq/freq
    #     central_freq = pywt.central_frequency(wavelet)
    #     frequencies = central_freq * sampling_rate / scales
    # else:
    #     # For other wavelets, use a general approximation
    #     frequencies = sampling_rate / scales

    # Create the CWT plot
    im = axs[1].imshow(
        np.abs(coef),
        aspect="auto",
        cmap=cmap,
        extent=[time[0], time[-1], frequencies[-1], frequencies[0]],
    )
    axs[1].set_title(title)
    axs[1].set_xlabel("Time [s]" if sampling_rate != 1.0 else "Samples")
    axs[1].set_ylabel("Frequency [Hz]")

    # Add colorbar
    plt.colorbar(im, ax=axs[1], orientation="vertical", label="Magnitude")

    plt.tight_layout()
    return fig


class ModelBasedLMSFilter:
    """
    LMS Filter for cancelling known interfering signals.
    Uses a model of the interferer instead of just a reference microphone.
    """

    def __init__(self, filter_length, mu, initial_weights=None):
        """
        Initialize the LMS filter for model-based interference cancellation.

        Args:
            filter_length (int): Number of filter taps
            mu (float): Step size parameter
            initial_weights (ndarray, optional): Initial filter coefficients
        """
        self.filter_length = filter_length
        self.mu = mu

        # Initialize filter weights
        if initial_weights is None:
            self.weights = np.zeros(filter_length)
        else:
            self.weights = initial_weights

        # Buffer for storing interferer model output
        self.reference_buffer = np.zeros(filter_length)

    def update(self, primary_sample, interferer_sample):
        """
        Process one sample and update filter weights.

        Args:
            primary_sample (float): Current sample from primary input (signal + interferer)
            interferer_sample (float): Current sample from interferer model

        Returns:
            float: Filtered output sample (estimated clean signal)
        """
        # Update reference buffer with new interferer sample
        self.reference_buffer = np.roll(self.reference_buffer, 1)
        self.reference_buffer[0] = interferer_sample

        # Calculate filtered output
        filtered_interferer = np.dot(self.weights, self.reference_buffer)

        # Error is the difference between primary and filtered interferer
        error = primary_sample - filtered_interferer

        # Update weights using LMS rule
        self.weights = self.weights + self.mu * error * self.reference_buffer

        return error

    def process_signal(self, primary_signal, interferer_model_output):
        """
        Process entire signals at once.

        Args:
            primary_signal (ndarray): Primary input signal (signal + interferer)
            interferer_model_output (ndarray): Output from the interferer model

        Returns:
            ndarray: Filtered output signal (cleaned)
        """
        output_signal = np.zeros(len(primary_signal))

        # Reset filter state
        self.weights = np.zeros(self.filter_length)
        self.reference_buffer = np.zeros(self.filter_length)

        # Process each sample
        for i in range(len(primary_signal)):
            output_signal[i] = self.update(
                primary_signal[i], interferer_model_output[i]
            )

        return output_signal


@jit(nopython=True)
def ukf_reference_denoiser(
    primary_signal,
    reference_signal,
    process_noise=0.01,
    measurement_noise=0.1,
    dt=1.0,
    alpha=0.1,
    beta=2.0,
    kappa=0,
    state_dim=4,
    measurement_dim=2,
):
    """
    Apply Unscented Kalman Filter to denoise a signal using a reference signal.

    This implementation models the relationship between the primary and reference signals,
    using the UKF to estimate the clean signal.

    Args:
        primary_signal (ndarray): Primary input signal (desired signal + noise)
        reference_signal (ndarray): Reference signal correlated with the noise
        process_noise (float): Process noise covariance magnitude
        measurement_noise (float): Measurement noise covariance magnitude
        dt (float): Time step between measurements
        alpha (float): UKF parameter controlling spread of sigma points
        beta (float): UKF parameter for prior knowledge of distribution
        kappa (float): UKF parameter, secondary scaling parameter
        state_dim (int): Dimension of state vector
        measurement_dim (int): Dimension of measurement vector

    Returns:
        ndarray: Denoised signal
    """
    if len(primary_signal) != len(reference_signal):
        raise ValueError("Primary and reference signals must have the same length")

    # State vector: [signal, signal_velocity, noise, noise_velocity]
    def fx(x, dt):
        """State transition function: x_k = f(x_{k-1})"""
        # Simple kinematic model with velocity
        F = np.array(
            [
                [1, dt, 0, 0],  # signal = signal + velocity*dt
                [0, 1, 0, 0],  # signal_velocity remains constant
                [0, 0, 1, dt],  # noise = noise + noise_velocity*dt
                [0, 0, 0, 1],  # noise_velocity remains constant
            ]
        )
        return F @ x

    def hx(x):
        """Measurement function: z = h(x)"""
        # Measurement vector: [primary_signal, reference_signal]
        # primary_signal = signal + noise
        # reference_signal is assumed to be predominantly noise with some transformation
        H = np.array(
            [
                [1, 0, 1, 0],  # primary = signal + noise
                [0, 0, 1, 0],  # reference ~= noise (primarily)
            ]
        )
        return H @ x

    # Initialize UKF
    points = MerweScaledSigmaPoints(n=state_dim, alpha=alpha, beta=beta, kappa=kappa)
    ukf = UnscentedKalmanFilter(
        dim_x=state_dim, dim_z=measurement_dim, dt=dt, fx=fx, hx=hx, points=points
    )

    # Process and measurement noise covariances
    ukf.Q = np.eye(state_dim) * process_noise
    ukf.R = np.eye(measurement_dim) * measurement_noise

    # Initial state estimate
    ukf.x = np.array([primary_signal[0], 0, reference_signal[0], 0])

    # Initial state covariance
    ukf.P = np.eye(state_dim)

    # Initialize output array for the denoised signal
    denoised_signal = np.zeros_like(primary_signal)

    # Adaptive estimation of the relationship between reference and noise
    alpha_adapt = 0.95  # Adaptation rate
    noise_gain = 1.0  # Initial gain estimate

    # Process each sample
    for i in range(len(primary_signal)):
        # Current measurements
        z = np.array([primary_signal[i], reference_signal[i]])

        # Predict and update
        ukf.predict()
        ukf.update(z)

        # Extract the clean signal estimate (first element of state vector)
        denoised_signal[i] = ukf.x[0]

        # Adaptively update the measurement function based on observed relationship
        # between reference and estimated noise
        if i > 0:
            estimated_noise = primary_signal[i] - denoised_signal[i]
            # Update noise gain (relationship between reference and actual noise)
            if abs(reference_signal[i]) > 1e-10:  # Avoid division by zero
                current_gain = estimated_noise / reference_signal[i]
                noise_gain = alpha_adapt * noise_gain + (1 - alpha_adapt) * current_gain

    return denoised_signal, estimated_noise


def spectral_subtraction(signal, noise, alpha=1.0, beta=0.0):
    """
    Perform spectral subtraction of noise from signal.

    Parameters:
    ----------
    signal : array_like
        The input signal (potentially noisy signal in time domain)
    noise : array_like
        The noise signal to be subtracted (time domain)
    alpha : float, optional
        Subtraction factor (default: 1.0)
    beta : float, optional
        Spectral floor parameter to prevent musical noise (default: 0.0)

    Returns:
    -------
    array_like
        The enhanced signal after spectral subtraction (time domain)
    """
    # Ensure both signals have the same length
    min_length = min(len(signal), len(noise))
    signal = signal[:min_length]
    noise = noise[:min_length]

    # Convert to frequency domain
    signal_fft = np.fft.fft(signal)
    noise_fft = np.fft.fft(noise)

    # Compute magnitude and phase spectra
    signal_mag = np.abs(signal_fft)
    noise_mag = np.abs(noise_fft)
    signal_phase = np.angle(signal_fft)

    # Perform spectral subtraction
    subtracted_mag = signal_mag - alpha * noise_mag

    # Apply spectral floor to avoid negative values and reduce musical noise
    subtracted_mag = np.maximum(subtracted_mag, beta * signal_mag)

    # Recombine magnitude with original phase
    enhanced_fft = subtracted_mag * np.exp(1j * signal_phase)

    # Convert back to time domain
    enhanced_signal = np.real(np.fft.ifft(enhanced_fft))

    return enhanced_signal


def cepstral_subtraction(signal, noise_model, nfft):
    ccep_sig, _ = complex_cepstrum(signal, nfft)
    # rcep = real_cepstrum(signal, nfft)
    ccep_noise, _ = complex_cepstrum(noise_model, nfft)
    ccep = ccep_sig - ccep_noise
    icep = inverse_complex_cepstrum(ccep, nfft / 2)
    return icep


# def cepstral_subtraction(signal, noise_model, alpha=1.0, beta=0.01):
#     """
#     Perform cepstral subtraction of a noise model from a signal.

#     Parameters:
#     ----------
#     signal : array_like
#         The input noisy signal (time domain)
#     noise_model : array_like
#         The noise model to subtract (time domain)
#     alpha : float, optional
#         Subtraction factor (default: 1.0)
#     beta : float, optional
#         Floor parameter to prevent excessive suppression (default: 0.01)

#     Returns:
#     -------
#     array_like
#         The enhanced signal after cepstral subtraction (time domain)
#     """
#     # Ensure both signals have the same length
#     min_length = min(len(signal), len(noise_model))
#     signal = signal[:min_length]
#     noise_model = noise_model[:min_length]

#     # Step 1: Compute FFT of both signals
#     signal_fft = np.fft.fft(signal)
#     noise_fft = np.fft.fft(noise_model)

#     # Step 2: Calculate log magnitude spectra
#     signal_log_mag = np.log(np.abs(signal_fft) + 1e-10)
#     noise_log_mag = np.log(np.abs(noise_fft) + 1e-10)

#     # Step 3: Convert to cepstral domain
#     signal_cepstrum = np.real(np.fft.ifft(signal_log_mag))
#     noise_cepstrum = np.real(np.fft.ifft(noise_log_mag))

#     # Step 4: Perform cepstral subtraction
#     enhanced_cepstrum = signal_cepstrum - alpha * noise_cepstrum

#     # Step 5: Convert back to log spectral domain
#     enhanced_log_mag = np.real(np.fft.fft(enhanced_cepstrum))

#     # Step 6: Apply spectral floor to avoid excessive suppression
#     signal_mag = np.abs(signal_fft)
#     enhanced_mag = np.exp(enhanced_log_mag)
#     enhanced_mag = np.maximum(enhanced_mag, beta * signal_mag)

#     # Step 7: Reconstruct with original phase
#     signal_phase = np.angle(signal_fft)
#     enhanced_fft = enhanced_mag * np.exp(1j * signal_phase)

#     # Step 8: Convert back to time domain
#     enhanced_signal = np.real(np.fft.ifft(enhanced_fft))

#     return enhanced_signal
