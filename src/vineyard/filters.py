from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np


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

    return denoised_signal
