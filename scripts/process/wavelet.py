import numpy as np
import matplotlib.pyplot as plt
import pywt


# def plot_signal_cwt(
#     signal_data,
#     sampling_rate=1.0,
#     wavelet="morl",
#     scales=None,
#     title="Continuous Wavelet Transform",
#     cmap="viridis",
#     figsize=(10, 8),
# ):
#     """
#     Plot a signal in Continuous Wavelet Transform (CWT) space.

#     Parameters:
#     -----------
#     signal_data : array_like
#         The input signal to analyze
#     sampling_rate : float, optional
#         Sampling rate of the signal in Hz (default: 1.0)
#     wavelet : str, optional
#         Wavelet to use (default: 'morl' for Morlet wavelet)
#         Options include: 'morl', 'cmor', 'gaus', 'mexh', etc.
#     scales : array_like, optional
#         Scales for the CWT (default: None, will generate appropriate scales)
#     title : str, optional
#         Title for the plot (default: 'Continuous Wavelet Transform')
#     cmap : str, optional
#         Colormap for the CWT plot (default: 'viridis')
#     figsize : tuple, optional
#         Figure size (width, height) in inches (default: (10, 8))

#     Returns:
#     --------
#     fig : matplotlib figure
#         The figure object containing the plots
#     """
#     # Generate appropriate scales if not provided
#     if scales is None:
#         # Create logarithmically spaced scales
#         width = min(len(signal_data), 1024)  # Use smaller of signal length or 1024
#         scales = np.arange(1, width // 2)

#     # Compute the CWT
#     coef, freqs = pywt.cwt(signal_data, scales, wavelet, 1.0 / sampling_rate)

#     # Create figure with subplots
#     fig, axs = plt.subplots(
#         2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 3]}
#     )

#     # Plot the original signal
#     time = np.arange(len(signal_data)) / sampling_rate
#     axs[0].plot(time, signal_data)
#     axs[0].set_title("Original Signal")
#     axs[0].set_xlabel("Time [s]" if sampling_rate != 1.0 else "Samples")
#     axs[0].set_ylabel("Amplitude")

#     # Plot the CWT
#     # Convert scales to frequencies for better interpretation
#     if wavelet == "morl":
#         # For Morlet, the relationship is approximately scale = central_freq/freq
#         central_freq = pywt.central_frequency(wavelet)
#         frequencies = central_freq * sampling_rate / scales
#     else:
#         # For other wavelets, use a general approximation
#         frequencies = sampling_rate / scales

#     # Create the CWT plot
#     im = axs[1].imshow(
#         np.abs(coef),
#         aspect="auto",
#         cmap=cmap,
#         extent=[time[0], time[-1], frequencies[-1], frequencies[0]],
#     )
#     axs[1].set_title(title)
#     axs[1].set_xlabel("Time [s]" if sampling_rate != 1.0 else "Samples")
#     axs[1].set_ylabel("Frequency [Hz]")

#     # Add colorbar
#     plt.colorbar(im, ax=axs[1], orientation="vertical", label="Magnitude")

#     plt.tight_layout()
#     return fig


# # Example usage
# if __name__ == "__main__":
#     # Create a sample signal with multiple components
#     sampling_rate = 1000  # Hz
#     duration = 1.0  # seconds
#     t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

#     # Signal with varying frequency components
#     f1, f2, f3 = 5, 50, 100
#     signal_data = (
#         np.sin(2 * np.pi * f1 * t)
#         + np.sin(2 * np.pi * f2 * t) * np.exp(-t / 0.4)
#         + np.sin(2 * np.pi * f3 * t) * (t > 0.6)
#     )

#     # Add some noise
#     np.random.seed(0)
#     signal_data += 0.1 * np.random.randn(len(t))

#     # Plot the signal in CWT space
#     fig = plot_signal_cwt(
#         signal_data,
#         sampling_rate=sampling_rate,
#         wavelet="morl",
#         scales=np.arange(1, 200),
#         title="CWT of a Signal with Multiple Components",
#     )
#     plt.show()

#     # You can also try different wavelets
#     fig = plot_signal_cwt(
#         signal_data,
#         sampling_rate=sampling_rate,
#         wavelet="cmor1.5-1.0",  # Complex Morlet wavelet
#         scales=np.arange(1, 200),
#         title="CWT with Complex Morlet Wavelet",
#         cmap="jet",
#     )
#     plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import pywt

# def wavelet_denoise(signal, wavelet='db4', level=None, threshold_mode='soft', 
#                     threshold_method='universal'):
#     """
#     Denoise a signal using wavelet thresholding.
    
#     Parameters:
#     -----------
#     signal : array_like
#         The noisy input signal
#     wavelet : str, optional
#         Wavelet to use (default: 'db4')
#     level : int, optional
#         Decomposition level (default: None, auto-calculated)
#     threshold_mode : str, optional
#         'soft' or 'hard' thresholding (default: 'soft')
#     threshold_method : str, optional
#         Method for threshold calculation: 'universal' or 'bayes' (default: 'universal')
    
#     Returns:
#     --------
#     array_like
#         Denoised signal
#     """
#     # Calculate appropriate decomposition level if not specified
#     if level is None:
#         level = min(pywt.dwt_max_level(len(signal), wavelet), 5)
    
#     # Decompose signal using wavelet transform
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
    
#     # Calculate threshold
#     if threshold_method == 'universal':
#         # Universal threshold (VisuShrink)
#         sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise level
#         threshold = sigma * np.sqrt(2 * np.log(len(signal)))
#     else:  # BayesShrink (simplified)
#         threshold = []
#         for i in range(1, len(coeffs)):
#             sigma = np.median(np.abs(coeffs[i])) / 0.6745
#             threshold.append(sigma * np.sqrt(2 * np.log(len(coeffs[i]))))
    
#     # Apply thresholding
#     if threshold_method == 'universal':
#         new_coeffs = [coeffs[0]]  # Keep approximation coefficients
#         for i in range(1, len(coeffs)):
#             if threshold_mode == 'soft':
#                 new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
#             else:
#                 new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
#     else:
#         new_coeffs = [coeffs[0]]  # Keep approximation coefficients
#         for i in range(1, len(coeffs)):
#             if threshold_mode == 'soft':
#                 new_coeffs.append(pywt.threshold(coeffs[i], threshold[i-1], mode='soft'))
#             else:
#                 new_coeffs.append(pywt.threshold(coeffs[i], threshold[i-1], mode='hard'))
    
#     # Reconstruct signal
#     denoised_signal = pywt.waverec(new_coeffs, wavelet)
    
#     # Adjust denoised signal length to match original signal
#     return denoised_signal[:len(signal)]

# # Example usage
# fs = 1000  # Sampling frequency (Hz)
# t = np.arange(0, 1, 1/fs)  # 1 second signal

# # Create a clean signal with multiple components
# clean_signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# # Add noise
# np.random.seed(42)
# noise = 0.5 * np.random.randn(len(t))
# noisy_signal = clean_signal + noise

# # Denoise the signal
# denoised_signal = wavelet_denoise(noisy_signal, wavelet='db8', threshold_mode='soft')

# # Plot results
# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plt.plot(t, clean_signal)
# plt.title('Original Clean Signal')
# plt.subplot(3, 1, 2)
# plt.plot(t, noisy_signal)
# plt.title('Noisy Signal')
# plt.subplot(3, 1, 3)
# plt.plot(t, denoised_signal)
# plt.title('Denoised Signal')
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# from scipy import signal

# def detect_signal_wavelet(data, template, wavelet='morl', scales=None, threshold=None):
#     """
#     Detect occurrences of a template signal in noisy data using wavelet matching.
    
#     Parameters:
#     -----------
#     data : array_like
#         The signal to analyze
#     template : array_like
#         The template signal to detect
#     wavelet : str, optional
#         Wavelet to use (default: 'morl')
#     scales : array_like, optional
#         Scales for CWT (default: None, auto-calculated)
#     threshold : float, optional
#         Detection threshold (default: None, auto-calculated)
    
#     Returns:
#     --------
#     tuple
#         (detection locations, detection scores, cwt coefficients)
#     """
#     # Generate appropriate scales if not provided
#     if scales is None:
#         width = min(len(data), 1024)
#         scales = np.arange(1, width // 4)
    
#     # Compute CWT for both data and template
#     coef_data, _ = pywt.cwt(data, scales, wavelet)
#     coef_template, _ = pywt.cwt(template, scales, wavelet)
    
#     # Normalize the template coefficients
#     normalized_template = []
#     for i in range(len(scales)):
#         template_row = coef_template[i]
#         norm = np.sqrt(np.sum(np.abs(template_row)**2))
#         if norm > 0:
#             normalized_template.append(template_row / norm)
#         else:
#             normalized_template.append(template_row)
    
#     # Calculate cross-correlation for each scale
#     detection_scores = np.zeros(len(data) - len(template) + 1)
#     for i in range(len(scales)):
#         # Use cross-correlation to find matches
#         corr = signal.correlate(np.abs(coef_data[i]), np.abs(normalized_template[i]), mode='valid')
#         detection_scores += corr
    
#     # Normalize scores
#     detection_scores = detection_scores / len(scales)
    
#     # Determine threshold if not provided
#     if threshold is None:
#         threshold = 0.7 * np.max(detection_scores)
    
#     # Find peaks above threshold
#     peaks, _ = signal.find_peaks(detection_scores, height=threshold, distance=len(template)//2)
    
#     return peaks, detection_scores, coef_data

# # Example usage
# fs = 1000  # Sampling frequency (Hz)
# t = np.arange(0, 4, 1/fs)  # 4 second signal

# # Create a template signal (a chirp)
# template_t = np.arange(0, 0.5, 1/fs)  # 0.5 second template
# template = signal.chirp(template_t, f0=10, f1=50, t1=0.5, method='linear')

# # Create test signal with multiple occurrences of the template
# data = np.zeros_like(t)
# # Add template at specific locations
# loc1, loc2, loc3 = int(0.5*fs), int(1.5*fs), int(2.8*fs)
# data[loc1:loc1+len(template)] += template
# data[loc2:loc2+len(template)] += template
# data[loc3:loc3+len(template)] += template

# # Add noise
# np.random.seed(42)
# data += 0.5 * np.random.randn(len(data))

# # Detect template in the noisy signal
# peaks, scores, coeffs = detect_signal_wavelet(data, template, scales=np.arange(1, 100))

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.subplot(3, 1, 1)
# plt.plot(template_t, template)
# plt.title('Template Signal')

# plt.subplot(3, 1, 2)
# plt.plot(t, data)
# plt.title('Noisy Signal with Template Occurrences')
# # Mark the true locations
# for loc in [loc1, loc2, loc3]:
#     plt.axvline(x=loc/fs, color='g', linestyle='--', alpha=0.5)

# plt.subplot(3, 1, 3)
# plt.plot(np.arange(len(scores))/fs, scores)
# plt.title('Detection Scores')
# plt.axhline(y=0.7*np.max(scores), color='r', linestyle='--', label='Threshold')
# # Mark detected peaks
# for peak in peaks:
#     plt.plot(peak/fs, scores[peak], 'ro')
# plt.legend()

# plt.tight_layout()
# plt.draw()

# # Also show the CWT scalogram
# plt.figure(figsize=(12, 6))
# plt.imshow(np.abs(coeffs), aspect='auto', cmap='viridis',
#           extent=[0, t[-1], 1, len(coeffs)])
# plt.title('CWT Scalogram of Signal')
# plt.ylabel('Scale')
# plt.xlabel('Time (s)')
# plt.colorbar(label='Magnitude')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from matplotlib.colors import LogNorm

def plot_scalogram(signal_data, sampling_rate=1.0, wavelet='morl', 
                  scales=None, log_scale=True, log_norm=False,
                  cmap='viridis', figsize=(10, 6), title=None,
                  show_signal=True, colorbar_label='Magnitude'):
    """
    Plot a scalogram (time-frequency representation) using Continuous Wavelet Transform.
    
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
    log_scale : bool, optional
        Use logarithmic scale for y-axis (default: True)
    log_norm : bool, optional
        Use logarithmic normalization for color intensity (default: False)
    cmap : str, optional
        Colormap for the scalogram (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6))
    title : str, optional
        Title for the plot (default: None, will use default title)
    show_signal : bool, optional
        Whether to show the original signal above the scalogram (default: True)
    colorbar_label : str, optional
        Label for the colorbar (default: 'Magnitude')
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the scalogram
    """
    # Time array
    time = np.arange(len(signal_data)) / sampling_rate
    
    # Generate appropriate scales if not provided
    if scales is None:
        if log_scale:
            # Logarithmically spaced scales
            scales = np.logspace(0, np.log10(min(len(signal_data) // 2, 512)), 128)
        else:
            # Linearly spaced scales
            scales = np.arange(1, min(len(signal_data) // 2, 512))
    
    # Compute the CWT
    if 'cmor' in wavelet or 'shan' in wavelet or 'fbsp' in wavelet:
        # For complex wavelets
        coef, freqs = pywt.cwt(signal_data, scales, wavelet, 1.0/sampling_rate)
        scalogram = np.abs(coef)**2  # Power scalogram
    else:
        # For real wavelets
        coef, freqs = pywt.cwt(signal_data, scales, wavelet, 1.0/sampling_rate)
        scalogram = np.abs(coef)  # Magnitude scalogram
    
    # Create figure
    if show_signal:
        fig, axs = plt.subplots(2, 1, figsize=figsize, 
                              gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.2})
        
        # Plot the original signal
        axs[0].plot(time, signal_data)
        axs[0].set_title('Original Signal')
        axs[0].set_xlabel('')  # No x-label on top plot
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, alpha=0.3)
        
        # Main plot will be in axs[1]
        ax = axs[1]
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Convert frequencies for better interpretation
    if wavelet == 'morl':
        # For Morlet, central_freq ≈ 0.8125 for standard Morlet
        central_freq = pywt.central_frequency(wavelet)
        frequencies = central_freq * sampling_rate / scales
    else:
        # General approximation for other wavelets
        frequencies = sampling_rate / (2 * scales)
    
    # Create the scalogram
    if log_norm:
        norm = LogNorm(vmin=scalogram.min() + 1e-10, vmax=scalogram.max())
        im = ax.pcolormesh(time, frequencies, scalogram, cmap=cmap, norm=norm, shading='auto')
    else:
        im = ax.pcolormesh(time, frequencies, scalogram, cmap=cmap, shading='auto')
    
    # Set y-axis scale
    if log_scale:
        ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('Time [s]' if sampling_rate != 1.0 else 'Time [samples]')
    ax.set_ylabel('Frequency [Hz]')
    
    if title is None:
        title = f'Scalogram (CWT with {wavelet} wavelet)'
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label=colorbar_label)
    
    # Invert y-axis so low frequencies are at the bottom
    ax.invert_yaxis()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Create a sample signal with multiple components
    fs = 1000  # Hz
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    
    # Create a chirp signal (frequency increases from 5Hz to 250Hz)
    chirp_signal = signal.chirp(t, f0=5, f1=250, t1=2, method='logarithmic')
    
    # Add some transients
    transient1 = 2 * np.exp(-(t-0.3)**2/(2*0.01**2)) * np.sin(2*np.pi*50*t)
    transient2 = 3 * np.exp(-(t-1.3)**2/(2*0.01**2)) * np.sin(2*np.pi*150*t)
    
    # Combine signals
    combined_signal = chirp_signal + transient1 + transient2
    
    # Add some noise
    np.random.seed(42)
    noisy_signal = combined_signal + 0.2 * np.random.randn(len(t))
    
    # Plot standard scalogram with linear frequency scale
    fig1 = plot_scalogram(noisy_signal, sampling_rate=fs, wavelet='morl', 
                        log_scale=False, title='Linear Frequency Scale Scalogram')
    
    # Plot scalogram with logarithmic frequency scale (better for visualizing wide frequency range)
    fig2 = plot_scalogram(noisy_signal, sampling_rate=fs, wavelet='morl', 
                        log_scale=True, title='Logarithmic Frequency Scale Scalogram')
    
    # Plot scalogram with logarithmic intensity normalization
    fig3 = plot_scalogram(noisy_signal, sampling_rate=fs, wavelet='morl', 
                        log_scale=True, log_norm=True, 
                        title='Scalogram with Log Scale and Log Normalization')
    
    # Plot scalogram with a complex wavelet (better for analyzing phase)
    fig4 = plot_scalogram(noisy_signal, sampling_rate=fs, wavelet='cmor1.0-1.0', 
                        log_scale=True, title='Scalogram with Complex Morlet Wavelet')
    
    plt.show()