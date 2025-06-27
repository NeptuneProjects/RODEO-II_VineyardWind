#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import dascore as dc
import dotenv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, hilbert, spectrogram
from tqdm import tqdm
from tritonoa.data.stream import DataStream, DataStreamStats

import vwdas.paths as paths
from vineyard.plotting import SAVEFIG_KWARGS
from vineyard.readers import read_tdms

dotenv.load_dotenv()

legs = {
    "leg1": [1600, 2670],
    "leg2": [2670, 3400],
    "leg3": [3400, 4200],
}


def template_match(data, template, mode="valid", analytic=False):
    """
    Correlate each channel of a 2D data block with a 1D template in a vectorized manner.
7
    Parameters:
    -----------
    data : numpy.ndarray
        2D array with dimensions (channel × time)
    template : numpy.ndarray
        1D array representing the template (shorter than data in time dimension)
    mode : str, optional
        The correlation mode. Options are:
        - 'valid': returns output of length max(M, N) - min(M, N) + 1
        - 'same': returns output of length max(M, N)
        - 'full': returns output of length M + N - 1
        Default is 'valid'.

    Returns:
    --------
    numpy.ndarray
        2D array of correlation results with dimensions (channel × time_corr),
        where time_corr depends on the mode parameter.
    """
    # Ensure data and template are numpy arrays
    data = np.asarray(data)
    template = np.asarray(template)

    if analytic:
        data = hilbert(data)
        template = hilbert(template)


    # Use the correlate function with axis=1 to correlate along the time dimension
    # for all channels simultaneously
    return correlate(data, template[np.newaxis, :], mode=mode, method="direct")


def normalize_channelwise(data):
    """
    Normalize each channel of a 2D data block to have a maximum absolute value of 1.

    Parameters:
    -----------
    data : numpy.ndarray
        2D array with dimensions (channel x time)

    Returns:
    --------
    numpy.ndarray
        Normalized 2D array with the same shape as input data.
    """

    # Normalize each channel to have a maximum absolute value of 1
    max_values = np.max(np.abs(data), axis=1, keepdims=True)
    return data / max_values



def plot_cc_with_channels(data, label):
        fig = plt.figure()
        plt.imshow(np.abs(data), aspect="auto", cmap="jet", interpolation="none")
        plt.colorbar(label="Correlation Coefficient")
        plt.xlabel("Time")
        plt.ylabel("Channel")
        plt.title(f"cc_{label}")

        fig.savefig(
            paths.reports.figures / f"temp_match_{label}.png", **SAVEFIG_KWARGS
        )
        plt.close()


def plot_trace(data, label: str, fs=250.0):
    time_vec = np.arange(0, len(data)) / fs

    fig = plt.figure()
    plt.plot(time_vec, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(label)
    plt.grid()
    fig.savefig(
        paths.reports.figures / f"trace_{label}.png", **SAVEFIG_KWARGS
    )
    plt.close()


def plot_spectrogram(data, label: str, fs=250.0):
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=64, noverlap=32, nfft=1024)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="jet")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(label)
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("dB/Hz")

    fig.savefig(
        paths.reports.figures / f"spectrogram_{label}.png", **SAVEFIG_KWARGS
    )
    plt.close()


def main():
    # template = np.load("data/acoustic/pile_driving_pattern.npy")
    leg_name = "leg2"
    leg_start = legs[leg_name][0]
    leg_end = legs[leg_name][1]

    datadir = Path(os.getenv("DASDATADIR"))
    logging.info(f"Reading DAS data from: {datadir}")

    patch, properties = read_tdms(
        datadir,
        time=("2023-12-01T21:06:00", "2023-12-01T21:09:00"),
        channel=legs[leg_name],
    )


    patch = patch.pass_filter(time=(2.0, None))
    patch = patch.set_dims(distance="channel")
    # patch.viz.wiggle(show=True, alpha=0.1, scale=0.1)
    # plt.plot(trace)
    # plt.show()

    
    for channel in tqdm(range(0, leg_end - leg_start)):
        trace = patch.data.T[channel]
        f, t, Sxx = spectrogram(trace, fs=properties.sampling_rate_hz, nperseg=64, noverlap=48, nfft=1024)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="magma")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Channel {channel + leg_start}")
        cbar = fig.colorbar(im, ax=ax)
        fig.savefig(paths.reports.figures / "fin_whale_search" / f"specgram_{leg_name}_{channel + leg_start}.png", **SAVEFIG_KWARGS)
        plt.close()

    

    # patch.viz.waterfall(show=True)
    # ufilt = None
    # for lfilt in [1.0, 20.0, 40.0, 60.0]:
    #     patch_filt = patch.pass_filter(time=(lfilt, ufilt))

    #     channels = [0, 20, 40, 60, 80, 100]
    #     for chan in channels:
    #         data = patch_filt.data.T[chan]
    #         cc = patch.correlate(distance=chan, samples=True).squeeze().data.T
    #         cc = normalize_channelwise(cc)

    #         label = f"ch_{chan}__filt_{lfilt}-{ufilt}"

    #         # plot_cc_with_channels(cc, label)
    #         # plot_trace(data, label)

    #         if lfilt == 1.0:
    #             plot_spectrogram(data, label)

    # plot_cc_with_channels(patch_filt, f"Filter_{lfilt}-{ufilt}")


    # cc = template_match(data.T, template)
    # cc = normalize_channelwise(cc)

    # if lfilt is None:
    #     lfilt = 0
    # if ufilt is None:
    #     ufilt = "inf"

    # fig = plt.figure()
    # plt.imshow(np.abs(cc), aspect="auto", cmap="jet", interpolation="none")
    # plt.colorbar(label="Correlation Coefficient")
    # plt.xlabel("Time")
    # plt.ylabel("Channel")
    # plt.title(f"Template Matching | Filter: ({lfilt}, {ufilt}) Hz")
    # # fig.savefig(paths.reports.figures / f"temp_match_{lfilt}-{ufilt}.png", **savefig_kwargs)
    # plt.draw()

    # cc = patch_filt.correlate(distance=100, samples=True).squeeze().data.T
    # cc = normalize_channelwise(cc)
    # # cc.viz.waterfall(show=True, cmap="jet", scale=0.1, scale_type="absolute")

    # # cc = template_match(data.T, template, analytic=True)
    

    # fig = plt.figure()
    # plt.imshow(np.abs(cc), aspect="auto", cmap="jet", interpolation="none")
    # plt.colorbar(label="Correlation Coefficient")
    # plt.xlabel("Time")
    # plt.ylabel("Channel")
    # plt.title(f"Template Matching - Analytic Signal | Filter: ({lfilt}, {ufilt}) Hz")
    # # # fig.savefig(paths.reports.figures / f"temp_match_hilb_{lfilt}-{ufilt}.png", **savefig_kwargs)
    # # plt.draw()

    # plt.show()
    # patch.correlate


    # patch_filt.viz.waterfall(show=True, scale=1.0, scale_type="absolute")
    # patch_filt = patch_filt.savgol_filter(time=1.0, polyorder=3)

    # fig = plt.figure(figsize=(12, 6))
    # ax = plt.gca()
    # ax = patch_filt.viz.wiggle(ax=ax, alpha=0.15, scale=0.1)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Distance (m)")
    # plt.show()

    # cc_patch = patch_filt.correlate(distance=50, samples=True).squeeze()
    # cc_patch.viz.wiggle(show=True)

    # return
    # patch_filt.viz.spectrogram(show=True)

    # corr = patch_filt.correlate(distance=500, samples=True)
    # fig, ax = plt.subplots(figsize=(20, 12))
    # corr.squeeze().viz.waterfall(ax=ax, scale=1000, scale_type="absolute")
    # plt.draw()

    # fk_patch = patch_filt.dft(patch_filt.dims)
    # fk_patch.abs().viz.waterfall(show=True)

    # filt_sound = np.array([1_200, 1_400, 2_000, 2_200])
    # patch_sfilt = patch.slope_filter(filt=filt_sound)
    # patch_sfilt.viz.waterfall(show=True, scale=10, scale_type="absolute")
    # plt.figure(figsize=(18, 12))
    # ax = plt.gca()
    # ax = patch_sfilt.viz.wiggle(ax=ax, alpha=0.1, scale=0.1)
    # plt.draw()

    # plt.show()

    # def plot_filtered_data(
    #     # blast: Blast,
    #     filter_type: str,
    #     freq: float | tuple[float, float] | None = None,
    #     afk_exponent: float | None = None,
    #     figsize: tuple[float, float] = (18, 12),
    #     normalize_traces: bool = True,
    # ) -> None:
    #     blast_filt = blast.copy()
    #     if filter_type == "highpass":
    #         logging.info(f"Highpass filter with cutoff frequency: {freq} Hz")
    #         blast_filt.highpass(cutoff_freq=freq)
    #     if filter_type == "lowpass":
    #         logging.info(f"Lowpass filter with cutoff frequency: {freq} Hz")
    #         blast_filt.lowpass(cutoff_freq=freq)
    #     if filter_type == "bandpass":
    #         logging.info(f"Bandpass filter with cutoff frequencies: {freq} Hz")
    #         blast_filt.bandpass(min_freq=freq[0], max_freq=freq[1])
    #     if filter_type == "afk":
    #         logging.info(f"AFK filter with exponent: {afk_exponent}")
    #         blast_filt.afk_filter(exponent=afk_exponent)

    #     fig, ax = plt.subplots(figsize=figsize)
    #     blast_filt.plot(axes=ax, normalize_traces=normalize_traces)
    #     ax.set_title(f"{filter_type.capitalize()} Filtered Data")
    #     return fig

    # def plot_data(blast: Blast, figsize: tuple[float, float] = (18, 12)) -> None:
    #     fig, ax = plt.subplots(figsize=figsize)
    #     im = blast.plot(axes=ax)
    #     im.set_clim(-0.5, 0.5)
    #     ax.set_title("Raw Data")
    #     return fig

    # attrs = dc.scan(datadir)
    # spool = dc.spool(datadir)

    # patch = spool[1]
    # fs = get_dim_sampling_rate(patch, "time")
    # patch.update_attrs(sampling_rate=fs)

    # data = patch.data
    # [print(a) for a in patch.attrs]
    # strain_rate_data = convert_to_strain_rate(
    #     data,
    #     tsdmFileProperties.scale_factor,
    #     tsdmFileProperties.calibration_factor,
    #     patch.attrs["sampling_rate"],
    # )

    # scaled_data = convert_to_strain_rate()
    # return
    # logging.info(patch)

    # first_ch = 2800
    # last_ch = 4000

    # # fig, ax = plt.subplots(figsize=(18, 12))
    # # ax = patch.viz.waterfall(ax=ax, scale=0.2)
    # # plt.draw()

    # dist_len = patch.coord_shapes["distance"][0]
    # channel_number = np.arange(dist_len)

    # fig, ax = plt.subplots(figsize=(18, 12))
    # out.viz.waterfall(ax=ax)
    # plt.draw()

    # plt.show()
    # return

    # # filtered_patch = patch.detrend("time").normalize("time").pass_filter(time=(10 * Hz, 60 * Hz))
    # # filtered_patch.viz.waterfall(show=True)

    # return

    # fig = plot_data(blast, figsize=(18, 12))
    # plt.draw()

    # fig = plot_filtered_data(blast, "highpass", 20.0, figsize=(18, 12))
    # plt.draw()

    # fig = plot_filtered_data(blast, "lowpass", 20.0, figsize=(18, 12))
    # plt.draw()

    # fig = plot_filtered_data(blast, "bandpass", [40.0, 60.0], figsize=(18, 12))
    # plt.draw()

    # fig = plot_filtered_data(blast, "afk", afk_exponent=0.8, figsize=(18, 12))
    # plt.draw()

    # plt.show()

    # blast_hp = blast.copy()
    # blast_hp.highpass(cutoff_freq=20.0)

    # blast_bp = blast.copy()
    # blast_bp.bandpass(min_freq=10.0, max_freq=30.0)

    # blast_afk = blast_bp.copy()
    # blast_afk.afk_filter(
    #     exponent=0.8,
    # )

    # fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ax = axes[0]
    # blast.plot(ax)
    # ax.set_title("Original Data")

    # ax = axes[1]
    # blast_bp.plot(ax)
    # ax.set_title("Bandpass Filtered Data")

    # ax = axes[2]
    # blast_afk.plot(ax)
    # ax.set_title("AFK Filtered Data")

    # blast = blast.trim_channels(begin=800, end=2112)

    # logging.info(properties)

    # fig, ax = plt.subplots()
    # blast.plot(ax)
    # ax.set_title("Raw Data Visualization")
    # plt.draw()

    # fig, ax = plt.subplots()
    # blast_hp.plot(ax)
    # ax.set_title("Highpass Filtered Data")
    # plt.draw()

    # # fig, ax = plt.subplots()
    # # blast_afk.plot(ax)
    # # ax.set_title("AFK Filtered Data")

    # # ch = 3050

    # plt.figure()
    # i = 0
    # for i, tr in enumerate(blast_hp.data[3040:3050]):
    #     tr /= np.max(np.abs(tr))
    #     plt.plot(tr + i)

    # # plt.plot(blast_hp.data[3040])
    # plt.show()
    # return

    # for ch in range(3090, 3200, 10):
    #     f, t, Sxx = signal.spectrogram(
    #         blast_hp.data[ch],
    #         fs=blast.sampling_rate,
    #         nperseg=256,
    #         noverlap=128,
    #         nfft=1024,
    #     )

    #     fig, ax = plt.subplots()
    #     ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    #     ax.set_xlabel("Time [s]")
    #     ax.set_ylabel("Frequency [Hz]")
    #     ax.set_title(f"Spectrogram of Channel {ch}")

    #     plt.draw()
    # plt.show()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
