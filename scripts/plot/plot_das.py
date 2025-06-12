#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import dascore as dc
import dotenv
import matplotlib.pyplot as plt
import numpy as np

from vineyard.readers import read_tdms

dotenv.load_dotenv()


def main():
    datadir = Path(os.getenv("DASDATADIR"))
    logging.info(f"Reading data from: {datadir}")

    patch, properties = read_tdms(
        datadir, time=("2023-12-01T21:06:30", "2023-12-01T21:07:00"), channel=(0, 4200)
    )
    print(patch.coords)

    patch_filt = patch.pass_filter(time=(20, None))
    # patch_filt.viz.waterfall(show=True, scale=10, scale_type="absolute")
    # patch_filt = patch_filt.savgol_filter(time=1.0, polyorder=3)



    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax = patch_filt.viz.wiggle(ax=ax, alpha=0.1, scale=0.1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance (m)")

    plt.draw()
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

    plt.show()



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
    return
    logging.info(patch)

    first_ch = 2800
    last_ch = 4000

    # fig, ax = plt.subplots(figsize=(18, 12))
    # ax = patch.viz.waterfall(ax=ax, scale=0.2)
    # plt.draw()

    dist_len = patch.coord_shapes["distance"][0]
    channel_number = np.arange(dist_len)


    fig, ax = plt.subplots(figsize=(18, 12))
    out.viz.waterfall(ax=ax)
    plt.draw()

    plt.show()
    return

    # filtered_patch = patch.detrend("time").normalize("time").pass_filter(time=(10 * Hz, 60 * Hz))
    # filtered_patch.viz.waterfall(show=True)

    return

    fig = plot_data(blast, figsize=(18, 12))
    plt.draw()

    fig = plot_filtered_data(blast, "highpass", 20.0, figsize=(18, 12))
    plt.draw()

    fig = plot_filtered_data(blast, "lowpass", 20.0, figsize=(18, 12))
    plt.draw()

    fig = plot_filtered_data(blast, "bandpass", [40.0, 60.0], figsize=(18, 12))
    plt.draw()

    fig = plot_filtered_data(blast, "afk", afk_exponent=0.8, figsize=(18, 12))
    plt.draw()

    plt.show()

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
