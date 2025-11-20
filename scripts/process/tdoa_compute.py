#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tritonoa.data.reader import read_hdf5
from scipy.signal import find_peaks, find_peaks_cwt, hilbert

from vineyard.align_detections import (
    correlate_all_references,
    merge_correlations,
    correlations_to_dataframe,
)
from vineyard.config import get_path


def compute_time_gates(
    distance_lut: Path, ref_sound_speed: float = 1500.0
) -> dict[tuple[str, str], np.ndarray]:
    d_3dvha_vla1, d_3dvha_vla2, d_vla1_vla2 = load_distances(distance_lut)
    t_3dvha_vla1 = d_3dvha_vla1 / ref_sound_speed
    t_3dvha_vla2 = d_3dvha_vla2 / ref_sound_speed
    t_vla1_vla2 = d_vla1_vla2 / ref_sound_speed
    return {
        ("3dvha", "vla1"): t_3dvha_vla1,
        ("3dvha", "vla2"): t_3dvha_vla2,
        ("vla1", "vla2"): t_vla1_vla2,
    }


def get_peaks(
    streams: dict[str, any], heights: list[float]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    times = {}
    for height, (name, ds) in zip(heights, streams.items()):
        data = np.abs(hilbert(streams[name].copy().data[0]))
        peaks, _ = find_peaks(data, distance=3 * ds.stats.sampling_rate, height=height)
        times[name] = ds.time_vector[peaks]
    return times


def load_datastreams(data_path: Path, starttime: np.datetime64):
    streams = {}
    for name in ["3dvha", "vla1", "vla2"]:
        streams[name], channel = (
            read_hdf5(data_path / f"{name}_pc.h5").trim(starttime=starttime),
            0,
        )
    return streams, channel


def load_distances(distance_lut: Path) -> tuple[float, float, float]:
    distance_lut = pl.read_csv(get_path("distance_lut"))
    d_3dvha_vla1 = (
        distance_lut.filter(pl.col("from_equipment") == "3DVHA")
        .filter(pl.col("to_equipment") == "VLA1")["distance_meters"]
        .item()
    )
    d_3dvha_vla2 = (
        distance_lut.filter(pl.col("from_equipment") == "3DVHA")
        .filter(pl.col("to_equipment") == "VLA2")["distance_meters"]
        .item()
    )
    d_vla1_vla2 = (
        distance_lut.filter(pl.col("from_equipment") == "VLA1")
        .filter(pl.col("to_equipment") == "VLA2")["distance_meters"]
        .item()
    )
    return d_3dvha_vla1, d_3dvha_vla2, d_vla1_vla2


def main():
    streams, _ = load_datastreams(
        get_path("pulse_comp_data"), starttime=np.datetime64("2023-12-01T22:14:00")
    )

    heights = [0.15, 0.15, 0.15]
    times = get_peaks(streams, heights)
    time_gates = compute_time_gates(get_path("distance_lut"))

    correlations = correlate_all_references(times, time_gates)
    merged_results = merge_correlations(correlations)
    df_tdoa = correlations_to_dataframe(merged_results)
    save_dir = get_path("tdoa_data")
    save_dir.mkdir(parents=True, exist_ok=True)
    df_tdoa.write_csv(save_dir / "tdoa.csv")


if __name__ == "__main__":
    main()
