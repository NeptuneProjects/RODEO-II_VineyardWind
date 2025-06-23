#!/usr/bin/env python3
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymap3d as pm
from scipy.signal import correlation_lags, correlate, hilbert
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream

from vineyard.config import get_path, SENSORS
from vineyard.tdoa import tdoa
from vineyard.signal import resample_datastreams


def calculate_bearing(reference_easting, reference_northing, target_easting, target_northing):
    """
    Calculate the bearing from a reference sensor to a target point.
    
    Bearing is measured clockwise from North (0 degrees), with East at 90 degrees,
    South at 180 degrees, and West at 270 degrees.
    
    Parameters:
    -----------
    reference_easting: float
        The easting coordinate of the reference sensor
    reference_northing: float
        The northing coordinate of the reference sensor
    target_easting: float
        The easting coordinate of the target point
    target_northing: float
        The northing coordinate of the target point
        
    Returns:
    --------
    float: The bearing in degrees, between 0 and 360
    """
    # Calculate the differences in eastings and northings
    delta_e = target_easting - reference_easting
    delta_n = target_northing - reference_northing
    
    # Calculate the bearing using arctan2
    # arctan2 returns angle in radians from the positive x-axis
    # We adjust to get bearing from North, clockwise
    bearing = np.degrees(np.arctan2(delta_e, delta_n))
    
    # Normalize to [0, 360) degrees
    bearing = (bearing + 360) % 360
    
    return bearing


def compute_northing_easting(lat, lon, lat0, lon0):
    easting, northing, _ = pm.geodetic2enu(lat, lon, 0, lat0, lon0, 0)
    return easting, northing


def condition_data(
    ds: DataStream,
    target_fs: float,
    filt_type: str = "bandpass",
    freq: float | Iterable[float] = [15.0, 35.0],
) -> DataStream:
    return resample_datastreams(ds.taper(max_percentage=0.05), target_fs)[0].filter(
        filt_type, freq
    )


def get_sensor_positions():
    df = pl.read_csv(get_path("equipment_config"))

    lat0, lon0 = (
        df.filter(pl.col("equipment") == "VLA2")
        .select(["latitude", "longitude"])
        .row(0)
    )

    df = df.with_columns(
        pl.struct("latitude", "longitude").map_elements(
            lambda cols: dict(
                zip(
                    ("easting", "northing"),
                    compute_northing_easting(
                        cols["latitude"], cols["longitude"], lat0, lon0
                    ),
                )
            ),
            return_dtype=pl.Struct(
                [
                    pl.Field("easting", pl.Float64),
                    pl.Field("northing", pl.Float64),
                ]
            ),
        ).alias("result")
    ).unnest("result")

    return df["easting"].to_list()[0:-1], df["northing"].to_list()[0:-1]


def get_tdoas(time_start, time_end, plot: bool = True):
    target_fs = 2000

    channels = [7, 3, 1]
    data = []
    for sensor, channel in zip(SENSORS, channels):
        data.append(
            condition_data(
                read_inventory(
                    get_path(f"{sensor}_inventory"),
                    time_start=time_start,
                    time_end=time_end,
                    channels=channel,
                ),
                target_fs,
                "bandpass",
                [15.0, 35.0],
            )
        )

    t = data[0].time_vector

    lags = correlation_lags(len(t), len(t), mode="full")
    tlag = lags / target_fs

    xcorr = {}
    for i in range(3):
        corr = correlate(
            data[2].data[0].squeeze(),
            data[i].data[0].squeeze(),
            mode="full",
            method="fft",
        )
        xcorr[f"3_{i+1}"] = {
            "data": corr / np.max(np.abs(corr)),
            "tdoa": lags[np.argmax(corr)] / target_fs,
        }

    if plot:
        fig, axs = plt.subplots(nrows=3, sharex=True)
        for ds, ax, sensor in zip(data, axs, SENSORS):
            ax.plot(t, ds.data[0])
            ax.set_title(sensor)
        plt.draw()

        plt.figure()
        for i in range(3):
            corr = xcorr[f"3_{i+1}"]["data"]
            tdoa = xcorr[f"3_{i+1}"]["tdoa"]
            plt.plot(tlag, hilbert(corr), label=f"3-{i+1}")
            plt.axvline(tdoa, color="k", linestyle="--")
        plt.legend()
        plt.draw()

    return {k: -v["tdoa"] for k, v in xcorr.items()}


def main():

    # time_start = np.datetime64("2023-12-01T22:25:54")
    # time_end = np.datetime64("2023-12-01T22:25:57")
    time_starts = [
        np.datetime64("2023-12-01T22:25:15"),
        np.datetime64("2023-12-01T22:25:54")
    ]
    time_ends = [
        np.datetime64("2023-12-01T22:25:18"),
        np.datetime64("2023-12-01T22:25:57")
    ]

    for i in range(2):
        time_start = time_starts[i]
        time_end = time_ends[i]

        denom = 1000.0

        tdoas = get_tdoas(time_start, time_end, plot=True)
        easting, northing = get_sensor_positions()
        
        easting = [e / denom for e in easting]
        northing = [n / denom for n in northing]

        xp = np.mean(easting)
        yp = np.mean(northing)
        speed = 1500.0 / denom

        x0, x1, x2 = easting
        y0, y1, y2 = northing
        t0, t1, t2 = [tdoas["3_1"], tdoas["3_2"], tdoas["3_3"]]

        x, y, F = tdoa(
            [x0, x1, x2],
            [y0, y1, y2],
            [t0, t1, t2],
            speed,
            xp,
            -yp,
        )
        r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
        bearing = calculate_bearing(x2, y2, x, y)
        
        plt.figure()
        plt.axis("equal")
        plt.grid()
        plt.scatter(x0, y0, color="tab:orange", label="3DVHA")
        plt.scatter(x1, y1, color="tab:blue", label="VLA1")
        plt.scatter(x2, y2, color="tab:green", label="VLA2")
        plt.scatter(x, y, color="tab:red", label="Whale")
        plt.plot([x2, x], [y2, y], color="r", linestyle="--", label=None)
        plt.legend()
        plt.xlabel("Easting (km)")
        plt.ylabel("Northing (km)")
        plt.title(f"Range: {r2:,.1f} km, Bearing: {bearing:.1f} deg")
        plt.draw()
    
    plt.show()

def demo():
    measurements = [
        [34.888, -103.826, 0.0],
        [34.931, -103.805, 0.6],
        [34.921, -103.781, 2.5],
    ]

    lat0, lon0, t0 = measurements[0]
    for i in range(len(measurements)):
        lat, lon, t = measurements[i]
        e, n, _ = pm.geodetic2enu(lat, lon, 0, lat0, lon0, 0)
        measurements[i] = [e, n, t]

    speed = 343.0  # m/s

    xp = np.mean([x for x, y, t in measurements])
    yp = np.mean([y for x, y, t in measurements])

    x0, y0, t0 = measurements[0]
    x1, y1, t1 = measurements[1]
    x2, y2, t2 = measurements[2]

    x, y, F = tdoa(
        [x0, x1, x2],
        [y0, y1, y2],
        [t0, t1, t2],
        speed,
        xp,
        yp,
    )
    lat, lon, _ = pm.enu2geodetic(x, y, 0, lat0, lon0, 0)
    print(f"Estimated position: {lat:.6f}, {lon:.6f} (lat, lon)")

    # Create reasonable x, y bounds for visualization
    max_x = max(x0, x1, x2, x)
    min_x = min(x0, x1, x2, x)
    range_x = max_x - min_x
    min_x -= range_x * 0.2
    max_x += range_x * 0.2

    max_y = max(y0, y1, y2, y)
    min_y = min(y0, y1, y2, y)
    range_y = max_y - min_y
    min_y -= range_y * 0.2
    max_y += range_y * 0.2

    # Create a grid of input coordinates
    xs = np.linspace(min_x, max_x, 100)
    ys = np.linspace(min_y, max_y, 100)
    xs, ys = np.meshgrid(xs, ys)

    # Evaluate the system across the grid
    A, B, C = F((xs, ys))

    # Plot the results
    plt.scatter(x0, y0, color="tab:red")
    plt.scatter(x1, y1, color="tab:green")
    plt.scatter(x2, y2, color="tab:blue")
    plt.scatter(x, y, color="k")
    plt.contour(xs, ys, A, [0], colors="y")
    plt.contour(xs, ys, B, [0], colors="m")
    plt.contour(xs, ys, C, [0], colors="c")
    plt.show()


if __name__ == "__main__":
    main()
