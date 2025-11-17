import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
from scipy.interpolate import CubicSpline

from vineyard.config import get_path

def compute_whale_range(
    angular_speed: ArrayLike | float, tangential_speed: float = 35.0
) -> NDArray | float:
    """Compute the whale range given angular speed and max tangential speed.

    Args:
        angular_speed: Angular speed in degrees per second.
        max_tangential_speed: Tangential speed in km/h. Default is 40.0 km/h.

    Returns:
        Whale range in kilometers.
    """
    return (tangential_speed / 3600) / np.deg2rad(angular_speed)

if __name__ == "__main__":
    df = pl.read_csv(get_path("tdoa_data") / "tdoa_with_locations.csv").cast({"timestamp": pl.Datetime})

    times = df["timestamp"].to_numpy().astype(int) / 1e6
    bearings = df["vla1_brg"].to_numpy()
    dwdt = df["dwdt"].to_numpy()

    interp = CubicSpline(times, bearings)
    t_i = np.linspace(times[0], times[-1], 100)
    b_i = interp(t_i)
    plt.plot(t_i, b_i)
    plt.show()


    angular_speeds = df["dwdt"].to_numpy()
    # angular_speed = 0.025  # deg/s
    whale_range = compute_whale_range(angular_speeds)
    
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(df["timestamp"], df["vla1_brg"])

    plt.subplot(3, 1, 2)
    plt.plot(df["timestamp"], angular_speeds)

    plt.subplot(3, 1, 3)
    plt.plot(df["timestamp"], whale_range)
    plt.ylim(0, 100)
    plt.show()
    # print(
    #     f"Whale range for angular speed {angular_speed:.6f} deg/s: {whale_range:,.2f} km"
    # )
