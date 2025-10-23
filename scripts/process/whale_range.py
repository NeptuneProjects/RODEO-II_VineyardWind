import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_whale_range(
    angular_speed: ArrayLike | float, tangential_speed: float = 45.0
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
    angular_speed = 10 / 600  # deg/s
    whale_range = compute_whale_range(angular_speed)
    print(
        f"Whale range for angular speed {angular_speed:.6f} deg/s: {whale_range:,.2f} km"
    )
