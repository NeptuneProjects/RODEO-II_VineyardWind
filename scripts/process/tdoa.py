#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm

from vineyard.tdoa import tdoa


def main():
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
