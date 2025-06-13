#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm
from scipy.optimize import leastsq


def functions(x0, y0, x1, y1, x2, y2, d01, d02, d12):
    """Given observers at (x0, y0), (x1, y1), (x2, y2) and TDOA between observers d01, d02, d12, this closure
    returns a function that evaluates the system of three hyperbolae for given event x, y.
    """

    def fn(args):
        x, y = args
        a = (
            np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
            - np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
            - d01
        )
        b = (
            np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0))
            - np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
            - d02
        )
        c = (
            np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0))
            - np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
            - d12
        )
        return [a, b, c]

    return fn


def jacobian(x0, y0, x1, y1, x2, y2, d01, d02, d12):
    def fn(args):
        x, y = args
        adx = (x - x1) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0)) - (
            x - x0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        bdx = (x - x2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            x - x0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        cdx = (x - x2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            x - x1
        ) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
        ady = (y - y1) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0)) - (
            y - y0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        bdy = (y - y2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            y - y0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        cdy = (y - y2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            y - y1
        ) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))

        return [[adx, ady], [bdx, bdy], [cdx, cdy]]

    return fn


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

    F = functions(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )
    J = jacobian(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )

    pos, _ = leastsq(F, x0=[xp, yp], Dfun=J)
    x = pos[0]
    y = pos[1]
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
