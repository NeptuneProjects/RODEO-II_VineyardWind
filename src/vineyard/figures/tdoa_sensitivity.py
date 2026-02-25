"""TDOA localization cost-function figure.

Single-panel figure showing the MSE cost function with hyperbolic lines of
position for a query source.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from vineyard.readers import read_sensor_positions


def _cost_grid(
    e0: float,
    n0: float,
    e1: float,
    n1: float,
    e2: float,
    n2: float,
    qe: float,
    qn: float,
    grid_e: np.ndarray,
    grid_n: np.ndarray,
) -> np.ndarray:
    """Compute MSE of TDOA hyperbolic residuals over a 2D grid (km²).

    Returns:
        2D array of MSE values in km².
    """
    r0 = np.sqrt((grid_e - e0) ** 2 + (grid_n - n0) ** 2)
    r1 = np.sqrt((grid_e - e1) ** 2 + (grid_n - n1) ** 2)
    r2 = np.sqrt((grid_e - e2) ** 2 + (grid_n - n2) ** 2)

    qr0 = np.sqrt((qe - e0) ** 2 + (qn - n0) ** 2)
    qr1 = np.sqrt((qe - e1) ** 2 + (qn - n1) ** 2)
    qr2 = np.sqrt((qe - e2) ** 2 + (qn - n2) ** 2)

    a = (r1 - r0) - (qr1 - qr0)
    b = (r2 - r0) - (qr2 - qr0)
    c = (r2 - r1) - (qr2 - qr1)

    return (a**2 + b**2 + c**2) / 3.0


def plot_tdoa_sensitivity(
    sensor_data: Path,
    grid_extent_km: tuple[float, float, float, float] | None = None,
    grid_resolution: int = 300,
    query_point_km: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (4.5, 4.5),
) -> Figure:
    """Plot the TDOA cost-function landscape for a query source.

    Shows the log₁₀ MSE of the TDOA hyperbolic residuals for a query source,
    with hyperbolic lines of position (LOPs) overlaid.  Each LOP is the locus
    of equal range difference between one sensor pair; all three LOPs intersect
    at the true source.  A narrow, well-isolated intersection indicates good
    localization.

    Args:
        sensor_data: Path to sensor positions CSV (see read_sensor_positions).
        grid_extent_km: (e_min, e_max, n_min, n_max) bounds of the search grid
            in km, in the same ENU frame as the sensor positions.  Defaults to
            ±50 km around the sensor centroid.
        grid_resolution: Number of grid points per axis.
        query_point_km: (easting_km, northing_km) of the query source in the
            same ENU frame as the sensor positions (km from the reference
            point).  Defaults to 20 km due north of the sensor centroid.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib Figure.
    """
    # ---- Sensor positions (meters → km) ----
    sensor_e_m, sensor_n_m, _, _ = read_sensor_positions(sensor_data)
    sensor_e = [e / 1000.0 for e in sensor_e_m]
    sensor_n = [n / 1000.0 for n in sensor_n_m]
    e0, e1, e2 = sensor_e
    n0, n1, n2 = sensor_n

    # ---- Search grid ----
    cx = float(np.mean(sensor_e))
    cy = float(np.mean(sensor_n))
    if grid_extent_km is None:
        e_min, e_max, n_min, n_max = cx - 50.0, cx + 50.0, cy - 50.0, cy + 50.0
    else:
        e_min, e_max, n_min, n_max = grid_extent_km
    e_vec = np.linspace(e_min, e_max, grid_resolution)
    n_vec = np.linspace(n_min, n_max, grid_resolution)
    grid_e, grid_n = np.meshgrid(e_vec, n_vec)

    # ---- Query source ----
    if query_point_km is None:
        qe, qn = cx, cy + 20.0  # 20 km due north — worst case for E–W arrays
    else:
        qe, qn = float(query_point_km[0]), float(query_point_km[1])

    # ---- Cost field ----
    cost = _cost_grid(e0, n0, e1, n1, e2, n2, qe, qn, grid_e, grid_n)

    # ---- Figure ----
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    log_cost = np.log10(cost + 1e-10)
    im = ax.pcolormesh(
        grid_e,
        grid_n,
        log_cost,
        cmap="bone",
        vmin=-4,
        vmax=2,
        shading="auto",
        rasterized=True,
    )
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(r"$\mathdefault{log}_{10}(\mathdefault{MSE})$ (km²)")

    # Range differences over the grid and at the query source
    r0_g = np.sqrt((grid_e - e0) ** 2 + (grid_n - n0) ** 2)
    r1_g = np.sqrt((grid_e - e1) ** 2 + (grid_n - n1) ** 2)
    r2_g = np.sqrt((grid_e - e2) ** 2 + (grid_n - n2) ** 2)
    qr0 = float(np.sqrt((qe - e0) ** 2 + (qn - n0) ** 2))
    qr1 = float(np.sqrt((qe - e1) ** 2 + (qn - n1) ** 2))
    qr2 = float(np.sqrt((qe - e2) ** 2 + (qn - n2) ** 2))

    _LOP_COLORS = ["c", "m", "y"]
    _LOP_LABELS = [
        "3DVHA-VLA1 line of constant TDOA",
        "3DVHA-VLA2 line of constant TDOA",
        "VLA1-VLA2 line of constant TDOA",
    ]
    for diff_grid, true_diff, color in zip(
        [r1_g - r0_g, r2_g - r0_g, r2_g - r1_g],
        [qr1 - qr0, qr2 - qr0, qr2 - qr1],
        _LOP_COLORS,
    ):
        ax.contour(
            grid_e,
            grid_n,
            diff_grid,
            levels=[true_diff],
            colors=[color],
            linewidths=1.5,
        )

    for se, sn in zip(sensor_e, sensor_n):
        ax.plot(se, sn, "kv", ms=7, zorder=5)
    ax.plot(qe, qn, "r*", ms=12, zorder=6, markeredgecolor="k")

    legend_handles = [
        Line2D([0], [0], marker="v", color="k", label="Sensors", ls="none", ms=5),
        Line2D(
            [0],
            [0],
            marker="*",
            color="red",
            label="Source",
            ls="none",
            ms=7,
            markeredgecolor="k",
        ),
        *[
            Line2D([0], [0], color=c, label=l, linewidth=1.5)
            for c, l in zip(_LOP_COLORS, _LOP_LABELS)
        ],
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="lower right", framealpha=1.0)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig
