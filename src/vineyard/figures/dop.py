"""Geometric dilution of precision (GDOP) figure."""

from pathlib import Path

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from vineyard.readers import read_sensor_positions


def _gdop_grid(
    e0: float,
    n0: float,
    e1: float,
    n1: float,
    e2: float,
    n2: float,
    grid_e: np.ndarray,
    grid_n: np.ndarray,
) -> np.ndarray:
    """Compute GDOP over a 2D grid given three sensor positions (all in km).

    Returns:
        2D array of GDOP values (NaN where geometry is singular or at sensor
        locations).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r0 = np.sqrt((grid_e - e0) ** 2 + (grid_n - n0) ** 2)
        r1 = np.sqrt((grid_e - e1) ** 2 + (grid_n - n1) ** 2)
        r2 = np.sqrt((grid_e - e2) ** 2 + (grid_n - n2) ** 2)

        # Jacobian rows: partial derivatives of each hyperbolic range-difference
        # equation with respect to source (x, y).  Matches tdoa.jacobian().
        adx = (grid_e - e1) / r1 - (grid_e - e0) / r0
        ady = (grid_n - n1) / r1 - (grid_n - n0) / r0
        bdx = (grid_e - e2) / r2 - (grid_e - e0) / r0
        bdy = (grid_n - n2) / r2 - (grid_n - n0) / r0
        cdx = (grid_e - e2) / r2 - (grid_e - e1) / r1
        cdy = (grid_n - n2) / r2 - (grid_n - n1) / r1

        # Symmetric 2×2 J^T J
        m00 = adx**2 + bdx**2 + cdx**2
        m11 = ady**2 + bdy**2 + cdy**2
        m01 = adx * ady + bdx * bdy + cdx * cdy

        det = m00 * m11 - m01**2
        gdop = np.sqrt((m00 + m11) / det)

    gdop[det <= 0] = np.nan
    return gdop


def plot_dop(
    sensor_data: Path,
    grid_extent_km: tuple[float, float, float, float] | None = None,
    grid_resolution: int = 300,
    figsize: tuple[float, float] = (4.5, 4.5),
) -> Figure:
    """Plot geometric dilution of precision (GDOP) for the sensor array geometry.

    Shows where the sensor geometry gives precise (low GDOP) or imprecise
    (high GDOP) localization over the search domain.  For a nearly E–W linear
    array the GDOP will be high directly north and south of the array because
    all sensor pairs share nearly the same bearing to those sources.

    Args:
        sensor_data: Path to sensor positions CSV (see read_sensor_positions).
        grid_extent_km: (e_min, e_max, n_min, n_max) bounds of the search grid
            in km, in the same ENU frame as the sensor positions.  Defaults to
            ±50 km around the sensor centroid.
        grid_resolution: Number of grid points per axis.
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

    # ---- GDOP field ----
    gdop = _gdop_grid(e0, n0, e1, n1, e2, n2, grid_e, grid_n)

    # ---- Figure ----
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    _gdop_vmax = 10000
    im = ax.pcolormesh(
        grid_e,
        grid_n,
        gdop,
        cmap=cmo.thermal,
        norm=LogNorm(vmax=_gdop_vmax),
        shading="auto",
        rasterized=True,
    )
    CS = ax.contour(
        grid_e,
        grid_n,
        gdop,
        levels=10 ** np.arange(0, np.log10(_gdop_vmax) + 1),
        colors="k",
        linewidths=0.5,
        alpha=1.0,
    )
    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()
    ax.clabel(CS, inline=1, fontsize=6, fmt=fmt)

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("DOP (km/km)")

    for se, sn in zip(sensor_e, sensor_n):
        ax.plot(se, sn, "kv", ms=7, zorder=5)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig
