"""TDOA localization sensitivity figure.

Two-panel figure showing:
    (a) Geometric dilution of precision (GDOP) over the search domain.
    (b) MSE cost function with hyperbolic lines of position for a query source.
"""

from pathlib import Path

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from vineyard.figures.common import add_panel_label
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

    GDOP (geometric dilution of precision) is sqrt(trace(inv(J^T J))), where J
    is the 3×2 Jacobian of the hyperbolic TDOA equations evaluated at each grid
    point. High GDOP indicates poor localization geometry.

    For a 2×2 symmetric matrix M, trace(inv(M)) = trace(M) / det(M), so
    GDOP = sqrt((m00 + m11) / (m00*m11 - m01^2)).

    $$\text{GDOP} = \sqrt{\frac{\text{tr}(\mathbf{J}^\top \mathbf{J})}{\det(\mathbf{J}^\top \mathbf{J})}}$$

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

    For a source at (qe, qn) the true range differences are computed, then the
    squared residuals of all three hyperbolic equations are evaluated at every
    grid point.  The MSE surface is minimised exactly at the true source
    location and rises away from it; its shape reveals localization precision.

    $$\mathcal{C}(\mathbf{x}) = \frac{1}{3} \sum_{(i,j),\in,\mathcal{P}} \bigl[\Delta_{ij}(\mathbf{x}) - \Delta_{ij}(\mathbf{q})\bigr]^2$$

    where $\Delta_{ij}(\mathbf{p}) = |\mathbf{p} - \mathbf{s}_j| - |\mathbf{p} - \mathbf{s}_i|$ is the signed range difference at position $\mathbf{p}$, $\mathbf{q}$ is the true source, and $\mathcal{P} = {(0,1),(0,2),(1,2)}$ is the set of sensor pairs.

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
    figsize: tuple[float, float] = (8.0, 4.0),
) -> Figure:
    """Plot TDOA localization sensitivity for the sensor array geometry.

    Panel (a) — GDOP heatmap: shows where the sensor geometry gives precise
    (low GDOP) or imprecise (high GDOP) localization.  For a nearly E–W linear
    array the GDOP will be high directly north and south of the array because
    all sensor pairs share nearly the same bearing to those sources.

    Panel (b) — Cost function landscape: log₁₀ MSE of the TDOA hyperbolic
    residuals for a query source, with hyperbolic lines of position (LOPs)
    overlaid.  Each LOP is the locus of equal range difference between one
    sensor pair; all three LOPs intersect at the true source.  A narrow,
    well-isolated intersection indicates good localization.

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

    # ---- Sensitivity fields ----
    gdop = _gdop_grid(e0, n0, e1, n1, e2, n2, grid_e, grid_n)
    cost = _cost_grid(e0, n0, e1, n1, e2, n2, qe, qn, grid_e, grid_n)

    # ---- Figure ----
    fig, (ax_gdop, ax_cost) = plt.subplots(1, 2, figsize=figsize)

    _SENSOR_NAMES = ["3DVHA", "VLA1", "VLA2"]

    # ---- Panel (a): GDOP ----
    _gdop_vmax = 10000
    im_g = ax_gdop.pcolormesh(
        grid_e,
        grid_n,
        gdop,
        cmap=cmo.thermal,
        norm=LogNorm(vmax=_gdop_vmax),
        shading="auto",
        rasterized=True,
    )
    CS = ax_gdop.contour(
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
    ax_gdop.clabel(CS, inline=1, fontsize=6, fmt=fmt)
    _div_g = make_axes_locatable(ax_gdop)
    _cax_g = _div_g.append_axes("right", size="4%", pad=0.05)
    cb_g = fig.colorbar(im_g, cax=_cax_g, extend="max")
    cb_g.set_label("DOP (km/km)")

    for se, sn in zip(sensor_e, sensor_n):
        ax_gdop.plot(se, sn, "kv", ms=7, zorder=5)

    ax_gdop.set_xlabel("Easting (km)")
    ax_gdop.set_ylabel("Northing (km)")
    ax_gdop.set_aspect("equal")
    add_panel_label(ax_gdop, "a")

    # ---- Panel (b): Cost function + hyperbolic LOPs ----
    log_cost = np.log10(cost + 1e-10)
    im_c = ax_cost.pcolormesh(
        grid_e,
        grid_n,
        log_cost,
        cmap="bone",
        vmax=2,
        shading="auto",
        rasterized=True,
    )
    _div_c = make_axes_locatable(ax_cost)
    _cax_c = _div_c.append_axes("right", size="4%", pad=0.05)
    cb_c = fig.colorbar(im_c, cax=_cax_c, extend="max")
    cb_c.set_label(r"$\mathdefault{log}_{10}(\mathdefault{MSE})$ (km²)")

    # Range differences over the grid and at the query source
    r0_g = np.sqrt((grid_e - e0) ** 2 + (grid_n - n0) ** 2)
    r1_g = np.sqrt((grid_e - e1) ** 2 + (grid_n - n1) ** 2)
    r2_g = np.sqrt((grid_e - e2) ** 2 + (grid_n - n2) ** 2)
    qr0 = float(np.sqrt((qe - e0) ** 2 + (qn - n0) ** 2))
    qr1 = float(np.sqrt((qe - e1) ** 2 + (qn - n1) ** 2))
    qr2 = float(np.sqrt((qe - e2) ** 2 + (qn - n2) ** 2))

    _LOP_COLORS = ["#E07B54", "#66C2A5", "#8DA0CB"]
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
        ax_cost.contour(
            grid_e,
            grid_n,
            diff_grid,
            levels=[true_diff],
            colors=[color],
            linewidths=1.5,
        )

    # amin_idx = np.unravel_index(np.nanargmin(cost), cost.shape)
    # amin_e = float(grid_e[amin_idx])
    # amin_n = float(grid_n[amin_idx])

    for se, sn in zip(sensor_e, sensor_n):
        ax_cost.plot(se, sn, "kv", ms=7, zorder=5)
    ax_cost.plot(qe, qn, "r*", ms=10, zorder=6)
    # ax_cost.plot(amin_e, amin_n, "c+", ms=10, mew=1.5, zorder=7)

    legend_handles = [
        Line2D([0], [0], marker="v", color="k", label="Sensors", ls="none", ms=5),
        Line2D([0], [0], marker="*", color="red", label="Source", ls="none", ms=7),
        *[
            Line2D([0], [0], color=c, label=l, linewidth=1.5)
            for c, l in zip(_LOP_COLORS, _LOP_LABELS)
        ],
        # Line2D(
        #     [0],
        #     [0],
        #     marker="+",
        #     color="cyan",
        #     label="MSE argmin",
        #     ls="none",
        #     ms=10,
        #     mew=1.5,
        # ),
    ]
    ax_cost.legend(
        handles=legend_handles, fontsize=6, loc="lower right", framealpha=1.0
    )

    ax_cost.set_xlabel("Easting (km)")
    ax_cost.set_ylabel("Northing (km)")
    ax_cost.set_aspect("equal")
    add_panel_label(ax_cost, "b")

    fig.tight_layout()
    return fig
