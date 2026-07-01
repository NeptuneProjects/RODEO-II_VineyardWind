"""DOA bearing residual figure.

Single-panel figure showing the normalized DOA residual (dB) as a function
of trial bearing for a simulated far-field source, illustrating the
north-south ambiguity of the east-west linear array.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_tdoa_sensitivity(
    query_bearing_deg: float = 350.0,
    figsize: tuple[float, float] = (3.0, 1.5),
) -> Figure:
    """Plot the normalized DOA residual as a function of trial bearing.

    Shows 10 log10 of the normalized least-squares residual of the
    plane-wave TDOA fit as a function of trial bearing. The residual
    vanishes at the true bearing and its mirror image across the array
    axis, illustrating the north-south ambiguity of the east-west array.

    Args:
        query_bearing_deg: True source bearing (degrees CW from North).
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib Figure.
    """
    c = 1.5  # km/s

    # ‖g‖² is a constant that cancels in normalization; only sₑ matters
    s_e_true = -np.sin(np.radians(query_bearing_deg)) / c

    # Axis centered on North: bearings run −180° … 0° (N) … +180°
    # Shift formula: bearing > 180° maps to bearing − 360°
    bearings = np.linspace(-180.0, 180.0, 7201)
    s_e_trial = -np.sin(np.radians(bearings)) / c
    residuals = (s_e_trial - s_e_true) ** 2
    residuals /= residuals.max()
    residuals_db = 10.0 * np.log10(np.maximum(residuals, 1e-6))

    # Mirror bearing: sin(θ_mirror) = sin(θ_true) ⟹ θ_mirror = 180° − θ_true
    theta_mirror_deg = (180.0 - query_bearing_deg) % 360.0

    # Convert true and mirror bearings to centered coordinates
    query_centered = (
        query_bearing_deg if query_bearing_deg <= 180.0 else query_bearing_deg - 360.0
    )
    mirror_centered = (
        theta_mirror_deg if theta_mirror_deg <= 180.0 else theta_mirror_deg - 360.0
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Shade the southern half-space (excluded by geographic prior).
    # In the centered axis, south half-space [90°, 270°] → [90°, 180°] and [−180°, −90°].
    ax.axvspan(
        -180.0,
        -90.0,
        color="lightgray",
        alpha=0.5,
        zorder=0,
        label="Southern half-space",
    )
    ax.axvspan(90.0, 180.0, color="lightgray", alpha=0.5, zorder=0)

    ax.plot(bearings, residuals_db, "k-", linewidth=1.0, zorder=2)

    ax.axvline(
        query_centered,
        color="tab:blue",
        linewidth=1.0,
        linestyle="--",
        label=f"True bearing ({query_bearing_deg:.0f}°)",
        zorder=3,
    )
    ax.axvline(
        mirror_centered,
        color="tab:red",
        linewidth=1.0,
        linestyle="--",
        label=f"Mirror bearing ({theta_mirror_deg:.0f}°)",
        zorder=3,
    )

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-65.0, 5.0)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["S (180°)", "W (270°)", "N (0°)", "E (90°)", "S (180°)"])
    ax.set_xlabel("Bearing")
    ax.set_ylabel("Normalized residual (dB)")
    ax.legend(fontsize=7, loc="lower right", ncol=1, framealpha=1.0)

    fig.tight_layout()
    return fig
