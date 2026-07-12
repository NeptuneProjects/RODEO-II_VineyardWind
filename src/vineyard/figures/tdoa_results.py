"""Plot whale DOA estimation results."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from vineyard.figures.common import add_panel_label

_cmap = matplotlib.cm.get_cmap("turbo")
colorbar_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "custom", _cmap(np.linspace(0.0, 0.9, 256))
)


def plot_whale_data(
    whale_data: Path,
    time_ranges: Sequence[tuple[np.datetime64, np.datetime64]],
    brg_ylim: tuple[float, float] | None = None,
    brg_ref: float | None = None,
) -> Figure:
    whale_df = pl.read_csv(whale_data, try_parse_dates=True)

    site_a_color = "tab:blue"
    site_b_color = "tab:orange"
    site_c_color = "tab:green"

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, offset_formats=["%Y-%b-%d"] * 6)

    time = whale_df["timestamp"].to_numpy()
    # Center bearings on North: map (180°, 360°] → (−180°, 0°] so north-facing
    # bearings cluster around 0 instead of wrapping across the 0°/360° boundary.
    brg_mean = ((whale_df["doa_brg"].to_numpy() + 180.0) % 360.0) - 180.0
    brg_unc = whale_df["doa_brg_unc"].to_numpy()
    snr_a = whale_df["snr_p_db_3dvha"].to_numpy()
    snr_b = whale_df["snr_p_db_vla1"].to_numpy()
    snr_c = whale_df["snr_p_db_vla2"].to_numpy()
    sigma_t_a = whale_df["sigma_t_3dvha"].to_numpy() * 1e3
    sigma_t_b = whale_df["sigma_t_vla1"].to_numpy() * 1e3
    sigma_t_c = whale_df["sigma_t_vla2"].to_numpy() * 1e3

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(8, 4),
        gridspec_kw={"height_ratios": [2, 1, 1, 1], "hspace": 0.14},
    )
    ci_color = "tab:red"

    ax0 = axes[0]
    ax0.scatter(time, brg_mean, color="k", s=10, zorder=20)
    if brg_ylim is not None:
        # brg_ylim given as compass bearings; convert to centered coordinates
        lo = ((brg_ylim[0] + 180.0) % 360.0) - 180.0
        hi = ((brg_ylim[1] + 180.0) % 360.0) - 180.0
        ax0.set_ylim(min(lo, hi), max(lo, hi))
    ax0.grid()
    if brg_ref is not None:
        brg_ref_centered = ((brg_ref + 180.0) % 360.0) - 180.0
        ax0.axhline(
            brg_ref_centered, color="k", linestyle="--", linewidth=0.75, zorder=5
        )
    # Tick labels show actual compass bearings (e.g. −30 → "330°")
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v % 360:.0f}°"))
    ax0.set_ylabel("DOA (°T)")
    add_panel_label(ax0, "a")

    ax1 = axes[1]
    ax1.scatter(time, brg_unc, color=ci_color, s=5, zorder=10)
    ax1.set_ylim(0, 0.06)
    ax1.grid()
    ax1.set_ylabel(r"$\sigma_\theta$ (°)")
    add_panel_label(ax1, "b")

    ax2 = axes[2]
    ax2.scatter(time, sigma_t_a, color=site_a_color, s=1, zorder=10, label="Site A")
    ax2.scatter(time, sigma_t_b, color=site_b_color, s=1, zorder=10, label="Site B")
    ax2.scatter(time, sigma_t_c, color=site_c_color, s=1, zorder=10, label="Site C")
    ax2.set_ylim(0, 6.0)
    ax2.grid()
    ax2.set_ylabel(r"$\sigma_t$ (ms)")
    add_panel_label(ax2, "c")

    ax3 = axes[3]
    ax3.scatter(time, snr_a, color=site_a_color, s=1, zorder=10, label="Site A")
    ax3.scatter(time, snr_b, color=site_b_color, s=1, zorder=10, label="Site B")
    ax3.scatter(time, snr_c, color=site_c_color, s=1, zorder=10, label="Site C")
    ax3.set_ylim(0, 40)
    ax3.grid()
    ax3.set_ylabel("SNR (dB)")
    add_panel_label(ax3, "d")

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    for start, end in time_ranges:
        for ax in axes:
            ax.axvspan(
                start, end, facecolor="gray", edgecolor="none", alpha=0.3, zorder=5
            )

    ax3.legend(
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.62, 1.5),
        framealpha=1.0,
        markerscale=3.0,
    )

    return fig
