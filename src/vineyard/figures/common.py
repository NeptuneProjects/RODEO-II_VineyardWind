"""Common plotting utilities shared across all figure types."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from pydantic import BaseModel


class MapConfig(BaseModel):
    """Configuration for map figure creation."""

    bathy_data: Path | None = None
    sensor_data: Path | None = None
    turbine_data: Path | None = None
    active_turbine_name: str | None = None
    whale_bearings: Path | None = None
    output: Path = "reports/figures/map.png"
    dpi: int = 300


class PlottingConfig(BaseModel):
    """Configuration for plotting operations.

    Attributes:
        map: Optional configuration for map figure creation
    """

    mpl_style: Path | None = None
    map: MapConfig | None = None


def add_panel_label(ax, label: str) -> None:
    """Add a panel label (e.g., 'a', 'b') to the upper left of an axis.

    Args:
        ax: Matplotlib axis to add label to
        label: Label text (typically a single letter)
    """
    anchored_text = AnchoredText(
        label,
        loc="upper left",
        prop=dict(fontsize=8, fontweight="bold"),
        frameon=True,
        pad=0.0,
        borderpad=0.5,
    )
    anchored_text.patch.set_boxstyle("square,pad=0.3")
    anchored_text.patch.set_edgecolor("black")
    anchored_text.patch.set_facecolor("white")
    anchored_text.zorder = 50
    ax.add_artist(anchored_text)


def save_and_show_figure(
    fig: Figure, output: Path, dpi: int = 300, show: bool = False
) -> None:
    """Save figure to file and optionally display it.

    Args:
        fig: Matplotlib figure to save
        output: Output file path
        dpi: Resolution in dots per inch
        show: Whether to display the figure interactively
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    logging.info(f"Figure saved to {output.resolve()}")

    if show:
        plt.show()
    plt.close(fig)
