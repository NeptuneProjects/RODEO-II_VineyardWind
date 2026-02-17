"""Common plotting utilities shared across all figure types."""

import logging
from pathlib import Path
from typing import Any

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


class SignalTemplateConfig(BaseModel):
    """Configuration for signal template figure creation."""

    inventory_dir: Path | None = None
    col_titles: list[str] = ["3DVHA", "VLA1", "VLA2"]
    example_signal: dict[str, Any] | None = None
    whale_sensors: list[dict[str, Any]] | None = None
    strike_sensors: list[dict[str, Any]] | None = None
    filt_type: str | None = None
    filt_freq: list[float] | float | None = None
    nperseg: int = 4096
    hop: int = 2048
    nfft: int | None = None
    flim: tuple[float, float] | None = None
    whale_ylim: tuple[float, float] | None = None
    strike_ylim: tuple[float, float] | None = None
    figsize: tuple[float, float] = (12.0, 8.0)
    output: Path = "reports/figures/signal_templates.png"


class PlottingConfig(BaseModel):
    """Configuration for plotting operations.

    Attributes:
        map: Optional configuration for map figure creation
    """

    mpl_style: Path | None = None
    map: MapConfig | None = None
    signal_template: SignalTemplateConfig | None = None
    savefig_kwargs: dict[str, Any] = {}
    calibration_dir: Path | None = None


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
    fig: Figure, output: Path, show: bool = False, savefig_kwargs: dict = {}
) -> None:
    """Save figure to file and optionally display it.

    Args:
        fig: Matplotlib figure to save
        output: Output file path
        dpi: Resolution in dots per inch
        show: Whether to display the figure interactively
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, **savefig_kwargs)
    logging.info(f"Figure saved to {output.resolve()}")

    if show:
        plt.show()
    plt.close(fig)
