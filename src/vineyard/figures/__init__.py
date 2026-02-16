"""Figure creation module for vineyard wind analysis.

This package provides functionality for creating publication-quality figures
from vineyard wind data, organized by figure type.

Public API:
    Maps:
        - create_maps: Create a two-panel map figure
        - create_and_save_maps: Load data, create, and save map figure

    Common utilities:
        - PlottingConfig: Configuration for plotting operations
        - MapConfig: Configuration for map figures
        - add_panel_label: Add panel labels to axes
        - save_and_show_figure: Save and optionally display figures

    Orchestration:
        - make_figures: Create all configured figures from PlottingConfig
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from vineyard.figures.common import (
    MapConfig,
    PlottingConfig,
    add_panel_label,
    save_and_show_figure,
)
from vineyard.figures.maps import create_and_save_maps, create_maps

__all__ = [
    # Common utilities
    "PlottingConfig",
    "MapConfig",
    "add_panel_label",
    "save_and_show_figure",
    # Map figures
    "create_maps",
    "create_and_save_maps",
    # Orchestration
    "make_figures",
]


def make_figures(config: PlottingConfig, show: bool = False) -> None:
    """Create all configured figures based on PlottingConfig.

    This is the main entry point for figure generation, called by the workflow.
    It creates any figures that are configured in the PlottingConfig.

    Args:
        config: Configuration specifying which figures to create
        show: Whether to display figures interactively after saving
    """
    if config.mpl_style is not None:
        plt.style.use(config.mpl_style)

    figures_created = []

    # Create map figure if configured
    if config.map is not None:
        logging.info("Creating map figure...")
        create_and_save_maps(
            bathy_data=config.map.bathy_data,
            sensor_data=config.map.sensor_data,
            turbine_data=config.map.turbine_data,
            active_turbine_name=config.map.active_turbine_name,
            whale_bearings=config.map.whale_bearings,
            output=config.map.output,
            dpi=config.map.dpi,
            show=show,
        )
        figures_created.append("map")

    if not figures_created:
        logging.warning("No figures configured in PlottingConfig")
    else:
        logging.info(
            f"Created {len(figures_created)} figure(s): {', '.join(figures_created)}"
        )
