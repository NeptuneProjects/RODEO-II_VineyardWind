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

import matplotlib.pyplot as plt

from vineyard.figures.common import (
    MapConfig,
    PlottingConfig,
    add_panel_label,
    save_and_show_figure,
)
from vineyard.figures.maps import create_map_panels, create_maps
from vineyard.figures.signals import plot_signal_templates

__all__ = [
    # Common utilities
    "PlottingConfig",
    "MapConfig",
    "add_panel_label",
    "save_and_show_figure",
    # Map figures
    "create_map_panels",
    "create_maps",
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

    if config.map is not None:
        logging.info("Creating map figure...")
        fig = create_map_panels(
            bathy_data=config.map.bathy_data,
            sensor_data=config.map.sensor_data,
            turbine_data=config.map.turbine_data,
            active_turbine_name=config.map.active_turbine_name,
            whale_bearings=config.map.whale_bearings,
        )
        save_and_show_figure(
            fig, config.map.output, show=show, savefig_kwargs=config.savefig_kwargs
        )
        logging.info(f"Map figure saved to {config.map.output}")

    if config.signal_template is not None:
        logging.info("Creating signal template figure...")
        fig = plot_signal_templates(
            inventory_dir=config.signal_template.inventory_dir,
            whale_sensors=config.signal_template.whale_sensors,
            strike_sensors=config.signal_template.strike_sensors,
            col_titles=config.signal_template.col_titles,
            filt_type=config.signal_template.filt_type,
            filt_freq=config.signal_template.filt_freq,
            nperseg=config.signal_template.nperseg,
            hop=config.signal_template.hop,
            nfft=config.signal_template.nfft,
            flim=config.signal_template.flim,
            whale_ylim=config.signal_template.whale_ylim,
            strike_ylim=config.signal_template.strike_ylim,
            calibration_dir=config.calibration_dir,
            figsize=config.signal_template.figsize,
        )
        save_and_show_figure(
            fig,
            config.signal_template.output,
            show=show,
            savefig_kwargs=config.savefig_kwargs,
        )
        logging.info(f"Signal template figure saved to {config.signal_template.output}")
