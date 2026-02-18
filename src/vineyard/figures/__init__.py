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
    WhaleTrackingConfig,
    PlottingConfig,
    add_panel_label,
    save_and_show_figure,
)
from vineyard.figures.denoise import plot_denoising
from vineyard.figures.experiment import plot_experiment_setup
from vineyard.figures.whale import plot_whale_tracking
from vineyard.figures.signals import plot_signals
from vineyard.figures.templates import plot_strike_template

__all__ = [
    # Common utilities
    "PlottingConfig",
    "WhaleTrackingConfig",
    "add_panel_label",
    "save_and_show_figure",
    # Map figures
    "create_map_panels",
    # Signal figures
    "plot_signals",
    # Template construction figures
    "plot_strike_template",
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

    if config.experiment is not None:
        logging.info("Creating experiment setup figure...")
        fig = plot_experiment_setup(
            bathy_data=config.experiment.bathy_data,
            sensor_data=config.experiment.sensor_data,
            turbine_data=config.experiment.turbine_data,
            active_turbine_name=config.experiment.active_turbine_name,
            image_file=config.experiment.image_file,
        )
        save_and_show_figure(
            fig,
            config.experiment.output,
            show=show,
            savefig_kwargs=config.savefig_kwargs,
        )
        logging.info(f"Experiment setup figure saved to {config.experiment.output}")

    if config.whale_tracking is not None:
        logging.info("Creating whale tracking figure...")
        fig = plot_whale_tracking(
            bathy_data=config.whale_tracking.bathy_data,
            sensor_data=config.whale_tracking.sensor_data,
            turbine_data=config.whale_tracking.turbine_data,
            active_turbine_name=config.whale_tracking.active_turbine_name,
            whale_bearings=config.whale_tracking.whale_bearings,
        )
        save_and_show_figure(
            fig,
            config.whale_tracking.output,
            show=show,
            savefig_kwargs=config.savefig_kwargs,
        )
        logging.info(f"Whale tracking figure saved to {config.whale_tracking.output}")

    if config.signal_template is not None:
        logging.info("Creating signal template figure...")
        fig = plot_signals(
            inventory_dir=config.signal_template.inventory_dir,
            example_signal=config.signal_template.example_signal,
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

    if config.template_construction is not None:
        logging.info("Creating template construction figures...")
        tc = config.template_construction
        fig = plot_strike_template(
            strike_index_path=tc.strike_index_path,
            strike_corr_path=tc.strike_corr_path,
            inventory_path=tc.inventory_path,
            calibration_dir=config.calibration_dir,
            sensor_name=tc.sensor_name,
            channel=tc.channel,
            strike_indices=tc.strike_indices,
            start_time=tc.start_time,
            end_time=tc.end_time,
            buffer_start=tc.buffer_start,
            buffer_end=tc.buffer_end,
            window_size=tc.window_size,
            ylim=tc.ylim,
            taper_pc=tc.taper_pc,
            dec_factor=tc.dec_factor,
            filt_type=tc.filt_type,
            filt_freq=tc.filt_freq,
        )
        outfile = tc.output_dir / f"{tc.sensor_name}_template_construction.png"
        save_and_show_figure(
            fig,
            outfile,
            show=show,
            savefig_kwargs=config.savefig_kwargs,
        )
        logging.info(f"Template construction figure saved to {outfile}")

    if config.denoising is not None:
        logging.info("Creating denoising figure...")
        dn = config.denoising
        fig = plot_denoising(
            data_dir=dn.data_dir,
            sensor=dn.sensor,
            template_data=dn.template_data,
            time_start=dn.time_start,
            time_end=dn.time_end,
            freq_time_start=dn.freq_time_start,
            freq_time_end=dn.freq_time_end,
            filt_type=dn.filt_type,
            filt_freq=dn.filt_freq,
            window=dn.window,
            nperseg=dn.nperseg,
            hop=dn.hop,
            flim=dn.flim,
        )
        save_and_show_figure(
            fig,
            dn.output,
            show=show,
            savefig_kwargs=config.savefig_kwargs,
        )
        logging.info(f"Denoising figure saved to {dn.output}")
