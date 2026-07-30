"""Figure creation module for vineyard wind analysis.

This package provides functionality for creating publication-quality figures
from vineyard wind data, organized by figure type.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vineyard.figures.common import (
    DOPConfig,
    PlottingConfig,
    PRCurveConfig,
    SNRComparisonConfig,
    TDOASensitivityConfig,
    WhaleTrackingConfig,
    add_panel_label,
    save_and_show_figure,
)
from vineyard.figures.corr import plot_correlations
from vineyard.figures.denoise import plot_denoising
from vineyard.figures.dop import plot_dop
from vineyard.figures.experiment import plot_experiment_setup
from vineyard.figures.pr_curve import plot_pr_curve
from vineyard.figures.signals import plot_signals
from vineyard.figures.snr import plot_snr_comparison
from vineyard.figures.tdoa_results import plot_whale_data
from vineyard.figures.tdoa_sensitivity import plot_tdoa_sensitivity
from vineyard.figures.templates import plot_strike_template

__all__ = [
    "DOPConfig",
    "PRCurveConfig",
    "PlottingConfig",
    "SNRComparisonConfig",
    "TDOASensitivityConfig",
    "WhaleTrackingConfig",
    "add_panel_label",
    "make_figures",
    "plot_correlations",
    "plot_denoising",
    "plot_dop",
    "plot_experiment_setup",
    "plot_pr_curve",
    "plot_signals",
    "plot_snr_comparison",
    "plot_strike_template",
    "plot_tdoa_sensitivity",
    "plot_whale_data",
    "save_and_show_figure",
]

logger = logging.getLogger(__name__)


def _render(
    name: str,
    plot_fn: Callable[..., Figure],
    output: Path,
    show: bool,
    savefig_kwargs: dict[str, Any],
    **plot_kwargs: Any,
) -> None:
    """Call a plot function and save/show its figure, with standardized logging.

    Centralizing this ensures every figure type is saved, shown, and logged
    identically, regardless of which module produced it.
    """
    logger.info(f"Creating {name} figure...")
    fig = plot_fn(**plot_kwargs)
    save_and_show_figure(fig, output, show=show, savefig_kwargs=savefig_kwargs)
    logger.info(f"{name} figure saved to {output}")


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
        _render(
            "Experiment setup",
            plot_experiment_setup,
            config.experiment.output,
            show,
            config.savefig_kwargs,
            bathy_data=config.experiment.bathy_data,
            sensor_data=config.experiment.sensor_data,
            turbine_data=config.experiment.turbine_data,
            active_turbine_name=config.experiment.active_turbine_name,
            image_file=config.experiment.image_file,
        )

    if config.signal_template is not None:
        st = config.signal_template
        _render(
            "Signal template",
            plot_signals,
            st.output,
            show,
            config.savefig_kwargs,
            inventory_dir=st.inventory_dir,
            example_signal=st.example_signal,
            whale_sensors=st.whale_sensors,
            strike_sensors=st.strike_sensors,
            col_titles=st.col_titles,
            filt_type=st.filt_type,
            filt_freq=st.filt_freq,
            nperseg=st.nperseg,
            hop=st.hop,
            nfft=st.nfft,
            flim=st.flim,
            whale_ylim=st.whale_ylim,
            strike_ylim=st.strike_ylim,
            calibration_dir=config.calibration_dir,
            figsize=st.figsize,
        )

    if config.template_construction is not None:
        tc = config.template_construction
        _render(
            "Template construction",
            plot_strike_template,
            tc.output,
            show,
            config.savefig_kwargs,
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

    if config.correlation is not None:
        corr = config.correlation
        _render(
            "Correlation",
            plot_correlations,
            corr.output,
            show,
            config.savefig_kwargs,
            corr_file=corr.corr_file,
            window=corr.window,
            strike_window_size=corr.strike_window_size,
        )

    if config.denoising is not None:
        dn = config.denoising
        _render(
            "Denoising",
            plot_denoising,
            dn.output,
            show,
            config.savefig_kwargs,
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

    if config.dop is not None:
        dop = config.dop
        _render(
            "DOP",
            plot_dop,
            dop.output,
            show,
            config.savefig_kwargs,
            sensor_data=dop.sensor_data,
            grid_extent_km=dop.grid_extent_km,
            grid_resolution=dop.grid_resolution,
            figsize=dop.figsize,
        )

    if config.tdoa_sensitivity is not None:
        ts = config.tdoa_sensitivity
        _render(
            "TDOA sensitivity",
            plot_tdoa_sensitivity,
            ts.output,
            show,
            config.savefig_kwargs,
            query_bearing_deg=ts.query_bearing_deg,
            figsize=ts.figsize,
        )

    if config.snr_comparison is not None:
        sc = config.snr_comparison
        _render(
            "SNR comparison",
            plot_snr_comparison,
            sc.output,
            show,
            config.savefig_kwargs,
            snr_file=sc.snr_file,
            noise_reduction_dir=sc.noise_reduction_dir,
        )

    if config.pr_curve is not None:
        _render(
            "PR curve",
            plot_pr_curve,
            config.pr_curve.output,
            show,
            config.savefig_kwargs,
            pr_curve_data=config.pr_curve.pr_curve_data,
        )

    if config.whale_tracking is not None:
        wt = config.whale_tracking
        _render(
            "Whale tracking",
            plot_whale_data,
            wt.output,
            show,
            config.savefig_kwargs,
            whale_data=wt.whale_data,
            time_ranges=wt.time_ranges or [],
            brg_ylim=wt.brg_ylim,
            brg_ref=wt.brg_ref,
        )
