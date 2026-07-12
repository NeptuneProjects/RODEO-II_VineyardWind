"""Common plotting utilities shared across all figure types."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from pydantic import BaseModel, model_validator
from tritonoa.data.time import TIME_PRECISION


class CorrelationConfig(BaseModel):
    """Configuration for correlation figure creation."""

    corr_file: Path | None = None
    window: float = 300.0
    strike_window_size: int | None = None
    output: Path = "reports/figures/strike_correlation.png"


class DenoiseConfig(BaseModel):
    """Configuration for denoising figure creation."""

    data_dir: Path | None = None
    sensor: str = "vla1"
    template_data: Path | None = None
    time_start: str | None = None
    time_end: str | None = None
    freq_time_start: str | None = None
    freq_time_end: str | None = None
    filt_type: str | None = None
    filt_freq: list[float] | float | None = None
    window: str = "hann"
    nperseg: int = 16384
    hop: int = 8192
    flim: list[float] | None = None
    output: Path = "reports/figures/denoising.png"

    @model_validator(mode="after")
    def convert_to_np_datetime(self) -> "DenoiseConfig":
        """Convert time_ranges from lists of strings to lists of numpy datetime64 tuples."""
        if self.time_start is not None:
            self.time_start = np.datetime64(self.time_start, TIME_PRECISION)
        if self.time_end is not None:
            self.time_end = np.datetime64(self.time_end, TIME_PRECISION)
        if self.freq_time_start is not None:
            self.freq_time_start = np.datetime64(self.freq_time_start, TIME_PRECISION)
        if self.freq_time_end is not None:
            self.freq_time_end = np.datetime64(self.freq_time_end, TIME_PRECISION)
        return self


class ExperimentConfig(BaseModel):
    """Configuration for experiment figure creation."""

    bathy_data: Path | None = None
    sensor_data: Path | None = None
    turbine_data: Path | None = None
    active_turbine_name: str | None = None
    image_file: Path | None = None
    output: Path = "reports/figures/experiment_setup.png"


class WhaleTrackingConfig(BaseModel):
    """Configuration for whale tracking figure creation."""

    whale_data: Path | None = None
    time_ranges: list[list[str]] | None = None
    output: Path = "reports/figures/whale_tracking.png"
    brg_ylim: tuple[float, float] | None = None  # bearing panel y-axis limits (degrees)
    brg_ref: float | None = None  # optional reference bearing line (degrees)

    @model_validator(mode="after")
    def convert_to_np_datetime(self) -> "WhaleTrackingConfig":
        """Convert time_ranges from lists of strings to lists of numpy datetime64 tuples."""
        if self.time_ranges is not None:
            self.time_ranges = [
                (
                    np.datetime64(start, TIME_PRECISION),
                    np.datetime64(end, TIME_PRECISION),
                )
                for start, end in self.time_ranges
            ]
        return self


class SignalsConfig(BaseModel):
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


class TemplateConstructionConfig(BaseModel):
    """Configuration for template construction figure.

    Attributes:
        sensor_name: Name of the sensor to plot templates for
        strike_indices: List of strike indices to plot
        output_dir: Directory to save the plots (defaults to reports/figures/template_construction/)

    The following attributes are typically synced from ProcessConfig in workflow:
        strike_index_path: Path to strike index CSV (from process.strike.strike_index)
        strike_corr_path: Path to correlation matrix (from process.strike.strike_corr)
        inventory_path: Path to sensor inventory (derived from process.inventory_path)
        channel: Channel number (from process.template.sensors)
        start_time: Start time (from process.start_time)
        end_time: End time (from process.end_time)
        buffer_start: Buffer before strike peak (from process.template.buffer_start)
        buffer_end: Buffer after strike peak (from process.template.buffer_end)
        window_size: Rolling window size (from process.template.window_size)
        ylim: Y-axis limits (from process.template.sensors)
        taper_pc: Taper percentage (from process.template.taper_pc)
        dec_factor: Decimation factor (from process.template.dec_factor)
        filt_type: Filter type (from process.template.filt_type)
        filt_freq: Filter frequency (from process.template.filt_freq)
    """

    sensor_name: str = "vla1"
    strike_indices: list[int] = [500, 1500, 2500]
    output_dir: Path = Path("reports/figures/template_construction/")

    # These will be synced from ProcessConfig
    strike_index_path: Path | None = None
    strike_corr_path: Path | None = None
    inventory_path: Path | None = None
    channel: int | None = None
    start_time: str | None = None
    end_time: str | None = None
    buffer_start: float | None = None
    buffer_end: float | None = None
    window_size: int | None = None
    ylim: tuple[float, float] | None = None
    taper_pc: float | None = None
    dec_factor: int | None = None
    filt_type: str | None = None
    filt_freq: float | list[float] | None = None


class DOPConfig(BaseModel):
    """Configuration for the geometric dilution of precision (GDOP) figure."""

    sensor_data: Path = "data/sensors.csv"
    grid_extent_km: tuple[float, float, float, float] | None = None
    grid_resolution: int = 300
    figsize: tuple[float, float] = (4.5, 4.5)
    output: Path = "reports/figures/dop.png"


class TDOASensitivityConfig(BaseModel):
    """Configuration for the DOA bearing residual figure."""

    query_bearing_deg: float = 350.0
    figsize: tuple[float, float] = (5.0, 2.0)
    output: Path = "reports/figures/tdoa/tdoa_sensitivity.png"


class SNRComparisonConfig(BaseModel):
    """Configuration for the SNR piling vs. quiet comparison figure."""

    snr_file: Path = Path("reports/evaluation/snr_comparison.csv")
    noise_reduction_dir: Path = Path("data/acoustic/denoised")
    output: Path = Path("reports/figures/snr_comparison.png")


class PRCurveConfig(BaseModel):
    """Configuration for the precision-recall curve figure."""

    pr_curve_data: Path = Path("reports/evaluation/pr_curve_data.csv")
    output: Path = Path("reports/scirep/figure07.png")


class PlottingConfig(BaseModel):
    """Configuration for plotting operations.

    Attributes:
        map: Optional configuration for map figure creation
        signal_template: Optional configuration for signal template figure
        template_construction: Optional configuration for template construction figure
    """

    mpl_style: Path | None = None
    experiment: ExperimentConfig | None = None
    whale_tracking: WhaleTrackingConfig | None = None
    signal_template: SignalsConfig | None = None
    template_construction: TemplateConstructionConfig | None = None
    denoising: DenoiseConfig | None = None
    correlation: CorrelationConfig | None = None
    dop: DOPConfig | None = None
    tdoa_sensitivity: TDOASensitivityConfig | None = None
    snr_comparison: SNRComparisonConfig | None = None
    pr_curve: PRCurveConfig | None = None
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


def format_tick_scientific(value: float, pos=None, mathtext: bool = True) -> str:
    """Format tick label in simple scientific notation (10 ** n).

    Args:
        value: The tick value to format.
        pos: Position (unused, required by matplotlib FuncFormatter).
        mathtext: If True, use math font. If False, use regular figure font (via \\mathregular).

    Returns:
        Formatted string: "0" for zero, "C x 10^n" for others with superscripts.
    """
    if value == 0:
        return "0"

    # Get the exponent and mantissa
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)

    # If mantissa is very close to +-1, just show +-10 ** exponent
    if np.isclose(abs(mantissa), 1.0, atol=0.01):
        sign = "-" if mantissa < 0 else ""
        if exponent == 0:
            return f"{sign}1"
        body = f"{sign}10^{{{exponent}}}"
    else:
        body = f"{mantissa:.1f} \\times 10^{{{exponent}}}"

    return f"${body}$" if mathtext else f"$\\mathregular{{{body}}}$"


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
