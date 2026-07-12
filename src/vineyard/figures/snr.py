"""SNR distribution figure: piling vs. quiet on the denoised channel."""

import string
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from vineyard.figures.common import add_panel_label

_COLOR_QUIET = sns.color_palette("pastel")[0]
_COLOR_PILING = sns.color_palette("pastel")[1]
_COLOR_NOISE_REDUCTION = sns.color_palette("pastel")[2]
_SENSORS = ["3dvha", "vla1", "vla2"]
_SENSOR_LABELS = {"3dvha": "Site A", "vla1": "Site B", "vla2": "Site C"}
_SITE_ORDER = [_SENSOR_LABELS[s] for s in _SENSORS]
_CONDITION_LABELS = {"piling": "Full-energy strikes", "quiet": "Inactive"}
_HUE_ORDER = [_CONDITION_LABELS["piling"], _CONDITION_LABELS["quiet"]]
_PALETTE = {
    _CONDITION_LABELS["piling"]: _COLOR_PILING,
    _CONDITION_LABELS["quiet"]: _COLOR_QUIET,
}


def _load_data(snr_file: Path) -> pl.DataFrame:
    df = pl.read_csv(snr_file)
    if "data" in df.columns:
        df = df.filter(pl.col("data") == "denoised")
    return (
        df.filter(pl.col("snr_db").is_finite())
        .filter(pl.col("var_t_ms2").is_finite())
        .with_columns(pl.col("sensor").replace(_SENSOR_LABELS).alias("site"))
        .with_columns(pl.col("condition").replace(_CONDITION_LABELS))
    )


def _load_noise_reduction(noise_reduction_dir: Path) -> pl.DataFrame:
    frames = [
        pl.read_csv(noise_reduction_dir / f"{sensor}_noise_reduction.csv")
        .filter(pl.col("rms_diff_db").is_finite())
        .with_columns(pl.lit(_SENSOR_LABELS[sensor]).alias("site"))
        for sensor in _SENSORS
    ]
    return pl.concat(frames)


def plot_snr_comparison(snr_file: Path, noise_reduction_dir: Path) -> Figure:
    """Plot noise reduction, piling vs. quiet SNR, and timing variance distributions.

    Returns a figure with 1 row x 3 columns (noise reduction, SNR, timing
    variance), with sites A, B, and C on the x-axis. The SNR and timing
    variance panels additionally split each site by piling condition.
    """
    df = _load_data(snr_file).to_pandas()
    nr_df = _load_noise_reduction(noise_reduction_dir).to_pandas()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3))
    letters = iter(string.ascii_lowercase)

    sns.boxenplot(
        data=nr_df,
        x="site",
        y="rms_diff_db",
        order=_SITE_ORDER,
        color=_COLOR_NOISE_REDUCTION,
        showfliers=False,
        ax=axes[0],
        k_depth=4,
    )
    axes[0].set_ylim(-2, 20)
    axes[0].grid(axis="y", linewidth=0.5, alpha=0.75)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Strike RMS level reduction (dB)")
    add_panel_label(axes[0], next(letters))

    sns.boxenplot(
        data=df,
        x="site",
        y="snr_db",
        hue="condition",
        order=_SITE_ORDER,
        hue_order=_HUE_ORDER,
        palette=_PALETTE,
        showfliers=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="site",
        y="snr_db",
        hue="condition",
        order=_SITE_ORDER,
        hue_order=_HUE_ORDER,
        dodge=True,
        palette="dark:k",
        size=2,
        alpha=0.5,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_ylim(0, 35)
    axes[1].grid(axis="y", linewidth=0.5, alpha=0.75)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].legend(
        title="Condition",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        ncols=2,
        bbox_to_anchor=(0.75, -0.1),
        borderaxespad=0,
    ).set_title("Pile driving status")
    add_panel_label(axes[1], next(letters))

    sns.boxenplot(
        data=df,
        x="site",
        y="var_t_ms2",
        hue="condition",
        order=_SITE_ORDER,
        hue_order=_HUE_ORDER,
        palette=_PALETTE,
        showfliers=False,
        ax=axes[2],
    )
    sns.stripplot(
        data=df,
        x="site",
        y="var_t_ms2",
        hue="condition",
        order=_SITE_ORDER,
        hue_order=_HUE_ORDER,
        dodge=True,
        palette="dark:k",
        size=2,
        alpha=0.5,
        ax=axes[2],
        legend=False,
    )
    axes[2].set_ylim(0, 10)
    axes[2].grid(axis="y", linewidth=0.5, alpha=0.75)
    axes[2].set_xlabel("")
    axes[2].set_ylabel(r"$\sigma_t^2$ (ms$^2$)")
    axes[2].get_legend().remove()
    add_panel_label(axes[2], next(letters))

    return fig
