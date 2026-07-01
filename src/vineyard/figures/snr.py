"""SNR distribution figure: piling vs. quiet on the denoised channel."""

import string
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from vineyard.figures.common import add_panel_label

# _COLOR_QUIET = "#006BA4"
_COLOR_QUIET = sns.color_palette("pastel")[0]
# _COLOR_PILING = "#FF800E"
_COLOR_PILING = sns.color_palette("pastel")[1]
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


def plot_snr_comparison(snr_file: Path) -> Figure:
    """Plot piling vs. quiet SNR and timing variance distributions as boxplots.

    Returns a figure with 1 row (SNR, timing variance) x 2 columns, with sites
    A, B, and C on the x-axis and condition indicated by color.
    """
    df = _load_data(snr_file).to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    letters = iter(string.ascii_lowercase)

    sns.boxenplot(
        data=df,
        x="site",
        y="snr_db",
        hue="condition",
        order=_SITE_ORDER,
        hue_order=_HUE_ORDER,
        palette=_PALETTE,
        showfliers=False,
        ax=axes[0],
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
        ax=axes[0],
        legend=False,
    )
    axes[0].set_ylim(0, 35)
    axes[0].grid(axis="y", linewidth=0.5, alpha=0.75)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("SNR (dB)")
    axes[0].legend(
        title="Condition",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        ncols=2,
        bbox_to_anchor=(0.75, -0.1),
        borderaxespad=0,
    ).set_title("Pile driving status")
    add_panel_label(axes[0], next(letters))

    sns.boxenplot(
        data=df,
        x="site",
        y="var_t_ms2",
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
        y="var_t_ms2",
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
    axes[1].set_ylim(0, 10)
    axes[1].grid(axis="y", linewidth=0.5, alpha=0.75)
    axes[1].set_xlabel("")
    axes[1].set_ylabel(r"$\sigma_t^2$ (ms$^2$)")
    axes[1].get_legend().remove()
    add_panel_label(axes[1], next(letters))

    return fig
