"""Precision-recall curves for denoised whale call detection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import auc

_SENSORS = ["3dvha", "vla1", "vla2"]
_SITES = {"3dvha": "Site A", "vla1": "Site B", "vla2": "Site C"}
_COLOR = "steelblue"


def compute_pr(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool TP/FP/FN records across sensors, return (thresholds, precision, recall)."""
    pooled = (
        pl.DataFrame(records)
        .group_by("threshold")
        .agg(pl.col("tp").sum(), pl.col("fp").sum(), pl.col("fn").sum())
        .sort("threshold")
    )
    tp = pooled["tp"].to_numpy().astype(float)
    fp = pooled["fp"].to_numpy().astype(float)
    fn = pooled["fn"].to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
    return pooled["threshold"].to_numpy(), precision, recall


def plot_pr_curve(pr_curve_data: Path) -> Figure:
    """Plot per-sensor precision-recall curves for denoised whale detection.

    Args:
        pr_curve_data: Path to CSV with columns sensor, threshold, tp, fp, fn.

    Returns:
        Figure with 1x3 subplots, one per sensor.
    """
    records = pl.read_csv(pr_curve_data).to_dicts()

    fig, axes = plt.subplots(1, len(_SENSORS), figsize=(2.5 * 3, 1.5), sharey=True)

    for ax, sensor in zip(axes, _SENSORS):
        sensor_records = [r for r in records if r["sensor"] == sensor]
        thresholds, precision, recall = compute_pr(sensor_records)

        valid = np.isfinite(precision) & np.isfinite(recall)
        rec_v, prec_v = recall[valid], precision[valid]
        order = np.argsort(rec_v)
        pr_auc = auc(rec_v[order], prec_v[order])
        ax.plot(rec_v, prec_v, color=_COLOR, lw=2, label=f"AUC = {pr_auc:.3f}")

        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                np.nan,
            )
        f1_idx = int(np.nanargmax(f1))
        if np.isfinite(precision[f1_idx]) and np.isfinite(recall[f1_idx]):
            ax.scatter(
                recall[f1_idx],
                precision[f1_idx],
                color="k",
                zorder=5,
                s=60,
                marker="*",
                edgecolors="k",
                linewidths=0.5,
                label=f"Optimal F1 thresh. = {thresholds[f1_idx]:.2f})",
            )

        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.4, 1.02)
        ax.grid()
        ax.legend(fontsize=6, loc="lower left", framealpha=1.0, title=_SITES[sensor])

    axes[0].set_ylabel("Precision")
    axes[0].set_xlabel("Recall")
    fig.tight_layout()
    return fig
