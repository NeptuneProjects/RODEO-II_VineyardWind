from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def main():
    file_raw = Path("data/acoustic/tdoa/tdoa_localization_raw.csv")
    file_clean = Path("data/acoustic/tdoa/localization.csv")
    df_raw = pl.read_csv(file_raw, try_parse_dates=True)
    df_clean = pl.read_csv(file_clean, try_parse_dates=True)
    out_file = Path("reports/figures/ambig_brg.png")

    print(df_raw)

    fig, axes = plt.subplots(ncols=2, figsize=(14, 3))
    ax = axes[0]
    ax.scatter(df_raw["timestamp"], df_raw["vla1_brg"])
    ax.set_title("Ambiguous Bearing Estimates")

    ax = axes[1]
    ax.scatter(df_clean["timestamp"], df_clean["vla1_brg"])
    ax.set_title("Corrected Bearing Estimates")

    for ax in axes:
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(
            locator, offset_formats=["%Y-%b-%d"] * 6
        )
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        ax.axhline(180, color="k", linestyle="--")
        ax.set_ylim(0, 360)
        ax.set_yticks(np.linspace(0, 360, 5))
        ax.grid()
        ax.set_ylabel("Bearing (degrees)")

    fig.savefig(out_file, dpi=300, bbox_inches="tight")

    return


if __name__ == "__main__":
    main()
