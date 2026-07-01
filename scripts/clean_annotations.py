#!/usr/bin/env python3
from pathlib import Path

import polars as pl


def main():
    cutoff_time_s = 16.0
    file = Path("data/acoustic/annotations.csv")
    df = (
        pl.read_csv(file, try_parse_dates=True)
        .sort(["sensor", "time"])
        .with_columns(
            (pl.col("time").diff().over("sensor").dt.total_microseconds() / 1e6).alias(
                "time_diff_s"
            )
        )
        .with_columns(
            pl.when(pl.col("time_diff_s") > cutoff_time_s)
            .then(None)
            .otherwise(pl.col("time_diff_s"))
            .alias("time_diff_s")
        )
    )
    df.write_csv(file.with_stem("annotations_cleaned"))
    print(df.select(["time_diff_s"]).describe())


if __name__ == "__main__":
    main()
