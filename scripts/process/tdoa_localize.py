#!/usr/bin/env python3
import numpy as np
import polars as pl
import pymap3d as pm

from vineyard.config import get_path
from vineyard.tdoa import tdoa as tdoa_solve


def compute_northing_easting(lat, lon, lat0, lon0):
    """Convert latitude/longitude to local ENU coordinates."""
    easting, northing, _ = pm.geodetic2enu(lat, lon, 0, lat0, lon0, 0)
    return easting, northing


def compute_lat_lon(easting, northing, lat0, lon0):
    """Convert local ENU coordinates back to latitude/longitude."""
    lat, lon, _ = pm.enu2geodetic(easting, northing, 0, lat0, lon0, 0)
    return lat, lon


def calculate_bearing(
    reference_easting, reference_northing, target_easting, target_northing
):
    """
    Calculate the bearing from a reference sensor to a target point.

    Bearing is measured clockwise from North (0 degrees), with East at 90 degrees,
    South at 180 degrees, and West at 270 degrees.

    Parameters:
    -----------
    reference_easting: float
        The easting coordinate of the reference sensor
    reference_northing: float
        The northing coordinate of the reference sensor
    target_easting: float
        The easting coordinate of the target point
    target_northing: float
        The northing coordinate of the target point

    Returns:
    --------
    float: The bearing in degrees, between 0 and 360
    """
    # Calculate the differences in eastings and northings
    delta_e = target_easting - reference_easting
    delta_n = target_northing - reference_northing

    # Calculate the bearing using arctan2
    # arctan2 returns angle in radians from the positive x-axis
    # We adjust to get bearing from North, clockwise
    bearing = np.degrees(np.arctan2(delta_e, delta_n))

    # Normalize to [0, 360) degrees
    bearing = (bearing + 360) % 360

    return bearing


def get_sensor_positions():
    """Load sensor positions from equipment config and compute ENU coordinates."""
    df = pl.read_csv(get_path("equipment_config"))

    # Use VLA2 as reference point for coordinate transformation
    lat0, lon0 = (
        df.filter(pl.col("equipment") == "VLA2")
        .select(["latitude", "longitude"])
        .row(0)
    )

    df = df.with_columns(
        pl.struct("latitude", "longitude")
        .map_elements(
            lambda cols: dict(
                zip(
                    ("easting", "northing"),
                    compute_northing_easting(
                        cols["latitude"], cols["longitude"], lat0, lon0
                    ),
                )
            ),
            return_dtype=pl.Struct(
                [
                    pl.Field("easting", pl.Float64),
                    pl.Field("northing", pl.Float64),
                ]
            ),
        )
        .alias("result")
    ).unnest("result")

    # Return sensor positions (excluding last row which is reference) and reference lat/lon
    sensor_eastings = df["easting"].to_list()
    sensor_northings = df["northing"].to_list()

    return sensor_eastings, sensor_northings, lat0, lon0


def localize_tdoa_data(tdoa_csv_path: str) -> pl.DataFrame:
    """
    Load TDOA data from CSV and compute locations using TDOA localization.

    Parameters:
    -----------
    tdoa_csv_path : str
        Path to the TDOA CSV file containing columns: timestamp, 3dvha, vla1, vla2

    Returns:
    --------
    pl.DataFrame
        DataFrame with original TDOA data plus computed location columns:
        - easting: East coordinate in meters
        - northing: North coordinate in meters
        - latitude: Latitude in degrees
        - longitude: Longitude in degrees
        - 3dvha_brg: True bearing from 3DVHA sensor to target (degrees, 0-360)
        - vla1_brg: True bearing from VLA1 sensor to target (degrees, 0-360)
        - vla2_brg: True bearing from VLA2 sensor to target (degrees, 0-360)
    """
    # Load TDOA data
    df = pl.read_csv(tdoa_csv_path)

    # Get sensor positions and reference coordinates
    sensor_eastings, sensor_northings, lat0, lon0 = get_sensor_positions()

    # Convert to km for numerical stability
    denom = 1000.0
    sensor_eastings_km = [e / denom for e in sensor_eastings]
    sensor_northings_km = [n / denom for n in sensor_northings]

    # Sound speed in water (m/s converted to km/s)
    speed = 1500.0 / denom

    # Initial guess at centroid of sensors
    xp = np.mean(sensor_eastings_km)
    yp = np.mean(sensor_northings_km)

    # Compute localization for each row
    eastings = []
    northings = []
    latitudes = []
    longitudes = []
    bearings_3dvha = []
    bearings_vla1 = []
    bearings_vla2 = []

    for row in df.iter_rows(named=True):
        # Extract TDOA values (in seconds)
        t0 = row["3dvha"]
        t1 = row["vla1"]
        t2 = row["vla2"]

        # Solve TDOA localization
        x_km, y_km, _ = tdoa_solve(
            sensor_eastings_km,
            sensor_northings_km,
            [t0, t1, t2],
            speed,
            xp,
            -yp,  # Note: sign flip for yp as in original code
        )

        # Convert back to meters
        easting_m = x_km * denom
        northing_m = y_km * denom

        # Convert to lat/lon
        lat, lon = compute_lat_lon(easting_m, northing_m, lat0, lon0)

        # Calculate bearings from each sensor to the target location
        brg_3dvha = calculate_bearing(
            sensor_eastings[0], sensor_northings[0], easting_m, northing_m
        )
        brg_vla1 = calculate_bearing(
            sensor_eastings[1], sensor_northings[1], easting_m, northing_m
        )
        brg_vla2 = calculate_bearing(
            sensor_eastings[2], sensor_northings[2], easting_m, northing_m
        )

        eastings.append(easting_m)
        northings.append(northing_m)
        latitudes.append(lat)
        longitudes.append(lon)
        bearings_3dvha.append(brg_3dvha)
        bearings_vla1.append(brg_vla1)
        bearings_vla2.append(brg_vla2)

    return df.with_columns(
        [
            pl.Series("easting", eastings),
            pl.Series("northing", northings),
            pl.Series("latitude", latitudes),
            pl.Series("longitude", longitudes),
            pl.Series("3dvha_brg", bearings_3dvha),
            pl.Series("vla1_brg", bearings_vla1),
            pl.Series("vla2_brg", bearings_vla2),
        ]
    )


def correct_ambiguous_bearings(df):
    unamb_bearings = df.filter(pl.col("vla1_brg") > 90)
    amb_bearings = df.filter(pl.col("vla1_brg") < 90)
    amb_bearings = amb_bearings.with_columns(
        pl.col("vla1_brg").add(2 * (90 - pl.col("vla1_brg")))
    )
    return (
        pl.concat([unamb_bearings, amb_bearings])
        .filter(pl.col("vla1_brg") < 175, pl.col("vla1_brg") > 155)
        .sort("timestamp")
    )


def main():
    """Main entry point for TDOA localization."""
    tdoa_csv = get_path("tdoa_data") / "tdoa.csv"
    df = localize_tdoa_data(str(tdoa_csv))
    corr_df = correct_ambiguous_bearings(df)

    dw = corr_df["vla1_brg"].diff().to_numpy()
    dt = corr_df.cast({"timestamp": pl.Datetime})["timestamp"].diff().to_numpy() / 1e6

    dwdt = dw / dt.astype(int)

    print(np.nanmax(dwdt))
    print(np.nanmean(dwdt))
    

    corr_df = corr_df.with_columns(pl.Series("dwdt", dwdt))

    print(corr_df)
    output_path = get_path("tdoa_data") / "tdoa_with_locations.csv"
    corr_df.write_csv(output_path)
    print(f"\nResults saved to: {output_path}")

    return corr_df


if __name__ == "__main__":
    main()
