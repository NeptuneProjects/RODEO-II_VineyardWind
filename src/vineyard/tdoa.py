from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pymap3d as pm
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from scipy.optimize import leastsq

from vineyard.readers import (
    read_distances,
    read_sensor_positions,
    read_whale_call_times,
)


class LocalizationConfig(BaseModel):
    """Configuration for TDOA estimation."""

    whale_call_data: Path = "data/acoustic/whale_detections.csv"
    distance_lut: Path = "data/distances.csv"
    sensor_data: Path = "data/sensors.csv"
    tdoa_file: Path = "data/acoustic/tdoa/tdoa.csv"
    localization_file: Path = "data/acoustic/tdoa/localization.csv"
    reference_site: str = "vla1"
    ambiguity_lower_bound: float = 90.0
    ambiguity_upper_bound: float = 270.0
    dwdt_threshold: float = 0.25  # degrees per second
    smoothing_window: int = 10
    speed_upper_bound: float = 35.0  # m/s


@dataclass
class CorrelatedDetection:
    """A detection event correlated across multiple sites."""

    site_3dvha: np.datetime64 | None = None
    site_vla1: np.datetime64 | None = None
    site_vla2: np.datetime64 | None = None
    reference_site: str = ""

    def is_complete(self) -> bool:
        """Check if detection is present at all three sites."""
        return all(
            [
                self.site_3dvha is not None,
                self.site_vla1 is not None,
                self.site_vla2 is not None,
            ]
        )

    def num_sites(self) -> int:
        """Count number of sites with detections."""
        return sum(
            [
                self.site_3dvha is not None,
                self.site_vla1 is not None,
                self.site_vla2 is not None,
            ]
        )


def compute_angular_velocity(
    df: pl.DataFrame, dwdt_threshold: float, smoothing_window: int
) -> pl.DataFrame:
    """Compute angular velocity from bearing changes over time.

    Args:
        df: DataFrame containing bearing columns (e.g., 'vla1_brg') and 'unix_time_us'
        dwdt_threshold: Threshold for angular velocity in degrees per second
        smoothing_window: Window size for smoothing angular velocity (number of calls)
    Returns:
        DataFrame with new column 'angular_velocity' representing the rate
        of change of bearing in degrees per second. Values with absolute
        angular velocity > dwdt_threshold deg/s are set to null (NaN).
    """
    return smooth_angular_velocity(
        df.with_columns(
            (df["vla1_brg"].diff() / (df["unix_time_us"].diff() / 1e6)).alias(
                "angular_velocity"
            )
        ).with_columns(
            pl.when(pl.col("angular_velocity").abs() > dwdt_threshold)
            .then(None)
            .otherwise(pl.col("angular_velocity"))
            .alias("angular_velocity")
        ),
        smoothing_window,
    )


def compute_bearing(
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


def compute_lat_lon(easting, northing, lat0, lon0):
    """Convert local ENU coordinates back to latitude/longitude."""
    lat, lon, _ = pm.enu2geodetic(easting, northing, 0, lat0, lon0, 0)
    return lat, lon


def compute_time_gates(
    distance_lut: Path, ref_sound_speed: float = 1500.0
) -> dict[tuple[str, str], np.ndarray]:
    d_3dvha_vla1, d_3dvha_vla2, d_vla1_vla2 = read_distances(distance_lut)
    t_3dvha_vla1 = d_3dvha_vla1 / ref_sound_speed
    t_3dvha_vla2 = d_3dvha_vla2 / ref_sound_speed
    t_vla1_vla2 = d_vla1_vla2 / ref_sound_speed
    return {
        ("3dvha", "vla1"): t_3dvha_vla1,
        ("3dvha", "vla2"): t_3dvha_vla2,
        ("vla1", "vla2"): t_vla1_vla2,
    }


def compute_whale_range(
    angular_speed: ArrayLike | float, tangential_speed: float = 35.0
) -> NDArray | float:
    """Compute the whale range given angular speed and max tangential speed.

    Args:
        angular_speed: Angular speed in degrees per second.
        tangential_speed: Tangential speed in km/h.

    Returns:
        Whale range in kilometers.
    """
    return (tangential_speed / 3600) / np.deg2rad(angular_speed)


def correct_ambiguous_bearings(
    df: pl.DataFrame, lower_bound: float, upper_bound: float
) -> pl.DataFrame:
    """Correct ambiguous bearings in the DataFrame by reflecting them across
    the specified bounds.

    Args:
        df: DataFrame containing bearing columns to correct
        lower_bound: Lower bound of the valid bearing range (e.g., 90 degrees).
        upper_bound: Upper bound of the valid bearing range (e.g., 270 degrees).

    Returns:
        DataFrame with corrected bearing columns.
    """
    BEARING_COLUMNS = ["3dvha_brg", "vla1_brg", "vla2_brg"]

    # Filter for unambiguous bearings (between lower_bound and upper_bound for all sensors)
    unamb_bearings = df.filter(
        (pl.col("3dvha_brg") > lower_bound)
        & (pl.col("3dvha_brg") < upper_bound)
        & (pl.col("vla1_brg") > lower_bound)
        & (pl.col("vla1_brg") < upper_bound)
        & (pl.col("vla2_brg") > lower_bound)
        & (pl.col("vla2_brg") < upper_bound)
    )

    # Filter for ambiguous bearings (outside lower_bound-upper_bound range for any sensor)
    amb_bearings = df.filter(
        (pl.col("3dvha_brg") < lower_bound)
        | (pl.col("3dvha_brg") > upper_bound)
        | (pl.col("vla1_brg") < lower_bound)
        | (pl.col("vla1_brg") > upper_bound)
        | (pl.col("vla2_brg") < lower_bound)
        | (pl.col("vla2_brg") > upper_bound)
    )
    # breakpoint()
    # Correct ambiguous bearings for all three columns
    amb_bearings = amb_bearings.with_columns(
        [
            pl.when(pl.col(col) < lower_bound)
            .then((2 * lower_bound - pl.col(col)) % 360)
            .when(pl.col(col) > upper_bound)
            .then((2 * upper_bound - pl.col(col)) % 360)
            .otherwise(pl.col(col))
            .alias(col)
            for col in BEARING_COLUMNS
        ]
    )
    # breakpoint()

    return (
        pl.concat([unamb_bearings, amb_bearings])
        # .filter(pl.col("vla1_brg") < 175, pl.col("vla1_brg") > 155)
        .sort("timestamp")
    )


def correlate_all_references(
    times: dict[str, np.ndarray], time_gates: dict[tuple[str, str], float]
) -> dict[str, list[CorrelatedDetection]]:
    """
    Run correlation using each site as reference and return all results.

    Parameters
    ----------
    times : dict[str, np.ndarray]
        Dictionary mapping site names to arrays of detection times
    time_gates : dict[tuple[str, str], float]
        Dictionary mapping site pairs to maximum time delays in seconds

    Returns
    -------
    dict[str, list[CorrelatedDetection]]
        Dictionary mapping reference site names to their correlation results
    """
    results = {}

    for site in times.keys():
        print(f"Correlating with {site} as reference...")
        correlations = correlate_detections_triplet(
            times, time_gates, reference_site=site
        )
        results[site] = correlations

        # Print statistics
        complete = sum(1 for c in correlations if c.is_complete())
        partial = sum(1 for c in correlations if 1 < c.num_sites() < 3)
        single = sum(1 for c in correlations if c.num_sites() == 1)

        print(f"  Complete (3 sites): {complete}")
        print(f"  Partial (2 sites): {partial}")
        print(f"  Single site only: {single}")
        print(f"  Total: {len(correlations)}\n")

    return results


def correlate_detections_triplet(
    times: dict[str, np.ndarray],
    time_gates: dict[tuple[str, str], float],
    reference_site: str = "vla1",
) -> list[CorrelatedDetection]:
    """
    Correlate detections across three sites using a greedy matching approach.

    This algorithm iterates through detections at a reference site and finds
    corresponding detections at the other sites within the specified time gates.

    Parameters
    ----------
    times : dict[str, np.ndarray]
        Dictionary mapping site names to arrays of detection times
    time_gates : dict[tuple[str, str], float]
        Dictionary mapping site pairs to maximum time delays in seconds
        Keys should be tuples like ('3dvha', 'vla1')
    reference_site : str, optional
        Which site to use as reference (default: '3dvha')

    Returns
    -------
    list[CorrelatedDetection]
        List of correlated detections
    """
    # Get site names
    sites = list(times.keys())
    if reference_site not in sites:
        raise ValueError(f"Reference site {reference_site} not in times dictionary")

    other_sites = [s for s in sites if s != reference_site]
    if len(other_sites) != 2:
        raise ValueError("This function expects exactly 3 sites")

    site_a, site_b = other_sites

    # Track which detections have been used
    used_a = set()
    used_b = set()

    correlations = []

    # Iterate through reference site detections
    for ref_time in times[reference_site]:

        # Find matches at site A
        matches_a = find_matches_in_window(
            ref_time,
            times[site_a],
            time_gates.get(
                (reference_site, site_a), time_gates.get((site_a, reference_site))
            ),
            used_a,
        )

        # Find matches at site B
        matches_b = find_matches_in_window(
            ref_time,
            times[site_b],
            time_gates.get(
                (reference_site, site_b), time_gates.get((site_b, reference_site))
            ),
            used_b,
        )

        # If we have matches at both sites, need to pair them
        if matches_a and matches_b:
            # Strategy: pair closest matches that also satisfy the A-B time gate
            time_gate_ab = time_gates.get(
                (site_a, site_b), time_gates.get((site_b, site_a))
            )

            for idx_a in matches_a:
                time_a = times[site_a][idx_a]

                # Check which B matches are compatible with this A
                compatible_b = []
                for idx_b in matches_b:
                    if idx_b in used_b:
                        continue
                    time_b = times[site_b][idx_b]
                    time_diff = abs((time_b - time_a) / np.timedelta64(1, "s"))
                    if time_diff <= time_gate_ab:
                        compatible_b.append(idx_b)

                if compatible_b:
                    # Choose closest B to A
                    best_idx_b = min(
                        compatible_b, key=lambda idx: abs(times[site_b][idx] - time_a)
                    )

                    # Create correlation
                    corr = CorrelatedDetection(reference_site=reference_site)
                    corr.__dict__[f"site_{reference_site}"] = ref_time
                    corr.__dict__[f"site_{site_a}"] = times[site_a][idx_a]
                    corr.__dict__[f"site_{site_b}"] = times[site_b][best_idx_b]

                    correlations.append(corr)
                    used_a.add(idx_a)
                    used_b.add(best_idx_b)
                    break  # Move to next reference time

        elif matches_a:
            # Only match at site A
            idx_a = matches_a[0]  # Take first/closest match
            corr = CorrelatedDetection(reference_site=reference_site)
            corr.__dict__[f"site_{reference_site}"] = ref_time
            corr.__dict__[f"site_{site_a}"] = times[site_a][idx_a]
            correlations.append(corr)
            used_a.add(idx_a)

        elif matches_b:
            # Only match at site B
            idx_b = matches_b[0]
            corr = CorrelatedDetection(reference_site=reference_site)
            corr.__dict__[f"site_{reference_site}"] = ref_time
            corr.__dict__[f"site_{site_b}"] = times[site_b][idx_b]
            correlations.append(corr)
            used_b.add(idx_b)

        else:
            # No matches - detection only at reference site
            corr = CorrelatedDetection(reference_site=reference_site)
            corr.__dict__[f"site_{reference_site}"] = ref_time
            correlations.append(corr)

    return correlations


def correlations_to_dataframe(
    correlations: list[CorrelatedDetection], reference_site: str = "3dvha"
) -> pl.DataFrame:
    """
    Convert correlated detections to a Polars DataFrame with TDOA values.

    The DataFrame contains a timestamp column (from the reference site) and
    TDOA columns for each site (time difference in seconds from reference).

    Parameters
    ----------
    correlations : List[CorrelatedDetection]
        List of correlated detections (should be complete triplets)
    reference_site : str, optional
        Site to use as time reference (default: '3dvha')

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - timestamp: Reference site detection time
        - 3dvha: TDOA in seconds (0.0 if reference site)
        - vla1: TDOA in seconds
        - vla2: TDOA in seconds

    Notes
    -----
    TDOA values are computed as: site_time - reference_time
    Positive values mean the signal arrived later at that site.
    """
    timestamps = []
    tdoa_3dvha = []
    tdoa_vla1 = []
    tdoa_vla2 = []

    for corr in correlations:
        # Get reference time
        ref_time = getattr(corr, f"site_{reference_site}")

        if ref_time is None:
            # Skip if reference site is missing
            continue

        # Convert numpy datetime64 to int64 (microseconds since epoch) for Polars
        timestamps.append(ref_time.astype("datetime64[us]").astype("int64"))

        # Compute TDOA for each site
        # TDOA = site_time - reference_time (in seconds)
        for site_name, tdoa_list in [
            ("3dvha", tdoa_3dvha),
            ("vla1", tdoa_vla1),
            ("vla2", tdoa_vla2),
        ]:
            site_time = getattr(corr, f"site_{site_name}")

            if site_time is None:
                # Should not happen for complete triplets
                tdoa_list.append(None)
            else:
                # Compute time difference in seconds
                time_diff = (site_time - ref_time) / np.timedelta64(1, "s")
                tdoa_list.append(float(time_diff))

    # Create DataFrame with timestamps as integers, then cast to datetime
    return pl.DataFrame(
        {
            "timestamp": pl.Series(timestamps, dtype=pl.Int64),
            "3dvha": pl.Series(tdoa_3dvha, dtype=pl.Float64),
            "vla1": pl.Series(tdoa_vla1, dtype=pl.Float64),
            "vla2": pl.Series(tdoa_vla2, dtype=pl.Float64),
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


def estimate_range(df: pl.DataFrame, tangential_speed: float = 35.0) -> pl.DataFrame:
    """Estimate whale range from smoothed angular velocity.

    Args:
        df: DataFrame containing 'angular_velocity_smoothed' column
        tangential_speed: Maximum tangential speed of whale in km/h (default: 35.0)

    Returns:
        DataFrame with new column 'whale_range_km' containing estimated range in kilometers.
    """
    ranges = compute_whale_range(
        np.abs(df["angular_velocity_smoothed"].to_numpy()),
        tangential_speed=tangential_speed,
    )
    return df.with_columns(
        pl.Series("whale_range_km", ranges).fill_nan(None).cast(pl.Float64)
    )


def estimate_tdoa(
    whale_call_data: Path, distance_lut: Path, reference_site: str
) -> pl.DataFrame:
    """Estimate TDOA from whale call detection times and distance LUT.

    Args:
        whale_call_data: Path to whale call detection times.
        distance_lut: Path to distance lookup table.

    Returns:
        DataFrame with correlated detections and TDOA values.
    """
    times = read_whale_call_times(whale_call_data)
    time_gates = compute_time_gates(distance_lut)
    correlations = correlate_all_references(times, time_gates)
    merged_results = merge_correlations(correlations)
    return correlations_to_dataframe(
        merged_results, reference_site=reference_site
    ).with_columns(pl.col("timestamp").dt.epoch(time_unit="us").alias("unix_time_us"))


def find_matches_in_window(
    reference_time: np.datetime64,
    search_times: np.ndarray,
    max_delay: float,
    used_indices: set,
) -> list[int]:
    """
    Find all matching times within a time gate window.

    Parameters
    ----------
    reference_time : np.datetime64
        Reference detection time
    search_times : np.ndarray
        Array of detection times to search
    max_delay : float
        Maximum time delay in seconds (time gate)
    used_indices : set
        Set of indices already matched (to avoid double-counting)

    Returns
    -------
    List[int]
        Indices of matching detections in search_times
    """
    # Convert max_delay to timedelta
    time_gate = np.timedelta64(int(max_delay * 1e6), "us")

    # Find times within the window [reference_time - gate, reference_time + gate]
    lower_bound = reference_time - time_gate
    upper_bound = reference_time + time_gate

    # Boolean mask for times in window
    in_window = (search_times >= lower_bound) & (search_times <= upper_bound)

    # Get indices and filter out already used ones
    candidate_indices = np.where(in_window)[0]
    available_indices = [idx for idx in candidate_indices if idx not in used_indices]

    return available_indices


def functions(x0, y0, x1, y1, x2, y2, d01, d02, d12):
    """Given observers at (x0, y0), (x1, y1), (x2, y2) and TDOA between
    observers d01, d02, d12, this closure returns a function that evaluates
    the system of three hyperbolae for given event x, y.
    """

    def fn(args):
        x, y = args
        a = (
            np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
            - np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
            - d01
        )
        b = (
            np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0))
            - np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
            - d02
        )
        c = (
            np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0))
            - np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
            - d12
        )
        return [a, b, c]

    return fn


def localize(config: LocalizationConfig) -> None:
    """Run the full TDOA localization pipeline and save results to CSV.

    Args:
        config: TDOAConfig instance containing all necessary configuration
            parameters and file paths.
    """
    df = estimate_tdoa(
        config.whale_call_data, config.distance_lut, config.reference_site
    )
    df = localize_tdoa_data(df, config.sensor_data)
    df = correct_ambiguous_bearings(
        df, config.ambiguity_lower_bound, config.ambiguity_upper_bound
    )
    df = compute_angular_velocity(
        df,
        config.dwdt_threshold,
        config.smoothing_window,
    )
    df = estimate_range(
        df,
        config.speed_upper_bound,
    )
    df.write_csv(config.localization_file)


def localize_tdoa_data(df: pl.DataFrame, sensor_data: Path) -> pl.DataFrame:
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
    # Get sensor positions and reference coordinates
    sensor_eastings, sensor_northings, lat0, lon0 = read_sensor_positions(sensor_data)

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
        brg_3dvha = compute_bearing(
            sensor_eastings[0], sensor_northings[0], easting_m, northing_m
        )
        brg_vla1 = compute_bearing(
            sensor_eastings[1], sensor_northings[1], easting_m, northing_m
        )
        brg_vla2 = compute_bearing(
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


def jacobian(x0, y0, x1, y1, x2, y2, d01, d02, d12):
    def fn(args):
        x, y = args
        adx = (x - x1) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0)) - (
            x - x0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        bdx = (x - x2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            x - x0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        cdx = (x - x2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            x - x1
        ) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
        ady = (y - y1) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0)) - (
            y - y0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        bdy = (y - y2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            y - y0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        cdy = (y - y2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            y - y1
        ) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))

        return [[adx, ady], [bdx, bdy], [cdx, cdy]]

    return fn


def merge_correlations(
    results: dict[str, list[CorrelatedDetection]], tolerance: float = 0.001
) -> list[CorrelatedDetection]:
    """
    Merge correlations from different reference sites to get complete triplets only.

    This function:
    1. Collects all correlations from different reference site runs
    2. Merges complementary partial matches to form complete triplets
    3. Deduplicates complete triplets within the time tolerance
    4. Returns only complete 3-site detections

    Args:
        results: Results from correlate_all_references
        tolerance: Time tolerance in seconds for considering detections as duplicates

    Returns:
        List of unique complete triplets only (all 3 sites present)
    """
    # Convert tolerance to timedelta
    time_tolerance = np.timedelta64(int(tolerance * 1e6), "us")

    # Collect all correlations
    all_correlations = []
    for correlations in results.values():
        all_correlations.extend(correlations)

    # Sort by completeness (3-site matches first), then by earliest time
    def get_earliest_time(corr: CorrelatedDetection) -> np.datetime64:
        times = [corr.site_3dvha, corr.site_vla1, corr.site_vla2]
        valid_times = [t for t in times if t is not None]
        return min(valid_times) if valid_times else np.datetime64("NaT")

    all_correlations.sort(key=lambda c: (-c.num_sites(), get_earliest_time(c)))

    # Deduplicate by checking if detections are within tolerance
    def are_duplicates(c1: CorrelatedDetection, c2: CorrelatedDetection) -> bool:
        """Check if two correlations represent the same detection event."""
        matches = 0
        sites_to_check = ["site_3dvha", "site_vla1", "site_vla2"]

        for site in sites_to_check:
            t1 = getattr(c1, site)
            t2 = getattr(c2, site)

            # Both have detection at this site
            if t1 is not None and t2 is not None:
                if abs(t1 - t2) <= time_tolerance:
                    matches += 1
                else:
                    # Times at same site differ too much - not duplicates
                    return False

        # Consider duplicates if at least one site matches within tolerance
        return matches >= 1

    def merge_two_correlations(
        c1: CorrelatedDetection, c2: CorrelatedDetection
    ) -> CorrelatedDetection:
        """Merge two correlations by taking non-None values from both."""
        merged = CorrelatedDetection()

        # Prefer times from the more complete detection
        if c1.num_sites() >= c2.num_sites():
            primary, secondary = c1, c2
        else:
            primary, secondary = c2, c1

        merged.site_3dvha = primary.site_3dvha or secondary.site_3dvha
        merged.site_vla1 = primary.site_vla1 or secondary.site_vla1
        merged.site_vla2 = primary.site_vla2 or secondary.site_vla2
        merged.reference_site = primary.reference_site

        return merged

    # Deduplicate and merge
    unique_correlations = []
    skip_indices = set()

    for i, corr in enumerate(all_correlations):
        if i in skip_indices:
            continue

        # Try to merge with later correlations
        merged = corr
        for j in range(i + 1, len(all_correlations)):
            if j in skip_indices:
                continue

            if are_duplicates(merged, all_correlations[j]):
                # Merge and mark as processed
                merged = merge_two_correlations(merged, all_correlations[j])
                skip_indices.add(j)

        unique_correlations.append(merged)

    # Filter to only complete triplets
    complete_triplets = [c for c in unique_correlations if c.is_complete()]

    # Final sort by earliest time
    complete_triplets.sort(key=get_earliest_time)

    return complete_triplets


def smooth_angular_velocity(df: pl.DataFrame, window_size: int = 5) -> pl.DataFrame:
    """Smooth angular velocity using a rolling mean.

    Args:
        df: DataFrame containing 'angular_velocity' column
        window_size: Size of the rolling window (default: 5)

    Returns:
        DataFrame with new column 'angular_velocity_smoothed' containing the smoothed values.
    """
    return df.with_columns(
        pl.col("angular_velocity")
        .rolling_mean(window_size)
        .alias("angular_velocity_smoothed")
    )


def tdoa_solve(x, y, t, speed, xp, yp) -> tuple[float, float, Callable]:
    x0, y0, t0 = x[0], y[0], t[0]
    x1, y1, t1 = x[1], y[1], t[1]
    x2, y2, t2 = x[2], y[2], t[2]

    F = functions(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )
    J = jacobian(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )

    pos, _ = leastsq(F, x0=[xp, yp], Dfun=J)
    return pos[0], pos[1], F
