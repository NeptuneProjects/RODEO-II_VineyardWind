from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pymap3d as pm
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from scipy.optimize import leastsq

from vineyard.process import compute_sigma_tdoa_per_detection
from vineyard.readers import (
    read_distances,
    read_sensor_positions,
    read_whale_call_times,
)


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


class LocalizationConfig(BaseModel):
    """Configuration for TDOA estimation."""

    whale_call_data: Path = "data/acoustic/whale_detections.csv"
    distance_lut: Path = "data/distances.csv"
    sensor_data: Path = "data/sensors.csv"
    tdoa_file: Path = "data/acoustic/tdoa/tdoa.csv"
    localization_file: Path = "data/acoustic/tdoa/localization.csv"
    raw_localization_file: Path = "data/acoustic/tdoa/localization_raw.csv"
    reference_site: str = "vla1"
    ambiguity_lower_bound: float = 90.0
    ambiguity_upper_bound: float = 270.0
    # --- Bearing uncertainty (optional) ---
    # Provide pc_data_path + template_duration_s + f_low_hz + f_high_hz to
    # enable per-detection SNR-based σ_TDOA estimation.  Omit all to skip
    # bearing uncertainty columns, or set var_tdoa (s²) for a constant fallback.
    pc_data_path: Path | None = None
    channel: int = 0
    template_duration_s: float | None = None
    f_low_hz: float | None = None
    f_high_hz: float | None = None
    filt_type: str | None = None
    filt_freq: list[float] | float | None = None
    snr_window_s: float = 5.0
    noise_correction: str = "rayleigh"
    var_tdoa: float | None = None  # constant TDOA variance fallback (s²) if pc_data_path not set
    source_north: bool = True  # sources are north of the array (confirmed by external sensor)


def compute_bearing(
    reference_easting, reference_northing, target_easting, target_northing
):
    """Calculate the bearing from a reference sensor to a target point.

    Bearing is measured clockwise from North (0 degrees), with East at 90 degrees,
    South at 180 degrees, and West at 270 degrees.

    Args:
        reference_easting: The easting coordinate of the reference sensor.
        reference_northing: The northing coordinate of the reference sensor.
        target_easting: The easting coordinate of the target point.
        target_northing: The northing coordinate of the target point.

    Returns:
        The bearing in degrees, between 0 and 360.
    """
    # Calculate the differences in eastings and northings
    delta_e = target_easting - reference_easting
    delta_n = target_northing - reference_northing

    # Calculate the bearing using arctan2
    # arctan2 returns angle in radians from the positive x-axis
    # Adjust to get bearing from North, clockwise
    bearing = np.degrees(np.arctan2(delta_e, delta_n))

    # Normalize to [0, 360) degrees
    bearing = (bearing + 360) % 360

    return bearing


def compute_lat_lon(easting, northing, lat0, lon0):
    """Convert local ENU coordinates back to latitude/longitude."""
    lat, lon, _ = pm.enu2geodetic(easting, northing, 0, lat0, lon0, 0)
    return lat, lon


def compute_time_gates(
    distance_lut: Path, ref_sound_speed: float = 1500.0, design_factor: float = 1.1
) -> dict[tuple[str, str], np.ndarray]:
    d_3dvha_vla1, d_3dvha_vla2, d_vla1_vla2 = read_distances(distance_lut)
    t_3dvha_vla1 = design_factor * d_3dvha_vla1 / ref_sound_speed
    t_3dvha_vla2 = design_factor * d_3dvha_vla2 / ref_sound_speed
    t_vla1_vla2 = design_factor * d_vla1_vla2 / ref_sound_speed
    return {
        ("3dvha", "vla1"): t_3dvha_vla1,
        ("3dvha", "vla2"): t_3dvha_vla2,
        ("vla1", "vla2"): t_vla1_vla2,
    }


def correct_ambiguous_bearings(
    df: pl.DataFrame,
    lower_bound: float,
    upper_bound: float,
    source_north: bool = False,
) -> pl.DataFrame:
    """Correct ambiguous bearings by reflecting them into the valid half-space.

    The reflection formula for an E-W array is (180° − b) % 360°, which maps
    each bearing to its mirror image across the array axis.

    Args:
        df: DataFrame containing bearing columns to correct.
        lower_bound: Boundary at the eastern end of the array axis (e.g., 90°).
        upper_bound: Boundary at the western end of the array axis (e.g., 270°).
        source_north: If True, sources are north of the array; valid bearings
            are outside (lower_bound, upper_bound) and south-half-space bearings
            are reflected northward. If False (default), sources are south;
            valid bearings are inside (lower_bound, upper_bound).

    Returns:
        DataFrame with corrected bearing columns.
    """
    # Include doa_brg when present (far-field pipeline output)
    BEARING_COLUMNS = [
        c for c in ["doa_brg", "3dvha_brg", "vla1_brg", "vla2_brg"] if c in df.columns
    ]

    if source_north:
        # Valid = north half-space: bearing < lower_bound or > upper_bound
        # Ambiguous = south half-space: bearing in (lower_bound, upper_bound)
        in_south = (
            (pl.col("3dvha_brg") > lower_bound) & (pl.col("3dvha_brg") < upper_bound)
        ) | (
            (pl.col("vla1_brg") > lower_bound) & (pl.col("vla1_brg") < upper_bound)
        ) | (
            (pl.col("vla2_brg") > lower_bound) & (pl.col("vla2_brg") < upper_bound)
        )
        unamb_bearings = df.filter(~in_south)
        amb_bearings = df.filter(in_south)
        amb_bearings = amb_bearings.with_columns(
            [
                pl.when((pl.col(col) > lower_bound) & (pl.col(col) < upper_bound))
                .then((180.0 - pl.col(col)) % 360.0)
                .otherwise(pl.col(col))
                .alias(col)
                for col in BEARING_COLUMNS
            ]
        )
    else:
        # Valid = south half-space: bearing in (lower_bound, upper_bound)
        # Ambiguous = north half-space: bearing < lower_bound or > upper_bound
        unamb_bearings = df.filter(
            (pl.col("3dvha_brg") > lower_bound)
            & (pl.col("3dvha_brg") < upper_bound)
            & (pl.col("vla1_brg") > lower_bound)
            & (pl.col("vla1_brg") < upper_bound)
            & (pl.col("vla2_brg") > lower_bound)
            & (pl.col("vla2_brg") < upper_bound)
        )
        amb_bearings = df.filter(
            (pl.col("3dvha_brg") < lower_bound)
            | (pl.col("3dvha_brg") > upper_bound)
            | (pl.col("vla1_brg") < lower_bound)
            | (pl.col("vla1_brg") > upper_bound)
            | (pl.col("vla2_brg") < lower_bound)
            | (pl.col("vla2_brg") > upper_bound)
        )
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

    return pl.concat([unamb_bearings, amb_bearings]).sort("timestamp")


def correlate_all_references(
    times: dict[str, np.ndarray], time_gates: dict[tuple[str, str], float]
) -> dict[str, list[CorrelatedDetection]]:
    """Run correlation using each site as reference and return all results.

    Args:
        times: Dictionary mapping site names to arrays of detection times.
        time_gates: Dictionary mapping site pairs to maximum time delays in seconds.

    Returns:
        Dictionary mapping reference site names to their correlation results.
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
    """Correlate detections across three sites using a greedy matching approach.

    This algorithm iterates through detections at a reference site and finds
    corresponding detections at the other sites within the specified time gates.

    Args:
        times: Dictionary mapping site names to arrays of detection times.
        time_gates: Dictionary mapping site pairs to maximum time delays in seconds.
            Keys should be tuples like ('3dvha', 'vla1').
        reference_site: Which site to use as reference. Defaults to 'vla1'.

    Returns:
        List of correlated detections.
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


def correlate_detections_triplet_gated(
    times: dict[str, np.ndarray],
    time_gates: dict[tuple[str, str], float],
    reference_site: str,
) -> list[CorrelatedDetection]:
    """Correlate detections anchored on a single reference site, returning only
    complete triplets.

    Unlike ``correlate_detections_triplet``, this function:

    - Uses a single user-specified reference site (no multi-reference voting or
      merging step).
    - Discards any detection not confirmed at **all three** sites.
    - Assumes detections are spaced beyond the time gate, so at most one match
      per window is expected; the first candidate is taken.
    - Still enforces the mutual A-B time-gate to guard against coincidental
      in-window matches at the two non-reference sites.

    Args:
        times: Detection times per site.
        time_gates: Maximum time delay in seconds for each site pair.
        reference_site: Site that must have a detection for any triplet to be recorded.

    Returns:
        Complete triplets only.
    """
    sites = list(times.keys())
    if reference_site not in sites:
        raise ValueError(f"Reference site '{reference_site}' not found in times dict")

    other_sites = [s for s in sites if s != reference_site]
    if len(other_sites) != 2:
        raise ValueError("Expected exactly 3 sites")

    site_a, site_b = other_sites
    used_a: set[int] = set()
    used_b: set[int] = set()
    correlations: list[CorrelatedDetection] = []

    gate_ref_a = time_gates.get(
        (reference_site, site_a), time_gates.get((site_a, reference_site))
    )
    gate_ref_b = time_gates.get(
        (reference_site, site_b), time_gates.get((site_b, reference_site))
    )
    gate_ab = time_gates.get((site_a, site_b), time_gates.get((site_b, site_a)))
    for ref_time in times[reference_site]:
        matches_a = find_matches_in_window(ref_time, times[site_a], gate_ref_a, used_a)
        matches_b = find_matches_in_window(ref_time, times[site_b], gate_ref_b, used_b)

        if not matches_a or not matches_b:
            print(
                f"Skipping reference time {ref_time} - matches A: {matches_a}, B: {matches_b}"
            )
            continue

        idx_a = matches_a[0]
        idx_b = matches_b[0]

        # Enforce mutual A–B consistency
        diff_ab = abs(
            (times[site_a][idx_a] - times[site_b][idx_b]) / np.timedelta64(1, "s")
        )
        if diff_ab > gate_ab:
            print(
                f"Skipping reference time {ref_time} - A-B difference {diff_ab} exceeds gate {gate_ab}"
            )
            continue

        corr = CorrelatedDetection(reference_site=reference_site)
        corr.__dict__[f"site_{reference_site}"] = ref_time
        corr.__dict__[f"site_{site_a}"] = times[site_a][idx_a]
        corr.__dict__[f"site_{site_b}"] = times[site_b][idx_b]
        correlations.append(corr)
        used_a.add(idx_a)
        used_b.add(idx_b)

    return correlations


def estimate_tdoa(
    whale_call_data: Path,
    distance_lut: Path,
    reference_site: str,
) -> pl.DataFrame:
    """Estimate TDOA requiring detections at all three sites, anchored on
    ``reference_site``.

    This is a simplified alternative to ``estimate_tdoa``: it uses a single
    reference site as a mandatory gate, discards partial detections, and
    skips the multi-reference voting / merge step.

    Args:
        whale_call_data: Path to whale call detection times.
        distance_lut: Path to distance lookup table.
        reference_site: Site that must have a detection for a triplet to be
            kept (e.g. ``'vla2'``).

    Returns:
        DataFrame with correlated triplets and TDOA values.
    """
    times = read_whale_call_times(whale_call_data)
    time_gates = compute_time_gates(distance_lut)
    correlations = correlate_detections_triplet_gated(
        times, time_gates, reference_site=reference_site
    )
    print(f"Complete triplets anchored on '{reference_site}': {len(correlations)}")
    return correlations_to_dataframe(
        correlations, reference_site=reference_site
    ).with_columns(pl.col("timestamp").dt.epoch(time_unit="us").alias("unix_time_us"))


def correlations_to_dataframe(
    correlations: list[CorrelatedDetection], reference_site: str = "3dvha"
) -> pl.DataFrame:
    """Convert correlated detections to a Polars DataFrame with TDOA values.

    The DataFrame contains a timestamp column (from the reference site) and
    TDOA columns for each site (time difference in seconds from reference).
    TDOA values are computed as site_time - reference_time; positive values
    mean the signal arrived later at that site.

    Args:
        correlations: List of correlated detections (should be complete triplets).
        reference_site: Site to use as time reference. Defaults to '3dvha'.

    Returns:
        DataFrame with columns: timestamp (reference site detection time),
        3dvha (TDOA in seconds, 0.0 if reference site), vla1 (TDOA in seconds),
        and vla2 (TDOA in seconds).
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


def estimate_tdoa_greedy(
    whale_call_data: Path, distance_lut: Path, reference_site: str
) -> pl.DataFrame:
    """Estimate TDOA from whale call detection times and distance LUT.

    Args:
        whale_call_data: Path to whale call detection times.
        distance_lut: Path to distance lookup table.
        reference_site: Site to use as time reference.

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
    """Find all matching times within a time gate window.

    Args:
        reference_time: Reference detection time.
        search_times: Array of detection times to search.
        max_delay: Maximum time delay in seconds (time gate).
        used_indices: Set of indices already matched (to avoid double-counting).

    Returns:
        Indices of matching detections in search_times.
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
    """Return a function evaluating the system of three hyperbolae for a given event.

    Args:
        x0: Easting of observer 0.
        y0: Northing of observer 0.
        x1: Easting of observer 1.
        y1: Northing of observer 1.
        x2: Easting of observer 2.
        y2: Northing of observer 2.
        d01: TDOA (in distance units) between observers 0 and 1.
        d02: TDOA (in distance units) between observers 0 and 2.
        d12: TDOA (in distance units) between observers 1 and 2.

    Returns:
        A callable that takes (x, y) and returns the residuals of the three
        hyperbolic equations.
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

    var_tdoa: float | NDArray | pl.DataFrame | None = config.var_tdoa
    if (
        config.pc_data_path is not None
        and config.template_duration_s is not None
        and config.f_low_hz is not None
        and config.f_high_hz is not None
    ):
        var_tdoa = compute_sigma_tdoa_per_detection(
            df,
            config.pc_data_path,
            config.channel,
            config.template_duration_s,
            config.f_low_hz,
            config.f_high_hz,
            filt_type=config.filt_type,
            filt_freq=config.filt_freq,
            window_s=config.snr_window_s,
            noise_correction=config.noise_correction,
        )

    df = localize_tdoa_data(df, config.sensor_data, var_tdoa=var_tdoa)
    df.write_csv(config.raw_localization_file)
    df = correct_ambiguous_bearings(
        df,
        config.ambiguity_lower_bound,
        config.ambiguity_upper_bound,
        source_north=config.source_north,
    )
    df.write_csv(config.localization_file)


def _bearing_uncertainty_deg(
    target_x_km: float,
    target_y_km: float,
    sensor_x_km: float,
    sensor_y_km: float,
    C_pos: NDArray,
) -> float:
    """Propagate position covariance to 1-sigma bearing uncertainty (degrees).

    Uses the gradient of atan2(Δe, Δn) with respect to target position (x, y).
    C_pos must be in km², matching the km coordinate system used throughout.
    """
    delta_e = target_x_km - sensor_x_km
    delta_n = target_y_km - sensor_y_km
    r_sq = delta_e**2 + delta_n**2
    if r_sq == 0.0:
        return float("nan")
    grad = np.array([delta_n / r_sq, -delta_e / r_sq])  # rad/km
    return float(np.degrees(np.sqrt(grad @ C_pos @ grad)))


def localize_tdoa_data(
    df: pl.DataFrame,
    sensor_data: Path,
    var_tdoa: float | ArrayLike | pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute locations from TDOA data using least-squares localization.

    Args:
        df: DataFrame containing TDOA columns: timestamp, 3dvha, vla1, vla2.
        sensor_data: Path to the sensor positions CSV file.
        var_tdoa: TDOA variance in seconds².  May be:
            - None: bearing uncertainty columns are filled with NaN.
            - float: one constant value applied to every detection.
            - array-like of length len(df): per-detection TDOA variance values.
              NaN entries produce NaN bearing uncertainty for that detection.
            - pl.DataFrame: output of compute_sigma_tdoa_per_detection, with
              columns sigma_t_3dvha, sigma_t_vla1, sigma_t_vla2, var_tdoa.
              The combined var_tdoa column is used for covariance propagation;
              all columns are appended to the output.

    Returns:
        DataFrame with original TDOA data plus computed location columns:
        easting (m), northing (m), latitude (degrees), longitude (degrees),
        3dvha_brg, vla1_brg, vla2_brg (true bearing from each sensor to
        target, in degrees 0–360), 3dvha_brg_unc, vla1_brg_unc, vla2_brg_unc
        (1-sigma bearing uncertainty in degrees), and — when var_tdoa is a
        DataFrame — sigma_t_3dvha, sigma_t_vla1, sigma_t_vla2, var_tdoa.
    """
    # Get sensor positions and reference coordinates
    sensor_eastings, sensor_northings, lat0, lon0 = read_sensor_positions(sensor_data)

    # Convert to km for numerical stability
    denom = 1000.0
    sensor_eastings_km = [e / denom for e in sensor_eastings]
    sensor_northings_km = [n / denom for n in sensor_northings]

    # Sound speed in water (m/s converted to km/s)
    speed = 1500.0 / denom

    # When var_tdoa is a DataFrame, extract the combined column for covariance
    # propagation and stash the full DataFrame to append to the output.
    sigma_data_df: pl.DataFrame | None = None
    if isinstance(var_tdoa, pl.DataFrame):
        sigma_data_df = var_tdoa
        var_tdoa = sigma_data_df["var_tdoa"].to_numpy()

    # Normalise var_tdoa to a list for uniform per-row access.
    n_rows = len(df)
    if var_tdoa is None:
        var_tdoa_list: list[float | None] = [None] * n_rows
    elif isinstance(var_tdoa, (int, float)):
        var_tdoa_list = [float(var_tdoa)] * n_rows
    else:
        arr = np.asarray(var_tdoa, dtype=float)
        var_tdoa_list = [None if np.isnan(v) else float(v) for v in arr]

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
    unc_3dvha = []
    unc_vla1 = []
    unc_vla2 = []

    for i, row in enumerate(df.iter_rows(named=True)):
        # Extract TDOA values (in seconds)
        t0 = row["3dvha"]
        t1 = row["vla1"]
        t2 = row["vla2"]

        # Solve TDOA localization
        x_km, y_km, _, cov_x = tdoa_solve(
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

        # Propagate TDOA variance to bearing uncertainty for each sensor
        row_var = var_tdoa_list[i]
        if row_var is not None and cov_x is not None:
            var_d_km = speed**2 * row_var  # range-difference variance (km²)
            C_pos = cov_x * var_d_km  # position covariance (km²)
            u3 = _bearing_uncertainty_deg(
                x_km, y_km, sensor_eastings_km[0], sensor_northings_km[0], C_pos
            )
            u1 = _bearing_uncertainty_deg(
                x_km, y_km, sensor_eastings_km[1], sensor_northings_km[1], C_pos
            )
            u2 = _bearing_uncertainty_deg(
                x_km, y_km, sensor_eastings_km[2], sensor_northings_km[2], C_pos
            )
        else:
            u3 = u1 = u2 = float("nan")

        eastings.append(easting_m)
        northings.append(northing_m)
        latitudes.append(lat)
        longitudes.append(lon)
        bearings_3dvha.append(brg_3dvha)
        bearings_vla1.append(brg_vla1)
        bearings_vla2.append(brg_vla2)
        unc_3dvha.append(u3)
        unc_vla1.append(u1)
        unc_vla2.append(u2)

    result = df.with_columns(
        [
            pl.Series("easting", eastings),
            pl.Series("northing", northings),
            pl.Series("latitude", latitudes),
            pl.Series("longitude", longitudes),
            pl.Series("3dvha_brg", bearings_3dvha),
            pl.Series("vla1_brg", bearings_vla1),
            pl.Series("vla2_brg", bearings_vla2),
            pl.Series("3dvha_brg_unc", unc_3dvha),
            pl.Series("vla1_brg_unc", unc_vla1),
            pl.Series("vla2_brg_unc", unc_vla2),
        ]
    )
    if sigma_data_df is not None:
        result = result.with_columns(sigma_data_df.get_columns())
    return result


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
    """Merge correlations from different reference sites to get complete triplets only.

    Collects all correlations from different reference site runs, merges
    complementary partial matches to form complete triplets, deduplicates
    within the time tolerance, and returns only complete 3-site detections.

    Args:
        results: Results from correlate_all_references.
        tolerance: Time tolerance in seconds for considering detections as duplicates.

    Returns:
        List of unique complete triplets only (all 3 sites present).
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


def tdoa_solve(x, y, t, speed, xp, yp) -> tuple[float, float, Callable, NDArray | None]:
    x0, y0, t0 = x[0], y[0], t[0]
    x1, y1, t1 = x[1], y[1], t[1]
    x2, y2, t2 = x[2], y[2], t[2]

    F = functions(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )
    J = jacobian(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )

    pos, cov_x, *_ = leastsq(F, x0=[xp, yp], Dfun=J, full_output=True)
    return pos[0], pos[1], F, cov_x
