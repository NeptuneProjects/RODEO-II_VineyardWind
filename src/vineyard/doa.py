"""Far-field direction-of-arrival estimation from TDOA measurements.

Replaces the near-field hyperbolic optimizer in tdoa.py with a direct
linear least-squares solution for the plane-wave slowness vector.

For a plane wave arriving from bearing "bearing" (clockwise from North),
the arrival time at sensor i is:

    arrival_time_i = t0 + sensor_position_i * s

where s = direction_vector / speed is the slowness vector (direction_vector
points from source toward array). The TDOA between sensor pairs is linear
in s:

    tdoa_ij = (sensor_position_i - sensor_position_j) * s

This array is arranged nearly east-west (N-S baselines are O(m) vs O(km)
E-W baselines), making the full 2D system rank-deficient in the N-S
direction. Only the east-west component of slowness is reliably observable:

    delta_e_ij * s_e ~= tdoa_ij   (1-D lstsq, eastings only)

The N-S component is then recovered from the physical constraint |s| = 1/speed:

    s_n = +/- sqrt(1 / speed**2 - s_e**2)

with the sign chosen by a half-space prior on source location (sources are
north of the array, so s_n < 0 -- the wave propagates southward).

Bearing and uncertainty follow:

    bearing = atan2(-s_e, -s_n)      [degrees, clockwise from North]
    sigma_bearing = sigma_tdoa / (abs(s_n) * sqrt(sum_sq_e))   [radians; convert to degrees]
"""

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from vineyard.process import compute_sigma_tdoa_per_detection
from vineyard.readers import read_sensor_positions
from vineyard.tdoa import LocalizationConfig, correct_ambiguous_bearings, estimate_tdoa


def doa_solve(
    sensor_eastings: list[float],
    tdoas: list[float],
    speed: float,
    source_south: bool = False,
) -> tuple[float, float, float, float]:
    """Solve for far-field DOA from TDOA measurements using a 1-D lstsq.

    The N-S sensor baselines in this array are O(m) vs O(km) E-W, making
    the 2-D system ill-conditioned. Only s_e is solved from data; s_n is
    recovered from the sound-speed constraint |s| = 1/speed.

    Parameters
    ----------
    sensor_eastings :
        Sensor easting positions in km.
    tdoas :
        TDOA for each sensor in seconds (reference sensor = 0.0).
    speed :
        Sound speed in km/s (typically 1.5).
    source_south :
        If True, source is south of the array (s_n > 0). If False (default),
        source is north of the array, so the wave propagates southward and
        s_n < 0.

    Returns
    -------
    bearing :
        Direction-of-arrival in degrees, clockwise from North.
    s_e :
        East component of slowness vector (s/km).
    s_n :
        North component of slowness vector (s/km).
    sum_sq_e :
        Sum of delta_e**2 over unique sensor pairs (km**2); used for
        uncertainty propagation.
    """
    n = len(sensor_eastings)
    delta_e: list[float] = []
    delta_tau: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            delta_e.append(sensor_eastings[i] - sensor_eastings[j])
            delta_tau.append(tdoas[i] - tdoas[j])

    de = np.array(delta_e)
    dt = np.array(delta_tau)

    # 1-D lstsq: s_e = sum(delta_e * delta_tau) / sum(delta_e**2)
    sum_sq_e = float(de @ de)
    s_e = float(de @ dt) / sum_sq_e

    # Recover s_n from |s| = 1/speed; clamp s_e to the physical range first
    s_e_clamped = float(np.clip(s_e, -1.0 / speed, 1.0 / speed))
    s_n_mag = float(np.sqrt(max(0.0, (1.0 / speed) ** 2 - s_e_clamped**2)))
    s_n = s_n_mag if source_south else -s_n_mag

    bearing = float(np.degrees(np.arctan2(-s_e_clamped, -s_n)) % 360)
    return bearing, s_e_clamped, s_n, sum_sq_e


def _bearing_uncertainty_deg(
    s_n: float,
    sum_sq_e: float,
    var_tdoa: float,
) -> float:
    """Propagate TDOA variance to 1-sigma DOA bearing uncertainty (degrees).

    With s_n fixed by the sound-speed constraint, d(bearing)/d(s_e) = 1/s_n,
    giving:

        sigma_bearing = sigma_tdoa / (abs(s_n) * sqrt(sum_sq_e))   [radians]
    """
    if s_n == 0.0 or sum_sq_e == 0.0:
        return float("nan")
    sigma_tau = float(np.sqrt(var_tdoa))
    sigma_theta_rad = sigma_tau / (abs(s_n) * np.sqrt(sum_sq_e))
    return float(np.degrees(sigma_theta_rad))


def localize_doa_data(
    df: pl.DataFrame,
    sensor_data: Path,
    var_tdoa: float | ArrayLike | pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute far-field DOA from TDOA data using the plane-wave model.

    Parameters
    ----------
    df :
        DataFrame with columns timestamp, 3dvha, vla1, vla2 (TDOAs in
        seconds, relative to the reference site used during correlation).
    sensor_data :
        Path to sensor positions CSV.
    var_tdoa :
        TDOA variance in seconds squared. Accepts the same shapes as
        localize_tdoa_data in tdoa.py: None, scalar float, array-like of
        length len(df), or a pl.DataFrame from
        compute_sigma_tdoa_per_detection.

    Returns
    -------
    DataFrame with original TDOA columns plus:
        doa_brg             — array-level far-field bearing (deg CW from N)
        doa_brg_unc         — 1-sigma bearing uncertainty (degrees)
        3dvha_brg, vla1_brg, vla2_brg         — equal to doa_brg
        3dvha_brg_unc, vla1_brg_unc, vla2_brg_unc — equal to doa_brg_unc

    The three per-sensor bearing columns are filled with the same value so
    that downstream consumers (figures, ambiguity correction) work unchanged.
    Position columns (easting, northing, lat, lon) are not produced because
    range is unobservable in the far-field model.
    """
    sensor_eastings, _, _, _ = read_sensor_positions(sensor_data)

    # Convert m to km for numerical stability (matching tdoa.py convention)
    denom = 1000.0
    speed_km_s = 1500.0 / denom  # km/s
    e_km = [e / denom for e in sensor_eastings]

    sigma_data_df: pl.DataFrame | None = None
    if isinstance(var_tdoa, pl.DataFrame):
        sigma_data_df = var_tdoa
        var_tdoa = sigma_data_df["var_tdoa"].to_numpy()

    n_rows = len(df)
    if var_tdoa is None:
        var_tdoa_list: list[float | None] = [None] * n_rows
    elif isinstance(var_tdoa, (int, float)):
        var_tdoa_list = [float(var_tdoa)] * n_rows
    else:
        arr = np.asarray(var_tdoa, dtype=float)
        var_tdoa_list = [None if np.isnan(v) else float(v) for v in arr]

    bearings: list[float] = []
    uncertainties: list[float] = []

    for i, row in enumerate(df.iter_rows(named=True)):
        tdoas = [row["3dvha"], row["vla1"], row["vla2"]]
        bearing, _, s_n, sum_sq_e = doa_solve(e_km, tdoas, speed_km_s)
        bearings.append(bearing)

        row_var = var_tdoa_list[i]
        unc = (
            _bearing_uncertainty_deg(s_n, sum_sq_e, row_var)
            if row_var is not None
            else float("nan")
        )
        uncertainties.append(unc)

    result = df.with_columns(
        [
            pl.Series("doa_brg", bearings),
            pl.Series("doa_brg_unc", uncertainties),
            pl.Series("3dvha_brg", bearings),
            pl.Series("vla1_brg", bearings),
            pl.Series("vla2_brg", bearings),
            pl.Series("3dvha_brg_unc", uncertainties),
            pl.Series("vla1_brg_unc", uncertainties),
            pl.Series("vla2_brg_unc", uncertainties),
        ]
    )
    if sigma_data_df is not None:
        result = result.with_columns(sigma_data_df.get_columns())
    return result


def localize_farfield(config: LocalizationConfig) -> None:
    """Run the far-field DOA pipeline and save results to CSV.

    Drop-in replacement for tdoa.localize. Uses the plane-wave slowness
    model instead of the near-field hyperbolic optimizer.
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

    df = localize_doa_data(df, config.sensor_data, var_tdoa=var_tdoa)
    df.write_csv(config.raw_localization_file)
    df = correct_ambiguous_bearings(
        df,
        config.ambiguity_lower_bound,
        config.ambiguity_upper_bound,
        source_north=config.source_north,
    )
    df.write_csv(config.localization_file)
