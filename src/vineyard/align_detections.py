"""
TDOA Correlation for Multi-Site Detection Alignment

This module provides functions to correlate acoustic detections across multiple
sensor sites using time-difference of arrival (TDOA) analysis.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CorrelatedDetection:
    """A detection event correlated across multiple sites."""

    site_3dvha: Optional[np.datetime64] = None
    site_vla1: Optional[np.datetime64] = None
    site_vla2: Optional[np.datetime64] = None
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


def find_matches_in_window(
    reference_time: np.datetime64,
    search_times: np.ndarray,
    max_delay: float,
    used_indices: set,
) -> List[int]:
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


def correlate_detections_triplet(
    times: Dict[str, np.ndarray],
    time_gates: Dict[Tuple[str, str], float],
    reference_site: str = "3dvha",
) -> List[CorrelatedDetection]:
    """
    Correlate detections across three sites using a greedy matching approach.

    This algorithm iterates through detections at a reference site and finds
    corresponding detections at the other sites within the specified time gates.

    Parameters
    ----------
    times : Dict[str, np.ndarray]
        Dictionary mapping site names to arrays of detection times
    time_gates : Dict[Tuple[str, str], float]
        Dictionary mapping site pairs to maximum time delays in seconds
        Keys should be tuples like ('3dvha', 'vla1')
    reference_site : str, optional
        Which site to use as reference (default: '3dvha')

    Returns
    -------
    List[CorrelatedDetection]
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


def correlate_all_references(
    times: Dict[str, np.ndarray], time_gates: Dict[Tuple[str, str], float]
) -> Dict[str, List[CorrelatedDetection]]:
    """
    Run correlation using each site as reference and return all results.

    Parameters
    ----------
    times : Dict[str, np.ndarray]
        Dictionary mapping site names to arrays of detection times
    time_gates : Dict[Tuple[str, str], float]
        Dictionary mapping site pairs to maximum time delays in seconds

    Returns
    -------
    Dict[str, List[CorrelatedDetection]]
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


def merge_correlations(
    results: Dict[str, List[CorrelatedDetection]], tolerance: float = 0.001
) -> List[CorrelatedDetection]:
    """
    Merge correlations from different reference sites to get complete triplets only.

    This function:
    1. Collects all correlations from different reference site runs
    2. Merges complementary partial matches to form complete triplets
    3. Deduplicates complete triplets within the time tolerance
    4. Returns only complete 3-site detections

    Parameters
    ----------
    results : Dict[str, List[CorrelatedDetection]]
        Results from correlate_all_references
    tolerance : float
        Time tolerance in seconds for considering detections as duplicates

    Returns
    -------
    List[CorrelatedDetection]
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


def correlations_to_dataframe(
    correlations: List[CorrelatedDetection], reference_site: str = "3dvha"
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
        timestamps.append(ref_time.astype('datetime64[us]').astype('int64'))

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
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )
