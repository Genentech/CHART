"""
Quantile normalisation across wells.

Every well's values for a feature are mapped onto the distribution of that
feature pooled over the whole screen, so that per-well differences in
staining or exposure do not survive into the analysis.

Missing values are left out of the mapping and stay missing.  cellmapp
ranked them instead, which sorted them to the top of their well and put
them at the top of the pooled reference, so a missing value could come
back as a plausible measurement and the largest real value in an
unaffected well could come back missing.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from numba import njit, prange

from ..filtering.features import NoFeaturesError

logger = logging.getLogger(__name__)

#: A well needs at least this many cells to be ranked against the pooled
#: distribution.
MIN_CELLS_PER_WELL = 2


@njit
def rank_average(series: np.ndarray) -> np.ndarray:
    """Rank *series*, giving tied values their average rank."""
    n = len(series)
    sorted_indices = np.argsort(series)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        # Find the end of the tied group
        while j + 1 < n and series[sorted_indices[j]] == series[sorted_indices[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1
    return ranks


@njit(parallel=True)
def normalize_group(group: np.ndarray, sorted_reference: np.ndarray,
                    valid_counts: np.ndarray) -> np.ndarray:
    """Map one well's values onto the pooled distribution, per feature.

    Missing values take no part in the mapping: they are neither ranked
    within the well nor read from the reference, and they come back out
    missing.

    Args:
        group: One well's values, ``(n_cells, n_features)``
        sorted_reference: The pooled values, sorted per feature,
                          ``(n_total_cells, n_features)``.  Sorting puts
                          each feature's missing values last.
        valid_counts: How many leading rows of each reference column are
                      not missing, ``(n_features,)``

    Returns:
        The mapped values, shaped like *group*.
    """
    num_features = sorted_reference.shape[1]
    num_cells = group.shape[0]
    result = np.empty((num_features, num_cells), dtype=sorted_reference.dtype)

    for i in prange(num_features):
        series = group[:, i]

        for j in range(num_cells):
            result[i, j] = np.nan

        present = np.empty(num_cells, dtype=np.int64)
        num_present = 0
        for j in range(num_cells):
            if not np.isnan(series[j]):
                present[num_present] = j
                num_present += 1

        num_valid = valid_counts[i]
        if num_valid == 0 or num_present == 0:
            continue

        sorted_all_values = sorted_reference[:num_valid, i]
        values = np.empty(num_present, dtype=sorted_reference.dtype)
        for j in range(num_present):
            values[j] = series[present[j]]

        ranks = rank_average(values)

        rank_max = ranks.max()
        if rank_max == 0:
            # One value, so there is no ordering to map.  The reference
            # median is the least misleading value available.
            interpolated_values = np.full(num_present,
                                          np.median(sorted_all_values))
        else:
            # Scale the ranks onto the reference.  Note this reaches
            # num_valid while the interpolation grid stops one short, so
            # the very top of each well is clamped to the largest reference
            # value.  Kept as cellmapp had it: correcting the scale would
            # shift every normalised value.
            interp_x = ranks * (num_valid / rank_max)
            interpolated_values = np.interp(
                interp_x,
                np.arange(num_valid),
                sorted_all_values
            )

        for j in range(num_present):
            result[i, present[j]] = interpolated_values[j]

    return result.T


def normalize_across_wells(data: pd.DataFrame,
                           well_column: str = 'Well',
                           feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Quantile normalise *data* well by well.

    Args:
        data: A table whose index includes *well_column*
        well_column: The index level naming the well
        feature_columns: Columns to normalise.  The default is every
                         numeric column, so pass this explicitly if the
                         table carries numeric identifiers that are not
                         measurements.

    Returns:
        A table with the same index as *data*, holding the normalised
        feature columns, missing in exactly the places *data* was.
        Columns outside *feature_columns* are dropped.

    Raises:
        ValueError: if a well has fewer than :data:`MIN_CELLS_PER_WELL`
            cells, which cannot be ranked
        RuntimeError: if normalisation did not preserve which values are
            missing
    """
    if well_column not in (data.index.names or ()):
        raise KeyError(f"Cannot normalise across wells: the table has no "
                       f"'{well_column}' index level, only "
                       f"{', '.join(str(n) for n in data.index.names or ())}")

    if feature_columns is None:
        feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        ignored = [c for c in data.columns if c not in feature_columns]
        if ignored:
            logger.info(f"Leaving out {len(ignored)} non-numeric column(s): "
                        f"{', '.join(map(str, ignored))}")

    if not len(feature_columns):
        raise NoFeaturesError(
            f"Nothing to normalise: none of the {len(data.columns)} columns "
            f"in the table are features. Columns seen: "
            + (', '.join(str(col) for col in data.columns) or 'none') + ".")

    logger.info(f"Normalising {len(feature_columns)} features across wells")

    well_labels = data.index.get_level_values(well_column)
    wells = well_labels.unique()
    logger.info(f"Wells: {', '.join(map(str, wells))}")

    counts = {well: int((well_labels == well).sum()) for well in wells}
    too_small = {w: n for w, n in counts.items() if n < MIN_CELLS_PER_WELL}
    if too_small:
        detail = ', '.join(f"{w} ({n})" for w, n in too_small.items())
        raise ValueError(f"Cannot quantile normalise a well with fewer than "
                         f"{MIN_CELLS_PER_WELL} cells: {detail}")

    # float64 throughout: the kernel requires the group and the reference to
    # share a dtype, and np.interp returns float64 regardless.
    feature_values = data[feature_columns].to_numpy(dtype=np.float64)
    missing = np.isnan(feature_values)

    logger.info("Building the pooled reference distribution...")
    # np.sort puts missing values last, so each column's reference is its
    # leading valid_counts[i] entries.
    sorted_reference = np.sort(feature_values, axis=0)
    valid_counts = (~missing).sum(axis=0).astype(np.int64)

    if missing.any():
        affected = int((missing.any(axis=0)).sum())
        logger.warning(
            f"{int(missing.sum())} missing value(s) across {affected} of "
            f"{len(feature_columns)} features take no part in normalisation "
            f"and stay missing")
    empty = [c for c, n in zip(feature_columns, valid_counts) if n == 0]
    if empty:
        logger.warning(f"{len(empty)} feature(s) have no values at all and "
                       f"cannot be normalised: {', '.join(map(str, empty))}")

    # Results are written back into the original row positions rather than
    # concatenated per well, so the index keeps its order and its names.
    normalized = np.empty_like(feature_values)
    for well in wells:
        logger.info(f"Normalising well {well} ({counts[well]} cells)...")
        rows = well_labels == well
        normalized[rows] = normalize_group(feature_values[rows],
                                           sorted_reference, valid_counts)

    # Normalisation reorders values within a feature; it must not create or
    # remove them.  Checking is cheap next to the mapping itself.
    if not np.array_equal(np.isnan(normalized), missing):
        raise RuntimeError(
            f"Normalisation changed which values are missing: "
            f"{int(missing.sum())} missing before, "
            f"{int(np.isnan(normalized).sum())} after. This is a bug in "
            f"{__name__}.")

    result = pd.DataFrame(normalized, index=data.index, columns=feature_columns)
    logger.info(f"Quantile normalisation completed: {result.shape[0]} rows, "
                f"{result.shape[1]} features")
    return result
