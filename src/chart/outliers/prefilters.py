"""
Discarding columns and rows that are not worth analysing.

Three prefilters run before outlier detection: two drop columns, one drops
rows.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)

#: The prefilters, by the name used to switch them off.  The names also
#: appear in the filter list, as the reason a cell was removed.
MISSING = 'missing'
VARIATION = 'variation'
INCOMPLETE = 'incomplete'
PREFILTERS = (MISSING, VARIATION, INCOMPLETE)


@dataclass
class PrefilterResult:
    """What the prefilters removed."""
    columns_in: int = 0
    columns_out: int = 0
    rows_in: int = 0
    rows_out: int = 0
    #: Column name to the reason it went.
    dropped_columns: Dict[str, str] = field(default_factory=dict)
    #: Prefilter name to the cells it removed, so the filter list can say
    #: why each cell went rather than only how many.
    dropped_cells: Dict[str, pd.Index] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)


def robust_cv(values: pd.Series) -> float:
    """The robust coefficient of variation: MAD over median.
    """
    return median_abs_deviation(values, scale='normal') / (values.median()
                                                           + np.finfo(float).eps)


def drop_missing_columns(data: pd.DataFrame, threshold: int,
                         result: PrefilterResult) -> pd.DataFrame:
    """Drop columns with more than *threshold* missing values.
    """
    counts = data.isna().sum(axis=0)
    doomed = counts[counts > threshold]
    if doomed.empty:
        worst = int(counts.max()) if len(counts) else 0
        logger.info(f"Missing-value prefilter: nothing exceeds {threshold} missing "
                    f"values (the worst column has {worst})")
        return data

    for column, count in doomed.items():
        result.dropped_columns[str(column)] = f"{count} missing values"
    logger.warning(f"Missing-value prefilter: dropping {len(doomed)} column(s) with "
                   f"more than {threshold} missing values: "
                   f"{', '.join(map(str, doomed.index))}")
    return data.drop(columns=doomed.index)


def drop_low_variation_columns(data: pd.DataFrame, threshold: float,
                               result: PrefilterResult) -> pd.DataFrame:
    """Drop columns whose robust coefficient of variation is below *threshold*.
    """
    keep, dropped = [], {}
    for column in data.columns:
        values = data[column].dropna()
        if values.empty:
            dropped[str(column)] = "no values"
            continue
        rcv = robust_cv(values)
        if abs(rcv) >= threshold:
            keep.append(column)
        else:
            dropped[str(column)] = f"robust CV {rcv:.6f} below {threshold}"

    if not dropped:
        logger.info(f"Variation prefilter: every column varies by at least "
                    f"{threshold}")
        return data

    result.dropped_columns.update(dropped)
    logger.warning(f"Variation prefilter: dropping {len(dropped)} column(s): "
                   + '; '.join(f"{name} ({why})" for name, why in dropped.items()))
    return data[keep]


def drop_incomplete_rows(data: pd.DataFrame,
                         result: PrefilterResult) -> pd.DataFrame:
    """Drop every row holding a missing value.
    """
    incomplete = data.isna().sum(axis=1) > 0
    affected = int(incomplete.sum())
    if not affected:
        logger.info("Incomplete-row prefilter: no row has a missing value")
        return data

    culprits = data.isna().sum(axis=0)
    culprits = culprits[culprits > 0].sort_values(ascending=False)
    logger.warning(f"Incomplete-row prefilter: dropping {affected} of {len(data)} "
                   f"cells ({affected / len(data) * 100:.1f}%) for missing "
                   f"values in: "
                   + ', '.join(f"{name} ({count})"
                               for name, count in culprits.items()))
    # Recorded by cell, not just counted, so the filter list can account for
    # every row missing from the output.
    result.dropped_cells[INCOMPLETE] = data.index[incomplete.to_numpy()]
    return data[~incomplete]


def apply_prefilters(data: pd.DataFrame,
                   missing_value_threshold: int = 100000,
                   rcv_threshold: float = 0.01,
                   skip: Sequence[str] = ()
                   ) -> Tuple[pd.DataFrame, PrefilterResult]:
    """Run the prefilters in order, returning the reduced table.

    Args:
        data: The normalised feature table
        missing_value_threshold: Passed to :func:`drop_missing_columns`
        rcv_threshold: Passed to :func:`drop_low_variation_columns`
        skip: Prefilters not to run; see :data:`PREFILTERS`

    Returns:
        The prefiltered table and an account of what was removed.
    """
    unknown = [name for name in skip if name not in PREFILTERS]
    if unknown:
        raise ValueError(f"Unknown prefilter(s): {', '.join(unknown)}. "
                         f"Available prefilters: {', '.join(PREFILTERS)}")

    result = PrefilterResult(columns_in=len(data.columns), rows_in=len(data),
                          skipped=[s for s in PREFILTERS if s in skip])
    for name in result.skipped:
        logger.info(f"Not running the {name} prefilter")

    if MISSING not in skip:
        data = drop_missing_columns(data, missing_value_threshold, result)
    if VARIATION not in skip:
        data = drop_low_variation_columns(data, rcv_threshold, result)
    if INCOMPLETE not in skip:
        data = drop_incomplete_rows(data, result)

    result.columns_out = len(data.columns)
    result.rows_out = len(data)
    logger.info(f"Prefiltering: {result.columns_in} -> {result.columns_out} columns, "
                f"{result.rows_in} -> {result.rows_out} cells")
    return data, result
