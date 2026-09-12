"""
Column selection for filtering.
"""

import logging
from typing import List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Column prefixes worth keeping: the three segmentation compartments.
KEEP_PREFIXES = ('cell_', 'nuclei_', 'cytosol_')

#: Substrings marking columns that describe position or shape orientation
#: rather than a measurement, and so are not useful as features.
EXCLUDE_PATTERNS = ('bbox', 'centroid', 'location', 'centermass',
                    'to_origin', '_x', '_y', 'orientation')


class NoFeaturesError(ValueError):
    """Column selection left no feature columns."""


def filter_features(features: pd.DataFrame,
                    keep_prefixes: Sequence[str] = KEEP_PREFIXES,
                    exclude_patterns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Keep only the feature columns worth carrying forward.

    Two steps: keep columns matching *keep_prefixes*, then drop any of
    those containing one of *exclude_patterns*.

    Args:
        features: DataFrame to filter
        keep_prefixes: Compartment prefixes to keep
        exclude_patterns: Substrings to exclude; ``None`` uses
                          :data:`EXCLUDE_PATTERNS`

    Returns:
        The DataFrame with only the retained columns.

    Raises:
        NoFeaturesError: if no column survives the selection
    """
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS

    original_columns = len(features.columns)

    # Note: this matches the prefix anywhere in the column name rather than
    # only at the start, because that is what the upstream cellmapp code did
    # and anchoring it could drop columns that are currently kept.
    matched = features.filter(regex='|'.join(keep_prefixes))
    logger.debug(f"After prefix filtering: {len(matched.columns)} columns")

    kept, removed_by = matched, {}
    for pattern in exclude_patterns:
        before = len(kept.columns)
        kept = kept[[col for col in kept.columns if pattern not in col]]
        if len(kept.columns) < before:
            removed_by[pattern] = before - len(kept.columns)
        logger.debug(f"After removing '{pattern}': {len(kept.columns)} columns")

    if not len(kept.columns):
        raise NoFeaturesError(_no_features_message(
            features, matched, removed_by, keep_prefixes, exclude_patterns))

    logger.info(f"Feature filtering: {original_columns} -> {len(kept.columns)} columns")
    return kept


def _no_features_message(features: pd.DataFrame,
                         matched: pd.DataFrame,
                         removed_by: dict,
                         keep_prefixes: Sequence[str],
                         exclude_patterns: Sequence[str]) -> str:
    """Say which half of the selection emptied the table."""
    if not len(matched.columns):
        sample = ', '.join(str(col) for col in features.columns[:5])
        if len(features.columns) > 5:
            sample += f", and {len(features.columns) - 5} more"
        return (f"No feature columns: none of the {len(features.columns)} "
                f"columns contain any of the prefixes "
                f"{', '.join(keep_prefixes)}. Columns seen: "
                f"{sample or 'none'}.")

    return (f"No feature columns: the prefixes matched "
            f"{len(matched.columns)} columns and the exclude patterns "
            f"removed all of them ("
            + ', '.join(f"'{pattern}' removed {count}"
                        for pattern, count in removed_by.items())
            + f"). Narrow exclude_patterns, currently "
              f"{', '.join(exclude_patterns)}.")


def merge_objects_features(objects: pd.DataFrame,
                           features: pd.DataFrame) -> pd.DataFrame:
    """Combine an object table and a feature table on their shared index.
    """
    extra = [col for col in features.columns if col not in objects.columns]
    merged = objects.join(features[extra], how='inner')
    logger.info(f"Merged objects and features: {len(merged)} objects, "
                f"{len(merged.columns)} columns "
                f"({len(extra)} contributed by the feature table)")
    return merged
