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
    """
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS

    original_columns = len(features.columns)

    # Note: this matches the prefix anywhere in the column name rather than
    # only at the start, because that is what the upstream cellmapp code did
    # and anchoring it could drop columns that are currently kept.
    features = features.filter(regex='|'.join(keep_prefixes))
    logger.debug(f"After prefix filtering: {len(features.columns)} columns")

    for pattern in exclude_patterns:
        features = features[[col for col in features.columns if pattern not in col]]
        logger.debug(f"After removing '{pattern}': {len(features.columns)} columns")

    logger.info(f"Feature filtering: {original_columns} -> {len(features.columns)} columns")
    return features


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
