"""
Column selection for filtering.
"""

import logging
import re
from typing import List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Patterns worth keeping: the three segmentation compartments.
KEEP_PATTERNS = ('cell_', 'nuclei_', 'cytosol_')

#: Patterns marking columns that describe position or shape orientation
#: rather than a measurement, and so are not useful as features.  Matched
#: case-insensitively, as these have to recognise a naming convention
#: rather than name particular columns: CellProfiler writes
#: ``Nuclei_Location_Center_X`` where the reference screen writes
#: ``nuclei_centroid-0``.
EXCLUDE_PATTERNS = ('bbox', 'centroid', 'location', 'centermass',
                    'to_origin', '_x', '_y', 'orientation')


class NoFeaturesError(ValueError):
    """Column selection left no feature columns."""


def _compile_each(patterns: Sequence[str], key: str,
                  flags: int = 0) -> List['re.Pattern']:
    """Compile *patterns*, saying which one is at fault if any is broken.

    Compiling them one at a time rather than as a single alternation
    keeps the reported position pointing into the pattern the caller
    wrote.
    """
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, flags))
        except re.error as exc:
            raise ValueError(f"'{pattern}' in {key} is not a valid regular "
                             f"expression: {exc}") from exc
    return compiled


def filter_features(features: pd.DataFrame,
                    keep_patterns: Sequence[str] = KEEP_PATTERNS,
                    exclude_patterns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Keep only the feature columns worth carrying forward.

    Two steps: keep columns matching *keep_patterns*, then drop any of
    those matching one of *exclude_patterns*.

    Args:
        features: DataFrame to filter
        keep_patterns: Unanchored regular expressions; a column is a
                       feature when one of them matches its name.
                       Matched as written, since these name the columns
                       of one screen.
        exclude_patterns: Unanchored regular expressions, matched
                          case-insensitively because they describe a
                          naming convention rather than one screen's
                          columns; ``None`` uses :data:`EXCLUDE_PATTERNS`

    Returns:
        The DataFrame with only the retained columns.

    Raises:
        ValueError: if a pattern is not a valid regular expression
        NoFeaturesError: if no column survives the selection
    """
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS

    original_columns = len(features.columns)

    # Unanchored on purpose: re.search is what the upstream cellmapp code
    # used, and anchoring would drop columns currently kept.  Callers
    # wanting a true prefix can write '^cell_'.
    _compile_each(keep_patterns, 'feature_patterns')
    matched = features.filter(regex='|'.join(keep_patterns))
    logger.debug(f"After pattern filtering: {len(matched.columns)} columns")

    kept, removed_by = matched, {}
    for pattern in _compile_each(exclude_patterns, 'exclude_patterns',
                                 re.IGNORECASE):
        before = len(kept.columns)
        kept = kept[[col for col in kept.columns if not pattern.search(col)]]
        if len(kept.columns) < before:
            removed_by[pattern.pattern] = before - len(kept.columns)
        logger.debug(f"After removing '{pattern.pattern}': "
                     f"{len(kept.columns)} columns")

    if not len(kept.columns):
        raise NoFeaturesError(_no_features_message(
            features, matched, removed_by, keep_patterns, exclude_patterns))

    logger.info(f"Feature filtering: {original_columns} -> {len(kept.columns)} columns")
    return kept


def _no_features_message(features: pd.DataFrame,
                         matched: pd.DataFrame,
                         removed_by: dict,
                         keep_patterns: Sequence[str],
                         exclude_patterns: Sequence[str]) -> str:
    """Say which half of the selection emptied the table."""
    if not len(matched.columns):
        sample = ', '.join(str(col) for col in features.columns[:5])
        if len(features.columns) > 5:
            sample += f", and {len(features.columns) - 5} more"
        return (f"No feature columns: none of the {len(features.columns)} "
                f"columns match any of the patterns "
                f"{', '.join(keep_patterns)}. Columns seen: "
                f"{sample or 'none'}.")

    return (f"No feature columns: the patterns matched "
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
