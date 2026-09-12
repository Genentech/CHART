"""
Collapsing rows to one per guide or per gene.
"""

import logging

import pandas as pd

from .wells import INDEX_LEVELS

logger = logging.getLogger(__name__)

#: Levels that collapse rows.  ``cell`` is absent because the data is
#: already one row per cell, so aggregating there would do nothing.  Note
#: that ``Well`` is not among the grouping levels, so wells are pooled.
AGGREGATION_LEVELS = ('guide', 'gene')

#: How to combine the rows of a group.
METHODS = ('median', 'mean', 'std')


def aggregate(data: pd.DataFrame, level: str,
              method: str = 'median') -> pd.DataFrame:
    """Group *data* down to one row per *level*.

    Args:
        data: A table indexed by at least the levels *level* needs
        level: One of :data:`AGGREGATION_LEVELS`
        method: One of :data:`METHODS`

    Returns:
        The aggregated table, indexed by the levels *level* keeps.
    """
    if level not in AGGREGATION_LEVELS:
        raise ValueError(f"Cannot aggregate at level {level!r}. "
                         f"Available levels: {', '.join(AGGREGATION_LEVELS)}")
    if method not in METHODS:
        raise ValueError(f"Unknown aggregation method {method!r}. "
                         f"Available methods: {', '.join(METHODS)}")

    levels = list(INDEX_LEVELS[level])
    missing = [name for name in levels if name not in (data.index.names or ())]
    if missing:
        raise KeyError(f"Cannot aggregate at {level} level: the table has no "
                       f"index level(s) {', '.join(missing)}")

    aggregated = getattr(data.groupby(level=levels), method)()
    logger.info(f"Aggregated to {level} level by {method}: "
                f"{len(data)} rows -> {len(aggregated)}")
    return aggregated
