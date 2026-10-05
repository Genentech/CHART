"""
Stacking per-well tables into one table.

Each well's rows keep their identity through the index, extended with a
``Well`` level so that rows from different wells stay distinguishable.
"""

import logging
from typing import Dict

import pandas as pd

from ..io.config import INDEX_LEVELS

logger = logging.getLogger(__name__)

LEVELS = tuple(INDEX_LEVELS)


def reindex_for_level(data: pd.DataFrame, well: str, level: str) -> pd.DataFrame:
    """Rebuild *data*'s index for *level*, tagging the rows with *well*.

    Args:
        data: A table indexed by ``(Label, Guide, Gene)``
        well: The well the rows came from
        level: Which levels to keep; see :data:`INDEX_LEVELS`

    Returns:
        *data* with its index rebuilt.  Row count and columns are unchanged.
    """
    if level not in INDEX_LEVELS:
        raise ValueError(f"Unknown level: {level}. "
                         f"Available levels: {', '.join(LEVELS)}")

    names = INDEX_LEVELS[level]
    missing = [name for name in names if name not in (data.index.names or ())]
    if missing:
        raise KeyError(f"Table for well {well} has no index level(s) "
                       f"{', '.join(missing)}; expected an index of "
                       f"{', '.join(INDEX_LEVELS['cell'])}")

    data = data.copy()
    if 'Well' in data.columns:
        # The well becomes an index level, so a column of the same name
        # would be a confusing duplicate.  cellmapp overwrote and dropped it.
        logger.info(f"Dropping the existing 'Well' column for well {well}; "
                    f"the well is recorded in the index")
        data = data.drop(columns='Well')

    arrays = [data.index.get_level_values(name) for name in names]
    arrays.append(pd.Index([well] * len(data), name='Well'))
    data.index = pd.MultiIndex.from_arrays(arrays, names=(*names, 'Well'))
    return data


def concat_wells(tables: Dict[str, pd.DataFrame], level: str) -> pd.DataFrame:
    """Stack the per-well *tables* into one, reindexed for *level*.

    Only the columns present in every well survive.
    """
    if not tables:
        raise ValueError("No tables to combine")

    wells_with_column: Dict[str, int] = {}
    for table in tables.values():
        for column in table.columns:
            wells_with_column[column] = wells_with_column.get(column, 0) + 1

    dropped = sorted(col for col, count in wells_with_column.items()
                     if count < len(tables))
    if dropped:
        logger.warning(f"Dropping {len(dropped)} column(s) that not every well "
                       f"has: {', '.join(dropped)}")

    reindexed = [reindex_for_level(table, well, level)
                 for well, table in tables.items()]
    combined = pd.concat(reindexed, join='inner')

    logger.info(f"Combined {len(tables)} wells at {level} level: "
                f"{combined.shape[0]} objects, {combined.shape[1]} columns")
    if level != 'cell' and combined.index.has_duplicates:
        # Expected at these levels, since no aggregation happens; say so,
        # because a duplicated index makes lookups ambiguous downstream.
        logger.warning(f"The combined index has repeated entries: '{level}' level "
                       f"drops the levels that told the rows apart, and rows are "
                       f"not aggregated. Rows remain one per cell")
    return combined
