"""
Row selection for filtering.
"""

import logging
from functools import reduce
from typing import List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INDEX_NAMES = ('Label', 'Guide', 'Gene')


def arrange_index(df: pd.DataFrame) -> pd.DataFrame:
    """Move label, guide and gene identity from columns into the index.

    Requires the ``label``, ``sgRNA`` and ``gene_symbol`` columns.
    """
    missing = [col for col in ('label', 'sgRNA', 'gene_symbol')
               if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot build the {'/'.join(INDEX_NAMES)} index: "
                       f"missing column(s) {', '.join(missing)}")

    df = df.copy()
    df.insert(0, "Label", df['label'])
    df.insert(0, "Guide", df['sgRNA'])
    df.insert(0, "Gene", df['gene_symbol'])
    df.set_index(list(INDEX_NAMES), inplace=True)
    return df


def drop_unassigned(data: pd.DataFrame) -> pd.DataFrame:
    """Drop cells with no barcode or gene assignment."""
    before = len(data)
    data = data[data.index.get_level_values('Guide').notna() &
                data.index.get_level_values('Gene').notna()]
    dropped = before - len(data)
    if dropped:
        logger.info(f"Dropped {dropped}/{before} unassigned cells "
                    f"({dropped/before*100:.1f}% had no Guide/Gene)")
    return data


def apply_filters(data: pd.DataFrame,
                  inclusion_filters: Sequence[pd.Series],
                  exclusion_filters: Sequence[pd.Series]) -> pd.DataFrame:
    """Keep the cells that pass every inclusion filter and no exclusion one.
    """
    if not inclusion_filters and not exclusion_filters:
        logger.warning("No filters available, returning the data unfiltered")
        return data

    labels = set(data.index.get_level_values('Label').unique())

    if inclusion_filters:
        included = reduce(np.intersect1d, inclusion_filters)
        labels &= set(included)
        logger.info(f"Inclusion filters: {len(included)} labels pass all of them")

    for exclusion_filter in exclusion_filters:
        excluded = set(exclusion_filter)
        labels -= excluded
        logger.info(f"Exclusion filter: removed {len(excluded)} labels")

    filtered = data[data.index.get_level_values('Label').isin(labels)]
    logger.info(f"Filtered data: {len(filtered)} objects remaining, from {len(data)}")
    if len(filtered) == 0 and len(data) > 0:
        logger.warning("Every object was filtered out; check that the filter "
                       "labels refer to the same objects as this table")
    return filtered
