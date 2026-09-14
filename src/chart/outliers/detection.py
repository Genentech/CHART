"""
Finding outlier cells, one gene at a time.

Cells carrying guides against the same gene should behave alike, so the
detection is run per gene rather than over the screen as a whole: a cell is
an outlier relative to the other cells perturbed the same way.
"""

import logging
from typing import List, Union

import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

#: A gene needs at least this many cells before the forest means anything.
MIN_CELLS_PER_GENE = 2

#: The index level that defines a group.
GROUP_LEVEL = 'Gene'


def detect_outliers_per_gene(data: pd.DataFrame,
                             contamination: Union[str, float] = 'auto',
                             random_state: int = 42,
                             n_estimators: int = 100) -> pd.DataFrame:
    """Flag outlier cells within each gene with an isolation forest.

    Args:
        data: The prefiltered feature table, indexed by cell
        contamination: ``'auto'``, or the proportion to flag
        random_state: Seed, so a rerun flags the same cells
        n_estimators: Trees per forest

    Returns:
        One row per outlier cell, with a column for each index level of
        *data*.  Empty, but with those columns, when nothing is flagged.
    """
    if GROUP_LEVEL not in (data.index.names or ()):
        raise KeyError(f"Cannot detect outliers per gene: the table has no "
                       f"'{GROUP_LEVEL}' index level, only "
                       f"{', '.join(str(n) for n in data.index.names or ())}")

    logger.info(f"Detecting outliers per gene: contamination={contamination}, "
                f"n_estimators={n_estimators}, random_state={random_state}")

    flagged: List[pd.DataFrame] = []
    too_small, clean = [], 0

    # One pass over the groups.  cellmapp rescanned the whole table for each
    # gene, which is the same answer at many times the cost.
    for gene, gene_data in data.groupby(level=GROUP_LEVEL, sort=False):
        if len(gene_data) < MIN_CELLS_PER_GENE:
            too_small.append(str(gene))
            continue

        forest = IsolationForest(n_estimators=n_estimators,
                                 contamination=contamination,
                                 random_state=random_state)
        # -1 marks an outlier, 1 an inlier.
        predictions = forest.fit_predict(gene_data)
        outliers = gene_data.index[predictions == -1]

        if len(outliers):
            flagged.append(outliers.to_frame(index=False))
            logger.info(f"Gene {gene}: {len(outliers)} of {len(gene_data)} "
                        f"cells flagged")
        else:
            clean += 1

    if too_small:
        logger.warning(f"Skipped {len(too_small)} gene(s) with fewer than "
                       f"{MIN_CELLS_PER_GENE} cells: {', '.join(too_small)}")
    if clean:
        logger.info(f"{clean} gene(s) had no outliers")

    if not flagged:
        logger.warning("No outliers detected in any gene")
        return pd.DataFrame(columns=list(data.index.names))

    outlier_cells = pd.concat(flagged, axis=0, ignore_index=True)
    logger.info(f"Outliers detected: {len(outlier_cells)} of {len(data)} cells "
                f"({len(outlier_cells) / len(data) * 100:.2f}%)")
    return outlier_cells


def apply_outlier_filters(data: pd.DataFrame,
                          outlier_cells: pd.DataFrame) -> pd.DataFrame:
    """Remove the flagged cells from *data*."""
    if len(outlier_cells) == 0:
        logger.info("No outliers to remove")
        return data

    filtered = data.drop(pd.MultiIndex.from_frame(outlier_cells))
    logger.info(f"Removed {len(data) - len(filtered)} outlier cells: "
                f"{len(filtered)} remaining of {len(data)}")
    if len(filtered) == 0:
        logger.warning("Every cell was flagged as an outlier; check the "
                       "contamination setting")
    return filtered
