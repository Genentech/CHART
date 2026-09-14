"""
Plots describing which cells were flagged as outliers.
"""

import logging
import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

#: Needed to place a cell within its well.
CENTROID_COLUMNS = ('nuclei_centroid-0', 'nuclei_centroid-1')


def _save(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _counts_by_level(data: pd.DataFrame, outliers: pd.DataFrame, level: str):
    """Cells per group, before and after, aligned on the groups present."""
    before = data.index.get_level_values(level).value_counts()
    after = outliers[level].value_counts().reindex(before.index, fill_value=0)
    return before, after


def plot_outliers_by_well(data: pd.DataFrame, outliers: pd.DataFrame,
                          output_dir: str) -> None:
    """Bar chart of the percentage of cells flagged in each well."""
    before, after = _counts_by_level(data, outliers, 'Well')
    percentage = (after / before * 100).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x=[str(w) for w in percentage.index], height=percentage.values)
    ax.set_ylabel('% cells considered outliers')
    ax.set_title('Outlier percentage by well')
    ax.tick_params(axis='x', rotation=45)
    _save(fig, output_dir, 'outliers_by_well.png')


def plot_outliers_by_group(data: pd.DataFrame, outliers: pd.DataFrame,
                           level: str, output_dir: str) -> None:
    """Scatter of cells per group against outliers per group."""
    before, after = _counts_by_level(data, outliers, level)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(before.values, after.values, s=1)
    ax.set_xlabel('Cells in the prefiltered data')
    ax.set_ylabel('Cells flagged as outliers')
    ax.set_title(f'# cells per {level.lower()}')
    _save(fig, output_dir, f'outliers_by_{level.lower()}.png')


def plot_outlier_statistics(data: pd.DataFrame, outliers: pd.DataFrame,
                            output_dir: str) -> None:
    """Draw the by-well, by-gene and by-guide summaries.
    """
    if len(outliers) == 0:
        logger.info("No outliers to plot")
        return

    logger.info("Plotting outlier statistics...")
    plot_outliers_by_well(data, outliers, output_dir)
    for level in ('Gene', 'Guide'):
        if level in (data.index.names or ()):
            plot_outliers_by_group(data, outliers, level, output_dir)


def plot_outlier_positions(outliers: pd.DataFrame, cell_data: pd.DataFrame,
                           output_dir: str,
                           wells: Optional[Sequence[str]] = None,
                           figsize: tuple = (7.5, 7),
                           point_size: float = 0.5) -> None:
    """Plot where the flagged cells sit within each well.

    Args:
        outliers: One row per outlier cell
        cell_data: The combined objects table, for centroid coordinates
        output_dir: Where to write the plots
        wells: Wells to draw; the default is every well with an outlier
        figsize: Size of each figure
        point_size: Scatter point size
    """
    if len(outliers) == 0:
        logger.info("No outlier positions to plot")
        return

    missing = [c for c in CENTROID_COLUMNS if c not in cell_data.columns]
    if missing:
        logger.warning(f"Cannot plot outlier positions: the objects table has "
                       f"no {', '.join(missing)} column(s)")
        return

    if wells is None:
        wells = sorted(outliers['Well'].unique())

    logger.info(f"Plotting outlier positions for {len(wells)} well(s)...")
    y_column, x_column = CENTROID_COLUMNS

    for well in wells:
        in_well = cell_data[cell_data.index.get_level_values('Well') == well]
        if in_well.empty:
            logger.warning(f"No cells for well {well} in the objects table")
            continue

        flagged_labels = set(outliers.loc[outliers['Well'] == well, 'Label'])
        is_outlier = in_well.index.get_level_values('Label').isin(flagged_labels)

        fig, ax = plt.subplots(figsize=figsize)
        ax.invert_yaxis()
        ax.scatter(in_well.loc[~is_outlier, x_column],
                   in_well.loc[~is_outlier, y_column],
                   s=point_size, c='lightgrey', label='retained')
        ax.scatter(in_well.loc[is_outlier, x_column],
                   in_well.loc[is_outlier, y_column],
                   s=point_size, c='crimson', label='outlier')
        ax.set_title(f'Outlier cells - {well} '
                     f'({int(is_outlier.sum())} of {len(in_well)})')
        ax.legend(markerscale=10)
        _save(fig, output_dir, f'outlier_cells_well_{well}.png')
