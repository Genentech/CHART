"""
Each archetype's most heavily weighted cells, cropped out of the images.

A diagnostic, for judging an archetype and its K by eye.  The layout is
the published figures': the stain in inverted grey at a display range set
per well, the cell outlined in red and its nucleus in blue, five cells to
a row and one PNG per row.  The annotation reads the rows by their names,
``archetype<N>_<channel>_grid_<row>.png``.
"""

import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..combining.workflow import OUTPUT_NAME, SUFFIXES
from ..dimred.channels import INDEX_COLUMNS
from ..io.config import (ARCHETYPES_SECTION, SCHEMA_SECTION, SECTION_NAME,
                         ArchetypesConfig, BywellConfig, join_path,
                         load_allwells_config, load_bywell_config)
from ..io.tables import load_table, save_table

try:
    from scallops.io import read_experiment
except ImportError:
    read_experiment = None

logger = logging.getLogger(__name__)

#: Where a channel's crops go, under its reports directory.
CROPS_DIR = 'crops'

#: The table of the cells shown, written beside the grids.
TOP_CELLS_NAME = 'top_cells'

#: Cells to a row, which is what the annotation reads.
CELLS_PER_GRID = 5

#: Pixels shown around each cell's bounding box.
BBOX_PAD = 8

#: The percentiles of a well's pixels a stain is shown between, unless
#: ``display_percentiles`` says otherwise for its channel.
DEFAULT_PERCENTILES = (0.0, 99.5)

#: The percentiles are taken over five patches this many pixels a side,
#: one central and one per quadrant, inside a margin of this fraction of
#: the image on every side.
PATCH_SIZE = 5000
MARGIN_FRAC = 0.10


class CropsNotSetUpError(RuntimeError):
    """The crops step was asked for a channel it has no image settings for."""


def top_cells(weights: pd.DataFrame, objects: pd.DataFrame, n: int,
              bbox_columns: Sequence[str]) -> pd.DataFrame:
    """The *n* most heavily weighted cells of each archetype, with their
    bounding boxes.

    Args:
        weights: One channel's weights, as the weights step wrote them
        objects: The across-wells objects table, indexed by cell and well
        n: How many cells per archetype
        bbox_columns: Where *objects* holds the bounding boxes

    Returns:
        One row per archetype and rank, in the order the archetypes come.
    """
    missing = [name for name in INDEX_COLUMNS if name not in weights.columns]
    if missing:
        raise ValueError(f"The weights have no {' or '.join(missing)} column")

    ranked = []
    for column in [c for c in weights.columns if c not in INDEX_COLUMNS]:
        top = weights.nlargest(n, column)
        ranked.append(pd.DataFrame({
            'Archetype': column,
            'Rank': np.arange(1, len(top) + 1),
            'Weight': top[column].to_numpy(),
            **{name: top[name].to_numpy() for name in INDEX_COLUMNS}}))
    return with_boxes(pd.concat(ranked, ignore_index=True), objects,
                      bbox_columns)


def with_boxes(cells: pd.DataFrame, objects: pd.DataFrame,
               bbox_columns: Sequence[str]) -> pd.DataFrame:
    """*cells*, which has ``Well`` and ``Label`` columns, with each cell's
    bounding box from the objects table."""
    missing = [name for name in bbox_columns if name not in objects.columns]
    if missing:
        raise ValueError(f"The objects table has no {', '.join(missing)} "
                         f"column, which '{SCHEMA_SECTION}.bbox_columns' "
                         f"names as holding the bounding boxes")

    boxes = pd.DataFrame({'Well': objects.index.get_level_values('Well'),
                          'Label': objects.index.get_level_values('Label'),
                          **{name: objects[name].to_numpy()
                             for name in bbox_columns}})
    cells = cells.merge(boxes, on=['Well', 'Label'], how='left',
                        validate='many_to_one')

    unboxed = cells[list(bbox_columns)].isna().any(axis=1)
    if unboxed.any():
        first = cells[unboxed].iloc[0]
        raise ValueError(f"{unboxed.sum()} of the cells to crop have no "
                         f"bounding box in the objects table, the first being "
                         f"cell {first['Label']} of well {first['Well']}; the "
                         f"table may come from another run than the weights")
    return cells


def crop_bounds(box: Sequence[float],
                shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """The first row and column of a cell's crop and one past the last,
    padded by :data:`BBOX_PAD` and kept inside an image of *shape*."""
    y0, x0, y1, x1 = (int(value) for value in box)
    return (max(y0 - BBOX_PAD, 0), max(x0 - BBOX_PAD, 0),
            min(y1 + BBOX_PAD, shape[0]), min(x1 + BBOX_PAD, shape[1]))


def display_range(image: Any, percentiles: Sequence[float]) -> Tuple[float, float]:
    """The pixel values a well's stain is shown between: two percentiles
    of its pixels in five patches spread over the image."""
    height, width = image.shape
    my, mx = int(MARGIN_FRAC * height), int(MARGIN_FRAC * width)
    sy, sx = height - 2 * my, width - 2 * mx
    origins = [(my + sy // 2, mx + sx // 2),
               (my + sy // 4, mx + sx // 4),
               (my + sy // 4, mx + 3 * sx // 4),
               (my + 3 * sy // 4, mx + sx // 4),
               (my + 3 * sy // 4, mx + 3 * sx // 4)]
    pixels = np.concatenate([
        np.asarray(image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]).ravel()
        for y, x in origins])
    low, high = np.percentile(pixels, percentiles)
    return float(low), float(high)


def draw_cell(ax, image: np.ndarray, cell: np.ndarray, nucleus: np.ndarray,
              display: Sequence[float], title: str) -> None:
    """One panel: a cell's crop, with its outline and its nucleus's."""
    ax.imshow(image, cmap='gray_r', vmin=display[0], vmax=display[1])
    for mask, colour in ((cell, 'red'), (nucleus, 'blue')):
        if mask.any():
            ax.contour(mask, colors=colour, linewidths=3, levels=[0.5],
                       alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # 'pt' as in the published figures, though these are pixels.
    ax.set_xlabel(f'{image.shape[1]} pt', fontsize=18)
    ax.set_ylabel(f'{image.shape[0]} pt', fontsize=18)
    ax.set_title(title, fontsize=8)


def write_grids(cells: pd.DataFrame, wells: Dict[str, Tuple[Any, ...]],
                channel: str, bbox_columns: Sequence[str],
                output_dir: str) -> List[str]:
    """Draw every archetype's cells in rows of :data:`CELLS_PER_GRID`.

    Args:
        cells: What :func:`top_cells` returned
        wells: For each well, its stain, cell labels, nucleus labels and
               display range, the images as planes of rows by columns
        channel: Which channel, for naming the files
        bbox_columns: Where *cells* holds the bounding boxes
        output_dir: Where to write the PNGs

    Returns:
        The filenames written.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    unoutlined = 0
    for archetype, rows in cells.groupby('Archetype', sort=False):
        number = archetype[len('Archetype'):]
        rows = rows.sort_values('Rank')
        for start in range(0, len(rows), CELLS_PER_GRID):
            group = rows.iloc[start:start + CELLS_PER_GRID]
            fig, axes = plt.subplots(1, len(group),
                                     figsize=(5 * len(group), 5))
            for ax, (_, row) in zip(np.atleast_1d(axes), group.iterrows()):
                stain, cell_labels, nuclei_labels, display = wells[row['Well']]
                y0, x0, y1, x1 = crop_bounds(row[list(bbox_columns)],
                                             stain.shape)
                cell = np.asarray(cell_labels[y0:y1, x0:x1]) == row['Label']
                if not cell.any():
                    unoutlined += 1
                draw_cell(ax, np.asarray(stain[y0:y1, x0:x1]), cell,
                          np.asarray(nuclei_labels[y0:y1, x0:x1]) == row['Label'],
                          display,
                          f"Archetype {number} | weight: {row['Weight']:.3f} | "
                          f"Well: {row['Well']} | Cell ID: {row['Label']}")
            fig.tight_layout()
            name = (f'archetype{number}_{channel}_grid_'
                    f'{start // CELLS_PER_GRID + 1}.png')
            fig.savefig(os.path.join(output_dir, name), dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            written.append(name)

    if unoutlined:
        logger.warning(f"{unoutlined} of the {len(cells)} cells shown for "
                       f"{channel} have no outline in the cell label image "
                       f"at their bounding box, so the label images may not "
                       f"be the segmentation the objects table came from")
    return written


def _plane(path: str, kind: str, key: str, index: Any = None) -> Any:
    """One image or label image from the zarr at *path*, as a lazily read
    plane of rows by columns."""
    experiment = read_experiment(path, dask=True, group_by=None)
    group = getattr(experiment, kind)
    if key not in group:
        raise FileNotFoundError(f"No {kind}/{key} in {path}")
    plane = group[key]
    if index is not None:
        plane = plane.isel(c=index)
    plane = plane.squeeze()
    if plane.ndim != 2:
        raise ValueError(f"{kind}/{key} in {path} has {plane.ndim} dimensions "
                         f"once the single ones are dropped, not rows and "
                         f"columns alone")
    return plane


def _open_well(bywell: BywellConfig, well: str, index: int) -> Tuple[Any, ...]:
    """A well's stain, cell labels and nucleus labels."""
    settings = bywell.well(well)
    paths = {}
    for key in ('pheno_dir', 'cell_labels_dir', 'nuclei_labels_dir'):
        if not getattr(settings, key):
            raise ValueError(f"No '{key}' is configured for well {well} under "
                             f"'{SECTION_NAME}', and the crops step needs it "
                             f"to show that well's cells")
        paths[key] = join_path(bywell.scallops_dir, getattr(settings, key))
    return (_plane(paths['pheno_dir'], 'images', well, index),
            _plane(paths['cell_labels_dir'], 'labels', f'{well}-cell'),
            _plane(paths['nuclei_labels_dir'], 'labels', f'{well}-nuclei'))


def write_channel_crops(source: Any, archetypes: ArchetypesConfig,
                        channel: str, requested: bool = False) -> List[str]:
    """Crop *channel*'s top cells for every archetype, and write the grids
    with a ``top_cells.parquet`` listing them.

    Args:
        source: The configuration file or mapping, for the image locations
                and the objects table, which the preprocessing sections name
        archetypes: The archetypes section, as loaded from *source*
        channel: Which channel
        requested: Whether the step was asked for by name.  A channel it
                   cannot run for is then an error rather than a warning.

    Returns:
        The filenames written, or nothing when the step was skipped.
    """
    reason = None
    if channel not in archetypes.image_channels:
        reason = (f"'{ARCHETYPES_SECTION}.image_channels' does not say which "
                  f"channel of the images holds {channel}")
    elif read_experiment is None:
        reason = "scallops, which reads the images, is not installed"
    if reason:
        if requested:
            raise CropsNotSetUpError(f"Cannot crop {channel}'s cells: {reason}")
        logger.warning(f"Skipping the crops for {channel}: {reason}")
        return []

    bywell = load_bywell_config(source)
    objects = load_table(load_allwells_config(source).combined_path,
                         OUTPUT_NAME, SUFFIXES['objects'])
    path = archetypes.weights_path(channel)
    logger.info(f"Reading {path}")
    bbox_columns = archetypes.schema.bbox_columns
    cells = top_cells(pd.read_parquet(path), objects, archetypes.top_cells,
                      bbox_columns)

    output_dir = join_path(archetypes.report_path(channel), CROPS_DIR)
    save_table(cells, output_dir, TOP_CELLS_NAME)

    percentiles = archetypes.display_percentiles.get(channel,
                                                     DEFAULT_PERCENTILES)
    fixed = archetypes.display_ranges.get(channel, {})
    wells = {}
    for well in cells['Well'].unique():
        stain, cell_labels, nuclei_labels = _open_well(
            bywell, well, archetypes.image_channels[channel])
        if well in fixed:
            display = tuple(fixed[well])
            logger.info(f"{channel} in {well}: shown from {display[0]:g} to "
                        f"{display[1]:g}, as configured")
        else:
            display = display_range(stain, percentiles)
            logger.info(f"{channel} in {well}: shown from {display[0]:g} to "
                        f"{display[1]:g}, its {percentiles[0]:g} and "
                        f"{percentiles[1]:g} percentiles")
        wells[well] = (stain, cell_labels, nuclei_labels, display)

    written = write_grids(cells, wells, channel, bbox_columns, output_dir)
    logger.info(f"Wrote {len(written)} grids for {channel} to {output_dir}")
    return written
