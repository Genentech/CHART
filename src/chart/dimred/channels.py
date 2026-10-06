"""
One channel's own feature space, and the PCA the archetypes start from.

A channel's columns are chosen by removing the others, not by looking for
its own name.  That is what drops the colocalisation features, which name
two channels and belong to neither: keeping TOM20 removes every column
mentioning any other channel, and a TOM20/DAPI1 correlation mentions
DAPI1.  The geometry measures describe a shape rather than a stain, so
they name no channel, survive that removal and have to go by name.
"""

import logging
from typing import List, Optional, Sequence

import pandas as pd

from ..io.config import SchemaConfig
from ..io.tables import save_table, table_columns
from .pca import center_on_controls, generate_pca_space

logger = logging.getLogger(__name__)

#: What identifies a cell, and what it was perturbed with.  The archetype
#: scripts read every column but these four as components, and expect
#: them last and in this order.
INDEX_COLUMNS = ('Well', 'Label', 'Guide', 'Gene')

#: What the file each channel's steps start from is called.
PCA_NAME = 'cell_pca_centered'

#: Geometry measures, dropped because they describe the shape of a cell
#: rather than anything the channel stained.
GEOMETRY_PATTERNS = (
    '_areaconvex', '_axismajorlength', '_axisminorlength', '_compactness',
    '_eccentricity', '_eulernumber', '_extent', '_feretdiameter-min',
    '_feretdiameter-max', '_formfactor', '_humoments_', '_orientation',
    '_perimeter', '_max-radius', '_median-radius', '_mean-radius',
    '_solidity')


class NoFeaturesError(ValueError):
    """Nothing in the table belongs to the channel asked for."""


def channel_features(columns: Sequence[str], channel: str,
                     all_channels: Sequence[str]) -> List[str]:
    """The columns making up *channel*'s space, in the order given.

    Args:
        columns: Every column in the feature table
        channel: The channel to keep
        all_channels: Every channel the table holds, which is what the
                      others are removed from

    Returns:
        The feature columns belonging to *channel* alone.

    Raises:
        ValueError: if *channel* is not one of *all_channels*
        NoFeaturesError: if no column survives
    """
    if channel not in all_channels:
        raise NoFeaturesError(
            f"'{channel}' is not one of the channels in the table "
            f"({', '.join(all_channels) or 'none named'}). A channel's "
            f"columns are found by removing the other channels', so a name "
            f"missing from that list would remove nothing and keep "
            f"everything.")

    others = [name for name in all_channels if name != channel]

    # Matching is on any part of the name, as it was in the pipeline this
    # comes from.  It would take one channel named inside another to go
    # wrong, which is worth knowing when channels are added.
    kept = [column for column in columns
            if column not in INDEX_COLUMNS
            and not any(other in column for other in others)
            and not any(shape in column for shape in GEOMETRY_PATTERNS)]

    if not kept:
        raise NoFeaturesError(
            f"None of the {len(columns)} columns belong to {channel} alone. "
            f"Every one of them mentions another channel or measures a "
            f"shape, which would leave the analysis nothing to work from.")

    return kept


def write_channel_pca(features_path: str, channel: str,
                      all_channels: Sequence[str], output_dir: str,
                      schema: Optional[SchemaConfig] = None) -> str:
    """Build *channel*'s centred PCA space where its steps look for it.

    Args:
        features_path: The guide-filtered cell table
        channel: The channel to build the space for
        all_channels: Every channel *features_path* holds
        output_dir: That channel's directory
        schema: Which guides count as controls

    Returns:
        The path written.
    """
    columns = table_columns(features_path)

    missing = [name for name in INDEX_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            f"{features_path} has no {' or '.join(missing)} column, so its "
            f"cells cannot be named or centred on the controls. Columns "
            f"found: {', '.join(columns[:4])}, and "
            f"{max(len(columns) - 4, 0)} more.")

    features = channel_features(columns, channel, all_channels)
    logger.info(f"{channel}: {len(features)} of {len(columns)} columns")

    data = pd.read_parquet(features_path,
                           columns=features + list(INDEX_COLUMNS))
    space = center_on_controls(
        generate_pca_space(data.set_index(list(INDEX_COLUMNS))), schema)

    table = space.reset_index()
    table = table[[name for name in table.columns
                   if name not in INDEX_COLUMNS] + list(INDEX_COLUMNS)]
    return save_table(table, output_dir, PCA_NAME)
